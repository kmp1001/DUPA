import torch
import copy
import timm
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from src.utils.no_grad import no_grad
from typing import Callable, Iterator, Tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler

def inverse_sigma(alpha, sigma):
    return 1/sigma**2
def snr(alpha, sigma):
    return alpha/sigma
def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, min=threshold)
def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, max=threshold)
def constant(alpha, sigma):
    return 1

import math
import torch

@torch.no_grad()  # Only for the final statistics no_grad; the gradient itself is still calculated by autograd.grad.
def _safe_item(x):
    return None if x is None else float(x)

def grad_dir_stats(
    net,
    loss_a,
    loss_b,
    wrt,  # block index: int
    name_a="A",
    name_b="B",
    eps=1e-12,
):
    """
    Calculate the relationship between the gradients of two losses in the same layer (net.blocks[wrt]):`
    Returns (cos_sim, angle_deg, norm_a, norm_b)
    - cos_sim: <0 indicates adversarial; =0 indicates orthogonal; >0 indicates in the same direction
    - angle_deg: >90° indicates mutual obstruction of updates
    If a loss has no gradient for this block, return None.
    """
    block = net.blocks[wrt]
    ps = [p for p in block.parameters() if p.requires_grad]
    if len(ps) == 0:
        return None

    # Calculate the gradient for each parameter (tuple, length = len(ps))
    ga = torch.autograd.grad(loss_a, ps, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, ps, retain_graph=True, allow_unused=True)

    # Flatten all the gradients of the parameters of this block and concatenate them into a large vector.
    vec_a = []
    vec_b = []
    has_a = False
    has_b = False

    for p, g1, g2 in zip(ps, ga, gb):
        # Fill None with 0 to ensure consistent dimensions; simultaneously record whether a gradient actually exists.
        if g1 is None:
            g1 = torch.zeros_like(p)
        else:
            has_a = True
        if g2 is None:
            g2 = torch.zeros_like(p)
        else:
            has_b = True

        vec_a.append(g1.detach().float().reshape(-1))
        vec_b.append(g2.detach().float().reshape(-1))

    if (not has_a) or (not has_b):
        # A certain loss does not produce any gradient for this block.
        return None

    g_a = torch.cat(vec_a)
    g_b = torch.cat(vec_b)

    na = torch.linalg.norm(g_a).clamp_min(eps)
    nb = torch.linalg.norm(g_b).clamp_min(eps)

    cos_sim = torch.dot(g_a, g_b) / (na * nb)
    cos_clamped = cos_sim.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle_deg = torch.acos(cos_clamped) * (180.0 / math.pi)

    return float(cos_sim), float(angle_deg), float(na), float(nb)


def grad_dir_stats1(loss_a, loss_b, wrt1, wrt2, name_a="A", name_b="B", eps=1e-12):
    """
    Returns (cos_sim, angle_deg, norm_a, norm_b), or None if a loss does not depend on wrt.
    """
    assert wrt1.requires_grad, "`wrt1` must require grad (do NOT detach it)."
    assert wrt2.requires_grad, "`wrt2` must require grad (do NOT detach it)."
    g_a = torch.autograd.grad(loss_a, wrt1, retain_graph=True, create_graph=False, allow_unused=True)[0]
    g_b = torch.autograd.grad(loss_b, wrt2, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if g_a is None or g_b is None:
        return None

    g_a = g_a.detach()
    g_b = g_b.detach()

    # Transform it into a "direction vector": if there is a batch dimension, first perform a mean operation on the batch.
    if g_a.dim() >= 2:
        # print(f"g_a.shape[1]:{g_a.shape[1]}",g_a.shape[1])
        g_a = g_a.reshape(g_a.shape[0], -1).mean(dim=0)
        g_b = g_b.reshape(g_b.shape[0], -1).mean(dim=0)
    else:
        g_a = g_a.reshape(-1)
        g_b = g_b.reshape(-1)

    na = torch.linalg.norm(g_a).clamp_min(eps)
    nb = torch.linalg.norm(g_b).clamp_min(eps)
    cos_sim = (g_a @ g_b) / (na * nb)
    cos_clamped = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    angle_deg = torch.acos(cos_clamped) * (180.0 / math.pi)

    return cos_sim.item(), angle_deg.item(), na.item(), nb.item()

def grad_norm_params(loss, params, retain_graph=True):
    grads = torch.autograd.grad(
        loss, params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True
    )
    sq = 0.0
    for g in grads:
        if g is None:
            continue
        sq += g.detach().pow(2).sum()
    return sq.sqrt()


def grad_norm_wrt(
    net,
    loss,
    block_num: int,
    retain_graph: bool = True,
    eps: float = 1e-12,
    mode: str = "rms",  # "rms" | "mean_abs" | "l2"
):
    """
    Calculates the gradient strength of the loss over net.blocks[block_num].
    Returns a scalar tensor (detached), or None if the loss does not depend on this block.
    mode:
    - "rms": sqrt(sum(g^2) / numel) # Recommended: Parameter size independent, fairer comparison of different blocks
    - "mean_abs": sum(|g|) / numel
    - "l2": sqrt(sum(g^2)) # Global L2, larger value for blocks with more parameters
    """
    block = net.blocks[block_num]
    ps = [p for p in block.parameters() if p.requires_grad]
    if len(ps) == 0:
        return None

    grads = torch.autograd.grad(
        loss, ps,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True
    )

    # 聚合所有参数梯度
    sum_sq = None
    sum_abs = None
    total_numel = 0
    has_any = False

    for p, g in zip(ps, grads):
        if g is None:
            continue
        has_any = True
        gf = g.detach().float()
        if sum_sq is None:
            sum_sq = gf.pow(2).sum()
            sum_abs = gf.abs().sum()
        else:
            sum_sq = sum_sq + gf.pow(2).sum()
            sum_abs = sum_abs + gf.abs().sum()
        total_numel += gf.numel()

    if not has_any:
        return None

    if mode == "l2":
        return torch.sqrt(sum_sq).cpu()
    elif mode == "rms":
        return torch.sqrt(sum_sq / max(total_numel, 1)).clamp_min(eps).cpu()
    elif mode == "mean_abs":
        return (sum_abs / max(total_numel, 1)).clamp_min(eps).cpu()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def disp_loss(z): # Dispersive Loss implementation (InfoNCE-L2 variant)
    z = z.reshape((z.shape[0],-1)) # flatten
    diff = torch.nn.functional.pdist(z).pow(2)/z.shape[1] # pairwise distance
    diff = torch.concat((diff, diff, torch.zeros(z.shape[0]).cuda()))  # match JAX implementation of full BxB matrix
    return torch.log(torch.exp(-diff).mean()) # calculate loss

#
def info_nce_l2_loss(condition, condition2, temperature=1.0, fp32_logits=True):
    dual_condition = torch.cat([condition,condition2],dim=0)
    dual_condition = dual_condition.to(condition.device)
    if fp32_logits:
        dual_condition = dual_condition.float()
    labels = torch.cat([torch.arange(condition.shape[0]) for i in range(2)],dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(dual_condition.device)
    dual_condition = dual_condition.reshape(dual_condition.shape[0],-1)
    dual_condition = F.normalize(dual_condition.float(), dim=1)
    # similarity_matrix = torch.matmul(dual_condition, dual_condition.T)
    # Note that cdist(x, x) does not only calculate x[i] - x[i], but calculates all combinations:
    dist2 = torch.cdist(dual_condition, dual_condition, p=2).pow(2)   # (N,N)
    # dist2 = dist2/dual_condition.shape[1] 
    similarity_matrix = -dist2
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(dual_condition.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(dual_condition.device) 

    return F.cross_entropy(logits, labels)    


def info_nce_cosine_loss(condition, condition2, temperature=1.0, fp32_logits=True):
    dual_condition = torch.cat([condition,condition2],dim=0)
    dual_condition = dual_condition.to(condition.device)
    if fp32_logits:
        dual_condition = dual_condition.float()
    labels = torch.cat([torch.arange(condition.shape[0]) for i in range(2)],dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(dual_condition.device)
    dual_condition = dual_condition.reshape(dual_condition.shape[0],-1)
    dual_condition = F.normalize(dual_condition.float(), dim=1)
    similarity_matrix = torch.matmul(dual_condition, dual_condition.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(dual_condition.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(dual_condition.device) 

    return F.cross_entropy(logits, labels)

class BN1dTokens(nn.Module):
    def __init__(self, d, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(d, affine=affine)

    def forward(self, x):              # x: [B, N, D] or [B, D]
        if x.dim() == 3:
            B, N, D = x.shape
            y = self.bn(x.reshape(B*N, D)).reshape(B, N, D)
            return y
        return self.bn(x)


class DINOv2(nn.Module):
    def __init__(self, weight_path:str):
        super(DINOv2, self).__init__()
        # need to be revised for dino weight
        self.encoder = torch.hub.load('facebookresearch_dinov2_main', 'dinov2_vitb14', source="local", pretrained=True)

        # Extract positional encoding. DINOv2 originally had a fixed resolution of 224×224, but our raw images are 256×256.
        # The number of patches has changed → pos embedding must be resampled.

        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        # Remove the classification header, as we only need the feature extraction part
        # Image
        # → Patch Embed
        # → Transformer blocks
        # → x_norm_patch tokens (features for each patch)
        # → head (classification/projection header)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_size
        self.precomputed_pos_embed = dict()

    def fetch_pos(self, h, w):
        # The same patch grid (e.g., 16×16) will appear repeatedly during training, so caching speeds up the process.
        key = (h, w)
        if key in self.precomputed_pos_embed:
            return self.precomputed_pos_embed[key]
        # Reshape the original 2D position-encoded mesh to [H0, W0, C],
        # Then use 2D interpolation (usually bilinear) to transform it to [h, w, C],
        # and then flatten it back to [h*w, C].
        value = timm.layers.pos_embed.resample_abs_pos_embed(
            self.pos_embed.data, [h, w],
        )
        self.precomputed_pos_embed[key] = value
        return value

    def forward(self, x, encoder_depth=None,encoder_depth_structual=None):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)

        # Resample to 224 size
        # DINOv2 is the ViT encoder; the "semantic meaning" of the patch token depends on its absolute position within the patch mesh.
        # Maintain geometric scale: map the "image in 256 space" to the "coordinate system in 224 space".

        x = torch.nn.functional.interpolate(x, (int(224*h/256), int(224*w/256)), mode='bicubic')
        b, c, h, w = x.shape
        # patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        patch_num_h, patch_num_w = h//self.patch_size, w//self.patch_size
        pos_embed_data = self.fetch_pos(patch_num_h, patch_num_w)
        self.encoder.pos_embed.data = pos_embed_data
        # shape: [B, N, C]
        # without attn logits
        # feature = self.encoder.forward_features(x)['x_norm_patchtokens']

        # with attn logits
        z = self.encoder.forward_features_attn(x,depth=encoder_depth,encoder_structual_depth=encoder_depth_structual)

        # note:
        # z['x_norm_patchtokens'].shape [2, 256, 768]
        # print("DINOv2 forward attn_act shape:", z['attn_act'].shape)

        # use following function to extract multiple features
        # z = self.encoder.forward_features_list(x,encoder_list=encoder_list)
        logits = z['attn_act']
        feature = z['x_norm_patchtokens']
        structual_feature = z['x_norm_structual_patchtokens']
        return feature,logits,structual_feature
    
class REPATrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            feat_loss_weight: float=0.5,
            disp_loss_weight: float=0.5,           
            lognorm_t=False,
            encoder_weight_path=None,
            align_layer=8,
            log_interval=2500,
            encoder_anchor=2,
            decoder_anchor=8,
            encoder_depth=9,
            encoder_depth_structual=7,
            proj_denoiser_dim=256,
            proj_hidden_dim=256,
            proj_encoder_dim=256,
            hidden_dim=288,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.encoder_depth = encoder_depth
        self.encoder_depth_structual = encoder_depth_structual
        self.feat_loss_weight = feat_loss_weight
        self.disp_loss_weight = disp_loss_weight
        self.align_layer = align_layer
        self.log_interval = log_interval
        self.encoder_anchor = encoder_anchor
        self.decoder_anchor = decoder_anchor
        self.encoder = DINOv2(encoder_weight_path)
        self.proj_encoder_dim = proj_encoder_dim
        no_grad(self.encoder)


        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )

        self.predictor = nn.Sequential(
            nn.Linear(proj_hidden_dim, 4*hidden_dim),
            # BN1dTokens(hidden_dim, affine=True),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, proj_hidden_dim),)
        self._init_projector(self.predictor, xavier_gain=1.0)

    
    @staticmethod
    def _init_projector(proj: nn.Module, xavier_gain: float = 0.5) -> None:
        """
        Initialize projector z_phi:
          - First Linear: Kaiming (He) init to preserve variance.
          - Subsequent Linears: reduced-gain Xavier init to stabilize gradients / avoid overfitting.
          - Bias: zeros.
        Works even if proj is nested Sequential.
        """
        # Collect Linear layers in forward order
        linears = []
        for m in proj.modules():
            if isinstance(m, nn.Linear):
                linears.append(m)

        if len(linears) == 0:
            return

        # 1) First Linear: Kaiming uniform (He et al., 2015)
        first = linears[0]
        # For SiLU, PyTorch doesn't have a dedicated nonlinearity option; relu is a common stable proxy
        nn.init.kaiming_uniform_(first.weight, a=0.0, nonlinearity="relu")
        if first.bias is not None:
            nn.init.zeros_(first.bias)
            # nn.init.kaiming_uniform_(first.bias, a=0.0, nonlinearity="relu")

        # 2) Remaining Linears: reduced-gain Xavier (Glorot & Bengio, 2010)
        for lin in linears[1:]:
            nn.init.xavier_uniform_(lin.weight, gain=xavier_gain)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
                # nn.init.xavier_uniform_(lin.bias, gain=xavier_gain)

    def _impl_trainstep(self, net, ema_net, raw_images, x, y, step):
        batch_size, c, height, width = x.shape
        if self.lognorm_t:
            base_t = torch.randn((batch_size), device=x.device, dtype=x.dtype).sigmoid()
            base_t2 = torch.randn((batch_size), device=x.device, dtype=x.dtype).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=x.dtype)
            base_t2 = torch.rand((batch_size), device=x.device, dtype=x.dtype)
        delta = (torch.rand((batch_size), device=x.device, dtype=x.dtype) * 0.2) - 0.1
        base_t2 = (base_t + delta).clamp_(0.0, 1.0)
        t = base_t
        t2 = base_t2

       # Todo: for simplicity, not just list the two branch code, instead, give a for cycle to bring many branches together to improve code readability.

        noise = torch.randn_like(x)
        noise2 = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        alpha2 = self.scheduler.alpha(t2)
        dalpha = self.scheduler.dalpha(t)
        dalpha2 = self.scheduler.dalpha(t2)
        sigma = self.scheduler.sigma(t)
        sigma2 = self.scheduler.sigma(t2)
        dsigma = self.scheduler.dsigma(t)
        dsigma2 = self.scheduler.dsigma(t2)

        x_t = alpha * x + noise * sigma
        x_t2 = alpha2 * x + noise2 * sigma2
        v_t = dalpha * x + dsigma * noise


        # comprise sit features (hooked)
        src_feature = []

        # You can set hook like following ways
        # def initial_hook(net, input, output):
        #     init_feature.append(output[0])
        # def after_hook(net,input, output):
        #     after_feature.append(output[0])

        def forward_hook(net, input, output):
            # print("output shape at align layer:", output.shape)
            src_feature.append(output) # not output[0], because output is x, not (x, attn_logits)

        if getattr(net, "blocks", None) is not None:
            handle = net.blocks[self.align_layer - 1].register_forward_hook(forward_hook)
        else:
            handle = net.encoder.blocks[self.align_layer - 1].register_forward_hook(forward_hook)

        out, _ , condition = net(x_t, t, y)
        _, _, condition2 = net(x_t2, t2, y)

        src_feature = self.proj(src_feature[0])

        handle.remove()

        #********** delete predictor, resume if needed (DUPA ours originally) **********
        p1 = self.predictor(condition)
        p2 = self.predictor(condition2)


        with torch.no_grad():
            dst_feature,_, _ = self.encoder(raw_images, encoder_depth=self.encoder_depth,encoder_depth_structual=self.encoder_depth_structual)
            # Are the number of patches in DINO the same as the number of tokens in the diffusion intermediate layer?
            if dst_feature.shape[1] != src_feature.shape[1]:
                dst_length = dst_feature.shape[1]
                # rescale_ratio = sqrt(N_src / N_dst)
                rescale_ratio = (src_feature.shape[1] / dst_feature.shape[1])**0.5
                dst_height = (dst_length)**0.5 * (height/width)**0.5
                dst_width =  (dst_length)**0.5 * (width/height)**0.5
                dst_feature = dst_feature.view(batch_size, int(dst_height), int(dst_width), self.proj_encoder_dim)
                dst_feature = dst_feature.permute(0, 3, 1, 2)
                # H_out ≈ H_dst * rescale_ratio
                # W_out ≈ W_dst * rescale_ratio
                # The dst feature is spatially resized to the resolution of the src feature.
                dst_feature = torch.nn.functional.interpolate(dst_feature, scale_factor=rescale_ratio, mode='bilinear', align_corners=False)
                dst_feature = dst_feature.permute(0, 2, 3, 1)
                # (B, N_src, C)
                dst_feature = dst_feature.view(batch_size, -1, self.proj_encoder_dim)

        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_feature, dim=-1)
        cos_loss = 1 - cos_sim

        # positive and negative version loss
        dispersive_loss = info_nce_l2_loss(p1, p2)

        # pure negative loss
        # dispersive_loss = disp_loss(condition)

        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight*(out - v_t)**2

        fm_whole_loss = fm_loss.mean()

        # log the gradient for analysis
        if (step+1) % self.log_interval  == 0:
            outputs = {}
            params = [p for p in net.parameters() if p.requires_grad]

            pairs = [
                ("fm", "cos", fm_loss.mean(), cos_loss.mean()),
                ("fm", "align", fm_loss.mean(), dispersive_loss.mean()),
                ("cos", "align", cos_loss.mean(), dispersive_loss.mean()),
            ]

            lines = []
            lines.append(f"[step {step}]")

            for na, nb, la, lb in pairs:
                out = grad_dir_stats(net, la, lb, self.encoder_anchor, na, nb)
                if out is None:
                    lines.append(f"[gradcos] {na} vs {nb}: None (one loss does not depend on wrt)")
                else:
                    cosv, ang, norm_a, norm_b = out
                    lines.append(
                        f"[gradcos] {na} vs {nb}: cos={cosv:+.4f}, ang={ang:6.2f}deg, "
                        f"|g_{na}|={norm_a:.3e}, |g_{nb}|={norm_b:.3e}"
                    )


            # set different pairs if possible

            # for na, nb, la, lb in pairs1:
            #     out = grad_dir_stats(la, lb, after, na, nb)
            #     if out is None:
            #         lines.append(f"[gradcos] {na} vs {nb}: None (one loss does not depend on wrt)")
            #     else:
            #         cosv, ang, norm_a, norm_b = out
            #         lines.append(
            #             f"[gradcos] {na} vs {nb}: cos={cosv:+.4f}, ang={ang:6.2f}deg, "
            #             f"|g_{na}|={norm_a:.3e}, |g_{nb}|={norm_b:.3e}"
                    # )

            # gn_pos_p    = grad_norm_wrt(net, align_loss.mean(), self.encoder_anchor)
            gn_align_p = grad_norm_wrt(net, dispersive_loss.mean(), self.encoder_anchor)
            gn_fm_p     = grad_norm_wrt(net, fm_loss.mean(), self.encoder_anchor)
            gn_cos_p     = grad_norm_wrt(net, cos_loss.mean(), self.encoder_anchor)

            gn_fm_p1     = grad_norm_wrt(net, fm_loss.mean(), self.decoder_anchor)

            # whole_pos_p    = grad_norm_params(align_loss.mean(), params)
            whole_align_p = grad_norm_params(dispersive_loss.mean(), params)
            whole_fm_p     = grad_norm_params(fm_loss.mean(), params)
            whole_cos_p     = grad_norm_params(cos_loss.mean(), params)

            lines.append("grad_norm 3rd block core:")
            lines.append(f"  cos_loss: {gn_cos_p}")
            lines.append(f"  dispersive_loss: {gn_align_p}")
            lines.append(f"  fm_loss: {gn_fm_p}")

            lines.append("grad_norm 8th block core:")
            lines.append(f"  fm_loss: {gn_fm_p1}")

            lines.append("grad_norm whole model params:")
            lines.append(f"  cos_loss: {whole_cos_p}")
            lines.append(f"  dispersive_loss: {whole_align_p}")
            lines.append(f"  fm_loss: {whole_fm_p}")
            lines.append("")  # 空行分隔
            lines.append(f"  fm_loss_value: {fm_loss.mean()}")
            lines.append(f"  dispersive_loss: {dispersive_loss.mean()}")
            lines.append(f"  cos_loss_value: {cos_loss.mean()}")



            outputs["grad_log_text"] = "\n".join(lines)
        else:
            outputs = None


        out = dict(
            fm_loss=fm_whole_loss.mean(),
            dispersive_loss= dispersive_loss.mean(), # Todo
            cos_loss=cos_loss.mean(),
            # dispersive_loss=dispersive_loss.mean(),
            # loss=fm_whole_loss.mean() + self.feat_loss_weight*dispersive_loss.mean()+self.disp_loss_weight *dispersive_loss.mean(),
            loss=fm_whole_loss.mean() + self.feat_loss_weight*cos_loss.mean() + self.disp_loss_weight *dispersive_loss.mean(),
        )
        return out, outputs

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars)
        self.predictor.state_dict(
            destination=destination,
            prefix=prefix + "predictor.",
            keep_vars=keep_vars)
