import math
import torch
import copy
import timm
from torch.nn import Parameter
import torch.nn.functional as F
from src.utils.no_grad import no_grad
from typing import Callable, Iterator, Tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler
from src.utils.cknna_metric import cknna, prepare_features
import torch.distributed as dist
from torchvision.ops import MLP
import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as Fv2
from torchvision.transforms import InterpolationMode
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


import torch
import torch.nn.functional as F
import math
import math
import torch

@torch.no_grad()  # 只对最后的统计 no_grad；梯度本身仍由 autograd.grad 计算
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
    计算两个损失在同一层（net.blocks[wrt]）参数梯度的关系：
    返回 (cos_sim, angle_deg, norm_a, norm_b)
    - cos_sim: <0 表示对抗；=0 正交；>0 同向
    - angle_deg: >90° 表示更新相互阻碍
    若某个 loss 对该 block 完全无梯度，则返回 None。
    """
    block = net.blocks[wrt]
    ps = [p for p in block.parameters() if p.requires_grad]
    if len(ps) == 0:
        return None

    # 对每个参数分别求梯度（tuple，长度=len(ps)）
    ga = torch.autograd.grad(loss_a, ps, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, ps, retain_graph=True, allow_unused=True)

    # 把该 block 的所有参数梯度 flatten 后拼成一个大向量
    vec_a = []
    vec_b = []
    has_a = False
    has_b = False

    for p, g1, g2 in zip(ps, ga, gb):
        # 用 0 填充 None，确保维度一致；同时记录是否“真的有梯度”
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
        # 某个 loss 对该 block 完全不产生梯度
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
    返回 (cos_sim, angle_deg, norm_a, norm_b)，若某个 loss 不依赖 wrt 则返回 None。
    """
    assert wrt1.requires_grad, "`wrt1` must require grad (do NOT detach it)."
    assert wrt2.requires_grad, "`wrt2` must require grad (do NOT detach it)."
    g_a = torch.autograd.grad(loss_a, wrt1, retain_graph=True, create_graph=False, allow_unused=True)[0]
    g_b = torch.autograd.grad(loss_b, wrt2, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if g_a is None or g_b is None:
        return None

    g_a = g_a.detach()
    g_b = g_b.detach()
    # g_a: (B, N, C), 计算每一个patch的平均梯度（数值），接下来对这个tensor（所有patch的平均梯度）计算相似度。

    # print(f"g_a.dim:{g_a.dim}",g_a.dim)
    # print(f"g_a.shape[0]:{g_a.shape[0]}",g_a.shape[0])

    # 变成一个“方向向量”：若有 batch 维就先对 batch 做 mean
    if g_a.dim() >= 2:
        # print(f"g_a.shape[1]:{g_a.shape[1]}",g_a.shape[1])
        g_a = g_a.reshape(g_a.shape[0], -1).mean(dim=0)
        g_b = g_b.reshape(g_b.shape[0], -1).mean(dim=0)
    else:
        g_a = g_a.reshape(-1)
        g_b = g_b.reshape(-1)

    na = torch.linalg.norm(g_a).clamp_min(eps)
    nb = torch.linalg.norm(g_b).clamp_min(eps)
    cos_sim = (g_a @ g_b) / (na * nb) # 除掉范数
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

# def grad_norm_wrt(loss, x, retain_graph=True):
#     g = torch.autograd.grad(
#         loss, x,
#         retain_graph=retain_graph,
#         create_graph=False,
#         allow_unused=True
#     )[0]
#     if g is None:
#         return None
#     return g.detach().norm(p=2)


def grad_norm_wrt(
    net,
    loss,
    block_num: int,
    retain_graph: bool = True,
    eps: float = 1e-12,
    mode: str = "rms",  # "rms" | "mean_abs" | "l2"
):
    """
    计算 loss 对 net.blocks[block_num] 的“梯度强度”统计。
    返回一个标量 tensor（detach 后），若该 loss 不依赖该 block 则返回 None。

    mode:
      - "rms": sqrt(sum(g^2) / numel)   # 推荐：参数规模无关，更公平比较不同 block
      - "mean_abs": sum(|g|) / numel
      - "l2": sqrt(sum(g^2))           # 全局L2，参数多的block值更大
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

class DINOv2(nn.Module):
    def __init__(self, weight_path:str):
        super(DINOv2, self).__init__()
        # 正常训练请修改一下
        self.encoder = torch.hub.load('/home/v100-03/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitb14', source="local", pretrained=True)
        # self.encoder =  torch.hub.load('/home/v100-03/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitb14_reg', source="local", pretrained=True)
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_size
        self.precomputed_pos_embed = dict()

    def fetch_pos(self, h, w):
        key = (h, w)
        if key in self.precomputed_pos_embed:
            return self.precomputed_pos_embed[key]
        value = timm.layers.pos_embed.resample_abs_pos_embed(
            self.pos_embed.data, [h, w],
        )
        self.precomputed_pos_embed[key] = value
        return value

    def forward(self, x, encoder_depth=None, encoder_depth_structual=None):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, (int(224*h/256), int(224*w/256)), mode='bicubic')
        b, c, h, w = x.shape
        # patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        patch_num_h, patch_num_w = h//self.patch_size, w//self.patch_size
        pos_embed_data = self.fetch_pos(patch_num_h, patch_num_w)
        self.encoder.pos_embed.data = pos_embed_data
        # feature = self.encoder.forward_features(x)['x_norm_patchtokens']

        z = self.encoder.forward_features_attn(x, depth=encoder_depth, encoder_structual_depth=encoder_depth_structual)

        # z = self.encoder.forward_features_attn(x,depth=encoder_depth,encoder_structual_depth=encoder_depth_structual)

        # z = self.encoder.forward_features_list(x,encoder_list=encoder_list)

        # z = self.encoder.forward_feature_and_salient_token(x)

        # z['x_norm_patchtokens'].shape [2, 256, 768]
        # print("DINOv2 forward attn_act shape:", z['attn_act'].shape)
        logits = z['attn_act'] # useless
        feature = z['x_norm_patchtokens']
        # salient_mask = z['salient_patch_mask']
        # structual_feature = z['x_norm_structual_patchtokens']
        # return feature,logits,structual_feature
        return feature, logits

        
# class sigreg(torch.nn.Module):
#     def __init__(self, knots=17, t_max=3.0, num_proj=256):
#         super().__init__()
#         t = torch.linspace(0, t_max, knots, dtype=torch.float32)
#         dt = t_max / (knots - 1)

#         weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
#         weights[[0, -1]] = dt

#         window = torch.exp(-t.square() / 2.0)

#         self.num_proj = num_proj
#         self.register_buffer("t", t)
#         self.register_buffer("phi", window)
#         self.register_buffer("weights", weights * window)

#     def forward(self, proj):
#         # proj: [V, B, D]
#         A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
#         A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8)

#         x_t = (proj @ A).unsqueeze(-1) * self.t.to(proj.dtype)   # [V, B, P, K]
#         err = (x_t.cos().mean(-3) - self.phi.to(proj.dtype)).square() + x_t.sin().mean(-3).square()
#         statistic = (err @ self.weights.to(proj.dtype)) * proj.size(-2)
#         return statistic.mean()


# # this is the loss A = var_loss + cov_loss, however, in our setting, we need to adjust weights of these two losses to make optimization towards the proper direction.
# def sigreg_weak_loss(x, sketch_dim=64):
#     """
#     Weak-SIGReg: Forces Covariance(x) ~ Identity via Sketching.
#     Prevents dimensional collapse in the representations.
#     """
#     N, C = x.size()
#     if C > sketch_dim:
#         S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
#         x = x @ S.T
#     else:
#         sketch_dim = C

#     # Centering & Covariance
#     x = x - x.mean(dim=0, keepdim=True)
#     cov = (x.T @ x) / (N - 1 + 1e-6)

#     # Target Identity
#     target = torch.eye(sketch_dim, device=x.device)

#     # Minimize distance to Identity
#     return torch.norm(cov - target, p='fro')

class FullGatherLayer(torch.autograd.Function):
    """
    DDP-safe all_gather with backward support.
    """
    @staticmethod
    def forward(ctx, x):
        if (not dist.is_available()) or (not dist.is_initialized()):
            return x
        outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(outputs, x.contiguous())
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if (not dist.is_available()) or (not dist.is_initialized()):
            return grad_output
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        grads = list(grad_output.chunk(world_size, dim=0))
        grad_local = grads[rank].contiguous()
        dist.all_reduce(grad_local)
        return grad_local


def gather_if_distributed(x: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
    if (not dist.is_available()) or (not dist.is_initialized()):
        return x
    if with_grad:
        return FullGatherLayer.apply(x)
    xs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


def grad_norm_wrt_tensor(loss, x, retain_graph=True, eps=1e-12):
    g = torch.autograd.grad(
        loss,
        x,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )[0]
    if g is None:
        return None
    return g.detach().float().norm(p=2).clamp_min(eps)


def compute_vicreg_var_loss(x, gamma=1.0, eps=1e-4):
    # x: [N, C]
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def compute_vicreg_cov_loss(x):
    # x: [N, C], assume already centered or recenter here
    N, C = x.size()
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)
    return off_diagonal(cov).pow_(2).sum().div(C)

# # this is the loss B strong version
# def sigreg_weak_loss(x, sketch_dim=64):
#     """
#     Forces ECF(x) ~ ECF(Gaussian).
#     Matches ALL Moments (Maximum Entropy Cloud).
#     Exact implementation of LeJEPA Algorithm 1.
#     """
#     N, C = x.size()

#     # 1. Projection (The Observer)
#     # Project channels down to sketch_dim
#     A = torch.randn(C, sketch_dim, device=x.device)
#     A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

#     # 2. Integration Points
#     t = torch.linspace(-5, 5, 17, device=x.device)

#     # 3. Theoretical Gaussian CF
#     exp_f = torch.exp(-0.5 * t**2)

#     # 4. Empirical CF
#     # proj: [N, sketch_dim]
#     proj = x @ A

#     # args: [N, sketch_dim, T]
#     args = proj.unsqueeze(2) * t.view(1, 1, -1)

#     # ecf: [sketch_dim, T] (Mean over batch)
#     ecf = torch.exp(1j * args).mean(dim=0)

#     # 5. Weighted L2 Distance
#     # |ecf - gauss|^2 * gauss_weight
#     diff_sq = (ecf - exp_f.unsqueeze(0)).abs().square()
#     err = diff_sq * exp_f.unsqueeze(0)

#     # 6. Integrate
#     loss = torch.trapz(err, t, dim=1) * N

#     return loss.mean()

import torch
import torch.nn.functional as F
import math


@torch.no_grad()
def _safe_corrcoef(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    """
    x, y: [N]
    返回 Pearson correlation
    """
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm() + eps
    if denom.item() == 0:
        return float("nan")
    return float((x @ y / denom).item())


@torch.no_grad()
def _centered_cov(z: torch.Tensor, eps: float = 1e-6):
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    n = z.shape[0]
    if n <= 1:
        d = z.shape[1]
        return torch.eye(d, device=z.device, dtype=torch.float32)
    cov = (z.T @ z) / (n - 1 + eps)
    return cov.float()


@torch.no_grad()
def _effective_rank_from_eigs(eigs: torch.Tensor, eps: float = 1e-12):
    """
    eigs: [D], non-negative
    """
    eigs = eigs.float().clamp_min(0)
    s = eigs.sum()
    if s.item() <= eps:
        return 0.0
    p = eigs / s
    ent = -(p * (p + eps).log()).sum()
    return float(torch.exp(ent).item())


@torch.no_grad()
def _pairwise_cosine_stats(z1: torch.Tensor, z2: torch.Tensor = None):
    """
    z1: [N, D]
    z2: [N, D] or None
    若 z2 is None: 返回 z1 内部两两 cosine 的 offdiag 统计
    若 z2 is not None:
      - pos_cos: 平均对角线 cosine
      - neg_cos: 平均非对角线 cosine
      - gap = pos - neg
    """
    z1 = F.normalize(z1.float(), dim=-1)
    if z2 is None:
        sim = z1 @ z1.T  # [N, N]
        n = sim.shape[0]
        if n <= 1:
            return {
                "offdiag_mean": float("nan"),
                "offdiag_std": float("nan"),
                "offdiag_max": float("nan"),
            }
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        vals = sim[mask]
        return {
            "offdiag_mean": float(vals.mean().item()),
            "offdiag_std": float(vals.std(unbiased=False).item()),
            "offdiag_max": float(vals.max().item()),
        }

    z2 = F.normalize(z2.float(), dim=-1)
    sim = z1 @ z2.T  # [N, N]
    n = sim.shape[0]
    if n == 0:
        return {
            "pos_cos": float("nan"),
            "neg_cos": float("nan"),
            "gap": float("nan"),
        }
    pos = sim.diag().mean()
    if n == 1:
        neg = torch.tensor(float("nan"), device=sim.device)
    else:
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        neg = sim[mask].mean()
    gap = pos - neg if torch.isfinite(neg) else torch.tensor(float("nan"), device=sim.device)
    return {
        "pos_cos": float(pos.item()),
        "neg_cos": float(neg.item()) if torch.isfinite(neg) else float("nan"),
        "gap": float(gap.item()) if torch.isfinite(gap) else float("nan"),
    }


@torch.no_grad()
def _feature_collapse_stats(
    z: torch.Tensor,
    eps_std: float = 1e-4,
    eig_eps: float = 1e-8,
    cov_jitter: float = 1e-4,
):
    z = z.float()
    n, d = z.shape

    feat_std = z.std(dim=0, unbiased=False)
    std_mean = float(feat_std.mean().item())
    std_min = float(feat_std.min().item())
    dead_dim_ratio = float((feat_std < eps_std).float().mean().item())

    cov = _centered_cov(z).float()
    cov = cov + cov_jitter * torch.eye(d, device=cov.device, dtype=cov.dtype)

    try:
        eigs = torch.linalg.eigvalsh(cov).clamp_min(0)
    except RuntimeError:
        try:
            eigs = torch.linalg.eigvalsh(cov.cpu()).clamp_min(0).to(cov.device)
        except RuntimeError:
            # 实在不行就跳过 eig 相关指标
            cos_stats = _pairwise_cosine_stats(z)
            return {
                "std_mean": std_mean,
                "std_min": std_min,
                "dead_dim_ratio": dead_dim_ratio,
                "effective_rank": float("nan"),
                "lambda1_ratio": float("nan"),
                "offdiag_mean": cos_stats["offdiag_mean"],
                "offdiag_std": cos_stats["offdiag_std"],
                "offdiag_max": cos_stats["offdiag_max"],
            }

    eigs_sorted, _ = torch.sort(eigs, descending=True)
    eig_sum = eigs_sorted.sum()

    if eig_sum.item() <= eig_eps:
        lambda1_ratio = 1.0
        eff_rank = 0.0
    else:
        lambda1_ratio = float((eigs_sorted[0] / eig_sum).item())
        eff_rank = _effective_rank_from_eigs(eigs_sorted)

    cos_stats = _pairwise_cosine_stats(z)

    return {
        "std_mean": std_mean,
        "std_min": std_min,
        "dead_dim_ratio": dead_dim_ratio,
        "effective_rank": eff_rank,
        "lambda1_ratio": lambda1_ratio,
        "offdiag_mean": cos_stats["offdiag_mean"],
        "offdiag_std": cos_stats["offdiag_std"],
        "offdiag_max": cos_stats["offdiag_max"],
    }


@torch.no_grad()
def _pc1_scores(z: torch.Tensor):
    z = z.float()
    zc = z - z.mean(dim=0, keepdim=True)

    try:
        _, _, vh = torch.linalg.svd(zc.float(), full_matrices=False)
        pc1 = vh[0]
        scores = zc @ pc1
    except Exception:
        cov = _centered_cov(zc).float()
        eigs, vecs = torch.linalg.eigh(cov)
        pc1 = vecs[:, -1]
        scores = zc @ pc1

    return scores


@torch.no_grad()
def _shortcut_stats(z: torch.Tensor, t: torch.Tensor = None, extra_scalar: torch.Tensor = None):
    """
    z: [N, D]
    t: [N] 可为 timestep
    extra_scalar: [N] 可为亮度/面积等简单属性
    """
    z = z.float()
    feat_norm = z.norm(dim=-1)
    pc1_scores = _pc1_scores(z)

    out = {
        "feat_norm_mean": float(feat_norm.mean().item()),
        "feat_norm_std": float(feat_norm.std(unbiased=False).item()),
    }

    if t is not None:
        t = t.detach().float().reshape(-1)
        out["corr_pc1_t"] = _safe_corrcoef(pc1_scores, t)
        out["corr_norm_t"] = _safe_corrcoef(feat_norm, t)

    if extra_scalar is not None:
        extra_scalar = extra_scalar.detach().float().reshape(-1)
        out["corr_pc1_extra"] = _safe_corrcoef(pc1_scores, extra_scalar)
        out["corr_norm_extra"] = _safe_corrcoef(feat_norm, extra_scalar)

    return out


@torch.no_grad()
def collect_multiview_diagnostics(
    mid1_mean: torch.Tensor,     # [B, D], proj1 前
    mid2_mean: torch.Tensor,     # [B, D], proj1 前
    p1: torch.Tensor,            # [B, D'], proj1 后
    p2: torch.Tensor,            # [B, D'], proj1 后
    t: torch.Tensor = None,      # [B]
    t1: torch.Tensor = None,     # [B]
    raw_images: torch.Tensor = None,  # [B, C, H, W]
    token_subsample: int = 4096,
):
    """
    返回一个 dict，包含 proj1 前/后的 collapse 指标。
    这里直接接收 mid1_mean，避免在函数内重复 mean。
    """
    diag = {}
    stats_mid1_mean = _feature_collapse_stats(mid1_mean)
    stats_p1 = _feature_collapse_stats(p1)
    mid_cross = _pairwise_cosine_stats(mid1_mean, mid2_mean)
    p_cross = _pairwise_cosine_stats(p1, p2)

    for k, v in stats_mid1_mean.items():
        diag[f"mid1_mean_{k}"] = v
    for k, v in stats_p1.items():
        diag[f"p1_{k}"] = v
    for k, v in mid_cross.items():
        diag[f"mid_img_cross_{k}"] = v
    for k, v in p_cross.items():
        diag[f"p_img_cross_{k}"] = v

    return diag


class REPATrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            patch_size: int = 2, # or patch_size = 2,
            loss_weight_fn:Callable=constant,
            feat_loss_weight: float=0.5,
            attn_loss_weight: float=0.5,
            structual_loss_weight: float=0.5,
            lognorm_t=False,
            encoder_weight_path=None,
            align_layer=[1,2,3,4],
            dino_layer_timestep=[12,10,8,6],
            dino_layer_block=[10,9,8,7],
            encoder_anchor=3,
            decoder_anchor=22,
            proj_denoiser_dim=256,
            proj_hidden_dim=256,
            proj_encoder_dim=256,
            encoder_depth=9,
            encoder_begin=7,
            encoder_end=10,
            log_interval=10,
            decoder_depth=23,
            encoder_depth_structual=7,
            proj_dim = 16,
            # transform_num = 3,
            inv_loss_weight = 0.5,
            sigreg_loss_weight = 0.02,
            cov_loss_weight = 0.01,
            var_loss_weight = 0.25,
            disp_loss_weight = 0.01,
            vicreg_warmup_steps: int = 0,
            vicreg_warmup_start_scale: float = 1.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.patch_size = patch_size
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight
        self.attn_loss_weight = attn_loss_weight
        self.structual_loss_weight = structual_loss_weight
        self.decoder_depth = decoder_depth
        self.encoder_depth_structual=encoder_depth_structual
        self.align_layer = align_layer
        self.dino_layer_timestep = dino_layer_timestep
        self.dino_layer_block = dino_layer_block
        self.encoder_anchor = encoder_anchor
        self.decoder_anchor = decoder_anchor
        self.log_interval = log_interval
        self.encoder = DINOv2(encoder_weight_path)
        self.proj_denoiser_dim = proj_denoiser_dim
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_encoder_dim = proj_encoder_dim
        self.encoder_depth = encoder_depth
        self.encoder_begin = encoder_begin
        self.encoder_end = encoder_end
        self.proj_dim = proj_dim
        self.inv_loss_weight = inv_loss_weight
        self.sigreg_loss_weight = sigreg_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.disp_loss_weight = disp_loss_weight
        self.vicreg_warmup_steps = int(vicreg_warmup_steps)
        self.vicreg_warmup_start_scale = float(vicreg_warmup_start_scale)
        # self.transform_num = transform_num

        # self.proj1 = MLP(proj_denoiser_dim, [proj_hidden_dim, proj_hidden_dim, proj_dim], norm_layer=nn.BatchNorm1d)
        # self.sigreg = sigreg()
        # self.proj = nn.Conv2d(proj_denoiser_dim, proj_dim, kernel_size=3, padding=1)

        self.proj1 = nn.Sequential(
            nn.Linear(proj_denoiser_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(True),
            nn.Linear(proj_hidden_dim, proj_dim, bias=False),
        )

    def _round_up_to_even(self, v: float) -> int:
        """
        向上取整到偶数。
        例如 12.8 -> 14, 13.0 -> 14, 14.0 -> 14
        """
        v = math.ceil(v)
        if v % 2 == 1:
            v += 1
        return max(v, 2)


    def _sample_resized_crop_batch(
        self,
        x: torch.Tensor,
        scale_range: Tuple[float, float],
        out_size: Tuple[int, int],
        ratio_range: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ) -> torch.Tensor:
        """
        对 batch 中每张图单独做 random resized crop，然后 resize 到指定 out_size。
        输入:  x [B, C, H, W]
        输出:  [B, C, out_h, out_w]
        """
        B, C, H, W = x.shape
        out_h, out_w = out_size
        out = []
        area = H * W

        for i in range(B):
            target_area = area * torch.empty(1, device=x.device).uniform_(*scale_range).item()
            aspect = torch.empty(1, device=x.device).uniform_(*ratio_range).item()

            crop_h = int(round(math.sqrt(target_area / aspect)))
            crop_w = int(round(math.sqrt(target_area * aspect)))

            crop_h = max(1, min(crop_h, H))
            crop_w = max(1, min(crop_w, W))

            top = 0 if crop_h == H else int(torch.randint(0, H - crop_h + 1, (1,), device=x.device).item())
            left = 0 if crop_w == W else int(torch.randint(0, W - crop_w + 1, (1,), device=x.device).item())

            cropped = Fv2.resized_crop(
                x[i],
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
                size=[out_h, out_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            out.append(cropped)

        return torch.stack(out, dim=0)


    def _make_multi_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        whole: random 30%-100% area, resize 到原图大小
        part1/part2: two independent random 5%-30% area, resize 到原图 0.4 倍
        且 part 的输出高宽都向上取到偶数，方便后续 patchify。
        """
        B, C, H, W = x.shape

        whole_size = (H, W)

        part_h = self._round_up_to_even(H * 0.4)
        part_w = self._round_up_to_even(W * 0.4)
        part_size = (part_h, part_w)

        whole = self._sample_resized_crop_batch(
            x,
            scale_range=(0.40, 1.00),
            out_size=whole_size,
        )
        part1 = self._sample_resized_crop_batch(
            x,
            scale_range=(0.05, 0.40),
            out_size=part_size,
        )
        part2 = self._sample_resized_crop_batch(
            x,
            scale_range=(0.05, 0.40),
            out_size=part_size,
        )

        return whole, part1, part2

    def _impl_trainstep(self, net, ema_net, raw_images, x, y, step):

        batch_size, c, height, width = x.shape

        # old requirements, not enable now. 
        # x: [batch_size, c, height, width]
        # 按照 height,width (spatial) 做三组，第一组取出(random)30%-100% token; 第二组，第三组  (spatial) 取出 5%-30%（两个random）token
        # 得到 whole, part1, part2
        if self.lognorm_t: # 强调小 t（低噪）
            base_t = torch.randn((batch_size), device=x.device, dtype=x.dtype).sigmoid()
            base_t1 = torch.rand((batch_size), device=x.device, dtype=x.dtype).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=x.dtype)
            base_t1 = torch.rand((batch_size), device=x.device, dtype=x.dtype)
        t = base_t
        t1 = base_t1

        # whole, part1, part2 = self._make_multi_views(x)

        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)

        alpha1 = self.scheduler.alpha(t1)
        dalpha1 = self.scheduler.dalpha(t1)
        sigma1 = self.scheduler.sigma(t1)
        dsigma1 = self.scheduler.dsigma(t1)

        noise2 = torch.randn_like(x)


        # noise1 = torch.randn_like(whole)
        # noise2 = torch.randn_like(part1)
        # noise3 = torch.randn_like(part2)
        # whole_t = alpha * whole + noise1 * sigma
        # part1_t = alpha * part1 + noise2 * sigma
        # part2_t = alpha * part2 + noise3 * sigma

        x_t = alpha * x + noise * sigma
        v_t = dalpha * x + dsigma * noise

        x_t2 = alpha1 * x + noise2 * sigma1

        # src_feature = []
        # def forward_hook(net, input, output):
        #     src_feature.append(output[0])


        # 兼容你原来的写法：layer_id 是 1-based，所以这里 -1
        # if getattr(net, "blocks", None) is not None:
        #     handle = net.blocks[layer_id - 1].register_forward_hook(forward_hook)
        # else:
        #     handle = net.encoder.blocks[layer_id - 1].register_forward_hook(forward_hook)

        # src_feature_dict = {}

        # def make_hook(layer_id):
        #     def forward_hook(module, input, output):
        #         # output[0] follows your original code
        #         src_feature_dict[layer_id] = output[0]
        #     return forward_hook

        # handles = []
        # for net_block_id, _, _ in active_pairs:
        #     if getattr(net, "blocks", None) is not None:
        #         handle = net.blocks[net_block_id - 1].register_forward_hook(make_hook(net_block_id))
        #     else:
        #         handle = net.encoder.blocks[net_block_id - 1].register_forward_hook(make_hook(net_block_id))
        #     handles.append(handle)


        # if getattr(net, "blocks", None) is not None:
        #     handle = net.blocks[self.align_layer[0] - 1].register_forward_hook(forward_hook)
        # else:
        #     handle = net.encoder.blocks[self.align_layer[0] - 1].register_forward_hook(forward_hook)

        # out, _, whole_output, part1_output, part2_output, attn_acts = net(x_t, whole_t, part1_t, part2_t, t, y)
        # interval = getattr(self, "_repa_viz_interval", None)
        # need_viz = (
        #     interval is not None
        #     and int(interval) > 0
        #     and (int(step) % int(interval) == 0)
        #     and int(step) > 0
        # )
        # encode_viz + 返回 enc_viz：避免对 net.module  setattr（与 compile/实际 forward 对象不一致时会导致 enc 列表为空）
        out, _, mid1, mid1_blocks, mid2, _, enc_viz = net(
            x_t, x_t2, t, t1, y, encode_viz=False
        )

        # whole_output, part1_output, part2_output 各自本来是[B,T,D] 要求拼成[B*3, T, D], 然后对T这一维度做pooling, 假设得到的是emb
        # whole_emb = whole_output.mean(dim=1)  # [B, D]
        # part1_emb = part1_output.mean(dim=1)  # [B, D]
        # part2_emb = part2_output.mean(dim=1)  # [B, D]

        # with torch.no_grad():
        #     dst_feature, logits = self.encoder(raw_images, encoder_depth = self.encoder_depth, encoder_depth_structual = self.encoder_depth_structual)

        # if dst_feature.shape[1] != mid1.shape[1]:
        #     dst_length = dst_feature.shape[1]
        #     rescale_ratio = (mid1.shape[1] / dst_feature.shape[1])**0.5
        #     dst_height = (dst_length)**0.5 * (height/width)**0.5
        #     dst_width =  (dst_length)**0.5 * (width/height)**0.5
        #     dst_feature = dst_feature.view(batch_size, int(dst_height), int(dst_width), self.proj_encoder_dim)
        #     dst_feature = dst_feature.permute(0, 3, 1, 2)
        #     dst_feature = torch.nn.functional.interpolate(dst_feature, scale_factor=rescale_ratio, mode='bilinear', align_corners=False)
        #     dst_feature = dst_feature.permute(0, 2, 3, 1)
        #     dst_feature = dst_feature.view(batch_size, -1, self.proj_encoder_dim)

        outputs = {}
        # if need_viz:
        #     topk = int(getattr(self, "_repa_viz_topk", 10))
        #     q = float(getattr(self, "_repa_viz_q", 0.95))
        #     exact = bool(getattr(self, "_repa_viz_exact", True))
        #     device = mid1.device
        #     # ---------- patch 级：N = B*T，整批所有 token 展平后两两对齐（与 compute.py 一致）----------
        #     m1f = mid1.reshape(-1, mid1.shape[-1])
        #     m2f = mid2.reshape(-1, mid2.shape[-1])
        #     df = dst_feature.reshape(-1, dst_feature.shape[-1])
        #     p1 = prepare_features(m1f, device, q=q, exact=exact)
        #     p2 = prepare_features(m2f, device, q=q, exact=exact)
        #     pd = prepare_features(df, device, q=q, exact=exact)
        #     n = min(p1.shape[0], pd.shape[0])
        #     c_m1d_patch = cknna(p1[:n], pd[:n], topk=topk)
        #     n2 = min(p2.shape[0], pd.shape[0])
        #     c_m2d_patch = cknna(p2[:n2], pd[:n2], topk=topk)
        #     # ---------- 图像级：每图对 token 取均值，N = B（需 B>=3 且 topk 合法）----------
        #     c_m1d_img = float("nan")
        #     c_m2d_img = float("nan")
        #     if batch_size >= 3:
        #         m1b = mid1.mean(dim=1)
        #         m2b = mid2.mean(dim=1)
        #         db = dst_feature.mean(dim=1)
        #         p1b = prepare_features(m1b, device, q=q, exact=exact)
        #         p2b = prepare_features(m2b, device, q=q, exact=exact)
        #         pdb = prepare_features(db, device, q=q, exact=exact)
        #         nb = min(p1b.shape[0], pdb.shape[0])
        #         if nb >= 3:
        #             tk = min(topk, nb - 1)
        #             if tk >= 2:
        #                 c_m1d_img = cknna(p1b[:nb], pdb[:nb], topk=tk)
        #                 c_m2d_img = cknna(p2b[:nb], pdb[:nb], topk=tk)
        #     enc_mid1 = []
        #     enc_mid2 = []
        #     if enc_viz is not None:
        #         vblocks1 = enc_viz.get("mid1") or []
        #         vblocks2 = enc_viz.get("mid2") or []
        #     else:
        #         vblocks1, vblocks2 = [], []
        #     ns = min(2, batch_size)
        #     for block_slots in vblocks1:
        #         enc_mid1.append([t[:ns].detach().cpu().float() for t in block_slots])
        #     for block_slots in vblocks2:
        #         enc_mid2.append([t[:ns].detach().cpu().float() for t in block_slots])
        #     outputs["repa_viz"] = {
        #         "step": int(step),
        #         "cknna_mid1_dino_patch": float(c_m1d_patch),
        #         "cknna_mid2_dino_patch": float(c_m2d_patch),
        #         "cknna_mid1_dino_image_mean": float(c_m1d_img),
        #         "cknna_mid2_dino_image_mean": float(c_m2d_img),
        #         "encoder_mid1": enc_mid1,
        #         "encoder_mid2": enc_mid2,
        #         "dino": dst_feature[:ns].detach().cpu().float(),
        #     }


        # whether mid1,mid2 collapse?
        # need a monitor..
        #  
        # 
        mid1_tokens = mid1
        mid2_tokens = mid2

        # B,T,C = src_feature[0].shape
        # H = W = int(math.isqrt(T))
        # assert H * W == T, f"conv projector needs square grid or pass hw; got T={T}"
        # x_map = src_feature[0].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # y = self.proj1(x_map)
        # src_feature = y.permute(0, 2, 3, 1).contiguous().reshape(B, T, -1)

        # similarity_loss = 1 - F.cosine_similarity(mid1_tokens, mid2_tokens.detach(), dim=-1)

        # reg_loss += ... 在 PyTorch 里通常等价于 in-place add，也就是直接修改原来的那个 tensor，而不是新建一个新的 tensor
        # mid1_flat = mid1_tokens.reshape(-1, mid1_tokens.shape[-1])
        # # mid2_flat = mid2_tokens.reshape(-1, mid2_tokens.shape[-1])
        # reg_loss_tok = sigreg_weak_loss(mid1_flat)  # token-level


        mid1_mean = mid1_tokens.mean(dim=1)          # [B, D]
        mid2_mean = mid2_tokens.mean(dim=1)

        p1 = self.proj1(mid1_mean)          # [B, proj_dim]
        p2 = self.proj1(mid2_mean)

        # Todo: 现在是做了mean，因为disp_loss事实上要对每个样本做，里面讲N D两个维度共同作为第二维度，从而做disp_loss.
        # 然而，之前所有的实现都是：首先对token维度取均值，因此不需要上面这一步，从逻辑上讲这个涉及到交叉熵损失中向量的长度大小。在plan B中未观察到异常。

        # disp = 0.5 * (disp_loss(mid1_mean) + disp_loss(mid2_mean))
        # disp1 = 0.5 * (disp_loss(mid1_mean1) + disp_loss(mid2_mean1))
        disp = p1.new_zeros(())

        p1_all = gather_if_distributed(p1, with_grad=True)
        p2_all = gather_if_distributed(p2, with_grad=True)
        
        # similarity_loss = (1 - F.cosine_similarity(mid1_mean, mid2_mean.detach(), dim=-1))
        # similarity_loss1 = (1 - F.cosine_similarity(mid1_mean1, mid2_mean1.detach(), dim=-1))

        similarity_loss = F.mse_loss(p1, p2)

        p1c = p1_all - p1_all.mean(dim=0, keepdim=True)
        p2c = p2_all - p2_all.mean(dim=0, keepdim=True)
        
        # reg_loss = sigreg_weak_loss(mid1_mean)
        var_loss = 0.5 * (compute_vicreg_var_loss(p1c) + compute_vicreg_var_loss(p2c))

        cov_loss = compute_vicreg_cov_loss(p1c) + compute_vicreg_cov_loss(p2c)

        # reg_loss1 = sigreg_weak_loss(mid1_mean1)

        # reg_loss = 0.0
        # block_num = len(mid1_blocks)

        # for blk in mid1_blocks:
        #     blk = self.proj1(blk)
        #     blk_mean = blk.mean(dim=1)               # [B, D]
        #     reg_loss = reg_loss + sigreg_weak_loss(blk_mean)
        
        # reg_loss = reg_loss / max(block_num, 1)
        
        # emb_views = torch.stack([whole_emb, part1_emb, part2_emb], dim=0)  # [3, B, D]
        # proj_emb = self.proj1(emb_views.reshape(3 * batch_size, -1)).reshape(3, batch_size, -1)

        # inv_loss = (proj_emb.mean(0) - proj_emb).square().mean()
        # sigreg_loss = self.sigreg(proj_emb)

        # inter_state = src_feature[0]


        # src_feature = self.proj1(src_feature[0])


        # B,T,C = src_feature[0].shape
        # H = W = int(math.isqrt(T))
        # assert H * W == T, f"conv projector needs square grid or pass hw; got T={T}"
        # x_map = src_feature[0].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # y = self.proj1(x_map)
        # src_feature = y.permute(0, 2, 3, 1).contiguous().reshape(B, T, -1)

        # handle.remove()

        # # remove all hooks
        # for handle in handles:
        #     handle.remove()


        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight*(out - v_t)**2
        # dispersive_loss = disp_loss(inter_state)
        # total_loss = fm_loss.mean() + 0.5*(self.inv_loss_weight * similarity_loss.mean() + self.sigreg_loss_weight * reg_loss.mean() + self.disp_loss_weight * disp) + 0.5*(self.inv_loss_weight * similarity_loss1.mean() + self.sigreg_loss_weight * reg_loss1.mean() + self.disp_loss_weight * disp1)

        if self.vicreg_warmup_steps > 0:
            progress = min(float(step) / float(self.vicreg_warmup_steps), 1.0)
            vicreg_scale = self.vicreg_warmup_start_scale + (1.0 - self.vicreg_warmup_start_scale) * progress
        else:
            vicreg_scale = 1.0

        total_loss = (
            fm_loss.mean()
            + vicreg_scale * self.inv_loss_weight * similarity_loss.mean()
            + vicreg_scale * self.var_loss_weight * var_loss.mean()
            + vicreg_scale * self.cov_loss_weight * cov_loss.mean()
        )

        if (step + 1) % self.log_interval == 0:
            outputs.update({
                "step": int(step) + 1,
                "fm_loss": float(fm_loss.mean().detach().item()),
                "similarity_loss": float(similarity_loss.mean().detach().item()),
                "var_loss": float(var_loss.mean().detach().item()),
                "cov_loss": float(cov_loss.mean().detach().item()),
                "vicreg_scale": float(vicreg_scale),
                "disp_loss": float(disp.detach().item()),
                "loss": float(total_loss.detach().item()),
                "loss_log_text": (
                    f"step:{int(step) + 1}, "
                    f"fm_loss:{fm_loss.mean().detach().item():.8f}, "
                    f"similarity_loss:{similarity_loss.mean().detach().item():.8f}, "
                    f"var_loss:{var_loss.mean().detach().item():.8f}, "
                    f"cov_loss:{cov_loss.mean().detach().item():.8f}, "
                    f"vicreg_scale:{vicreg_scale:.4f}, "
                    f"disp_loss:{disp.detach().item():.8f}, "
                    f"loss:{total_loss.detach().item():.8f}"
                ),
            })

            var_grad_p1 = grad_norm_wrt_tensor(var_loss, p1, retain_graph=True)
            cov_grad_p1 = grad_norm_wrt_tensor(cov_loss, p1, retain_graph=True)
            outputs.update({
                "var_grad_norm_p1": float(var_grad_p1.item()) if var_grad_p1 is not None else float("nan"),
                "cov_grad_norm_p1": float(cov_grad_p1.item()) if cov_grad_p1 is not None else float("nan"),
                "p1_all_requires_grad": float(1.0 if p1_all.requires_grad else 0.0),
            })

            # ===== new: collapse / shortcut monitor =====
            diag = collect_multiview_diagnostics(
                mid1_mean=mid1_mean.detach().float(),
                mid2_mean=mid2_mean.detach().float(),
                p1=p1.detach().float(),
                p2=p2.detach().float(),
                t=t.detach().float(),
                t1=t1.detach().float(),
                raw_images=raw_images.detach().float() if raw_images is not None else None,
                token_subsample=4096,
            )
            outputs.update(diag)

            outputs["collapse_log_text"] = (
                f"mid_pos:{diag['mid_img_cross_pos_cos']:.4f}, "
                f"mid_neg:{diag['mid_img_cross_neg_cos']:.4f}, "
                f"mid_gap:{diag['mid_img_cross_gap']:.4f}, "
                f"p_pos:{diag['p_img_cross_pos_cos']:.4f}, "
                f"p_neg:{diag['p_img_cross_neg_cos']:.4f}, "
                f"p_gap:{diag['p_img_cross_gap']:.4f}, "
                f"m1_erank:{diag['mid1_mean_effective_rank']:.2f}, "
                f"p1_erank:{diag['p1_effective_rank']:.2f}, "
                f"m1_l1ratio:{diag['mid1_mean_lambda1_ratio']:.4f}, "
                f"p1_l1ratio:{diag['p1_lambda1_ratio']:.4f}, "
                f"m1_offdiag:{diag['mid1_mean_offdiag_mean']:.4f}, "
                f"p1_offdiag:{diag['p1_offdiag_mean']:.4f}"
            )
        


        out = dict(
            fm_loss=fm_loss.mean(),
            inv_loss = similarity_loss.mean(),
            # sigreg_loss = reg_loss.mean(),
            var_loss = var_loss.mean(),
            cov_loss = cov_loss.mean(),
            # disp_loss = disp,
            # inv_loss1 = similarity_loss1.mean(),
            # sigreg_loss1 = reg_loss1.mean(),
            # disp_loss1 = disp1,
            # cos_loss=cos_loss.mean(),
            # dispersive_loss = dispersive_loss.mean(),
            # mention: just use structural_loss_weight to replace dispersive loss weight
            # loss=fm_loss.mean() + self.feat_loss_weight*cos_loss.mean() + self.structual_loss_weight*dispersive_loss.mean(),
            loss = total_loss,
        )
        return out, outputs

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # pass
        self.proj1.state_dict(
            destination=destination,
            prefix=prefix + "proj1.",
            keep_vars=keep_vars)

