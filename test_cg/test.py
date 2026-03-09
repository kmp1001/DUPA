# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
import torch.nn as nn
import math
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import math
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import copy
from PIL import Image
import timm
from decoupled_sit import SiT_models
from diffusers import AutoencoderKL
from training import *
from scheduling import LinearScheduler
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob

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

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
def make_next_run_dir(root="result_angle"):
    os.makedirs(root, exist_ok=True)
    # 找到 root 下已有的数字目录
    existing = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.isdigit():
            existing.append(int(name))
    next_id = (max(existing) + 1) if existing else 1
    run_dir = os.path.join(root, str(next_id))
    os.makedirs(run_dir, exist_ok=False)  # exist_ok=False 确保不覆盖
    return run_dir

def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")



def grad_dir_stats(
    net,
    loss_a,
    loss_b,
    wrt,   # block index
    eps=1e-12,
):
    """
    Calculate the gradient relationship between the two losses in the same layer (net.blocks[wrt]) across the entire global batch:
    Returns (cos_sim, angle_deg, norm_a, norm_b)

    Method:
    1) Calculate the local gradient vector for each rank.
    2) Apply all_reduce(sum) to the gradient vector.
    3) Calculate cos / angle on the global gradient vector.
    If a loss has no gradient for that block, return None.
    """
    if getattr(net, "blocks", None) is not None:
        block = net.blocks[wrt]
    else:
        block = net.encoder.blocks[wrt]

    ps = [p for p in block.parameters() if p.requires_grad]
    if len(ps) == 0:
        return None

    ga = torch.autograd.grad(loss_a, ps, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, ps, retain_graph=True, allow_unused=True)

    vec_a = []
    vec_b = []
    has_a = False
    has_b = False

    for p, g1, g2 in zip(ps, ga, gb):
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
        return None

    g_a = torch.cat(vec_a)
    g_b = torch.cat(vec_b)

    # Key point: First sum the gradient vector across ranks.
    if dist.is_initialized():
        dist.all_reduce(g_a, op=dist.ReduceOp.SUM)
        dist.all_reduce(g_b, op=dist.ReduceOp.SUM)

    na = torch.linalg.norm(g_a).clamp_min(eps)
    nb = torch.linalg.norm(g_b).clamp_min(eps)

    cos_sim = torch.dot(g_a, g_b) / (na * nb)
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    angle_deg = torch.acos(cos_sim) * (180.0 / math.pi)

    return float(cos_sim.item()), float(angle_deg.item()), float(na.item()), float(nb.item())

def get_block_grad_vector(net, loss, wrt):
    """
    Retrieve the gradient vector of the loss with respect to the block's wrt parameters (local batch of this rank).
    If the loss has no gradient for this block, return None.
    """
    if getattr(net, "blocks", None) is not None:
        block = net.blocks[wrt]
    else:
        block = net.encoder.blocks[wrt]

    ps = [p for p in block.parameters() if p.requires_grad]
    if len(ps) == 0:
        return None

    grads = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)

    vec = []
    has_grad = False
    for p, g in zip(ps, grads):
        if g is None:
            g = torch.zeros_like(p)
        else:
            has_grad = True
        vec.append(g.detach().float().reshape(-1))

    if not has_grad:
        return None

    return torch.cat(vec)


def get_global_grad_vector(net, loss, wrt):
    """
    Returns the global gradient vector of the loss with respect to the block wrt parameters over the entire global batch.
    Method: Local grad -> all_reduce(sum)
    """
    g = get_block_grad_vector(net, loss, wrt)
    if g is None:
        return None

    if dist.is_initialized():
        dist.all_reduce(g, op=dist.ReduceOp.SUM)

    return g

class DINOv2(nn.Module):
    def __init__(self):
        super(DINOv2, self).__init__()

        # need to be revised
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

        # z['x_norm_patchtokens'].shape [2, 256, 768]
        # print("DINOv2 forward attn_act shape:", z['attn_act'].shape)
        logits = z['attn_act']
        feature = z['x_norm_patchtokens']
        structual_feature = z['x_norm_structual_patchtokens']
        return feature,logits,structual_feature
    
def ddp_mean_keepgrad(x: torch.Tensor):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x / dist.get_world_size()



def map_ckpt_to_align_block(ckpt_index: int, total_ckpts: int, gradient_list, total_steps: int = 400_000) -> int:
    """
    1) nodes: step_i = floor((i+1)*T/N), i=0..N-1
    2) segments (boundary -> previous): seg = floor((step_i-1)*G/T)
    3) align_block = gradient_list[seg]
    """
    G = len(gradient_list)
    if G == 0:
        raise ValueError("gradient_list is empty")
    N = total_ckpts
    if N <= 0:
        raise ValueError("total_ckpts must be positive")
    if not (0 <= ckpt_index < N):
        raise ValueError(f"ckpt_index out of range: {ckpt_index}/{N}")

    # node on 1..T
    step_i = ((ckpt_index + 1) * total_steps) // N
    step_i = max(1, min(step_i, total_steps))

    # segment id: 0..G-1 (boundary goes to previous segment)
    seg = ((step_i - 1) * G) // total_steps
    seg = max(0, min(seg, G - 1))

    return int(gradient_list[seg])


def map_ckpt_to_mlp_id(ckpt_index: int, total_ckpts: int, num_MLP: int, total_steps: int = 400_000) -> int:
    """
    Decide which MLP (proj1..projM) a ckpt should use by time-axis binning on steps 1..total_steps.

    - num_MLP <= 1: return 0 (meaning single proj/proj0/proj)
    - num_MLP  > 1: return mlp_id in [1..num_MLP]

    Assumption: ckpts are N evenly spaced nodes on the training step axis (1..total_steps):
        step_i = floor((i+1)*T/N)

    Segment rule (boundary belongs to previous segment):
        seg = floor((step_i-1)*M/T)  in [0..M-1]
        mlp_id = seg + 1
    """
    if num_MLP <= 1:
        return 0
    if total_ckpts <= 0:
        raise ValueError("total_ckpts must be positive")
    if not (0 <= ckpt_index < total_ckpts):
        raise ValueError(f"ckpt_index out of range: {ckpt_index}/{total_ckpts}")

    T = total_steps
    M = num_MLP
    # node step in [1..T]
    step_i = ((ckpt_index + 1) * T) // total_ckpts
    step_i = max(1, min(step_i, T))

    # segment id in [0..M-1], boundary -> previous segment
    seg = ((step_i - 1) * M) // T
    seg = max(0, min(seg, M - 1))

    return int(seg + 1)


def parse_ckpt_epoch_step(path: str):
    """
    从 checkpoint 文件名中提取 epoch 和 step，用于数值排序。
    支持类似：
      epoch=9-step=50000.ckpt
      epoch=19-step=100000.ckpt

    返回:
      (epoch, step, filename)
    若解析失败，返回一个较大的默认值/或兜底键，避免报错。
    """
    name = os.path.basename(path)

    m = re.search(r"epoch=(\d+)-step=(\d+)", name)
    if m is not None:
        epoch = int(m.group(1))
        step = int(m.group(2))
        return (epoch, step, name)

    # 兜底：如果文件名不符合格式，就放到后面，再按名字排
    return (10**18, 10**18, name)


def list_ckpts(ckpt_path: str):
    """
    If ckpt_path is a file -> return [ckpt_path]
    If ckpt_path is a dir  -> return ckpts sorted by (epoch, step), excluding 'last' in filename
    """
    if ckpt_path is None:
        return []

    if os.path.isdir(ckpt_path):
        files = glob(os.path.join(ckpt_path, "*.ckpt"))
        files = [p for p in files if "last" not in os.path.basename(p)]
        files = sorted(files, key=parse_ckpt_epoch_step)
        return files

    return [ckpt_path]

def get_proj_module(model):
    # model: unwrapped module (model.module)
    if not hasattr(model, "proj"):
        raise AttributeError("Model has no attribute `proj` but you said network only uses `proj`.")
    return model.proj

def load_backbone_and_selected_proj(model, ckpt_path: str, ckpt_proj_key: str):
    """
    model: unwrapped module (model.module)
    ckpt_proj_key: 'proj' or 'proj1'/'proj2'/...
                  (this is ONLY the ckpt key name, NOT module name)
    Always load proj weights INTO model.proj.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_all = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # choose ema_denoiser or denoiser
    prefix = "ema_denoiser."
    keys = list(state_all.keys())
    if not any(k.startswith(prefix) for k in keys):
        prefix = "denoiser."
        if not any(k.startswith(prefix) for k in keys):
            raise KeyError(
                f"Cannot find 'ema_denoiser.' or 'denoiser.' keys in ckpt. "
                f"Top example key: {keys[0] if keys else 'EMPTY'}"
            )

    # 1) load backbone (allow missing proj*)
    state = {k[len(prefix):]: v for k, v in state_all.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(state, strict=False)

    bad_missing = [k for k in missing if not k.startswith("proj")]
    if bad_missing:
        raise RuntimeError(
            f"Backbone load incomplete beyond allowed missing prefixes ('proj*').\n"
            f"Example bad missing: {bad_missing[:20]}\nTotal bad missing: {len(bad_missing)}"
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading backbone.\n"
            f"Example unexpected: {unexpected[:20]}\nTotal unexpected: {len(unexpected)}"
        )

    # 2) pick proj weights from ckpt keys, load STRICTLY into model.proj
    # main prefix
    prefix1 = f"diffusion_trainer.{ckpt_proj_key}."
    proj_state = {k[len(prefix1):]: v for k, v in state_all.items() if k.startswith(prefix1)}

    # # compat fallback: sometimes single-MLP is stored as diffusion_trainer.proj. even if ckpt_proj_key='proj0'
    # if len(proj_state) == 0 and ckpt_proj_key in ("proj0",):
    #     alt_prefix = "diffusion_trainer.proj."
    #     proj_state = {k[len(alt_prefix):]: v for k, v in state_all.items() if k.startswith(alt_prefix)}
    #     prefix1 = alt_prefix
    
    # compat fallback 2 (YOUR special case): num_MLP==1 but stored as diffusion_trainer.proj1.*
    if len(proj_state) == 0 and ckpt_proj_key == "proj":
        alt_prefix = "diffusion_trainer.proj1."
        proj_state = {k[len(alt_prefix):]: v for k, v in state_all.items() if k.startswith(alt_prefix)}
        prefix1 = alt_prefix

    if len(proj_state) == 0:
        raise KeyError(
            f"Required {prefix1}* not found in ckpt: {ckpt_path}\n"
            f"Tip: check ckpt contains diffusion_trainer.{ckpt_proj_key}.*"
        )

    # IMPORTANT: always load into model.proj
    model.proj.load_state_dict(proj_state, strict=True)

    print(f"[OK] backbone loaded from {prefix} | proj loaded from diffusion_trainer.{ckpt_proj_key}.* into model.proj")



def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup data (unchanged):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    print(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Labels / noise (unchanged):
    ys = torch.full((local_batch_size,), 1000, device=device, dtype=torch.long)
    n = ys.size(0)

    # Scheduler / teacher (unchanged):
    scheduler = LinearScheduler()
    DINO = DINOv2().to(device).eval()

    # VAE (unchanged):
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)

    # Create model once (we will reload weights per-ckpt):
    model = SiT_models[args.model](input_size=latent_size)
    model = DDP(model.to(device), device_ids=[device])
    model.eval()

    # ---- collect ckpts ----
    ckpt_list = list_ckpts(args.ckpt)
    if len(ckpt_list) == 0:
        raise ValueError(f"No ckpt found from args.ckpt={args.ckpt}")

    if rank == 0:
        os.makedirs(args.output, exist_ok=True)
        print(f"[ckpt_list] total={len(ckpt_list)} | num_MLP={args.num_MLP}")
        print(f"[align_rule] gradient_list={args.gradient} -> per-ckpt align_block decided by ckpt index")

    # time steps (unchanged):
    t_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    W = list(range(10))

    # ---- run each ckpt ----
    for ckpt_i, ckpt_path in enumerate(ckpt_list):
        
        align_block = map_ckpt_to_align_block(ckpt_i, len(ckpt_list), args.gradient)


        if args.num_MLP <= 1:
            mlp_id = 0
            ckpt_proj_key = "proj"
        else:
            mlp_id = map_ckpt_to_mlp_id(ckpt_i, len(ckpt_list), args.num_MLP)  # 1..num_MLP
            ckpt_proj_key = f"proj{mlp_id}"

        # load weights
        if args.ckpt is not None:
            load_backbone_and_selected_proj(model.module, ckpt_path, ckpt_proj_key)


        # output folder per ckpt (rank0 only writes)
        ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
        run_dir = os.path.join(args.output, ckpt_tag)
        if rank == 0:
            os.makedirs(run_dir, exist_ok=True)

        P = 5

        # dataset-level accumulated global gradients
        G_fm = {pi: {} for pi in range(1, P + 1)}
        G_cos = {pi: {} for pi in range(1, P + 1)}
        G_fm_cnt = {pi: {w: 0 for w in W} for pi in range(1, P + 1)}
        G_cos_cnt = {pi: {w: 0 for w in W} for pi in range(1, P + 1)}
        # also keep some text logs (optional)
        log_lines = []
        if rank == 0:
            log_lines.append(f"ckpt={ckpt_path}")
            log_lines.append(f"mlp_id={mlp_id}, num_MLP={args.num_MLP}, ckpt_proj_key={ckpt_proj_key}")
            log_lines.append(f"gradient_rule={args.gradient}")
            log_lines.append(f"align_block={align_block}")
            log_lines.append(f"compute_wrt_blocks={W} (0..9)")
            log_lines.append("")

        # iterate dataset batches
        for x, y in loader:
            x = x.to(device)
            raw_images = x
            ys = y.to(device) if y is not None else None

            with torch.no_grad():
                x_lat = vae.encode(x).latent_dist.sample().mul_(0.18215)
                batch_size, c, height, width = x_lat.shape

            # build all timestep tensors
            base_ts = [torch.full((x_lat.size(0),), t, device=device, dtype=x_lat.dtype) for t in t_list]

            noise = torch.randn_like(x_lat)

            # build x_t_list and v_t_list with a loop
            x_t_list = []
            v_t_list = []

            for base_t in base_ts:
                alpha = scheduler.alpha(base_t)
                dalpha = scheduler.dalpha(base_t)
                sigma = scheduler.sigma(base_t)
                dsigma = scheduler.dsigma(base_t)

                x_t = alpha * x_lat + sigma * noise
                v_t = dalpha * x_lat + dsigma * noise

                x_t_list.append(x_t)
                v_t_list.append(v_t)

            # hook features
            src_feature = []

            def forward_hook(m, inp, out):
                # if block output is tuple, keep out[0]; otherwise use out directly
                feat = out[0] if isinstance(out, (tuple, list)) else out
                src_feature.append(feat)

            if getattr(model.module, "blocks", None) is not None:
                handle = model.module.blocks[align_block].register_forward_hook(forward_hook)
            else:
                handle = model.module.encoder.blocks[align_block].register_forward_hook(forward_hook)

            # forward multiple branches with a loop
            out_list = []
            for x_t, base_t in zip(x_t_list, base_ts):
                out, _, _ = model(x_t, base_t, ys)
                out_list.append(out)

            handle.remove()

            if len(src_feature) != len(base_ts):
                raise RuntimeError(
                    f"Expected {len(base_ts)} hooked features, got {len(src_feature)}. "
                    f"align_block={align_block}"
                )

            # project all hooked features with a loop
            proj_module = get_proj_module(model.module)
            src_proj_list = [proj_module(feat) for feat in src_feature]

            # bug fixed already! Note that dino input is in [0,1]
            raw_images_for_dino = (raw_images + 1.0) / 2.0
            with torch.no_grad():
                dst_feature, logits, structual_feature = DINO(
                    raw_images_for_dino,
                    encoder_depth=args.encoder_depth,
                    encoder_depth_structual=args.encoder_depth_structual
                )

            # token number alignment
            if dst_feature.shape[1] != src_proj_list[0].shape[1]:
                dst_length = dst_feature.shape[1]
                rescale_ratio = (src_proj_list[0].shape[1] / dst_feature.shape[1]) ** 0.5
                dst_height = (dst_length) ** 0.5 * (height / width) ** 0.5
                dst_width = (dst_length) ** 0.5 * (width / height) ** 0.5

                dst_feature = dst_feature.view(batch_size, int(dst_height), int(dst_width), args.proj_encoder_dim)
                dst_feature = dst_feature.permute(0, 3, 1, 2)
                dst_feature = torch.nn.functional.interpolate(
                    dst_feature,
                    scale_factor=rescale_ratio,
                    mode='bilinear',
                    align_corners=False
                )
                dst_feature = dst_feature.permute(0, 2, 3, 1)
                dst_feature = dst_feature.view(batch_size, -1, args.proj_encoder_dim)

            # compute all losses with loops
            cos_losses = [
                (1 - torch.nn.functional.cosine_similarity(src_proj, dst_feature, dim=-1)).mean()
                for src_proj in src_proj_list
            ]

            fm_losses = [
                ((out - v_t) ** 2).mean()
                for out, v_t in zip(out_list, v_t_list)
            ]

            # package results
            pairs = [
                (f"fm{i + 1}", f"cos{i + 1}", fm_losses[i], cos_losses[i])
                for i in range(len(base_ts))
            ]

            # compute angles for this batch, accumulate
            for pi, (na, nb, la, lb) in enumerate(pairs, start=1):
                for wrt in W:
                    g_fm = get_global_grad_vector(model.module, la, wrt)
                    g_cos = get_global_grad_vector(model.module, lb, wrt)

                    if g_fm is None or g_cos is None:
                        continue

                    # 初始化累计向量
                    if wrt not in G_fm[pi]:
                        G_fm[pi][wrt] = torch.zeros_like(g_fm, dtype=torch.float64)
                        G_cos[pi][wrt] = torch.zeros_like(g_cos, dtype=torch.float64)

                    # 累计“每个 batch 的 global gradient”
                    G_fm[pi][wrt] += g_fm.to(torch.float64)
                    G_cos[pi][wrt] += g_cos.to(torch.float64)

                    G_fm_cnt[pi][wrt] += 1
                    G_cos_cnt[pi][wrt] += 1


        # rank0: finalize averages + one plot + one log
        if rank == 0:
            angle_curves = []
            cos_curves = []

            for pi in range(1, P + 1):
                y_ang = []
                y_cos = []

                for w in W:
                    if (w not in G_fm[pi]) or (w not in G_cos[pi]):
                        y_ang.append(float("nan"))
                        y_cos.append(float("nan"))
                        continue

                    g_fm_total = G_fm[pi][w]
                    g_cos_total = G_cos[pi][w]

                    na = torch.linalg.norm(g_fm_total).item()
                    nb = torch.linalg.norm(g_cos_total).item()

                    if na < 1e-12 or nb < 1e-12:
                        y_ang.append(float("nan"))
                        y_cos.append(float("nan"))
                        continue

                    cosv = torch.dot(g_fm_total.float(), g_cos_total.float()).item() / (na * nb)
                    cosv = max(-1.0, min(1.0, cosv))
                    ang = math.degrees(math.acos(cosv))

                    y_ang.append(float(ang))
                    y_cos.append(float(cosv))

                angle_curves.append(y_ang)
                cos_curves.append(y_cos)

            # log some summary
            for pi in range(1, P + 1):
                log_lines.append(f"[pair#{pi}]")
                for j, w in enumerate(W):
                    log_lines.append(
                        f"  wrt={w:2d} | angle_total={angle_curves[pi-1][j]:.4f} "
                        f"(batches={G_fm_cnt[pi][w]}) | cos_total={cos_curves[pi-1][j]:+.4f} "
                        f"(batches={G_cos_cnt[pi][w]})"
                    )
                log_lines.append("")

            log_path = os.path.join(run_dir, "log.txt")
            write_lines(log_path, log_lines)

            # plot only ONE figure per ckpt (angle vs wrt)
            all_angles = [v for y in angle_curves for v in y if v == v]
            if len(all_angles) == 0:
                y_min, y_max = 0.0, 180.0
            else:
                y_min, y_max = float(min(all_angles)), float(max(all_angles))
                margin = max(1.0, 0.05 * (y_max - y_min + 1e-6))
                y_min -= margin
                y_max += margin

            plt.figure()
            for pi, y in enumerate(angle_curves, start=1):
                plt.plot(W, y, marker="o", label=f"pair#{pi}")
            plt.xlabel("wrt (block index)")
            plt.ylabel("angle (deg)")
            plt.ylim(y_min, y_max)
            plt.xticks(W)
            plt.legend()
            plt.grid(True)

            fig_path = os.path.join(run_dir, "angle_vs_wrt.png")
            plt.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close()

            print(f"[saved] {log_path}")
            print(f"[saved] {fig_path}")

    print("Done!")

if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num_MLP", type=int, default=1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=75)
    # parser.add_argument("--align_layer", type=int, default=4)
    parser.add_argument("--encoder_depth", type=int, default=4)
    parser.add_argument("--gradient", type=int, nargs="+", default=[0,1,2,3])
    parser.add_argument("--encoder_depth_structual", type=int, default=4)
    # /data/lzw_25/DDT_revise/models
    parser.add_argument("--vae", type=str)  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--proj_encoder_dim", type=int, default=768)
    parser.add_argument("--ckpt", type=str, default="/data/lzw_25/SiT_adapted_from_DDT/universal_flow_workdirs/exp_repa_flatten_condit8_dit4_fixt_xl-009/last.ckpt",
                        help="Optional path to a custom SiT checkpoint")

    args = parser.parse_args()
    main(args)