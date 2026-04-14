# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
from typing import Tuple
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
# from timm.layers import use_fused_attn
from torch.jit import Final
import torch.nn.functional as F
from torch.nn.init import zeros_
from torch.nn.modules.module import T
from torch.nn.functional import scaled_dot_product_attention
class Attention(nn.Module):
    # fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        attn_logits = (q * self.scale) @ k.transpose(-2, -1)
        x = scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=0.,
        #     )
        # else:
        #     q = q * self.scale
        #     attn = q @ k.transpose(-2, -1)
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)
        #     x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_logits

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

# def modulate(x, shift, scale):
#     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

# def modulate_per_token(x, shift, scale):
#     """Per-token modulation for (N, T, D) conditioning."""
#     return x * (1 + scale) + shift
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        # Todo! remove the following could enable CFG 
        # use_cfg_embedding = dropout_prob > 0
        # self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes
        # self.dropout_prob = dropout_prob # set to 0

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels,):
        # use_dropout = self.dropout_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class DDTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True
            )
        # if "fused_attn" in block_kwargs.keys():
        #     self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, return_act=None, viz_out=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        n1 = self.norm1(x)
        x_, attn_logits = self.attn(modulate(n1, shift_msa, scale_msa))
        x_after_msa = x + gate_msa * x_
        n2 = self.norm2(x_after_msa)
        x = x_after_msa + gate_mlp * self.mlp(modulate(n2, shift_mlp, scale_mlp))
        if viz_out is not None:
            viz_out.extend(
                (
                    n1.detach(),
                    x_.detach(),
                    x_after_msa.detach(),
                    n2.detach(),
                    x.detach(),
                )
            )
        return x, attn_logits
        # else:
        #     x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        #     x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        #     return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x

class PerTokenFinalLayer(FinalLayer):
    """
    The final layer of SiT.
    """

    def forward(self, x, c):
        """
        Args:
            x: (N, T, D) tokens
            c: (N, T, D) per-token conditioning
        """
        batch_size, seq_len, hidden_dim = c.shape
        c_flat = c.reshape(-1, hidden_dim)
        modulation_flat = self.adaLN_modulation(c_flat)
        modulation = modulation_flat.reshape(batch_size, seq_len, -1)

        shift, scale = modulation.chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

# def SiT_XL_2(**kwargs):
#     return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# def SiT_XL_4(**kwargs):
#     return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

# def SiT_XL_8(**kwargs):
#     return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

# def SiT_L_2(**kwargs):
#     return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

# def SiT_L_4(**kwargs):
#     return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

# def SiT_L_8(**kwargs):
#     return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

# def SiT_B_2(**kwargs):
#     return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

# def SiT_B_4(**kwargs):
#     return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

# def SiT_B_8(**kwargs):
#     return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

# def SiT_S_2(**kwargs):
#     return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

# def SiT_S_4(**kwargs):
#     return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

# def SiT_S_8(**kwargs):
#     return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


# SiT_models = {
#     'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
#     'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
#     'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
#     'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
# }

class DDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        # path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        num_groups=12,
        hidden_size=1152,
        # decoder_hidden_size=768,
        # encoder_depth=8,
        num_blocks=18,
        num_encoder_blocks=4,
        # num_heads=16,
        # mlp_ratio=4.0,
        learn_sigma=True,
        deep_supervision=0,
        # class_dropout_prob=0.1,
        num_classes=1000,
        # use_cfg=False, # 没有？
        # z_dims=[768],
        # projector_dim=2048,
        # **block_kwargs # fused_attn
        weight_path=None,
        load_ema=False,
        attn_begin=18,
        attn_end=21,
    ):
        super().__init__()
        # self.path_type = path_type   # 似乎没用。
        self.deep_supervision = deep_supervision
        self.num_groups = num_groups
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels 
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        # self.num_heads = num_heads  # 在Attention中
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_encoder_blocks = num_encoder_blocks
        self.attn_begin = attn_begin
        self.attn_end = attn_end
        # elf.z_dims = z_dims  #
        # self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.s_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        # self.s_embedder1 = PatchEmbed(
        #     14, patch_size, in_channels, hidden_size, bias=True
        #     )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type

        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.y_embedder = LabelEmbedder(num_classes+1, hidden_size)
        num_patches = self.x_embedder.num_patches
        # num_patches1 = self.s_embedder1.num_patches
        self.weight_path = weight_path
        self.load_ema = load_ema
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        # self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches1, hidden_size), requires_grad=False)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))  # for masked training, not used in sampling

        self.blocks = nn.ModuleList([
            DDTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_blocks)
        ])

        # self.projectors = nn.ModuleList([
        #     build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
        #     ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # pos_embed1 = get_2d_sincos_pos_embed(
        #     self.pos_embed1.shape[-1], int(self.s_embedder1.num_patches ** 0.5)
        #     )
        # self.pos_embed1.data.copy_(torch.from_numpy(pos_embed1).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w1 = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # w11 = self.s_embedder1.proj.weight.data
        # nn.init.xavier_uniform_(w11.view([w11.shape[0], -1]))
        # nn.init.constant_(self.s_embedder1.proj.bias, 0)  


        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    # def forward(self, x, whole, part1, part2, t, y, s=None):
    def forward(self, x, x1, t, t1, y, s=None, encode_viz: bool = False):
        """
        Forward pass of SiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        encode_viz: 若为 True，在返回元组最后一项附带 encoder 中间激活（避免依赖 setattr，兼容 compile/DDP）。
        """
        N, channel, height, width = x.shape
        # x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # N, T, D = x.shape
        # timestep and class embedding
        # t_embed = self.t_embedder(t)
        assert t.dim() == 1, f"t should be (B,), got {t.shape}"                   
        t_embed = self.t_embedder(t.view(-1)).view(N, -1, self.hidden_size) # (N, 1, D)
        t_embed1 = self.t_embedder(t1.view(-1)).view(N, -1, self.hidden_size) # (N, 1, D)
        y = self.y_embedder(y).view(N, 1, self.hidden_size)    # (N, 1, D)
        # c = t_embed + y                               # (N, 1, D)
        # c1 = t_embed1 + y
        c = t_embed                               # (N, 1, D)
        c1 = t_embed1
        attn_acts = []

        enc_viz = None
        if s is None:
            capture_enc = bool(encode_viz)
            viz_mid1 = [] if capture_enc else None
            viz_mid2 = [] if capture_enc else None

            mid1_blocks = []
            s = self.s_embedder(x) + self.pos_embed
            for i in range(self.num_encoder_blocks):
                slot = [] if capture_enc else None
                s, attn_logits = self.blocks[i](s, c, viz_out=slot)
                mid1_blocks.append(s)   # 每个 block 后的输出
                if slot is not None:
                    viz_mid1.append(slot)
            mid1 = s

            s = nn.functional.silu(t_embed + s + y)

            s1 = self.s_embedder(x1) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
            for i in range(self.num_encoder_blocks):
                slot2 = [] if capture_enc else None
                s1, _ = self.blocks[i](s1, c1, viz_out=slot2)
                if slot2 is not None:
                    viz_mid2.append(slot2)
            mid2 = s1

            if capture_enc:
                enc_viz = {"mid1": viz_mid1, "mid2": viz_mid2}
        else:
            # 与历史接口兼容：带 x1 的训练路径始终 s=None；若传入 s 则不再计算 mid1/mid2
            mid1 = mid2 = None


        # part1 = self.s_embedder1(part1) + self.pos_embed1  # (N, T, D), where T = H * W / patch_size ** 2
        # for i in range(self.num_encoder_blocks):
        #     part1, _ = self.blocks[i](part1, c)

        # part2 = self.s_embedder1(part2) + self.pos_embed1  # (N, T, D), where T = H * W / patch_size ** 2
        # for i in range(self.num_encoder_blocks):
        #     part2, _ = self.blocks[i](part2, c)

        # if s is None:
        #     s = self.s_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        #     for i in range(self.num_encoder_blocks):
        #         s, attn_logits = self.blocks[i](s, c)

        #     s = nn.functional.silu(t_embed + s)

                           
        x = self.x_embedder(x) + self.pos_embed
        for i in range(self.num_encoder_blocks, self.num_blocks):
            x, attn_logits = self.blocks[i](x, s)


        # for i in range(self.num_blocks):
        #     x, attn_logits = self.blocks[i](x, c)


        # for i, block in enumerate(self.blocks):
        #     if ((i + 1) <= self.encoder_depth) and (return_act is not None):
        #         x, attn_logits = block(x, c, return_act)
        #         attn_acts.append(attn_logits)
        #         if (i + 1) == self.encoder_depth:
        #             zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        #             # zs.append(x)
        #     else:
        #         x = block(x, c)

        x = self.final_layer(x, s)                # (B, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (B, out_channels, H, W)
        # attn_acts = torch.stack(attn_acts,dim=0)  # (L, B, H, T, T)

        return x, c, mid1, mid1_blocks, mid2, attn_acts, enc_viz

    def forward_sample(self, x, t, y, s=None):
        """
        Forward pass of SiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """
        N, channel, height, width = x.shape
        # x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # N, T, D = x.shape
        # timestep and class embedding
        # t_embed = self.t_embedder(t)
        assert t.dim() == 1, f"t should be (B,), got {t.shape}"                   
        t_embed = self.t_embedder(t.view(-1)).view(N, -1, self.hidden_size) # (N, 1, D)
        y = self.y_embedder(y).view(N, 1, self.hidden_size)    # (N, 1, D)
        c = t_embed                               # (N, 1, D)
        attn_acts = []

        if s is None:
            s = self.s_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
            for i in range(self.num_encoder_blocks):
                s, attn_logits = self.blocks[i](s, c)

            s = nn.functional.silu(t_embed + s + y)

                           
        x = self.x_embedder(x) + self.pos_embed
        for i in range(self.num_encoder_blocks, self.num_blocks):
            x, attn_logits = self.blocks[i](x, s)


        # for i in range(self.num_blocks):
        #     x, attn_logits = self.blocks[i](x, c)


        # for i, block in enumerate(self.blocks):
        #     if ((i + 1) <= self.encoder_depth) and (return_act is not None):
        #         x, attn_logits = block(x, c, return_act)
        #         attn_acts.append(attn_logits)
        #         if (i + 1) == self.encoder_depth:
        #             zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        #             # zs.append(x)
        #     else:
        #         x = block(x, c)

        x = self.final_layer(x, s)                # (B, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (B, out_channels, H, W)
        # attn_acts = torch.stack(attn_acts,dim=0)  # (L, B, H, T, T)

        return x, c, attn_acts

class PerTokenDDT(DDT):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(self, **kwargs):
        # Initialize parent class (creates standard DiTBlocks)
        super().__init__(**kwargs)

        hidden_size = kwargs["hidden_size"]
        num_groups = kwargs["num_groups"]
        patch_size = kwargs["patch_size"]
        in_channels = kwargs["in_channels"]
        out_channels = in_channels

        # Convert standard blocks to per-token versions, preserving weights
        self._convert_to_per_token_blocks(hidden_size, num_groups, patch_size, out_channels)   

    def _convert_to_per_token_blocks(self, hidden_size, num_groups, patch_size, out_channels):
        """Convert DDTBlocks to PerTokenDDTBlocks while preserving weights."""
        new_blocks = nn.ModuleList()
        for original_block in self.blocks:
            new_block = DDTBlock(hidden_size, num_groups)
            new_block.load_state_dict(original_block.state_dict())
            new_blocks.append(new_block)
        self.blocks = new_blocks

        original_final = self.final_layer
        new_final = PerTokenFinalLayer(hidden_size, patch_size, out_channels)
        new_final.load_state_dict(original_final.state_dict())
        self.final_layer = new_final

    def forward(self, x, t, y, s=None):
        """
        Forward pass of SiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, N) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """
        N, channel, height, width = x.shape
        # x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # N, T, D = x.shape
        # timestep and class embedding
        # t_embed = self.t_embedder(t)
                
        # t_embed = self.t_embedder(t.view(-1)).view(N, -1, self.hidden_size) # (N, 1, D)
        # y = self.y_embedder(y).view(N, 1, self.hidden_size)    # (N, 1, D)
        # c = t_embed + y                                # (N, 1, D)
        attn_acts = []
                
        x = self.x_embedder(x) + self.pos_embed
        batch_size, seq_len, hidden_dim = x.shape
        # Handle timestep embedding - per-token or broadcast
        if t.ndim == 1:
            t_emb = self.t_embedder(t).unsqueeze(1).expand(-1, seq_len, -1)
        elif t.ndim == 2:
            t_flat = t.reshape(-1)
            t_emb_flat = self.t_embedder(t_flat)
            t_emb = t_emb_flat.reshape(batch_size, seq_len, -1)
        else:
            raise ValueError(f"Timesteps must be 1D or 2D, got shape {t.shape}") 

        # Class embedding (broadcast to per-token)
        y_embed = self.y_embedder(y).unsqueeze(1).expand(-1, seq_len, -1)

        # Combine embeddings
        c = t_emb + y_embed

        for i in range(self.num_blocks):
            x, attn_logits = self.blocks[i](x, c)
        # for i, block in enumerate(self.blocks):
        #     if ((i + 1) <= self.encoder_depth) and (return_act is not None):
        #         x, attn_logits = block(x, c, return_act)
        #         attn_acts.append(attn_logits)
        #         if (i + 1) == self.encoder_depth:
        #             zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        #             # zs.append(x)
        #     else:
        #         x = block(x, c)
        x = self.final_layer(x, c)                # (B, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (B, out_channels, H, W)
        # attn_acts = torch.stack(attn_acts,dim=0)  # (L, B, H, T, T)

        return x, c, attn_acts