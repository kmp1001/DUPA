# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

# from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, Attention, Block

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    # def forward_features_list(self, x_list, masks_list):
    #     x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
    #     for blk in self.blocks:
    #         x = blk(x)

    #     all_x = x
    #     output = []
    #     for x, masks in zip(all_x, masks_list):
    #         x_norm = self.norm(x)
    #         output.append(
    #             {
    #                 "x_norm_clstoken": x_norm[:, 0],
    #                 "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
    #                 "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
    #                 "x_prenorm": x,
    #                 "masks": masks,
    #             }
    #         )
    #     return output

    def forward_features_attn(self, x, depth=9, encoder_structual_depth=7,masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        attn_act = []
        feature=None
        # for blk in self.blocks:
        #     x, attn = blk(x, return_act="attn")
        #     attn_act.append(attn)
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, return_act="attn")
            attn_act.append(attn)

            if i == depth:  # 第 10 个 block（0-based）
                x_at_10 = self.norm(x)
            if i == encoder_structual_depth:
                x_at_8 = self.norm(x)

        x_norm = self.norm(x)
        attn_act = torch.stack(attn_act, dim=0)
        return {
            "x_norm_clstoken": x_at_10[:, 0],
            "x_norm_regtokens": x_at_10[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_at_10[:, self.num_register_tokens + 1 :],
            "x_norm_structual_patchtokens": x_at_8[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,  # bug
            "masks": masks,
            "attn_act": attn_act,
        }
    # revise: 输入encoder_list, 每一个元素是下标（如2是第3个，正常[2]即可，为了efficient，建议先排序好，然后整体forward一次取出所有的feature，然后根据原来输入的顺序还原，返回一个多一维度的tensor，而非list
    def forward_features_list(self, x, encoder_list=(10, 9, 8, 7), masks=None,
                            return_attn=True, norm_at_depth=True):
        """
        encoder_list: iterable of int, 0-based block indices.
        Returns:
        feats: (K, B, N, C) tensor, aligned to the original order of encoder_list
        """
        x = self.prepare_tokens_with_masks(x, masks)

        # ---- sanitize encoder_list ----
        enc = list(encoder_list)
        if len(enc) == 0:
            raise ValueError("encoder_list must be non-empty")

        # 允许重复：输出也跟随重复；但为了 efficient，我们只在 forward 时取 unique 的
        unique_sorted = sorted(set(enc))
        max_idx = len(self.blocks) - 1
        for i in unique_sorted:
            if i < 0 or i > max_idx:
                raise IndexError(f"encoder index {i} out of range [0, {max_idx}]")

        # depth -> output position in unique_sorted
        want = set(unique_sorted)
        feat_map = {}  # depth -> (B, N, C)
        attn_act = [] if return_attn else None

        # ---- forward once ----
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, return_act="attn")
            if return_attn:
                attn_act.append(attn)

            if i in want:
                feat_map[i] = self.norm(x) if norm_at_depth else x
                # feat_map[i] = x

        # final tensors
        x_prenorm = x                    # output of last block (pre final norm)
        x_norm = self.norm(x_prenorm)    # final normed tokens

        # ---- restore original order, allow duplicates ----
        feats = torch.stack([feat_map[i] for i in enc], dim=0)  # (K, B, N, C)

        out = {           
            "x_norm_clstoken": feats[:, :, 0], # (K, B, N, C) in the same order as encoder_list
            "x_norm_regtokens": feats[:, :, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": feats[:, :, self.num_register_tokens + 1 :],
            "x_norm": x_norm,            # (B, N, C)
            "x_prenorm": x_prenorm,      # (B, N, C)
            "masks": masks,
        }

        if return_attn:
            out["attn_act"] = torch.stack(attn_act, dim=0)  # (D, ...)

        return out

    def forward_feature_and_salient_token(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        attn_act = []
        feature = None

        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, return_act="attn")
            attn_act.append(attn)

        x_norm = self.norm(x) 
        attn_act = torch.stack(attn_act, dim=0)
        attn = attn_act[-1]
        # attn.shape = [B, H, T, T]

        cls_to_patch_attn = attn[:, :, 0, self.num_register_tokens + 1:]   # [B, H, N]
        cls_to_patch_attn = cls_to_patch_attn.mean(dim=1)                   # [B, N]

        num_patches = cls_to_patch_attn.shape[-1]
        k = max(1, int(num_patches * 0.25))

        topk_idx = cls_to_patch_attn.topk(k, dim=-1).indices   # [B, k]

        salient_mask = torch.zeros_like(cls_to_patch_attn, dtype=torch.bool)  # [B, N]
        salient_mask.scatter_(1, topk_idx, True)

        # print(f"Emm num_register_token is:{self.num_register_tokens}")
        # print(f"x_norm shape is:{x_norm.shape}")


        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,  # bug
            "masks": masks,
            "attn_act": attn,
            "salient_patch_mask": salient_mask,   # 只对应 patch tokens
        }       
        
    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
