import os
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info


def _infer_token_grid_hw(num_tokens: int, fixed_hw: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """将 T 个 token 排成 h×w，优先用 fixed_hw（若 h*w==T），否则尽量正方形再分解因数。"""
    if fixed_hw is not None:
        fh, fw = int(fixed_hw[0]), int(fixed_hw[1])
        if fh * fw == num_tokens:
            return fh, fw
    r = int(np.floor(np.sqrt(num_tokens)))
    if r * r == num_tokens:
        return r, r
    best_h, best_w = 1, num_tokens
    for h in range(r, 0, -1):
        if num_tokens % h == 0:
            w = num_tokens // h
            if abs(h - w) < abs(best_h - best_w):
                best_h, best_w = h, w
    return best_h, best_w


def _save_spatial_dimmean_surface(
    z_td: np.ndarray,
    out_path: str,
    title: str,
    grid_hw: Optional[Tuple[int, int]] = None,
    dpi: int = 140,
):
    """
    z_td: (T, D) 激活。对最后一维 D 取平均得到每个 token 一个标量；
    将 T 个 token 按 h×w 铺在 xoy 平面上（行优先与 patch 顺序一致），
    竖直方向 z 为该位置上的 dim-mean。
    """
    if z_td.ndim == 3:
        if z_td.shape[0] != 1:
            raise ValueError(
                f"Expected (T, D) or (1,T,D); got {z_td.shape}. "
                "若为 (B,T,D) 请在调用处按样本传入 arr[si]。"
            )
        z_td = z_td[0]
    if z_td.ndim != 2:
        raise ValueError(f"Expected (T, D), got {z_td.shape}")
    t, _d = z_td.shape
    z_mean = z_td.astype(np.float64).mean(axis=1)
    h, w = _infer_token_grid_hw(t, fixed_hw=grid_hw)
    if h * w != t:
        raise ValueError(f"Cannot reshape T={t} to grid; inferred ({h},{w})")

    Z = z_mean.reshape(h, w)
    # x: 列（宽），y: 行（高），与图像 patch 从左到右、从上到下一致
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
    ax.set_xlabel("token grid x (width)")
    ax.set_ylabel("token grid y (height)")
    ax.set_zlabel("mean over dim")
    ax.set_title(f"{title}\n({h}×{w}={t} tokens)")
    plt.tight_layout()
    ddir = os.path.dirname(out_path)
    if ddir:
        os.makedirs(ddir, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


STAGE_NAMES = ("norm1", "attn_out", "after_msa", "norm2", "block_out")


class RepaEncoderVisualizeHook(Callback):
    """
    每隔 visualize_interval 个 optimizer step：
    - 从 outputs['repa_viz'] 读取 CKNNA(mid1,dino)、CKNNA(mid2,dino) 与 mid1/mid2 两条 encoder 路径各阶段激活；
    - rank0 写入磁盘：每个样本 1 张 DINO + num_encoder_blocks*5 张 mid1 + 同数 mid2（共 1+2*N*5）。
      每张图为 3D 曲面：xoy 为 token 的二维网格（默认按 sqrt(T) 推断，可用 viz_grid_hw 固定如 16×16），
      z 为对该 token 所有 channel 取平均后的值。
    """

    def __init__(
        self,
        save_dir: str = "repa_encoder_viz",
        visualize_interval: int = 1000,
        topk: int = 10,
        q: float = 0.95,
        exact_quantile: bool = True,
        dpi: int = 140,
        viz_grid_hw: Optional[Union[Tuple[int, int], List[int]]] = None,
    ):
        self.save_dir = save_dir
        self.visualize_interval = int(visualize_interval)
        self.topk = topk
        self.q = q
        self.exact_quantile = exact_quantile
        self.dpi = dpi
        if viz_grid_hw is not None and len(viz_grid_hw) == 2:
            self.viz_grid_hw = (int(viz_grid_hw[0]), int(viz_grid_hw[1]))
        else:
            self.viz_grid_hw = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(pl_module, "diffusion_trainer"):
            dt = pl_module.diffusion_trainer
            dt._repa_viz_interval = self.visualize_interval
            dt._repa_viz_topk = self.topk
            dt._repa_viz_q = self.q
            dt._repa_viz_exact = self.exact_quantile
            rank_zero_info(
                f"RepaEncoderVisualizeHook: interval={self.visualize_interval}, save under "
                f"{os.path.join(trainer.default_root_dir, self.save_dir)}"
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        if not isinstance(outputs, dict):
            return
        payload = outputs.get("outputs", outputs)
        if not isinstance(payload, dict):
            return
        viz = payload.get("repa_viz", None)
        if viz is None:
            return

        root = os.path.join(trainer.default_root_dir, self.save_dir, f"step_{viz['step']:08d}")
        os.makedirs(root, exist_ok=True)

        def _f(key, alt_key=None, default=float("nan")):
            v = viz.get(key)
            if v is None and alt_key is not None:
                v = viz.get(alt_key)
            if v is None:
                return default
            return float(v)

        log_path = os.path.join(root, "cknna.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "# patch: N=B*T 个 patch token（整 batch 展平），与 compute.py 的 feats (N,D) 一致\n"
                "# image_mean: 每张图对 token 维 mean 后 N=B；batch<3 或无有效 topk 时为 nan\n"
                f"step\t{viz['step']}\n"
                f"cknna_mid1_dino_patch\t{_f('cknna_mid1_dino_patch', 'cknna_mid1_dino'):.6f}\n"
                f"cknna_mid2_dino_patch\t{_f('cknna_mid2_dino_patch', 'cknna_mid2_dino'):.6f}\n"
                f"cknna_mid1_dino_image_mean\t{_f('cknna_mid1_dino_image_mean'):.6f}\n"
                f"cknna_mid2_dino_image_mean\t{_f('cknna_mid2_dino_image_mean'):.6f}\n"
                f"topk\t{self.topk}\tq\t{self.q}\texact\t{self.exact_quantile}\n"
            )

        dino = viz["dino"].numpy()
        enc_m1 = viz["encoder_mid1"]
        enc_m2 = viz.get("encoder_mid2") or []
        for si in range(dino.shape[0]):
            _save_spatial_dimmean_surface(
                dino[si],
                os.path.join(root, f"sample{si}_dino.png"),
                f"sample{si} DINO",
                grid_hw=self.viz_grid_hw,
                dpi=self.dpi,
            )
            for bi, block_stages in enumerate(enc_m1):
                for sj, name in enumerate(STAGE_NAMES):
                    arr = block_stages[sj].numpy()
                    # training 侧为 (ns, T, D)，按样本 si 取 (T, D)
                    patch_td = arr[si] if arr.ndim == 3 else arr
                    _save_spatial_dimmean_surface(
                        patch_td,
                        os.path.join(root, f"sample{si}_mid1_block{bi:02d}_{name}.png"),
                        f"sample{si} mid1 block{bi} {name}",
                        grid_hw=self.viz_grid_hw,
                        dpi=self.dpi,
                    )
            for bi, block_stages in enumerate(enc_m2):
                for sj, name in enumerate(STAGE_NAMES):
                    arr = block_stages[sj].numpy()
                    patch_td = arr[si] if arr.ndim == 3 else arr
                    _save_spatial_dimmean_surface(
                        patch_td,
                        os.path.join(root, f"sample{si}_mid2_block{bi:02d}_{name}.png"),
                        f"sample{si} mid2 block{bi} {name}",
                        grid_hw=self.viz_grid_hw,
                        dpi=self.dpi,
                    )

        rank_zero_info(f"RepaEncoderVisualizeHook saved to {root}")
