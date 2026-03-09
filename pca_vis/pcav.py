
# Todo: Wrote PCA analysis and t-SNE analysis, organize into two files, which can be used separately later.

import numpy as np
import torch
import cv2
from sklearn.decomposition import PCA

# ---------------------------
# helpers
# ---------------------------

def _minmax_01(x, eps=1e-12):
    mn = x.min(axis=0, keepdims=True)
    mx = x.max(axis=0, keepdims=True)
    return (x - mn) / (mx - mn + eps)

def _adjacent_cosine_on_grid(vecs, H=16, W=16, eps=1e-12):
    """
    vecs: (H*W, D) feature per token (here we use PCA 3D or original C-dim)
    returns: mean cosine similarity between 4-neighbors on grid
    """
    X = vecs.reshape(H, W, -1).astype(np.float64)
    # normalize along D
    n = np.linalg.norm(X, axis=-1, keepdims=True) + eps
    Xn = X / n

    sims = []
    # right neighbors
    sims.append((Xn[:, :-1, :] * Xn[:, 1:, :]).sum(axis=-1))
    # down neighbors
    sims.append((Xn[:-1, :, :] * Xn[1:, :, :]).sum(axis=-1))
    sims = np.concatenate([s.ravel() for s in sims], axis=0)
    return float(sims.mean()), float(sims.std())

def _fft_low_high_ratio(rgb_img, low_radius_frac=0.15, eps=1e-12):
    """
    rgb_img: (H,W,3) float in [0,1] or any range.
    Compute energy ratio low_freq / high_freq on grayscale (or per-channel averaged).
    """
    img = rgb_img.astype(np.float64)
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img
    H, W = gray.shape

    F = np.fft.fft2(gray)
    F = np.fft.fftshift(F)
    P = np.abs(F) ** 2  # power

    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max() + eps
    low_r = low_radius_frac * rmax

    low_mask = rr <= low_r
    low_energy = P[low_mask].sum()
    high_energy = P[~low_mask].sum() + eps
    return float(low_energy / high_energy), float(low_energy), float(high_energy)

def _principal_angles(Ua, Ub, eps=1e-12):
    """
    Ua, Ub: (C, k) orthonormal bases
    Return cosines and angles (degrees) of principal angles between subspaces.
    """
    # ensure float64 for stability
    Ua = Ua.astype(np.float64)
    Ub = Ub.astype(np.float64)
    M = Ua.T @ Ub
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
    return s, angles


# Todo: the four metric may be inaccurate due to implementation, pca is well-done and other methods need to be checked.

def pca_v(nfeatures, n_components=3, thr=0.0, labels=None,
          grid_hw=(16,16), out_size=256,
          print_metrics=True, name_prefix="feat"):
    """
    - Perform PCA(n_components) on each feature
    - Take the first 3D dimensions and perform min-max mapping
    - reshape(16,16,3) + resize -> rgb

    Additionally: Print metrics (as accurate as possible)
    1) explained_variance_ratio_[:3]
    2) Adjacency cosine similarity mean/variance (in PCA 3D space; more consistent visualization)

    (Optional) Can also be calculated in the original feature C-dimensional space (more "realistic" but more expensive)
    3) FFT low/high frequency energy ratio (on the visualized RGB image)
    4) Principal subspace similarity: top-3 PCA subspace principal angles of the previous features

    """
    H, W = grid_hw
    rgbs = []

    prev_pca_components = None  # (C, k) for subspace similarity
    prev_tag = None

    for idx, features in enumerate(nfeatures):
        # ---- to numpy ----
        if isinstance(features, torch.Tensor):
            f = features.squeeze(0).detach().cpu().numpy()
        else:
            f = np.asarray(features)
        # expected: (N, C) where N=H*W
        if f.ndim != 2:
            raise ValueError(f"Expected (N,C) after squeeze(0), got {f.shape}")

        N, C = f.shape
        if N != H * W:
            raise ValueError(f"Token count N={N} != H*W={H*W}. "
                             f"Set grid_hw accordingly or fix reshape logic.")

        # ---- PCA ----
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(f)  # (N, k)

        # ---- metrics: explained variance ----
        evr = pca.explained_variance_ratio_
        evr3 = evr[:min(3, len(evr))]

        # ---- normalize first 3 comps for visualization ----
        vis = pca_features[:, :3].copy()
        vis = _minmax_01(vis)

        rgb = vis.reshape(H, W, 3)
        rgb = cv2.resize(rgb, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
        rgbs.append(rgb)

        if print_metrics:
            tag = f"{name_prefix}[{idx}]"
            print(f"\n== {tag} ==")
            print(f"shape: N={N}, C={C}")
            print(f"explained_variance_ratio (top3): {np.array2string(evr3, precision=4)} "
                  f"| sum(top3)={evr3.sum():.4f}")

            # (2) token spatial adjacency similarity on PCA-3D
            mu_adj, std_adj = _adjacent_cosine_on_grid(pca_features[:, :3], H=H, W=W)
            print(f"adjacent cosine (PCA-3D): mean={mu_adj:.4f}, std={std_adj:.4f}")

            # (3) FFT low/high energy ratio on rgb visualization
            ratio, lowE, highE = _fft_low_high_ratio(rgb, low_radius_frac=0.15)
            print(f"FFT low/high energy ratio (rgb gray): {ratio:.4f} "
                  f"(lowE={lowE:.3e}, highE={highE:.3e})")

            # (4) subspace similarity vs previous (principal angles)
            # PCA components_: (k, C). We use basis in original C space: (C, k)
            U = pca.components_.T  # (C, k)
            # Orthonormalize (PCA components are orthonormal already, but keep stable)
            # QR for safety
            U, _ = np.linalg.qr(U)

            if prev_pca_components is not None:
                k = min(U.shape[1], prev_pca_components.shape[1])
                s, ang = _principal_angles(prev_pca_components[:, :k], U[:, :k])
                print(f"subspace sim vs prev ({prev_tag}): "
                      f"cos={np.array2string(s, precision=4)} "
                      f"angles(deg)={np.array2string(ang, precision=2)} "
                      f"| mean_angle={ang.mean():.2f}")
            else:
                print("subspace sim vs prev: N/A (first item)")

            prev_pca_components = U
            prev_tag = tag

    return rgbs


# Todo: Orgainize the t-SNE codes.

# t-SNE

# import numpy as np
# import torch
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# def pca_v(
#     nfeatures, n_components, thr, labels=None,
#     out_size=256, perplexity=15, n_iter=1000, random_state=0,
    # point_size=18,   #  Larger
    # cmap="tab20",
    # pool="mean",
    # alpha=1.0,       #  Darker
    # edgecolor="k",   #  More prominent black border (can also use None)
    # edgewidth=0.35,  #  Edge width
    # show_axis=True,  #  Show axis
    # tick_step=0.2    #  Tick interval
# ):
#     """
# TSNE: One point per image.
# Features: (B,N,C) -> pool -> (B,C) -> TSNE -> (B,2)
# Output remains unchanged: Returns rgbs(list), each element is a (H,W,3) float RGB, which can be directly displayed using imshow.
#     """
#     rgbs = []

#     for features in nfeatures:
#         # ---- to numpy ----
#         if isinstance(features, torch.Tensor):
#             feats = features.detach().cpu().float().numpy()
#         else:
#             feats = np.asarray(features, dtype=np.float32)


#         # ---- unify to (B,C) ----
#         if feats.ndim == 3:
#             # normalize X first
#             if pool == "mean":
#                 X = feats.mean(axis=1)  # (B,C)
#             elif pool == "max":
#                 X = feats.max(axis=1)
#             else:
#                 raise ValueError(f"Unknown pool='{pool}'")
#         elif feats.ndim == 2:
#             X = feats  # (B,C)
#         else:
#             raise ValueError(f"Unexpected features shape: {feats.shape}")
        
#         # X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        

#         # ---- labels -> (B,) optional ----
#         y = None
#         if labels is not None:
#             y_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
#             y_np = y_np.reshape(-1)
#             if y_np.shape[0] != X.shape[0]:
#                 raise ValueError(f"labels length {y_np.shape[0]} != num images {X.shape[0]}")
#             y = y_np

#         # ---- thr filter (optional) ----
#         # if thr is not None and thr > 0:
#         #     norms = np.linalg.norm(X, axis=1)
#         #     keep = norms >= thr
#         #     X = X[keep]
#         #     if y is not None:
#         #         y = y[keep]

#         # ---- TSNE ----
#         if X.shape[0] < 3:
#             Z = np.zeros((X.shape[0], 2), dtype=np.float32)
#         else:
#             # perp = min(perplexity, max(2, (X.shape[0] - 1) // 3))
#             tsne = TSNE(
#                 n_components=2,
#                 perplexity=perplexity,
#                 # n_iter=n_iter,
#                 # max_iter = 2500,
#                 init="pca",
#                 learning_rate="auto",
#                 # random_state=random_state
#             )
#             Z = tsne.fit_transform(X)  # (B,2)

#         # normalize to [0,1] so axes are stable across subplots
#         Z = (Z - Z.min(axis=0, keepdims=True)) / (Z.ptp(axis=0, keepdims=True) + 1e-8)

#         # ---- render scatter to an RGB image ----
#         fig = plt.figure(figsize=(out_size / 100.0, out_size / 100.0), dpi=100)
#         ax = plt.gca()

#
#         if show_axis:
#             ax.set_xlim(0, 1)
#             ax.set_ylim(0, 1)
#             ticks = np.arange(0.0, 1.0 + 1e-9, tick_step)
#             ax.set_xticks(ticks)
#             ax.set_yticks(ticks)
#             ax.grid(True, linewidth=0.3, alpha=0.35)  # 轻网格，便于读
#             ax.tick_params(labelsize=6)
#         else:
#             ax.set_axis_off()
#             ax.set_xlim(0, 1)
#             ax.set_ylim(0, 1)

#         scatter_kwargs = dict(
#             s=point_size,
#             alpha=alpha,
#             linewidths=edgewidth,
#         )
#         # edgecolor 只在有意义时设置，避免 warning
#         if edgecolor is not None:
#             scatter_kwargs["edgecolors"] = edgecolor

#         if y is None:
#             ax.scatter(Z[:, 0], Z[:, 1], **scatter_kwargs)
#         else:
#             ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, **scatter_kwargs)

#         plt.tight_layout(pad=0.2 if show_axis else 0)

#         fig.canvas.draw()
#         w, h = fig.canvas.get_width_height()

#         buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
#         img = buf[:, :, 1:4]  # RGB
#         plt.close(fig)

#         if img.shape[0] != out_size or img.shape[1] != out_size:
#             img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

#         rgbs.append(img.astype(np.float32) / 255.0)

#     return rgbs
