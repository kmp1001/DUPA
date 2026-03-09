#!/usr/bin/env python3
"""
show_pca.py
Visualize PCA feature maps of SiTs.
"""

import argparse
import os

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from diffusers import AutoencoderKL
from getfeature import get_f
from pcav import pca_v

import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
def load_sit_from_complex_ckpt(model, ckpt_path, device="cpu", prefer_ema=True, strict=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_all = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # state_all = ckpt["ema"] if "ema" in ckpt else ckpt
    # 选用 ema_denoiser 或 denoiser
    prefix = "ema_denoiser." if prefer_ema else "denoiser."
    keys = list(state_all.keys())
    if not any(k.startswith(prefix) for k in keys):
        # fallback：如果没有 ema_denoiser，就用 denoiser
        prefix = "denoiser."
        if not any(k.startswith(prefix) for k in keys):
            raise KeyError(f"Cannot find '{'ema_denoiser.' if prefer_ema else 'denoiser.'}' "
                           f"or 'denoiser.' keys in ckpt. Top example key: {keys[0] if keys else 'EMPTY'}")

    # strip prefix + 丢掉不属于 denoiser 的东西
    state = {k[len(prefix):]: v for k, v in state_all.items() if k.startswith(prefix)}

    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"Loaded with prefix '{prefix}' | missing={len(missing)} unexpected={len(unexpected)}")
    if (missing or unexpected) and strict:
        print("Example missing:", missing[:10])
        print("Example unexpected:", unexpected[:10])
    return prefix, missing, unexpected

# 用法
# prefix, missing, unexpected = load_sit_from_complex_ckpt(model, args.ckpt, device=device, prefer_ema=True, strict=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA visualization for SiTs")
    # I/O
    parser.add_argument( "--img_path", type=str, default="macaw.jpg",
                        help="Path to input image")
    parser.add_argument("--folder", type=str, default=None,
                        help="If provided, process all images under folder/<label_id>/*")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Optional: save result to this file; otherwise show with plt.show()")

    # Model
    parser.add_argument("--ckpt", type=str,
                       required=True,
                        help="Path to model checkpoint")

    # Hyper-parameters
    parser.add_argument("--label_id", type=int, default=88,
                        help="ImageNet class id")
    parser.add_argument("--layers", type=int, nargs="+", default=[23,24,25,26,27,28],
                        help="Layer indices to visualize")
    parser.add_argument("--times", type=float, nargs="+", default=[0.7, 0.5, 0.3, 0.1],
                        help="Sampling timesteps")
    parser.add_argument("--thr", type=float, default=0.00,
                        help="PCA threshold")
    parser.add_argument("--baseline", type=bool, default=False,
                        help="Use SiT baseline model (default: False)")
    # folder mode options
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Optional: limit images per subfolder (class). Default: no limit.")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optional: overall max images across all subfolders. Default: no limit.")
    parser.add_argument("--sort", choices=["name", "none"], default="name",
                        help="Sort images by name for deterministic order (default: name).")
    return parser.parse_args()


def iter_images_from_folder(root: Path, sort_mode: str = "name",
                            max_per_class: int | None = None,
                            max_images: int | None = None):
    """
    Yields (img_path, label_id_int) from:
      root/
        0/xxx.png
        1/yyy.jpg
        ...
    where subfolder name is the label id (0-999).
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"--folder is not a valid directory: {root}")

    # Subfolders sorted by integer label if possible, else by name
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    def label_key(p: Path):
        try:
            return int(p.name)
        except Exception:
            return 10**9  # put non-int folders at end

    subdirs.sort(key=lambda p: (label_key(p), p.name))

    total = 0
    for sd in subdirs:
        try:
            label = int(sd.name)
        except Exception:
            print(f"[WARN] Skip subfolder (name not int label): {sd}")
            continue

        imgs = [p for p in sd.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS]

        if sort_mode == "name":
            imgs.sort(key=lambda p: p.name.lower())

        if max_per_class is not None:
            imgs = imgs[:max_per_class]

        for img_path in imgs:
            yield img_path, label
            total += 1
            if max_images is not None and total >= max_images:
                return


def process_one_image(model, vae, image, label_id, args, device, fuse="mean"):
    """
    Single image: `image` is `PIL.Image`, `label_id` is an int
    Multiple images: `image` is `[PIL.Image,...]`, `label_id` is `[int,...]`
    Output: `rgbs(list)`, length = `n_time * n_layer` (exactly the same as the original)
    """
    #  list
    if isinstance(image, (list, tuple)):
        images = list(image)
        labels = list(label_id)
        assert len(images) == len(labels), "image list and label_id list must have same length"
    else:
        images = [image]
        labels = [int(label_id)]

    rgbs = []
    for t in args.times:
        features = get_f(
            model, vae, images, 256, device, labels,
            args.layers, t, baseline=args.baseline
        )
        pcas = pca_v(features, 3, args.thr, labels)
        rgbs.extend(pcas)
    return rgbs



def showpca(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    # need to be revised
    vae = AutoencoderKL.from_pretrained("path_to_models").to(device)

    # Load model
    if args.baseline:
        from sito import SiT_models
        model = SiT_models["SiT-XL/2"](input_size=32, num_classes=1000)
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt["ema"]
        model.load_state_dict(state, strict=False)
    else:
        from sit_wo_trick import SiT_models
        model = SiT_models["SiT-XL/2"](num_classes=1000)
        load_sit_from_complex_ckpt(model, args.ckpt, device=device, prefer_ema=True, strict=True)

    model.to(device).eval()
    print(f"Loaded model from {args.ckpt}")

    if args.folder is None:
        image = Image.open(args.img_path).convert("RGB")
        label_id = int(args.label_id)
    else:
        # folder mode: load all images and labels from subfolders
        folder = Path(args.folder)

        images = []
        labels = []

        subdirs = [p for p in folder.iterdir() if p.is_dir()]
        # Sort by label number
        subdirs.sort(key=lambda p: int(p.name))

        for sd in subdirs:
            try:
                lab = int(sd.name)
            except:
                continue
            img_paths = [p for p in sd.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            img_paths.sort(key=lambda p: p.name.lower())

            for p in img_paths:
                images.append(Image.open(p).convert("RGB"))
                labels.append(lab)

        image = images          # note: list
        label_id = labels       # note: list

    rgbs = process_one_image(model, vae, image, label_id, args, device)

    # Plot
    n_layer = len(args.layers)
    n_time = len(args.times)
    fig, axes = plt.subplots(n_time, n_layer, figsize=(n_layer * 2, n_time * 2))
    axes = axes.flatten() if n_time > 1 or n_layer > 1 else [axes]
    for ax, img in zip(axes, rgbs):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path, dpi=600)
        print(f"Saved to {args.save_path}")
    else:
        plt.show()



if __name__ == "__main__":
    showpca(parse_args())