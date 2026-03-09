#!/usr/bin/env python3
"""
From a dataset root containing many subfolders (e.g., 1000 class folders),
randomly pick N images from each subfolder and copy it into a new folder.

Usage:
  python pick_one_per_subfolder.py --src /path/to/dataset --dst /path/to/output --seed 42 --num-per-subfolder 3

Notes:
- Skips subfolders with no images.
- If a subfolder has fewer than N images, it will pick all of them.
- By default copies files; use --move to move instead.
- Output filenames are prefixed with subfolder name to avoid collisions.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=str, help="Dataset root directory")
    parser.add_argument("--dst", required=True, type=str, help="Output directory")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--recursive", action="store_true",
                        help="If set, search images recursively inside each subfolder")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks instead of copy")

    # NEW: number of images per subfolder
    parser.add_argument("--num-per-subfolder", type=int, default=1,
                        help="How many images to pick from each subfolder (default: 1)")

    args = parser.parse_args()

    if args.num_per_subfolder <= 0:
        raise ValueError("--num-per-subfolder must be a positive integer.")

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        raise FileNotFoundError(f"--src is not a directory: {src}")

    if args.symlink and args.move:
        raise ValueError("Choose only one of --symlink or --move (or neither for copy).")

    random.seed(args.seed)

    # Only direct subfolders under src
    subfolders = [p for p in src.iterdir() if p.is_dir()]
    subfolders.sort()

    picked = 0
    skipped = 0

    for sub in subfolders:
        if args.recursive:
            candidates = [p for p in sub.rglob("*") if is_image(p)]
        else:
            candidates = [p for p in sub.iterdir() if is_image(p)]

        if not candidates:
            skipped += 1
            continue

        k = min(args.num_per_subfolder, len(candidates))
        # random.sample guarantees no duplicates
        chosen_list = random.sample(candidates, k=k)

        for chosen in chosen_list:
            # Avoid name collisions by prefixing with folder name
            out_name = f"{sub.name}__{chosen.name}"
            out_path = dst / out_name

            # If collision still happens, append an index
            idx = 1
            while out_path.exists():
                out_name = f"{sub.name}__{chosen.stem}__{idx}{chosen.suffix}"
                out_path = dst / out_name
                idx += 1

            if args.symlink:
                os.symlink(chosen, out_path)
            elif args.move:
                shutil.move(str(chosen), str(out_path))
            else:
                shutil.copy2(str(chosen), str(out_path))

            picked += 1

    print(f"Done. Picked {picked} images into: {dst}")
    if skipped:
        print(f"Skipped {skipped} subfolders with no images.")


if __name__ == "__main__":
    main()