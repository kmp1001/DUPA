#!/usr/bin/env bash
set -e
# conda activate ddt

# Todo: organize a readme.md for better readability

# My assumption is that the file paths are as follows:

# BASE
# test_cg
# decoupled_sit.py
# test.py
# move.py
# run_XXXX_XXXX (generated in each run)
# dummy (generated in the first run, used in subsequent runs)
# result_angle


# The result folder corresponding to ckpt

# 1 (contains results)
# run.log
# Note that you should execute this command multiple times for each experiment's result ckpt folder (which may contain multiple folders, possibly 4 or 8), because the file paths and --gradient parameters passed in each time will result in different results.
# e.g., for alignment 0 1 2 3 (order doesn't matter), locate the corresponding ckpt folder, usually something like exp_repa_flatten_condit8_dit4_fixt_xl-009
# It will automatically find all ckpt files within it. Additionally, modify the --gradient parameter in line 40 below to 0 1 2 3 (1234 setting), 0 1 (alignment 0 1), 4 5 6 7 (alignment 4567), etc., to obtain different results.
# Tips: For alignments of 4567 and 7654, both use the --gradient parameter as 4 5 6 7; the order doesn't matter.

BASE=/data/lzw_25/DDT_revise
SRC=/data/lzw_25/dataset/ImageNet/val   # Modified to: ImageNet — training dataset path
OUT=$BASE/test_cg/run_$(date +%m%d_%H%M) # Automatically create a new directory without overwriting
CKPT=/data/lzw_25/SiT_adapted_from_DDT/universal_flow_workdirs/exp_repa_flatten_condit8_dit4_fixt_xl-027
VAE=$BASE/models                         # Modified to: specific path to models
TEST=$BASE/test_cg/test.py               # Modified to: path to test.py
MOVE=$BASE/test_cg/move.py               # Modified to: path to move.py

# The following two lines can be commented out after the first run.
mkdir -p "$OUT/dummy/1000" "$OUT/result_angle"
python "$MOVE" --src "$SRC" --dst "$OUT/dummy/1000" --seed 42 --num-per-subfolder 1

# For the second run, you need to add the path to the dataset obtained in the first run.
# OUT=/data/lzw_25/DDT_revise/test_cg/run_0228_1126        # Modified: The path obtained after the first run
for CKPTDIR in "$CKPT"; do
  NAME=$(basename "$CKPTDIR")
  RDIR=$OUT/result_angle/$NAME
  mkdir -p "$RDIR"
  torchrun --nnodes=1 --nproc_per_node=8 "$TEST" \
    --data-path "$OUT/dummy" --vae "$VAE" \
    --ckpt "$CKPTDIR" \
    --global-batch-size 80 --output "$RDIR" \
    --gradient 0 1 2 3 --num_MLP 4 \
    2>&1 | tee "$RDIR/run.log"
done

