#!/usr/bin/env bash

CONFIG="configs/SiT_B_2.yaml"
DEVICES="0,1,2,3,4,5,6,7"

# ====== 修改文件夹路径======
DIRS=(
  "sit_repa"
  "slide_4567"
  "slide_4567_4MLP"
  "slide_timstep_9876"
  "slide_46_2MLP"
  "dino_timestep_9876_no_sit_slide"
  "slide_2468"
  "slide_2345678"
)
# ============================================================

for DIR in "${DIRS[@]}"; do
  if [[ ! -d "$DIR" ]]; then
    echo "[WARN] Not a directory, skip: $DIR" >&2
    continue
  fi

  echo "=============================="
  echo "[DIR] $DIR"

  # Collect ckpts (exclude last.ckpt), sort for deterministic order
  mapfile -t CKPTS < <(ls -1 "$DIR"/*.ckpt 2>/dev/null | grep -vE '/last\.ckpt$' | sort || true)

  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[INFO] No ckpt found (excluding last.ckpt) in: $DIR"
    continue
  fi

  for CKPT in "${CKPTS[@]}"; do
    echo "------------------------------"
    echo "[RUN] $CKPT"
    echo "CUDA_VISIBLE_DEVICES=${DEVICES} python main.py predict -c ${CONFIG} --ckpt_path ${CKPT}"

    CUDA_VISIBLE_DEVICES="${DEVICES}" \
      python main.py predict -c "${CONFIG}" --ckpt_path "${CKPT}"

    echo "[DONE] $CKPT"
  done
done

echo "All done."