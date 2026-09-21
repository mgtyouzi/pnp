#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"

if [ -z "${PARAFFIN_DATASET:-}" ]; then
  PARAFFIN_DATASET=$'\347\237\263\350\234\241-2025'
fi
if [ -z "${FROZEN_DATASET:-}" ]; then
  FROZEN_DATASET=$'\345\206\260\345\206\273-2025'
fi

cd "${PROJECT_ROOT}"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/mean_std_check_${timestamp}}"
mkdir -p "${OUT_DIR}"

PARAFFIN_HIST="${PROJECT_ROOT}/datasets/${PARAFFIN_DATASET}/mean_std.npy"
FROZEN_HIST="${PROJECT_ROOT}/datasets/${FROZEN_DATASET}/mean_std.npy"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PARAFFIN_DATASET=${PARAFFIN_DATASET}"
echo "FROZEN_DATASET=${FROZEN_DATASET}"
echo "OUT_DIR=${OUT_DIR}"
echo "PARAFFIN_HIST=${PARAFFIN_HIST}"
echo "FROZEN_HIST=${FROZEN_HIST}"
echo "This script writes recomputed stats under OUT_DIR only; historical mean_std.npy is not overwritten."

python compute_mean_std.py \
  --dataset "${PARAFFIN_DATASET}" \
  --split train \
  --output "${OUT_DIR}/paraffin_mean_std_train.npy" \
  --compare "${PARAFFIN_HIST}" \
  2>&1 | tee "${OUT_DIR}/paraffin_train.log"

python compute_mean_std.py \
  --dataset "${PARAFFIN_DATASET}" \
  --split all \
  --output "${OUT_DIR}/paraffin_mean_std_all.npy" \
  --compare "${PARAFFIN_HIST}" \
  2>&1 | tee "${OUT_DIR}/paraffin_all.log"

python compute_mean_std.py \
  --dataset "${FROZEN_DATASET}" \
  --split train \
  --output "${OUT_DIR}/frozen_mean_std_train.npy" \
  --compare "${FROZEN_HIST}" \
  2>&1 | tee "${OUT_DIR}/frozen_train_vs_frozen_hist.log"

python compute_mean_std.py \
  --dataset "${FROZEN_DATASET}" \
  --split train \
  --output "${OUT_DIR}/frozen_mean_std_train_vs_paraffin.npy" \
  --compare "${PARAFFIN_HIST}" \
  2>&1 | tee "${OUT_DIR}/frozen_train_vs_paraffin_hist.log"

python compute_mean_std.py \
  --dataset "${FROZEN_DATASET}" \
  --split all \
  --output "${OUT_DIR}/frozen_mean_std_all.npy" \
  --compare "${FROZEN_HIST}" \
  2>&1 | tee "${OUT_DIR}/frozen_all_vs_frozen_hist.log"

echo "Done. Recomputed mean/std files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name "*.npy" -print | sort
echo "Use frozen train stats for a controlled frozen training run, for example:"
echo "MEAN_STD_PATH=\"${OUT_DIR}/frozen_mean_std_train.npy\" bash run_original_frozen_no_empty_train_100ep_gpu0.sh"