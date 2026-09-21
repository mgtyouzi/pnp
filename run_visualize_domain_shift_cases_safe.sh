#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MATCH_DIS="${MATCH_DIS:-15}"
DEDUP_INTERVAL="${DEDUP_INTERVAL:-15}"
MAX_IMAGES="${MAX_IMAGES:-16}"
MAX_PANEL_WIDTH="${MAX_PANEL_WIDTH:-960}"

# Safety guard: GPU 7 must never be used in this container.
NORMALIZED_CUDA=",${CUDA_VISIBLE_DEVICES// /,},"
if [[ "${NORMALIZED_CUDA}" == *",7,"* ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} includes GPU 7. Refusing to run."
  exit 2
fi

if [ -z "${FROZEN_DATASET:-}" ]; then
  FROZEN_DATASET=$'\345\206\260\345\206\273-2025'
fi

CKPT_PAR="${CKPT_PAR:-${PROJECT_ROOT}/pth/0318/best_model.pth}"
CKPT_FROZEN="${CKPT_FROZEN:-${PROJECT_ROOT}/pth/0326_bd/best_model.pth}"

cd "${PROJECT_ROOT}"

if [ -z "${DIAGNOSIS_DIR:-}" ]; then
  DIAGNOSIS_DIR="$(ls -td "${PROJECT_ROOT}"/experiments/error_diagnosis_* 2>/dev/null | head -n 1 || true)"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/domain_shift_vis_${timestamp}}"
mkdir -p "${OUT_DIR}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "FROZEN_DATASET=${FROZEN_DATASET}"
echo "CKPT_PAR=${CKPT_PAR}"
echo "CKPT_FROZEN=${CKPT_FROZEN}"
echo "DIAGNOSIS_DIR=${DIAGNOSIS_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MAX_IMAGES=${MAX_IMAGES}"
echo "MATCH_DIS=${MATCH_DIS}"
echo "DEDUP_INTERVAL=${DEDUP_INTERVAL}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python visualize_domain_shift_cases.py \
  --paraffin_checkpoint "${CKPT_PAR}" \
  --frozen_checkpoint "${CKPT_FROZEN}" \
  --diagnosis_dir "${DIAGNOSIS_DIR}" \
  --dataset "${FROZEN_DATASET}" \
  --output_dir "${OUT_DIR}" \
  --num_classes=1 \
  --num_workers="${NUM_WORKERS}" \
  --match_dis="${MATCH_DIS}" \
  --dedup_interval="${DEDUP_INTERVAL}" \
  --max_images="${MAX_IMAGES}" \
  --max_panel_width="${MAX_PANEL_WIDTH}" \
  2>&1 | tee "${OUT_DIR}/console.log"

echo "Done. Visualization output:"
echo "${OUT_DIR}"
