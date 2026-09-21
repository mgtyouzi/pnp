#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MATCH_DIS="${MATCH_DIS:-15}"
NEAR_RADIUS="${NEAR_RADIUS:-30}"
DEDUP_INTERVAL="${DEDUP_INTERVAL:-15}"
MAX_IMAGES="${MAX_IMAGES:-24}"
MAX_POINTS_PER_KIND="${MAX_POINTS_PER_KIND:-40}"
MAX_CROPS_PER_KIND="${MAX_CROPS_PER_KIND:-12}"
CROP_SIZE="${CROP_SIZE:-256}"

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
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/hard_case_diag_${timestamp}}"
mkdir -p "${OUT_DIR}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "FROZEN_DATASET=${FROZEN_DATASET}"
echo "CKPT_PAR=${CKPT_PAR}"
echo "CKPT_FROZEN=${CKPT_FROZEN}"
echo "DIAGNOSIS_DIR=${DIAGNOSIS_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MAX_IMAGES=${MAX_IMAGES}"
echo "MAX_POINTS_PER_KIND=${MAX_POINTS_PER_KIND}"
echo "MAX_CROPS_PER_KIND=${MAX_CROPS_PER_KIND}"
echo "MATCH_DIS=${MATCH_DIS}"
echo "NEAR_RADIUS=${NEAR_RADIUS}"
echo "DEDUP_INTERVAL=${DEDUP_INTERVAL}"
echo "CROP_SIZE=${CROP_SIZE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python export_hard_case_diagnostics.py \
  --paraffin_checkpoint "${CKPT_PAR}" \
  --frozen_checkpoint "${CKPT_FROZEN}" \
  --diagnosis_dir "${DIAGNOSIS_DIR}" \
  --dataset "${FROZEN_DATASET}" \
  --output_dir "${OUT_DIR}" \
  --num_classes=1 \
  --num_workers="${NUM_WORKERS}" \
  --match_dis="${MATCH_DIS}" \
  --near_radius="${NEAR_RADIUS}" \
  --dedup_interval="${DEDUP_INTERVAL}" \
  --max_images="${MAX_IMAGES}" \
  --max_points_per_kind="${MAX_POINTS_PER_KIND}" \
  --max_crops_per_kind="${MAX_CROPS_PER_KIND}" \
  --crop_size="${CROP_SIZE}" \
  2>&1 | tee "${OUT_DIR}/console.log"

echo "Done. Hard-case diagnostics output:"
echo "${OUT_DIR}"
