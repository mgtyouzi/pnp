#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
GPU_ID="1"
FROZEN_DATASET="${FROZEN_DATASET:-}"
if [ -z "${FROZEN_DATASET}" ]; then
  FROZEN_DATASET=$'\345\206\260\345\206\273-2025-wsi'
fi

MATCH_DIS="${MATCH_DIS:-15}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [ "${GPU_ID}" = "7" ]; then
  echo "ERROR: GPU 7 is forbidden. Refusing to run."
  exit 2
fi

cd "${PROJECT_ROOT}"

DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset-new/${FROZEN_DATASET}}"
MEAN_STD_PATH="${MEAN_STD_PATH:-${DATASET_PATH}/mean_std.npy}"
if [ -z "${CHECKPOINT:-}" ]; then
  LATEST_DIR="$(find "${PROJECT_ROOT}/experiments" -maxdepth 1 -type d -name 'wsi_frozen_no_empty_100ep_*' | sort | tail -n 1 || true)"
  if [ -z "${LATEST_DIR}" ]; then
    echo "ERROR: no wsi_frozen_no_empty_100ep_* experiment found."
    echo "You can set CHECKPOINT=/path/to/best_model.pth bash $0"
    exit 3
  fi
  CHECKPOINT="${LATEST_DIR}/best_model.pth"
fi

if [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}"
  echo "You can set CHECKPOINT=/path/to/best_model.pth bash $0"
  exit 3
fi
if [ ! -d "${DATASET_PATH}/test_image" ]; then
  echo "ERROR: test split not found under ${DATASET_PATH}"
  exit 4
fi
if [ ! -f "${MEAN_STD_PATH}" ]; then
  echo "ERROR: mean/std file not found: ${MEAN_STD_PATH}"
  exit 5
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="${EVAL_DIR:-$(dirname "${CHECKPOINT}")/final_eval_test_${timestamp}}"
mkdir -p "${EVAL_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENTRYPOINT=eval_p2p_no_empty_wsi.py"
echo "DATASET_PATH=${DATASET_PATH}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "EVAL_DIR=${EVAL_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MATCH_DIS=${MATCH_DIS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "MEAN_STD_PATH=${MEAN_STD_PATH}"
echo "Protocol: final WSI-level frozen test, empty-GT patches skipped."

python eval_p2p_no_empty_wsi.py \
  --dataset "${DATASET_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --eval_output_dir "${EVAL_DIR}" \
  --output_dir "${EVAL_DIR}" \
  --num_classes=1 \
  --num_workers="${NUM_WORKERS}" \
  --match_dis="${MATCH_DIS}" \
  --mean_std_path "${MEAN_STD_PATH}" \
  --eval_split test \
  --skip_empty_eval \
  2>&1 | tee "${EVAL_DIR}/console.log"

echo "Done. Output:"
echo "${EVAL_DIR}"
