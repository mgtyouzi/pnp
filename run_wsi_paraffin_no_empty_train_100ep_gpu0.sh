#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
GPU_ID="0"
PARAFFIN_DATASET="${PARAFFIN_DATASET:-}"
if [ -z "${PARAFFIN_DATASET}" ]; then
  PARAFFIN_DATASET=$'\347\237\263\350\234\241-2025-wsi'
fi

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-4e-5}"
EOS_COEF="${EOS_COEF:-0.5}"
MATCH_DIS="${MATCH_DIS:-15}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [ "${GPU_ID}" = "7" ]; then
  echo "ERROR: GPU 7 is forbidden. Refusing to run."
  exit 2
fi

cd "${PROJECT_ROOT}"

DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset-new/${PARAFFIN_DATASET}}"
MEAN_STD_PATH="${MEAN_STD_PATH:-${DATASET_PATH}/mean_std.npy}"
timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/wsi_paraffin_no_empty_100ep_${timestamp}}"
mkdir -p "${OUT_DIR}"

if [ ! -d "${DATASET_PATH}/train_image" ] || [ ! -d "${DATASET_PATH}/val_image" ]; then
  echo "ERROR: dataset-new WSI folders not found under ${DATASET_PATH}"
  echo "Run: bash run_create_dataset_new_wsi_safe.sh"
  exit 3
fi
if [ ! -f "${MEAN_STD_PATH}" ]; then
  echo "ERROR: mean/std file not found: ${MEAN_STD_PATH}"
  exit 4
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENTRYPOINT=train_p2p_no_empty_v2.py"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR}"
echo "EOS_COEF=${EOS_COEF}"
echo "MATCH_DIS=${MATCH_DIS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "MEAN_STD_PATH=${MEAN_STD_PATH}"
echo "Protocol: WSI-level paraffin train -> paraffin val; final test uses eval script."

python train_p2p_no_empty_v2.py \
  --dataset "${DATASET_PATH}" \
  --output_dir "${OUT_DIR}" \
  --num_classes=1 \
  --epochs="${EPOCHS}" \
  --batch_size="${BATCH_SIZE}" \
  --lr="${LR}" \
  --eos_coef="${EOS_COEF}" \
  --start_eval=-1 \
  --num_workers="${NUM_WORKERS}" \
  --match_dis="${MATCH_DIS}" \
  --mean_std_path "${MEAN_STD_PATH}" \
  --eval_split val \
  --filter_empty_train \
  --skip_empty_eval \
  2>&1 | tee "${OUT_DIR}/console.log"

echo "Done. Output:"
echo "${OUT_DIR}"
