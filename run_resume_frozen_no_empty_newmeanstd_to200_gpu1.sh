#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-30533}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-4e-5}"
EOS_COEF="${EOS_COEF:-0.5}"
MATCH_DIS="${MATCH_DIS:-15}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [ -z "${FROZEN_DATASET:-}" ]; then
  FROZEN_DATASET=$'\345\206\260\345\206\273-2025'
fi

DEFAULT_MEAN_STD_PATH="${PROJECT_ROOT}/experiments/mean_std_check_20260629_035548/frozen_mean_std_train.npy"
MEAN_STD_PATH="${MEAN_STD_PATH:-${DEFAULT_MEAN_STD_PATH}}"

# Safety guard: GPU 7 must never be used in this container.
NORMALIZED_CUDA=",${CUDA_VISIBLE_DEVICES// /,},"
if [[ "${NORMALIZED_CUDA}" == *",7,"* ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} includes GPU 7. Refusing to run."
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  echo "ERROR: this script is fixed to GPU 1. Got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  exit 2
fi

cd "${PROJECT_ROOT}"

if [ ! -f "${MEAN_STD_PATH}" ]; then
  echo "ERROR: MEAN_STD_PATH not found: ${MEAN_STD_PATH}"
  exit 3
fi

if [ -z "${RESUME:-}" ]; then
  if [ -n "${RESUME_DIR:-}" ]; then
    RESUME="${RESUME_DIR}/recent_model.pth"
  else
    shopt -s nullglob
    candidates=("${PROJECT_ROOT}"/experiments/original_frozen_no_empty_newmeanstd_100ep_*)
    shopt -u nullglob
    if [ "${#candidates[@]}" -eq 0 ]; then
      echo "ERROR: no 100epoch run directory found."
      echo "Set RESUME=/path/to/recent_model.pth or RESUME_DIR=/path/to/old_output_dir."
      exit 4
    fi
    RESUME="$(ls -dt "${candidates[@]}" | head -n 1)/recent_model.pth"
  fi
fi

if [ ! -f "${RESUME}" ]; then
  echo "ERROR: resume checkpoint not found: ${RESUME}"
  echo "Use the 100epoch recent checkpoint, for example:"
  echo "  RESUME_DIR=/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/experiments/original_frozen_no_empty_newmeanstd_100ep_xxx bash $0"
  exit 5
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/original_frozen_no_empty_newmeanstd_resume_to200_gpu1_${timestamp}}"
mkdir -p "${OUT_DIR}"
CONSOLE_LOG="${OUT_DIR}/console_resume_to200_gpu1_${timestamp}.log"

export MASTER_ADDR
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENTRYPOINT=train_p2p_no_empty_v2.py"
echo "FROZEN_DATASET=${FROZEN_DATASET}"
echo "RESUME=${RESUME}"
echo "OUT_DIR=${OUT_DIR}"
echo "CONSOLE_LOG=${CONSOLE_LOG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR}"
echo "EOS_COEF=${EOS_COEF}"
echo "MATCH_DIS=${MATCH_DIS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "MEAN_STD_PATH=${MEAN_STD_PATH}"
echo "Protocol: resume frozen train -> frozen test to 200 epochs on GPU 1; new output dir/log; old 100epoch logs/checkpoints are not overwritten"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun \
  --nproc_per_node=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train_p2p_no_empty_v2.py \
  --dataset "${FROZEN_DATASET}" \
  --output_dir "${OUT_DIR}" \
  --resume "${RESUME}" \
  --num_classes=1 \
  --epochs="${EPOCHS}" \
  --batch_size="${BATCH_SIZE}" \
  --lr="${LR}" \
  --eos_coef="${EOS_COEF}" \
  --start_eval=-1 \
  --num_workers="${NUM_WORKERS}" \
  --match_dis="${MATCH_DIS}" \
  --mean_std_path "${MEAN_STD_PATH}" \
  --filter_empty_train \
  --skip_empty_eval \
  2>&1 | tee "${CONSOLE_LOG}"

echo "Done. Output:"
echo "${OUT_DIR}"
