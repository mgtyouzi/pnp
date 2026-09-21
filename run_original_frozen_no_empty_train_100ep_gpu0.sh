#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-30521}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-4e-5}"
EOS_COEF="${EOS_COEF:-0.5}"
MATCH_DIS="${MATCH_DIS:-15}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MEAN_STD_PATH="${MEAN_STD_PATH:-}"
TRAIN_MEAN_STD_PATH="${TRAIN_MEAN_STD_PATH:-}"
TEST_MEAN_STD_PATH="${TEST_MEAN_STD_PATH:-}"

MEAN_STD_ARGS=()
if [ -n "${MEAN_STD_PATH}" ]; then
  MEAN_STD_ARGS+=(--mean_std_path "${MEAN_STD_PATH}")
fi
if [ -n "${TRAIN_MEAN_STD_PATH}" ]; then
  MEAN_STD_ARGS+=(--train_mean_std_path "${TRAIN_MEAN_STD_PATH}")
fi
if [ -n "${TEST_MEAN_STD_PATH}" ]; then
  MEAN_STD_ARGS+=(--test_mean_std_path "${TEST_MEAN_STD_PATH}")
fi

# Safety guard: GPU 7 must never be used in this container.
NORMALIZED_CUDA=",${CUDA_VISIBLE_DEVICES// /,},"
if [[ "${NORMALIZED_CUDA}" == *",7,"* ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} includes GPU 7. Refusing to run."
  exit 2
fi

if [ -z "${FROZEN_DATASET:-}" ]; then
  FROZEN_DATASET=$'\345\206\260\345\206\273-2025'
fi

cd "${PROJECT_ROOT}"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/original_frozen_no_empty_train_eval_${timestamp}}"
mkdir -p "${OUT_DIR}"

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
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR}"
echo "EOS_COEF=${EOS_COEF}"
echo "MATCH_DIS=${MATCH_DIS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "MEAN_STD_PATH=${MEAN_STD_PATH}"
echo "TRAIN_MEAN_STD_PATH=${TRAIN_MEAN_STD_PATH}"
echo "TEST_MEAN_STD_PATH=${TEST_MEAN_STD_PATH}"
echo "MEAN_STD_ARGS=${MEAN_STD_ARGS[*]:-<default dataset mean_std.npy>}"
echo "Protocol: original project, train/test empty-GT patches filtered out; eval metrics also skip any remaining empty-GT patches"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun \
  --nproc_per_node=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train_p2p_no_empty_v2.py \
  --dataset "${FROZEN_DATASET}" \
  --output_dir "${OUT_DIR}" \
  --num_classes=1 \
  --epochs="${EPOCHS}" \
  --batch_size="${BATCH_SIZE}" \
  --lr="${LR}" \
  --eos_coef="${EOS_COEF}" \
  --start_eval=-1 \
  --num_workers="${NUM_WORKERS}" \
  --match_dis="${MATCH_DIS}" \
  "${MEAN_STD_ARGS[@]}" \
  --filter_empty_train \
  --skip_empty_eval \
  2>&1 | tee "${OUT_DIR}/console.log"

echo "Done. Output:"
echo "${OUT_DIR}"
