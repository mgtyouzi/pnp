#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
timestamp="$(date +%Y%m%d_%H%M%S)"
EXP_DIR="${EXP_DIR:-${PROJECT_ROOT}/experiments/p2p_proto_k4x4_d128_w0p05_100ep_${timestamp}_bt${BATCH_SIZE}}"
IFS=',' read -r -a TRAIN_GPU_IDS <<< "${TRAIN_GPUS// /}"
TRAIN_NPROC="${#TRAIN_GPU_IDS[@]}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
echo "[Pipeline] unit tests"
python test_p2p_prototype.py
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python smoke_test_p2p_prototype_integration.py
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun \
  --nproc_per_node="${TRAIN_NPROC}" \
  --master_addr=127.0.0.1 \
  --master_port="${DDP_TEST_PORT:-29640}" \
  test_p2p_prototype_ddp.py

echo "[Pipeline] 100-epoch source training on physical GPUs ${TRAIN_GPUS}"
PROJECT_ROOT="${PROJECT_ROOT}" \
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
BATCH_SIZE="${BATCH_SIZE}" \
OUT_DIR="${EXP_DIR}" \
bash run_p2p_prototype_100ep_2gpu.sh

echo "[Pipeline] frozen-domain raw/fusion evaluation on physical GPU ${EVAL_GPU}"
PROJECT_ROOT="${PROJECT_ROOT}" \
GPU="${EVAL_GPU}" \
EXP_DIR="${EXP_DIR}" \
bash run_eval_p2p_prototype_all_ckpts_gpu.sh

echo "[SUCCESS] full pipeline completed: ${EXP_DIR}"
