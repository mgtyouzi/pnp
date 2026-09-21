#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
PARAFFIN_DATASET="${PARAFFIN_DATASET:-/home/data/yh_1/SET_3/p2p-src-zzh-2025/datasets/石蜡-2025}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DIAGNOSTIC_EPOCHS="${DIAGNOSTIC_EPOCHS:-8}"
DEBUG_BATCHES="${DEBUG_BATCHES:-100}"

cd "${PROJECT_ROOT}"

if [ ! -f "${BASELINE_CHECKPOINT}" ]; then
  echo "[ERROR] baseline checkpoint not found: ${BASELINE_CHECKPOINT}"
  exit 2
fi
if ! grep -q 'PROTOTYPE_IMPLEMENTATION_VERSION = "prototype_v2_1_hardbg_kmeans_ema_20260814"' \
  models/p2p_prototype.py; then
  echo "[ERROR] stale models/p2p_prototype.py"
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/p2p_proto_v2_pretrained_diag_${timestamp}_gpu${GPU}_bt${BATCH_SIZE}}"
mkdir -p "${OUT_DIR}"

echo "[Diagnostic] initialize P2P from ${BASELINE_CHECKPOINT}"
echo "[Diagnostic] epoch0 initializes prototypes; later epochs test separation and EMA"

PROJECT_ROOT="${PROJECT_ROOT}" \
PARAFFIN_DATASET="${PARAFFIN_DATASET}" \
OUT_DIR="${OUT_DIR}" \
CUDA_VISIBLE_DEVICES="${GPU}" \
BATCH_SIZE="${BATCH_SIZE}" \
EPOCHS="${DIAGNOSTIC_EPOCHS}" \
INIT_CHECKPOINT="${BASELINE_CHECKPOINT}" \
DEBUG_MAX_TRAIN_BATCHES="${DEBUG_BATCHES}" \
PROTO_START_EPOCH=0 \
PROTO_NUM_FG=4 \
PROTO_NUM_BG=8 \
PROTO_EMBEDDING_DIM=128 \
PROTO_TEMPERATURE=0.2 \
PROTO_LOSS_WEIGHT=0.01 \
PROTO_INITIAL_POSITIVE_RADIUS=10 \
PROTO_POSITIVE_RADIUS=15 \
PROTO_BACKGROUND_RADIUS=30 \
PROTO_MAX_POS_PER_IMAGE=64 \
PROTO_MAX_HARD_BG_PER_IMAGE=16 \
PROTO_MAX_RANDOM_BG_PER_IMAGE=16 \
PROTO_KMEANS_ITERATIONS=20 \
PROTO_MOMENTUM=0.99 \
PROTO_DEAD_PATIENCE=3 \
PROTO_INPUT_GRAD_SCALE=0.1 \
PROTO_DEBUG_INTERVAL=10 \
bash run_p2p_prototype_100ep_2gpu.sh

python summarize_prototype_v2_debug.py --experiment "${OUT_DIR}"
printf '%s\n' "${OUT_DIR}" > "${PROJECT_ROOT}/experiments/latest_p2p_prototype_v2_pretrained_diagnostic.txt"
echo "[SUCCESS] pretrained Prototype-v2 diagnostic completed: ${OUT_DIR}"
