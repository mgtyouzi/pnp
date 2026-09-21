#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEBUG_BATCHES="${DEBUG_BATCHES:-20}"

cd "${PROJECT_ROOT}"
timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/p2p_proto_v2_mechanism_smoke_${timestamp}_gpu${CUDA_VISIBLE_DEVICES}_bt${BATCH_SIZE}}"
mkdir -p "${OUT_DIR}"

if ! grep -q 'PROTOTYPE_IMPLEMENTATION_VERSION = "prototype_v2_1_hardbg_kmeans_ema_20260814"' \
  models/p2p_prototype.py; then
  echo "[ERROR] stale models/p2p_prototype.py; sync the complete Prototype-v2 change set"
  exit 2
fi
if ! grep -q 'test_epoch_finalize_keeps_fifo_history_and_ema_updates_existing_center' \
  test_p2p_prototype.py; then
  echo "[ERROR] stale test_p2p_prototype.py; the old refresh-every-epoch test is incompatible with Prototype-v2"
  exit 2
fi

echo "[Stage 1/4] tensor-level Prototype-v2 tests"
python test_p2p_prototype.py

echo "[Stage 2/4] CUDA integration smoke on physical GPU ${CUDA_VISIBLE_DEVICES}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python smoke_test_p2p_prototype_integration.py

echo "[Stage 3/4] 22 short epochs; epoch20 initializes, epoch21 exercises loss and EMA"
OUT_DIR="${OUT_DIR}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
BATCH_SIZE="${BATCH_SIZE}" \
EPOCHS=22 \
DEBUG_MAX_TRAIN_BATCHES="${DEBUG_BATCHES}" \
PROTO_DEBUG_INTERVAL=1 \
bash run_p2p_prototype_100ep_2gpu.sh

echo "[Stage 4/4] mechanism audit"
python summarize_prototype_v2_debug.py --experiment "${OUT_DIR}"

printf '%s\n' "${OUT_DIR}" > "${PROJECT_ROOT}/experiments/latest_p2p_prototype_v2_smoke.txt"
echo "[SUCCESS] Prototype-v2 staged verification passed: ${OUT_DIR}"
