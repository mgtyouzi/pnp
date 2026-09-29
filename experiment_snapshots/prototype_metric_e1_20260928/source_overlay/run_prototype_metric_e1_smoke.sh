#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
DATASET="${DATASET:-/path/to/Paraffin-2025}"
CHECKPOINT="${CHECKPOINT:-/path/to/0318/best_model.pth}"
OUT="${OUT:-experiments/prototype_metric_e1_smoke_$(date +%Y%m%d_%H%M%S)_gpu${GPU}}"
PYTHON="${PYTHON:-python}"
PROTO_LOSS_COEF="${PROTO_LOSS_COEF:-0.005}"

if [[ ! -d "${DATASET}/train_image" || ! -d "${DATASET}/test_image" ]]; then
  echo "[ERROR] Paraffin train/test folders missing: ${DATASET}" >&2
  exit 2
fi
if [[ ! -f "${DATASET}/mean_std.npy" || ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] mean/std or baseline checkpoint missing" >&2
  exit 2
fi

GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="${GPU}" | awk 'NR == 1 {gsub(/[[:space:]]/, "", $1); print $1}')"
if [[ ! "${GPU_USED}" =~ ^[0-9]+$ ]] || (( GPU_USED > 1000 )); then
  echo "[ABORT] GPU ${GPU} is unavailable or already using ${GPU_USED:-unknown} MiB" >&2
  exit 3
fi

mkdir -p "${OUT}"
echo "[E1-smoke] physical_gpu=${GPU} visible_gpu=0 output=${OUT}"
if CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" train_p2p_no_empty_v2.py \
  --dataset "${DATASET}" \
  --mean_std_path "${DATASET}/mean_std.npy" \
  --output_dir "${OUT}" \
  --epochs 1 \
  --start_eval 999 \
  --batch_size 1 \
  --num_workers 0 \
  --use_proto_e1 \
  --init_checkpoint "${CHECKPOINT}" \
  --proto_hidden_dim 128 \
  --proto_embed_dim 64 \
  --proto_max_pos 32 \
  --proto_max_neg 64 \
  --proto_pos_radius 15 \
  --proto_neg_radius 30 \
  --proto_margin 0.1 \
  --proto_beta 16 \
  --proto_loss_coef "${PROTO_LOSS_COEF}" \
  --proto_grad_audit_interval 1 \
  --debug_max_train_batches 5 > "${OUT}/console.log" 2>&1; then
  tail -n 80 "${OUT}/console.log"
  test -s "${OUT}/final_model.pth"
  echo "[E1-smoke] PASS checkpoint=${OUT}/final_model.pth"
else
  status=$?
  tail -n 120 "${OUT}/console.log" || true
  exit "${status}"
fi
