#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
GPU="${GPU:-6}"
DATASET="${DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
MEAN_STD="${MEAN_STD:-${DATASET}/mean_std.npy}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-3}"
DEBUG_MAX_TRAIN_BATCHES="${DEBUG_MAX_TRAIN_BATCHES:-100}"
FULL_EPOCHS="${FULL_EPOCHS:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"

cd "${PROJECT}"
if [ -z "${REFERENCE_PAIR_ROOT:-}" ]; then
  if [ -f experiments/latest_positive_only_train_proto_v2.txt ]; then
    REFERENCE_PAIR_ROOT="$(cat experiments/latest_positive_only_train_proto_v2.txt)"
  elif [ -f experiments/latest_positive_only_train_proto.txt ]; then
    REFERENCE_PAIR_ROOT="$(cat experiments/latest_positive_only_train_proto.txt)"
  else
    REFERENCE_PAIR_ROOT="${PROJECT}/experiments/positive_only_train_proto_paired_20260903_070311_gpu6_bt1"
  fi
fi
case "${REFERENCE_PAIR_ROOT}" in
  /*) ;;
  *) REFERENCE_PAIR_ROOT="${PROJECT}/${REFERENCE_PAIR_ROOT}" ;;
esac
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-${REFERENCE_PAIR_ROOT}/control_p2p/recent_model.pth}"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT}/experiments/prototype_reliability_rescue_gate_${RUN_ID}_gpu${GPU}_bt${BATCH_SIZE}}"
SMOKE_DIR="${PAIR_ROOT}/mechanism_smoke"
MODEL_DIR="${PAIR_ROOT}/prototype_reliability_rescue"

test -f "${CONTROL_CHECKPOINT}"
test -f "${INIT_CHECKPOINT}"
test -f "${MEAN_STD}"
mkdir -p "${PAIR_ROOT}/control_p2p" "${SMOKE_DIR}" "${MODEL_DIR}"
ln -sfn "${CONTROL_CHECKPOINT}" "${PAIR_ROOT}/control_p2p/recent_model.pth"
printf '%s\n' "${PAIR_ROOT}" > experiments/latest_prototype_reliability_rescue.txt

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[PRR] physical_gpu=${GPU}"
echo "[PRR] pair_root=${PAIR_ROOT}"
echo "[PRR] reused_control=${CONTROL_CHECKPOINT}"
echo "[PRR] protocol=3x100 mechanism gate -> 10 epoch paired source test"

if [ "${RUN_PREFLIGHT}" = "1" ]; then
  echo "[Stage 0/4] implementation preflight"
  python test_prototype_reliability_rescue.py
  python test_prototype_reliability_rescue_structure.py
  python test_prototype_reliability_rescue_gates.py
  python test_diagnostic_aggregation.py
  python test_paired_detection_threshold_sweep.py
  python -m py_compile \
    models/prototype_reliability_rescue.py models/detr.py train_p2p.py \
    diagnostic_aggregation.py prr_mechanism_gate.py prr_final_gate.py \
    paired_detection_threshold_sweep.py
fi

train_prr() {
  local output_dir="$1"
  local epochs="$2"
  local max_batches="$3"
  local start_eval="$4"
  local checkpoint_interval="$5"
  local debug_save="$6"
  python train_p2p_no_empty_v2.py \
    --dataset="${DATASET}" \
    --mean_std_path="${MEAN_STD}" \
    --batch_size="${BATCH_SIZE}" \
    --epochs="${epochs}" \
    --start_eval="${start_eval}" \
    --checkpoint_interval="${checkpoint_interval}" \
    --debug_max_train_batches="${max_batches}" \
    --debug_save_final_checkpoint="${debug_save}" \
    --init_checkpoint="${INIT_CHECKPOINT}" \
    --output_dir="${output_dir}" \
    --num_workers=0 \
    --reset_rng_after_init=1 \
    --deterministic_training=1 \
    --lr=4e-5 \
    --min_lr=1e-6 \
    --warmup_epochs=0 \
    --weight_decay=1e-4 \
    --clip_max_norm=0.1 \
    --eos_coef=0.5 \
    --reg_loss_coef=0.002 \
    --cls_loss_coef=1.0 \
    --set_cost_point=0.1 \
    --set_cost_class=1.0 \
    --num_classes=1 \
    --backbone=resnet50 \
    --position_embedding=sine \
    --enc_layers=6 \
    --dim_feedforward=2048 \
    --hidden_dim=256 \
    --dropout=0.1 \
    --nheads=8 \
    --row=2 \
    --col=2 \
    --match_dis=15 \
    --seed=0 \
    --filter_empty_train \
    --skip_empty_eval \
    --proto_enable \
    --proto_mode=prototype_reliability_rescue \
    --proto_start_epoch=0 \
    --proto_embedding_dim=64 \
    --proto_num_fg=4 \
    --proto_temperature=0.1 \
    --proto_loss_weight=1.0 \
    --proto_input_grad_scale=0.0 \
    --proto_positive_radius=15 \
    --proto_background_radius=30 \
    --proto_max_random_bg_per_image=16 \
    --proto_gt_support_queue_size=8192 \
    --proto_kmeans_iterations=20 \
    --proto_sampling_seed=0 \
    --proto_debug_interval=25 \
    --proto_debug_fail_fast=1 \
    --proto_inference_fusion=0 \
    --prr_hidden_dim=128 \
    --prr_warmup_epochs=2 \
    --prr_support_low_quantile=0.2 \
    --prr_support_high_quantile=0.8 \
    --prr_max_cell_probability=0.55 \
    --prr_target_margin=0.2 \
    --prr_margin_temperature=0.2 \
    --prr_rescue_weight=0.005 \
    --prr_logit_gradient_scale=0.005 \
    --prr_proto_rank_weight=0.005 \
    --prr_metric_weight=0.005 \
    --prr_center_weight=0.005 \
    --prr_balance_weight=0.001 \
    --prr_diversity_weight=0.001 \
    --prr_max_rescue_per_image=32 \
    2>&1 | tee "${output_dir}/console.log"
}

if [ "${RUN_SMOKE}" = "1" ]; then
  echo "[Stage 1/4] three-epoch bounded mechanism smoke"
  train_prr "${SMOKE_DIR}" "${SMOKE_EPOCHS}" \
    "${DEBUG_MAX_TRAIN_BATCHES}" 999 1 1
  python prr_mechanism_gate.py \
    --metrics="${SMOKE_DIR}/prototype_epoch_metrics.csv" \
    --output="${SMOKE_DIR}/mechanism_gate.json"
fi

echo "[Stage 2/4] ten-epoch paired source training"
train_prr "${MODEL_DIR}" "${FULL_EPOCHS}" 0 8 5 0
test -f "${MODEL_DIR}/recent_model.pth"

echo "[Stage 3/4] paired raw-P2P threshold diagnosis"
python paired_detection_threshold_sweep.py \
  --pair_root="${PAIR_ROOT}" \
  --prototype_subdir=prototype_reliability_rescue \
  --dataset="${DATASET}" \
  --mean_std_path="${MEAN_STD}" \
  --gpu=0 \
  --thresholds="${THRESHOLDS:-0.45:0.75:0.01}" \
  --match_distance=15 \
  --dedup_interval=15 \
  --near_radius=30 \
  --minimum_gain=0.002 \
  --num_workers=0 \
  --output_dir="${PAIR_ROOT}/raw_p2p_threshold_sweep"

echo "[Stage 4/4] final mechanism and detection gate"
python prr_final_gate.py \
  --metrics="${MODEL_DIR}/prototype_epoch_metrics.csv" \
  --threshold_decision="${PAIR_ROOT}/raw_p2p_threshold_sweep/paired_threshold_decision.json" \
  --output="${PAIR_ROOT}/prr_final_decision.json"

echo "[SUCCESS] ${PAIR_ROOT}"
