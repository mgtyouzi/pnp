#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
PARAFFIN_DATASET="${PARAFFIN_DATASET:-/home/data/yh_1/SET_3/p2p-src-zzh-2025/datasets/石蜡-2025}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODE="${MODE:-rapid}"

case "${MODE}" in
  rapid)
    DIAGNOSTIC_EPOCHS="${DIAGNOSTIC_EPOCHS:-8}"
    DEBUG_BATCHES="${DEBUG_BATCHES:-100}"
    RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
    DATASET_SMOKE_SAMPLES="${DATASET_SMOKE_SAMPLES:-200}"
    OFFLINE_FIT_IMAGES="${OFFLINE_FIT_IMAGES:-200}"
    OFFLINE_EVAL_IMAGES="${OFFLINE_EVAL_IMAGES:-200}"
    CLASSIFICATION_RELATIVE_LIMIT="${CLASSIFICATION_RELATIVE_LIMIT:-0.0}"
    REGRESSION_RELATIVE_LIMIT="${REGRESSION_RELATIVE_LIMIT:-0.02}"
    ;;
  faithful)
    DIAGNOSTIC_EPOCHS="${DIAGNOSTIC_EPOCHS:-4}"
    DEBUG_BATCHES="${DEBUG_BATCHES:-0}"
    RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"
    DATASET_SMOKE_SAMPLES="${DATASET_SMOKE_SAMPLES:-0}"
    OFFLINE_FIT_IMAGES="${OFFLINE_FIT_IMAGES:-300}"
    OFFLINE_EVAL_IMAGES="${OFFLINE_EVAL_IMAGES:-200}"
    CLASSIFICATION_RELATIVE_LIMIT="${CLASSIFICATION_RELATIVE_LIMIT:-0.0}"
    REGRESSION_RELATIVE_LIMIT="${REGRESSION_RELATIVE_LIMIT:-0.005}"
    ;;
  *)
    echo "[ERROR] MODE must be rapid or faithful, got: ${MODE}"
    exit 2
    ;;
esac

RUN_OFFLINE="${RUN_OFFLINE:-1}"
PROTO_NUM_FG="${PROTO_NUM_FG:-4}"
PROTO_NUM_BG="${PROTO_NUM_BG:-8}"

cd "${PROJECT_ROOT}"
for required in "${BASELINE_CHECKPOINT}" "${PARAFFIN_DATASET}/mean_std.npy"; do
  if [ ! -f "${required}" ]; then
    echo "[ERROR] required file not found: ${required}"
    exit 2
  fi
done
if [ ! -d "${PARAFFIN_DATASET}/train_image" ]; then
  echo "[ERROR] train_image not found: ${PARAFFIN_DATASET}/train_image"
  exit 2
fi
if ! grep -q 'PROTOTYPE_IMPLEMENTATION_VERSION = "prototype_v2_2_diagnostics_20260815"' models/p2p_prototype.py; then
  echo "[ERROR] stale prototype implementation"
  exit 2
fi
for required_marker in \
  'prototype_positive_fg_center_' \
  'prototype_hard_background_bg_center_' \
  'prototype_random_background_bg_center_' \
  'projector_probe_drift_cosine'; do
  if ! grep -q "${required_marker}" models/p2p_prototype.py; then
    echo "[ERROR] incomplete prototype-v2.2 implementation: missing ${required_marker}"
    exit 2
  fi
done

timestamp="$(date +%Y%m%d_%H%M%S)"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT_ROOT}/experiments/p2p_proto_v22_${MODE}_paired_${timestamp}_gpu${GPU}_bt${BATCH_SIZE}}"
CONTROL_DIR="${PAIR_ROOT}/control_p2p"
PROTOTYPE_DIR="${PAIR_ROOT}/prototype_v22"
OFFLINE_DIR="${PAIR_ROOT}/offline_source_train_to_test"
COMPARISON_DIR="${PAIR_ROOT}/comparison"
mkdir -p "${CONTROL_DIR}" "${PROTOTYPE_DIR}" "${OFFLINE_DIR}" "${COMPARISON_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[Design] mode=${MODE}, epochs=${DIAGNOSTIC_EPOCHS}, batches_per_epoch=${DEBUG_BATCHES}"
echo "[Design] Kfg/Kbg=${PROTO_NUM_FG}/${PROTO_NUM_BG}, hard/random=16/16"
echo "[Design] term weights positive/hard/random=2.0/1.0/0.25"
echo "[Design] no HBS, no inference fusion, source-only offline validation"

if [ "${RUN_PREFLIGHT}" = "1" ]; then
  echo "[Stage 1/7] tensor and CUDA integration tests"
  python test_p2p_prototype.py
  python smoke_test_p2p_prototype_integration.py
  python test_summarize_prototype_v2_debug.py
  python test_compare_prototype_v2_paired_debug.py
  python test_offline_prototype_feature_spaces.py
  python test_decide_prototype_v22.py

  echo "[Stage 2/7] deterministic real-data keypoint audit"
  python smoke_test_dataset_keypoint_safety.py \
    --dataset "${PARAFFIN_DATASET}" \
    --samples "${DATASET_SMOKE_SAMPLES}" \
    --seed 0 \
    --max_source_points_dropped 1 \
    --output "${PAIR_ROOT}/dataset_keypoint_safety.json"
else
  echo "[Stage 1-2/7] preflight skipped by RUN_PREFLIGHT=${RUN_PREFLIGHT}"
fi

COMMON_ARGS=(
  --dataset "${PARAFFIN_DATASET}"
  --num_classes=1
  --epochs="${DIAGNOSTIC_EPOCHS}"
  --batch_size="${BATCH_SIZE}"
  --lr=4e-5
  --min_lr=1e-6
  --weight_decay=1e-4
  --warmup_epochs=5
  --clip_max_norm=0.1
  --seed=0
  --reset_rng_after_init=1
  --deterministic_training=1
  --eos_coef=0.5
  --reg_loss_coef=0.002
  --cls_loss_coef=1.0
  --set_cost_point=0.1
  --set_cost_class=1.0
  --backbone=resnet50
  --position_embedding=sine
  --enc_layers=6
  --dim_feedforward=2048
  --hidden_dim=256
  --dropout=0.1
  --nheads=8
  --row=2
  --col=2
  --start_eval=999
  --num_workers=0
  --debug_max_train_batches="${DEBUG_BATCHES}"
  --debug_save_final_checkpoint=1
  --match_dis=15
  --mean_std_path "${PARAFFIN_DATASET}/mean_std.npy"
  --filter_empty_train
  --skip_empty_eval
  --init_checkpoint "${BASELINE_CHECKPOINT}"
)

echo "[Stage 3/7] paired control P2P from identical baseline/RNG/data order"
python train_p2p_no_empty_v2.py \
  "${COMMON_ARGS[@]}" \
  --output_dir "${CONTROL_DIR}" \
  2>&1 | tee "${CONTROL_DIR}/console.log"
test -f "${CONTROL_DIR}/recent_model.pth"

echo "[Stage 4/7] paired Prototype-v2.2"
python train_p2p_no_empty_v2.py \
  "${COMMON_ARGS[@]}" \
  --output_dir "${PROTOTYPE_DIR}" \
  --proto_enable \
  --proto_start_epoch=0 \
  --proto_embedding_dim=128 \
  --proto_num_fg="${PROTO_NUM_FG}" \
  --proto_num_bg="${PROTO_NUM_BG}" \
  --proto_fg_queue_size=4096 \
  --proto_bg_queue_size=8192 \
  --proto_temperature=0.2 \
  --proto_loss_weight=0.01 \
  --proto_initial_positive_radius=10 \
  --proto_positive_radius=15 \
  --proto_background_radius=30 \
  --proto_max_pos_per_image=64 \
  --proto_max_bg_per_image=32 \
  --proto_max_hard_bg_per_image=16 \
  --proto_max_random_bg_per_image=16 \
  --proto_loss_mode=source_weighted \
  --proto_positive_term_weight=2.0 \
  --proto_hard_bg_term_weight=1.0 \
  --proto_random_bg_term_weight=0.25 \
  --proto_kmeans_iterations=20 \
  --proto_update_mode=kmeans_ema \
  --proto_momentum=0.9 \
  --proto_min_assignment_share=0.02 \
  --proto_dead_patience=2 \
  --proto_input_grad_scale=0.1 \
  --proto_sampling_seed=0 \
  --proto_debug_interval=10 \
  --proto_debug_fail_fast=1 \
  --proto_inference_fusion=0 \
  2>&1 | tee "${PROTOTYPE_DIR}/console.log"
test -f "${PROTOTYPE_DIR}/recent_model.pth"

echo "[Stage 5/7] prototype health and paired causal attribution"
python summarize_prototype_v2_debug.py --experiment "${PROTOTYPE_DIR}"
python compare_prototype_v2_paired_debug.py \
  --control "${CONTROL_DIR}" \
  --prototype "${PROTOTYPE_DIR}" \
  --output "${COMPARISON_DIR}" \
  --classification_relative_limit "${CLASSIFICATION_RELATIVE_LIMIT}" \
  --regression_relative_limit "${REGRESSION_RELATIVE_LIMIT}"

if [ "${RUN_OFFLINE}" = "1" ]; then
  echo "[Stage 6/7] source-only support/query validation in raw and projected spaces"
  python offline_validate_p2p_prototypes.py \
    --checkpoint "${PROTOTYPE_DIR}/recent_model.pth" \
    --dataset "${PARAFFIN_DATASET}" \
    --mean_std_path "${PARAFFIN_DATASET}/mean_std.npy" \
    --output_dir "${OFFLINE_DIR}" \
    --fit_images "${OFFLINE_FIT_IMAGES}" \
    --eval_images "${OFFLINE_EVAL_IMAGES}" \
    --fg_k_values "${PROTO_NUM_FG}" \
    --bg_k_values "${PROTO_NUM_BG}" \
    --background_radius=30 \
    --max_pos_per_image=64 \
    --max_bg_per_image=32 \
    --max_hard_bg_per_image=16 \
    --max_random_bg_per_image=16 \
    --kmeans_iterations=20 \
    --num_workers=0 \
    --seed=0 \
    --save_feature_bank=1 \
    --include_projected_space=1 \
    --proto_embedding_dim=128 \
    --proto_num_fg "${PROTO_NUM_FG}" \
    --proto_num_bg "${PROTO_NUM_BG}"
else
  echo "[ERROR] RUN_OFFLINE=0 cannot produce the required v2.2 decision"
  exit 2
fi

echo "[Stage 7/7] combined go/no-go decision"
python decide_prototype_v22.py \
  --mechanism "${PROTOTYPE_DIR}/prototype_v2_audit.json" \
  --paired "${COMPARISON_DIR}/paired_debug_summary.json" \
  --offline "${OFFLINE_DIR}/offline_prototype_summary.json" \
  --mode "${MODE}" \
  --output "${PAIR_ROOT}/prototype_v22_decision.json"

printf '%s\n' "${PAIR_ROOT}" > "${PROJECT_ROOT}/experiments/latest_p2p_prototype_v22_${MODE}.txt"
echo "[SUCCESS] Prototype-v2.2 ${MODE} paired diagnostic completed: ${PAIR_ROOT}"
echo "[DECISION] ${PAIR_ROOT}/prototype_v22_decision.json"
