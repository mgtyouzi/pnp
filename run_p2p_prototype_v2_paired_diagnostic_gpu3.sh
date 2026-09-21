#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
PARAFFIN_DATASET="${PARAFFIN_DATASET:-/home/data/yh_1/SET_3/p2p-src-zzh-2025/datasets/石蜡-2025}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DIAGNOSTIC_EPOCHS="${DIAGNOSTIC_EPOCHS:-8}"
DEBUG_BATCHES="${DEBUG_BATCHES:-100}"
DATASET_SMOKE_SAMPLES="${DATASET_SMOKE_SAMPLES:-0}"

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
if ! grep -q 'PROTOTYPE_IMPLEMENTATION_VERSION = "prototype_v2_1_hardbg_kmeans_ema_20260814"' models/p2p_prototype.py; then
  echo "[ERROR] stale prototype implementation"
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT_ROOT}/experiments/p2p_proto_v2_paired_${timestamp}_gpu${GPU}_bt${BATCH_SIZE}}"
CONTROL_DIR="${PAIR_ROOT}/control_p2p"
PROTOTYPE_DIR="${PAIR_ROOT}/prototype_v2"
mkdir -p "${CONTROL_DIR}" "${PROTOTYPE_DIR}" "${PAIR_ROOT}/comparison"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[Stage 1/5] tensor-level prototype tests"
python test_p2p_prototype.py

echo "[Stage 2/5] CUDA integration smoke test"
python smoke_test_p2p_prototype_integration.py

echo "[Stage 3/5] full real-data augmentation/keypoint audit"
python smoke_test_dataset_keypoint_safety.py \
  --dataset "${PARAFFIN_DATASET}" \
  --samples "${DATASET_SMOKE_SAMPLES}" \
  --seed 0 \
  --max_source_points_dropped 1 \
  --output "${PAIR_ROOT}/dataset_keypoint_safety.json"

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
  --match_dis=15
  --mean_std_path "${PARAFFIN_DATASET}/mean_std.npy"
  --filter_empty_train
  --skip_empty_eval
  --init_checkpoint "${BASELINE_CHECKPOINT}"
)

echo "[Stage 4/5] paired control P2P initialized from the same baseline"
python train_p2p_no_empty_v2.py \
  "${COMMON_ARGS[@]}" \
  --output_dir "${CONTROL_DIR}" \
  2>&1 | tee "${CONTROL_DIR}/console.log"

echo "[Stage 5/5] paired Prototype-v2 initialized from the same baseline"
python train_p2p_no_empty_v2.py \
  "${COMMON_ARGS[@]}" \
  --output_dir "${PROTOTYPE_DIR}" \
  --proto_enable \
  --proto_start_epoch=0 \
  --proto_embedding_dim=128 \
  --proto_num_fg=4 \
  --proto_num_bg=8 \
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
  --proto_kmeans_iterations=20 \
  --proto_momentum=0.99 \
  --proto_dead_patience=3 \
  --proto_input_grad_scale=0.1 \
  --proto_sampling_seed=0 \
  --proto_debug_interval=10 \
  --proto_debug_fail_fast=1 \
  --proto_inference_fusion=0 \
  2>&1 | tee "${PROTOTYPE_DIR}/console.log"

python summarize_prototype_v2_debug.py --experiment "${PROTOTYPE_DIR}"
python compare_prototype_v2_paired_debug.py \
  --control "${CONTROL_DIR}" \
  --prototype "${PROTOTYPE_DIR}" \
  --output "${PAIR_ROOT}/comparison"

printf '%s\n' "${PAIR_ROOT}" > "${PROJECT_ROOT}/experiments/latest_p2p_prototype_v2_paired.txt"
echo "[SUCCESS] paired diagnostic completed: ${PAIR_ROOT}"
