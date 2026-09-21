#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
DATASET="${DATASET:-/home/data/yh_1/SET_3/p2p-src-zzh-2025/datasets/石蜡-2025}"
FROZEN_DATASET="${FROZEN_DATASET:-/home/data/yh_1/SET_3/p2p-src-zzh-2025/datasets/冰冻-2025}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${PROJECT_ROOT}/experiments/teacher_candidate_audit_0318_full}"
GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
PROTO_LOSS_WEIGHT="${PROTO_LOSS_WEIGHT:-0.02}"

if [ "${EPOCHS}" -ne 100 ]; then
  echo "[ERROR] this causal queue is locked to 100 epochs, got ${EPOCHS}"
  exit 2
fi
if [ "${CHECKPOINT_INTERVAL}" -ne 10 ]; then
  echo "[ERROR] checkpoint interval is locked to 10, got ${CHECKPOINT_INTERVAL}"
  exit 2
fi

cd "${PROJECT_ROOT}"
timestamp="$(date +%Y%m%d_%H%M%S)"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT_ROOT}/experiments/p2p_frozen_supervised_proto_100ep_${timestamp}_gpu${GPU}_bt${BATCH_SIZE}}"
BANK_DIR="${PAIR_ROOT}/frozen_bank"
CONTROL_DIR="${PAIR_ROOT}/control_p2p"
PROTOTYPE_DIR="${PAIR_ROOT}/p2p_frozen_supervised_proto"
mkdir -p "${BANK_DIR}" "${CONTROL_DIR}" "${PROTOTYPE_DIR}"
printf '%s\n' "${PAIR_ROOT}" > experiments/latest_frozen_supervised_proto_100ep.txt

for required in \
  "${BASELINE_CHECKPOINT}" \
  "${DATASET}/mean_std.npy" \
  "${FROZEN_DATASET}/mean_std.npy" \
  "${CANDIDATE_ROOT}/candidate_features.npz" \
  "${CANDIDATE_ROOT}/image_manifest.json"; do
  if [ ! -e "${required}" ]; then
    echo "[ERROR] required path not found: ${required}"
    exit 2
  fi
done

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[Stage 1/6] code and tensor preflight"
python test_checkpoint_schedule.py
python test_export_frozen_supervised_metric_bank.py
python test_frozen_supervised_metric_prototype.py
python test_frozen_supervised_proto_wiring.py
python -m py_compile \
  checkpoint_schedule.py \
  export_frozen_supervised_metric_bank.py \
  models/frozen_supervised_metric_prototype.py \
  models/detr.py \
  train_p2p.py \
  diagnose_cross_domain.py

echo "[Stage 2/6] export frozen supervised prototype bank"
CUDA_VISIBLE_DEVICES="${GPU}" python export_frozen_supervised_metric_bank.py \
  --features "${CANDIDATE_ROOT}/candidate_features.npz" \
  --manifest "${CANDIDATE_ROOT}/image_manifest.json" \
  --output_dir "${BANK_DIR}" \
  --gpu=0 \
  --embedding_dim=32 \
  --foreground_prototypes=4 \
  --hard_background_prototypes=4 \
  --random_background_prototypes=2 \
  --temperature=0.15 \
  --learning_rate=0.01 \
  --weight_decay=0.0001 \
  --epochs=40 \
  --batch_size=1024 \
  --patience=8 \
  --calibration_fraction=0.20 \
  --positive_per_group=4096 \
  --hard_negative_per_group=2048 \
  --random_negative_per_group=2048 \
  --hard_negative_weight=2.0 \
  --diversity_weight=0.02 \
  --projection_anchor_weight=0.001 \
  --kmeans_iterations=20 \
  --seed=0 \
  2>&1 | tee "${BANK_DIR}/console.log"
PROTO_BANK="${BANK_DIR}/frozen_supervised_metric_bank.pth"
test -f "${PROTO_BANK}"

common_train_args=(
  --dataset "${DATASET}"
  --mean_std_path "${DATASET}/mean_std.npy"
  --eval_split=test
  --num_classes=1
  --num_workers=0
  --batch_size="${BATCH_SIZE}"
  --epochs="${EPOCHS}"
  --start_eval=29
  --checkpoint_interval="${CHECKPOINT_INTERVAL}"
  --lr=0.00004
  --min_lr=0.000001
  --weight_decay=0.0001
  --warmup_epochs=5
  --clip_max_norm=0.1
  --seed=0
  --reg_loss_coef=0.002
  --cls_loss_coef=1.0
  --eos_coef=0.5
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
  --match_dis=15
  --filter_empty_train
  --skip_empty_eval
  --reset_rng_after_init=1
  --deterministic_training=1
  --debug_save_final_checkpoint=1
  --init_checkpoint "${BASELINE_CHECKPOINT}"
)

echo "CONTROL_TRAIN_START"
echo "[Stage 3/6] paired control P2P, ${EPOCHS} epochs on physical GPU ${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" python train_p2p_no_empty_v2.py \
  --output_dir "${CONTROL_DIR}" \
  "${common_train_args[@]}" \
  2>&1 | tee "${CONTROL_DIR}/console.log"
test -f "${CONTROL_DIR}/recent_model.pth"

echo "PROTOTYPE_TRAIN_START"
echo "[Stage 4/6] P2P + frozen supervised prototypes, ${EPOCHS} epochs"
CUDA_VISIBLE_DEVICES="${GPU}" python train_p2p_no_empty_v2.py \
  --output_dir "${PROTOTYPE_DIR}" \
  "${common_train_args[@]}" \
  --proto_enable \
  --proto_mode=frozen_supervised_metric \
  --proto_bank_path "${PROTO_BANK}" \
  --proto_start_epoch=0 \
  --proto_embedding_dim=32 \
  --proto_num_fg=4 \
  --proto_num_hard_bg=4 \
  --proto_num_random_bg=2 \
  --proto_temperature=0.15 \
  --proto_loss_weight="${PROTO_LOSS_WEIGHT}" \
  --proto_initial_positive_radius=10 \
  --proto_background_radius=30 \
  --proto_max_pos_per_image=32 \
  --proto_max_hard_bg_per_image=16 \
  --proto_max_random_bg_per_image=16 \
  --proto_hard_bg_term_weight=2.0 \
  --proto_sampling_seed=0 \
  --proto_debug_interval=250 \
  --proto_debug_fail_fast=1 \
  --proto_inference_fusion=0 \
  2>&1 | tee "${PROTOTYPE_DIR}/console.log"
test -f "${PROTOTYPE_DIR}/recent_model.pth"

verify_checkpoints() {
  local experiment="$1"
  local logical_epoch
  shopt -s nullglob
  for logical_epoch in 10 20 30 40 50 60 70 80 90 100; do
    local matches=("${experiment}/model_epoch_${logical_epoch}_"*.pth)
    if [ "${#matches[@]}" -ne 1 ]; then
      echo "[ERROR] ${experiment}: expected checkpoint ${logical_epoch}, found ${#matches[@]}"
      exit 2
    fi
  done
}
verify_checkpoints "${CONTROL_DIR}"
verify_checkpoints "${PROTOTYPE_DIR}"

echo "[Stage 5/6] target-domain evaluation for all control checkpoints"
PROJECT_ROOT="${PROJECT_ROOT}" \
FROZEN_DATASET="${FROZEN_DATASET}" \
GPU="${GPU}" \
MODE=control \
EXP_DIR="${CONTROL_DIR}" \
EVAL_ROOT="${PAIR_ROOT}/eval_frozen/control" \
bash run_eval_frozen_supervised_proto_checkpoints_gpu.sh \
  2>&1 | tee "${PAIR_ROOT}/eval_control.log"

echo "[Stage 6/6] target-domain evaluation for all prototype checkpoints"
PROJECT_ROOT="${PROJECT_ROOT}" \
FROZEN_DATASET="${FROZEN_DATASET}" \
GPU="${GPU}" \
MODE=prototype \
PROTO_BANK_PATH="${PROTO_BANK}" \
EXP_DIR="${PROTOTYPE_DIR}" \
EVAL_ROOT="${PAIR_ROOT}/eval_frozen/prototype" \
bash run_eval_frozen_supervised_proto_checkpoints_gpu.sh \
  2>&1 | tee "${PAIR_ROOT}/eval_prototype.log"

echo "[SUCCESS] paired 100-epoch experiment and all target evaluations completed"
echo "[SUCCESS] output=${PAIR_ROOT}"
