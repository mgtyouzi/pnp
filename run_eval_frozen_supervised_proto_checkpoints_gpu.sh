#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
FROZEN_DATASET="${FROZEN_DATASET:-/home/data/yh_1/SET_3/p2p-src-zzh-2025/datasets/冰冻-2025}"
GPU="${GPU:-3}"
EXP_DIR="${EXP_DIR:?set EXP_DIR}"
EVAL_ROOT="${EVAL_ROOT:?set EVAL_ROOT}"
MODE="${MODE:-control}"
PROTO_BANK_PATH="${PROTO_BANK_PATH:-}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-10,20,30,40,50,60,70,80,90,100}"

cd "${PROJECT_ROOT}"
mkdir -p "${EVAL_ROOT}/raw"
if [ "${MODE}" != "control" ] && [ "${MODE}" != "prototype" ]; then
  echo "[ERROR] MODE must be control or prototype"
  exit 2
fi
if [ "${MODE}" = "prototype" ] && [ ! -f "${PROTO_BANK_PATH}" ]; then
  echo "[ERROR] prototype bank not found: ${PROTO_BANK_PATH}"
  exit 2
fi

declare -a LABELS=()
declare -a CHECKPOINTS=()
add_checkpoint() {
  local label="$1"
  local checkpoint="$2"
  if [ -f "${checkpoint}" ]; then
    LABELS+=("${label}")
    CHECKPOINTS+=("${checkpoint}")
  fi
}

shopt -s nullglob
IFS=',' read -r -a REQUESTED <<< "${CHECKPOINT_EPOCHS// /}"
for logical_epoch in "${REQUESTED[@]}"; do
  matches=("${EXP_DIR}/model_epoch_${logical_epoch}_"*.pth)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "[ERROR] expected one checkpoint for logical epoch ${logical_epoch}, found ${#matches[@]}"
    exit 2
  fi
  add_checkpoint "epoch${logical_epoch}" "${matches[0]}"
done
add_checkpoint "best" "${EXP_DIR}/best_model.pth"
add_checkpoint "recent" "${EXP_DIR}/recent_model.pth"

common_args=(
  --dataset "${FROZEN_DATASET}"
  --num_classes=1
  --batch_size=1
  --num_workers=0
  --match_dis=15
  --near_radius=30
  --dedup_interval=15
  --backbone=resnet50
  --position_embedding=sine
  --enc_layers=6
  --dim_feedforward=2048
  --hidden_dim=256
  --dropout=0.1
  --nheads=8
  --row=2
  --col=2
  --skip_empty_gt_in_eval
  --strict_load
)

prototype_args=()
if [ "${MODE}" = "prototype" ]; then
  prototype_args=(
    --proto_enable
    --proto_mode=frozen_supervised_metric
    --proto_bank_path "${PROTO_BANK_PATH}"
    --proto_start_epoch=0
    --proto_embedding_dim=32
    --proto_num_fg=4
    --proto_num_hard_bg=4
    --proto_num_random_bg=2
    --proto_temperature=0.15
    --proto_loss_weight=0.02
    --proto_initial_positive_radius=10
    --proto_background_radius=30
    --proto_max_pos_per_image=32
    --proto_max_hard_bg_per_image=16
    --proto_max_random_bg_per_image=16
    --proto_hard_bg_term_weight=2.0
    --proto_inference_fusion=0
  )
fi

for index in "${!CHECKPOINTS[@]}"; do
  label="${LABELS[$index]}"
  checkpoint="${CHECKPOINTS[$index]}"
  output="${EVAL_ROOT}/raw/${label}"
  mkdir -p "${output}"
  echo "[Frozen-eval] mode=${MODE}, label=${label}, checkpoint=${checkpoint}"
  CUDA_VISIBLE_DEVICES="${GPU}" python diagnose_cross_domain.py \
    --checkpoint "${checkpoint}" \
    --case_name "${MODE}_${label}_paraffin_to_frozen" \
    --output_dir "${output}" \
    "${common_args[@]}" \
    "${prototype_args[@]}" \
    2>&1 | tee "${output}/console.log"
done

python summarize_prototype_eval.py --root "${EVAL_ROOT}" \
  2>&1 | tee "${EVAL_ROOT}/ranked_summary.log"
echo "[SUCCESS] ${MODE} frozen evaluation: ${EVAL_ROOT}"
