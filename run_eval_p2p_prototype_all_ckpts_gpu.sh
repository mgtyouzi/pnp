#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
FROZEN_DATASET="${FROZEN_DATASET:-/home/data/yh_1/SET_3/p2p-src-zzh-2025/datasets/冰冻-2025}"
GPU="${GPU:-0}"
EXP_DIR="${EXP_DIR:-}"
FUSION_ALPHAS="${FUSION_ALPHAS:-0.10}"
FUSION_CHECKPOINT_LABELS="${FUSION_CHECKPOINT_LABELS:-best,epoch50}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-30,50,70,90}"
NUM_WORKERS="${NUM_WORKERS:-0}"

PROTO_EMBEDDING_DIM="${PROTO_EMBEDDING_DIM:-128}"
PROTO_NUM_FG="${PROTO_NUM_FG:-4}"
PROTO_NUM_BG="${PROTO_NUM_BG:-4}"
PROTO_FG_QUEUE_SIZE="${PROTO_FG_QUEUE_SIZE:-4096}"
PROTO_BG_QUEUE_SIZE="${PROTO_BG_QUEUE_SIZE:-8192}"
PROTO_TEMPERATURE="${PROTO_TEMPERATURE:-0.1}"
PROTO_LOSS_WEIGHT="${PROTO_LOSS_WEIGHT:-0.05}"
PROTO_BACKGROUND_RADIUS="${PROTO_BACKGROUND_RADIUS:-30}"
PROTO_MAX_POS_PER_IMAGE="${PROTO_MAX_POS_PER_IMAGE:-64}"
PROTO_MAX_BG_PER_IMAGE="${PROTO_MAX_BG_PER_IMAGE:-32}"
PROTO_KMEANS_ITERATIONS="${PROTO_KMEANS_ITERATIONS:-10}"

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "[ERROR] PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 2
fi
cd "${PROJECT_ROOT}"

if [ -z "${EXP_DIR}" ] && [ -f experiments/latest_p2p_prototype_experiment.txt ]; then
  EXP_DIR="$(head -n 1 experiments/latest_p2p_prototype_experiment.txt)"
fi
if [ -z "${EXP_DIR}" ] || [ ! -d "${EXP_DIR}" ]; then
  echo "[ERROR] set EXP_DIR to a completed prototype experiment"
  exit 2
fi
if [ ! -d "${FROZEN_DATASET}/test_image" ]; then
  echo "[ERROR] frozen test_image not found: ${FROZEN_DATASET}/test_image"
  exit 2
fi
if [ ! -f "${FROZEN_DATASET}/mean_std.npy" ]; then
  echo "[ERROR] frozen mean/std not found: ${FROZEN_DATASET}/mean_std.npy"
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
EVAL_ROOT="${EVAL_ROOT:-${EXP_DIR}/eval_frozen_proto_raw_fusion_${timestamp}_gpu${GPU}}"
mkdir -p "${EVAL_ROOT}"

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
IFS=',' read -r -a REQUESTED_EPOCHS <<< "${CHECKPOINT_EPOCHS// /}"
for epoch in "${REQUESTED_EPOCHS[@]}"; do
  matches=("${EXP_DIR}/model_epoch_${epoch}_"*.pth)
  if [ "${#matches[@]}" -gt 0 ]; then
    last_index=$((${#matches[@]} - 1))
    add_checkpoint "epoch${epoch}" "${matches[$last_index]}"
  fi
done
add_checkpoint "best" "${EXP_DIR}/best_model.pth"
add_checkpoint "recent" "${EXP_DIR}/recent_model.pth"

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  echo "[ERROR] no checkpoints found under ${EXP_DIR}"
  exit 2
fi

export PYTHONUNBUFFERED=1
echo "[Prototype-eval] experiment=${EXP_DIR}"
echo "[Prototype-eval] target=${FROZEN_DATASET}"
echo "[Prototype-eval] physical_gpu=${GPU}"
echo "[Prototype-eval] checkpoints=${LABELS[*]}"
echo "[Prototype-eval] fusion_alphas=${FUSION_ALPHAS}"
echo "[Prototype-eval] fusion checkpoints=${FUSION_CHECKPOINT_LABELS} (pre-registered; not target-selected)"

run_one() {
    local mode="$1"
    local fusion="$2"
    local alpha="$3"
    local label="$4"
    local checkpoint="$5"
    local output="${EVAL_ROOT}/${mode}/${label}"
    mkdir -p "${output}"
    echo "[RUN] mode=${mode}, label=${label}, checkpoint=${checkpoint}"

    CUDA_VISIBLE_DEVICES="${GPU}" python diagnose_cross_domain.py \
      --checkpoint "${checkpoint}" \
      --case_name "p2p_proto_${mode}_${label}_paraffin_to_frozen" \
      --dataset "${FROZEN_DATASET}" \
      --output_dir "${output}" \
      --num_classes=1 \
      --batch_size=1 \
      --num_workers="${NUM_WORKERS}" \
      --match_dis=15 \
      --near_radius=30 \
      --dedup_interval=15 \
      --backbone=resnet50 \
      --position_embedding=sine \
      --enc_layers=6 \
      --dim_feedforward=2048 \
      --hidden_dim=256 \
      --dropout=0.1 \
      --nheads=8 \
      --row=2 \
      --col=2 \
      --proto_enable \
      --proto_start_epoch=0 \
      --proto_embedding_dim="${PROTO_EMBEDDING_DIM}" \
      --proto_num_fg="${PROTO_NUM_FG}" \
      --proto_num_bg="${PROTO_NUM_BG}" \
      --proto_fg_queue_size="${PROTO_FG_QUEUE_SIZE}" \
      --proto_bg_queue_size="${PROTO_BG_QUEUE_SIZE}" \
      --proto_temperature="${PROTO_TEMPERATURE}" \
      --proto_loss_weight="${PROTO_LOSS_WEIGHT}" \
      --proto_background_radius="${PROTO_BACKGROUND_RADIUS}" \
      --proto_max_pos_per_image="${PROTO_MAX_POS_PER_IMAGE}" \
      --proto_max_bg_per_image="${PROTO_MAX_BG_PER_IMAGE}" \
      --proto_kmeans_iterations="${PROTO_KMEANS_ITERATIONS}" \
      --proto_inference_fusion="${fusion}" \
      --proto_fusion_alpha="${alpha}" \
      --proto_fusion_clip=2.0 \
      --skip_empty_gt_in_eval \
      --strict_load \
      2>&1 | tee "${output}/console.log"
}

for index in "${!CHECKPOINTS[@]}"; do
  run_one "raw" 0 0 "${LABELS[$index]}" "${CHECKPOINTS[$index]}"
done

python summarize_prototype_eval.py --root "${EVAL_ROOT}" | tee "${EVAL_ROOT}/raw_ranking.log"

IFS=',' read -r -a FUSION_LABELS <<< "${FUSION_CHECKPOINT_LABELS// /}"
IFS=',' read -r -a ALPHAS <<< "${FUSION_ALPHAS// /}"
for alpha in "${ALPHAS[@]}"; do
  case "${alpha}" in
    0|0.0|0.00|0.000) continue ;;
  esac
  tag="${alpha/./p}"
  mode="fusion_a${tag}"
  for selected_label in "${FUSION_LABELS[@]}"; do
    for index in "${!CHECKPOINTS[@]}"; do
      if [ "${LABELS[$index]}" = "${selected_label}" ]; then
        run_one "${mode}" 1 "${alpha}" "${LABELS[$index]}" "${CHECKPOINTS[$index]}"
      fi
    done
  done
done

python summarize_prototype_eval.py --root "${EVAL_ROOT}" | tee "${EVAL_ROOT}/ranked_summary.log"
echo "[SUCCESS] evaluation output: ${EVAL_ROOT}"
