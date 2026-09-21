#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
PARAFFIN_DATASET="${PARAFFIN_DATASET:-/home/data/yh_1/SET_3/p2p-src-zzh-2025/datasets/石蜡-2025}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29641}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-4e-5}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
EOS_COEF="${EOS_COEF:-0.5}"
MATCH_DIS="${MATCH_DIS:-15}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-0}"
RESET_RNG_AFTER_INIT="${RESET_RNG_AFTER_INIT:-1}"
DETERMINISTIC_TRAINING="${DETERMINISTIC_TRAINING:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
DEBUG_MAX_TRAIN_BATCHES="${DEBUG_MAX_TRAIN_BATCHES:-0}"

PROTO_EMBEDDING_DIM="${PROTO_EMBEDDING_DIM:-128}"
PROTO_NUM_FG="${PROTO_NUM_FG:-4}"
PROTO_NUM_BG="${PROTO_NUM_BG:-8}"
PROTO_FG_QUEUE_SIZE="${PROTO_FG_QUEUE_SIZE:-4096}"
PROTO_BG_QUEUE_SIZE="${PROTO_BG_QUEUE_SIZE:-8192}"
PROTO_TEMPERATURE="${PROTO_TEMPERATURE:-0.2}"
PROTO_LOSS_WEIGHT="${PROTO_LOSS_WEIGHT:-0.01}"
PROTO_START_EPOCH="${PROTO_START_EPOCH:-20}"
PROTO_INITIAL_POSITIVE_RADIUS="${PROTO_INITIAL_POSITIVE_RADIUS:-10}"
PROTO_POSITIVE_RADIUS="${PROTO_POSITIVE_RADIUS:-15}"
PROTO_BACKGROUND_RADIUS="${PROTO_BACKGROUND_RADIUS:-30}"
PROTO_MAX_POS_PER_IMAGE="${PROTO_MAX_POS_PER_IMAGE:-64}"
PROTO_MAX_BG_PER_IMAGE="${PROTO_MAX_BG_PER_IMAGE:-32}"
PROTO_MAX_HARD_BG_PER_IMAGE="${PROTO_MAX_HARD_BG_PER_IMAGE:-16}"
PROTO_MAX_RANDOM_BG_PER_IMAGE="${PROTO_MAX_RANDOM_BG_PER_IMAGE:-16}"
PROTO_KMEANS_ITERATIONS="${PROTO_KMEANS_ITERATIONS:-20}"
PROTO_MOMENTUM="${PROTO_MOMENTUM:-0.99}"
PROTO_DEAD_PATIENCE="${PROTO_DEAD_PATIENCE:-3}"
PROTO_INPUT_GRAD_SCALE="${PROTO_INPUT_GRAD_SCALE:-0.1}"
PROTO_SAMPLING_SEED="${PROTO_SAMPLING_SEED:-0}"
PROTO_DEBUG_INTERVAL="${PROTO_DEBUG_INTERVAL:-200}"
PROTO_DEBUG_FAIL_FAST="${PROTO_DEBUG_FAIL_FAST:-1}"

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "[ERROR] PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 2
fi
if [ ! -d "${PARAFFIN_DATASET}/train_image" ]; then
  echo "[ERROR] paraffin train_image not found: ${PARAFFIN_DATASET}/train_image"
  exit 2
fi
if [ ! -f "${PARAFFIN_DATASET}/mean_std.npy" ]; then
  echo "[ERROR] mean/std not found: ${PARAFFIN_DATASET}/mean_std.npy"
  exit 2
fi
if [ -n "${INIT_CHECKPOINT}" ] && [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "[ERROR] INIT_CHECKPOINT not found: ${INIT_CHECKPOINT}"
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES// /}"
NPROC_PER_NODE="${#GPU_IDS[@]}"
if [ "${NPROC_PER_NODE}" -lt 1 ]; then
  echo "[ERROR] no GPU selected"
  exit 2
fi

cd "${PROJECT_ROOT}"
timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/p2p_proto_v2_k${PROTO_NUM_FG}x${PROTO_NUM_BG}_d${PROTO_EMBEDDING_DIM}_w${PROTO_LOSS_WEIGHT}_${EPOCHS}ep_${timestamp}_${NPROC_PER_NODE}gpu_bt${BATCH_SIZE}}"
mkdir -p "${OUT_DIR}"
printf '%s\n' "${OUT_DIR}" > "${PROJECT_ROOT}/experiments/latest_p2p_prototype_experiment.txt"

export MASTER_ADDR
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

INIT_ARGS=()
if [ -n "${INIT_CHECKPOINT}" ]; then
  INIT_ARGS+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi

echo "[P2P-Prototype] project=${PROJECT_ROOT}"
echo "[P2P-Prototype] dataset=${PARAFFIN_DATASET}"
echo "[P2P-Prototype] output=${OUT_DIR}"
echo "[P2P-Prototype] physical_gpus=${CUDA_VISIBLE_DEVICES}, nproc=${NPROC_PER_NODE}, batch_per_gpu=${BATCH_SIZE}"
echo "[P2P-Prototype] epochs=${EPOCHS}, lr=${LR}, eos=${EOS_COEF}, match_dis=${MATCH_DIS}"
echo "[P2P-Prototype] Kfg=${PROTO_NUM_FG}, Kbg=${PROTO_NUM_BG}, dim=${PROTO_EMBEDDING_DIM}, weight=${PROTO_LOSS_WEIGHT}"
echo "[P2P-Prototype] start=${PROTO_START_EPOCH}, positive_radius=${PROTO_INITIAL_POSITIVE_RADIUS}->${PROTO_POSITIVE_RADIUS}, background_radius=${PROTO_BACKGROUND_RADIUS}"
echo "[P2P-Prototype] bg_hard/random=${PROTO_MAX_HARD_BG_PER_IMAGE}/${PROTO_MAX_RANDOM_BG_PER_IMAGE}, momentum=${PROTO_MOMENTUM}, input_grad_scale=${PROTO_INPUT_GRAD_SCALE}"
echo "[P2P-Prototype] debug_max_train_batches=${DEBUG_MAX_TRAIN_BATCHES}, debug_interval=${PROTO_DEBUG_INTERVAL}"
echo "[P2P-Prototype] seed=${SEED}, reset_rng_after_init=${RESET_RNG_AFTER_INIT}, prototype_sampling_seed=${PROTO_SAMPLING_SEED}"
echo "[P2P-Prototype] inference fusion is disabled during training/source best selection"

if [ "${NPROC_PER_NODE}" -eq 1 ]; then
  LAUNCH=(python)
else
  LAUNCH=(torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${LAUNCH[@]}" train_p2p_no_empty_v2.py \
  --dataset "${PARAFFIN_DATASET}" \
  --output_dir "${OUT_DIR}" \
  --num_classes=1 \
  --epochs="${EPOCHS}" \
  --batch_size="${BATCH_SIZE}" \
  --lr="${LR}" \
  --min_lr="${MIN_LR}" \
  --weight_decay="${WEIGHT_DECAY}" \
  --warmup_epochs="${WARMUP_EPOCHS}" \
  --clip_max_norm=0.1 \
  --seed="${SEED}" \
  --reset_rng_after_init="${RESET_RNG_AFTER_INIT}" \
  --deterministic_training="${DETERMINISTIC_TRAINING}" \
  --eos_coef="${EOS_COEF}" \
  --reg_loss_coef=0.002 \
  --cls_loss_coef=1.0 \
  --set_cost_point=0.1 \
  --set_cost_class=1.0 \
  --backbone=resnet50 \
  --position_embedding=sine \
  --enc_layers=6 \
  --dim_feedforward=2048 \
  --hidden_dim=256 \
  --dropout=0.1 \
  --nheads=8 \
  --row=2 \
  --col=2 \
  --start_eval=29 \
  --num_workers="${NUM_WORKERS}" \
  --debug_max_train_batches="${DEBUG_MAX_TRAIN_BATCHES}" \
  --match_dis="${MATCH_DIS}" \
  --mean_std_path "${PARAFFIN_DATASET}/mean_std.npy" \
  --filter_empty_train \
  --skip_empty_eval \
  --proto_enable \
  --proto_start_epoch="${PROTO_START_EPOCH}" \
  --proto_embedding_dim="${PROTO_EMBEDDING_DIM}" \
  --proto_num_fg="${PROTO_NUM_FG}" \
  --proto_num_bg="${PROTO_NUM_BG}" \
  --proto_fg_queue_size="${PROTO_FG_QUEUE_SIZE}" \
  --proto_bg_queue_size="${PROTO_BG_QUEUE_SIZE}" \
  --proto_temperature="${PROTO_TEMPERATURE}" \
  --proto_loss_weight="${PROTO_LOSS_WEIGHT}" \
  --proto_initial_positive_radius="${PROTO_INITIAL_POSITIVE_RADIUS}" \
  --proto_positive_radius="${PROTO_POSITIVE_RADIUS}" \
  --proto_background_radius="${PROTO_BACKGROUND_RADIUS}" \
  --proto_max_pos_per_image="${PROTO_MAX_POS_PER_IMAGE}" \
  --proto_max_bg_per_image="${PROTO_MAX_BG_PER_IMAGE}" \
  --proto_max_hard_bg_per_image="${PROTO_MAX_HARD_BG_PER_IMAGE}" \
  --proto_max_random_bg_per_image="${PROTO_MAX_RANDOM_BG_PER_IMAGE}" \
  --proto_kmeans_iterations="${PROTO_KMEANS_ITERATIONS}" \
  --proto_momentum="${PROTO_MOMENTUM}" \
  --proto_dead_patience="${PROTO_DEAD_PATIENCE}" \
  --proto_input_grad_scale="${PROTO_INPUT_GRAD_SCALE}" \
  --proto_sampling_seed="${PROTO_SAMPLING_SEED}" \
  --proto_debug_interval="${PROTO_DEBUG_INTERVAL}" \
  --proto_debug_fail_fast="${PROTO_DEBUG_FAIL_FAST}" \
  --proto_inference_fusion=0 \
  --proto_fusion_alpha=0.1 \
  --proto_fusion_clip=2.0 \
  "${INIT_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/console.log"

echo "[SUCCESS] training output: ${OUT_DIR}"
