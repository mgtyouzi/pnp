#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
DATASET="${DATASET:-/home/data/yh_1/SET_3/p2p-src-zzh-2025/datasets/石蜡-2025}"
TEACHER="${TEACHER:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
GPU_IDS="${GPU_IDS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-30}"
START_EVAL="${START_EVAL:-29}"
MASTER_PORT="${MASTER_PORT:-29681}"
PROTO_WEIGHT="${PROTO_WEIGHT:-0.01}"
PROTO_TEMPERATURE="${PROTO_TEMPERATURE:-0.1}"
PROTO_K="${PROTO_K:-4}"
REBUILD_BANK="${REBUILD_BANK:-0}"

cd "${PROJECT_ROOT}"
for required in "${TEACHER}" "${DATASET}/mean_std.npy"; do
  if [ ! -f "${required}" ]; then
    echo "[ERROR] required file not found: ${required}"
    exit 2
  fi
done

if ! grep -q 'ft_fgproto_v1_20260816' models/frozen_teacher_fg_prototype.py; then
  echo "[ERROR] frozen-teacher foreground prototype v1 code is missing"
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
FIRST_GPU="${GPU_ARRAY[0]}"
timestamp="$(date +%Y%m%d_%H%M%S)"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/experiments/frozen_teacher_fg_bank_0318_k${PROTO_K}}"
BANK_PATH="${BANK_PATH:-${BANK_DIR}/frozen_teacher_fg_bank.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments/p2p_frozen_teacher_fg_k${PROTO_K}_w${PROTO_WEIGHT//./p}_${EPOCHS}ep_${timestamp}_gpu${GPU_IDS//,/_}_bt${BATCH_SIZE}}"
mkdir -p "${BANK_DIR}" "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[Stage 1/4] frozen-teacher prototype tests"
python test_frozen_teacher_fg_prototype_structure.py
python test_build_frozen_teacher_fg_bank_structure.py
python test_frozen_teacher_fg_prototype.py

if [ "${REBUILD_BANK}" = "1" ] || [ ! -f "${BANK_PATH}" ]; then
  echo "[Stage 2/4] build frozen teacher foreground bank on physical GPU ${FIRST_GPU}"
  CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python build_frozen_teacher_fg_bank.py \
    --checkpoint "${TEACHER}" \
    --dataset "${DATASET}" \
    --mean_std_path "${DATASET}/mean_std.npy" \
    --output_dir "${BANK_DIR}" \
    --gpu 0 \
    --support_fraction 0.8 \
    --num_prototypes "${PROTO_K}" \
    --negative_bank_size 8192 \
    --negative_sample_size 256 \
    --negative_energy_blocks 4 \
    --negative_sampling_seed 0 \
    --temperature "${PROTO_TEMPERATURE}" \
    --positive_radius 10 \
    --background_radius 30 \
    --max_positive_per_image 64 \
    --max_hard_negative_per_image 16 \
    --max_random_negative_per_image 16 \
    --kmeans_iterations 20 \
    --gate_auc 0.70 \
    --gate_balanced_accuracy 0.65 \
    --gate_min_cluster_share 0.05 \
    --seed 0 \
    --num_workers 0
else
  echo "[Stage 2/4] reuse bank: ${BANK_PATH}"
fi

echo "[Stage 3/4] verify deployable bank metadata"
python - "${BANK_PATH}" "${TEACHER}" <<'PY'
import hashlib
import sys
import torch

bank_path, teacher_path = sys.argv[1:3]
payload = torch.load(bank_path, map_location="cpu")
metadata = payload.get("metadata", {})
digest = hashlib.sha256()
with open(teacher_path, "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
if metadata.get("gate_pass") is not True:
    raise SystemExit("[ERROR] offline prototype gate did not pass")
if metadata.get("checkpoint_sha256") != digest.hexdigest():
    raise SystemExit("[ERROR] bank and 0318 teacher hashes differ")
print("bank_id =", payload["bank_id"])
print("offline_metrics =", metadata.get("metrics"))
print("cluster_shares =", metadata.get("cluster_shares"))
PY

echo "[Stage 4/4] train P2P + fixed teacher foreground prototypes"
COMMON_ARGS=(
  train_p2p_no_empty_v2.py
  --output_dir "${OUTPUT_DIR}"
  --dataset "${DATASET}"
  --mean_std_path "${DATASET}/mean_std.npy"
  --num_classes 1
  --num_workers 0
  --epochs "${EPOCHS}"
  --start_eval "${START_EVAL}"
  --batch_size "${BATCH_SIZE}"
  --lr 4e-5
  --min_lr 1e-6
  --weight_decay 1e-4
  --warmup_epochs 5
  --clip_max_norm 0.1
  --seed 0
  --reset_rng_after_init 1
  --deterministic_training 1
  --eos_coef 0.5
  --reg_loss_coef 0.002
  --cls_loss_coef 1.0
  --set_cost_point 0.1
  --set_cost_class 1.0
  --backbone resnet50
  --position_embedding sine
  --enc_layers 6
  --dim_feedforward 2048
  --hidden_dim 256
  --dropout 0.1
  --nheads 8
  --row 2
  --col 2
  --match_dis 15
  --filter_empty_train
  --skip_empty_eval
  --init_checkpoint "${TEACHER}"
  --proto_enable
  --proto_mode frozen_teacher_fg
  --proto_bank_path "${BANK_PATH}"
  --proto_start_epoch 0
  --proto_num_fg "${PROTO_K}"
  --proto_bg_queue_size 8192
  --proto_fixed_negative_sample_size 256
  --proto_temperature "${PROTO_TEMPERATURE}"
  --proto_loss_weight "${PROTO_WEIGHT}"
  --proto_initial_positive_radius 10
  --proto_background_radius 30
  --proto_max_pos_per_image 64
  --proto_max_hard_bg_per_image 16
  --proto_max_random_bg_per_image 16
  --proto_sampling_seed 0
  --proto_debug_interval 100
  --proto_debug_fail_fast 1
  --proto_inference_fusion 0
)

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
if [ "${WORLD_SIZE}" -eq 1 ]; then
  python "${COMMON_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/console.log"
else
  torchrun --nproc_per_node="${WORLD_SIZE}" --master_port="${MASTER_PORT}" \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/console.log"
fi

echo "[SUCCESS] ${OUTPUT_DIR}"
