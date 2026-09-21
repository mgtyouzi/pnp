#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
DATASET="${DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
GPU="${GPU:-4}"
GPUS="${GPUS:-${GPU}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MASTER_PORT="${MASTER_PORT:-29641}"
DUAL_SEPARATION_WEIGHT="${DUAL_SEPARATION_WEIGHT:-1.0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

cd "${PROJECT_ROOT}"
for required in "${DATASET}/mean_std.npy" "${INIT_CHECKPOINT}"; do
  if [ ! -f "${required}" ]; then
    echo "[ERROR] required file not found: ${required}"
    exit 2
  fi
done

timestamp="$(date +%Y%m%d_%H%M%S)"
gpu_tag="${GPUS//,/_}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments/discriminative_dual_proxy_gate_${timestamp}_gpu${gpu_tag}_bt${BATCH_SIZE}}"
mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

IFS=',' read -r -a gpu_ids <<< "${GPUS}"
nproc="${#gpu_ids[@]}"
TRAIN_LAUNCH=(python)
if [ "${nproc}" -gt 1 ]; then
  TRAIN_LAUNCH=(
    torchrun
    --nproc_per_node="${nproc}"
    --master_addr=127.0.0.1
    --master_port="${MASTER_PORT}"
  )
fi

echo "[Gate] output=${OUTPUT_DIR}"
echo "[Gate] physical_gpus=${GPUS}, processes=${nproc}, batch_size=${BATCH_SIZE}"
echo "[Gate] separation_weight=${DUAL_SEPARATION_WEIGHT}"
echo "[Gate] detector frozen; projector-only optimization for five epochs"

COMMON_MODEL_ARGS=(
  --num_classes=1
  --backbone=resnet50
  --position_embedding=sine
  --enc_layers=6
  --dim_feedforward=2048
  --hidden_dim=256
  --dropout=0.1
  --nheads=8
  --row=2
  --col=2
  --proto_enable
  --proto_mode=discriminative_dual_proxy
  --proto_embedding_dim=64
  --proto_dual_num_fg=4
  --proto_dual_num_bg=4
  --proto_dual_fg_queue_size=8192
  --proto_dual_bg_queue_size=8192
  --proto_gt_max_support_per_image=64
  --proto_max_hard_bg_per_image=32
  --proto_max_random_bg_per_image=32
  --proto_positive_radius=15
  --proto_background_radius=30
  --proto_temperature=0.2
  --proto_momentum=0.99
  --proto_gt_projector_momentum=0.999
  --proto_dual_supcon_weight=0.1
  --proto_dual_separation_weight="${DUAL_SEPARATION_WEIGHT}"
  --proto_dual_separation_margin=0.1
  --proto_dual_balance_weight=0.1
  --proto_gt_min_assignment_share=0.05
  --proto_dead_patience=2
  --proto_sampling_seed=0
  --proto_inference_fusion=0
)

if [ "${EVAL_ONLY}" != "1" ]; then
  python test_discriminative_dual_proxy_structure.py
  python test_discriminative_dual_proxy_wiring.py
  python test_discriminative_dual_proxy_gate.py
  python test_discriminative_dual_proxy_run_script.py
  python test_discriminative_dual_proxy.py

  "${TRAIN_LAUNCH[@]}" train_p2p_no_empty_v2.py \
    --dataset "${DATASET}" \
    --mean_std_path "${DATASET}/mean_std.npy" \
    --output_dir "${OUTPUT_DIR}" \
    --init_checkpoint "${INIT_CHECKPOINT}" \
    --epochs=5 \
    --start_eval=999 \
    --checkpoint_interval=1 \
    --debug_save_final_checkpoint=1 \
    --batch_size="${BATCH_SIZE}" \
    --lr=4e-5 \
    --min_lr=1e-6 \
    --weight_decay=1e-4 \
    --warmup_epochs=1 \
    --clip_max_norm=0.1 \
    --seed=0 \
    --reset_rng_after_init=1 \
    --deterministic_training=1 \
    --eos_coef=0.5 \
    --reg_loss_coef=0.002 \
    --cls_loss_coef=1.0 \
    --set_cost_point=0.1 \
    --set_cost_class=1.0 \
    --num_workers=0 \
    --match_dis=15 \
    --filter_empty_train \
    --skip_empty_eval \
    --proto_start_epoch=0 \
    --proto_gt_warmup_epochs=1 \
    --proto_loss_weight=1.0 \
    --proto_freeze_detector=1 \
    --proto_debug_interval=0 \
    --proto_debug_fail_fast=1 \
    "${COMMON_MODEL_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/console.log"
else
  echo "[Gate] evaluation-only recovery: reusing ${OUTPUT_DIR}/recent_model.pth"
fi

if [ ! -f "${OUTPUT_DIR}/recent_model.pth" ]; then
  echo "[ERROR] gate checkpoint not found: ${OUTPUT_DIR}/recent_model.pth"
  exit 2
fi

python evaluate_discriminative_dual_proxy_gate.py \
  --dataset "${DATASET}" \
  --mean_std_path "${DATASET}/mean_std.npy" \
  --checkpoint "${OUTPUT_DIR}/recent_model.pth" \
  --gate_output "${OUTPUT_DIR}/representation_gate_metrics.json" \
  --batch_size=1 \
  --num_workers=0 \
  --match_dis=15 \
  "${COMMON_MODEL_ARGS[@]}"

set +e
python decide_discriminative_dual_proxy_gate.py \
  --metrics "${OUTPUT_DIR}/representation_gate_metrics.json" \
  --checkpoint "${OUTPUT_DIR}/recent_model.pth" \
  --output "${OUTPUT_DIR}/decision.json" \
  --bank_output "${OUTPUT_DIR}/dual_proxy_bank.pth"
status=$?
set -e
if [ "${status}" -ne 0 ]; then
  echo "[STOP] representation gate failed; no detector training"
  echo "[RESULT] ${OUTPUT_DIR}/decision.json"
  exit "${status}"
fi

echo "[PASS] representation gate passed: ${OUTPUT_DIR}/dual_proxy_bank.pth"
echo "[NEXT] detector training remains a separate explicit command"
