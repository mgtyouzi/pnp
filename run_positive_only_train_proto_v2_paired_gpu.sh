#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
GPU="${GPU:-6}"
DATASET="${DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
MEAN_STD="${MEAN_STD:-${DATASET}/mean_std.npy}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT}"
REFERENCE_PAIR_ROOT="${REFERENCE_PAIR_ROOT:-$(cat experiments/latest_positive_only_train_proto.txt)}"
case "${REFERENCE_PAIR_ROOT}" in
  /*) ;;
  *) REFERENCE_PAIR_ROOT="${PROJECT}/${REFERENCE_PAIR_ROOT}" ;;
esac
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-${REFERENCE_PAIR_ROOT}/control_p2p/recent_model.pth}"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT}/experiments/positive_only_train_proto_v2_${RUN_ID}_gpu${GPU}_bt${BATCH_SIZE}}"
PROTOTYPE_DIR="${PAIR_ROOT}/positive_only_train_proto"

test -f "${CONTROL_CHECKPOINT}"
test -f "${MEAN_STD}"
test -f "${INIT_CHECKPOINT}"
mkdir -p "${PAIR_ROOT}/control_p2p" "${PROTOTYPE_DIR}"
ln -sfn "${CONTROL_CHECKPOINT}" "${PAIR_ROOT}/control_p2p/recent_model.pth"
printf '%s\n' "${PAIR_ROOT}" > experiments/latest_positive_only_train_proto_v2.txt

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[POTP-v2] physical_gpu=${GPU}"
echo "[POTP-v2] pair_root=${PAIR_ROOT}"
echo "[POTP-v2] reused_control=${CONTROL_CHECKPOINT}"
echo "[POTP-v2] routing=detached-all-GT-support + lowest-confidence-25%-matched-query"

python test_positive_only_train_proto.py
python test_positive_only_train_proto_wiring.py
python test_paired_detection_threshold_sweep.py
python -m py_compile models/positive_only_train_proto.py models/detr.py train_p2p.py paired_detection_threshold_sweep.py

python train_p2p_no_empty_v2.py \
  --dataset="${DATASET}" \
  --mean_std_path="${MEAN_STD}" \
  --batch_size="${BATCH_SIZE}" \
  --epochs="${EPOCHS}" \
  --start_eval=8 \
  --checkpoint_interval=5 \
  --init_checkpoint="${INIT_CHECKPOINT}" \
  --output_dir="${PROTOTYPE_DIR}" \
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
  --proto_mode=positive_only_train_proto \
  --proto_start_epoch=0 \
  --proto_embedding_dim=64 \
  --proto_num_fg=4 \
  --proto_temperature=0.1 \
  --proto_loss_weight=1.0 \
  --proto_input_grad_scale=0.05 \
  --proto_background_radius=30 \
  --proto_max_hard_bg_per_image=32 \
  --proto_max_random_bg_per_image=16 \
  --proto_gt_support_queue_size=8192 \
  --proto_kmeans_iterations=20 \
  --proto_sampling_seed=0 \
  --proto_debug_interval=200 \
  --proto_debug_fail_fast=1 \
  --proto_inference_fusion=0 \
  --potp_warmup_epochs=2 \
  --potp_hidden_dim=128 \
  --potp_positive_margin=0.4 \
  --potp_negative_margin=0.2 \
  --potp_diversity_margin=0.5 \
  --potp_pair_weight=0.005 \
  --potp_positive_weight=0.005 \
  --potp_negative_weight=0.01 \
  --potp_balance_weight=0.001 \
  --potp_diversity_weight=0.001 \
  --potp_hard_positive_fraction=0.25 \
  --potp_max_hard_positive_per_image=32 \
  --potp_detach_support_input=1 \
  2>&1 | tee "${PROTOTYPE_DIR}/console.log"

test -f "${PROTOTYPE_DIR}/recent_model.pth"

if [ "${RUN_THRESHOLD_SWEEP:-1}" = "1" ]; then
  python paired_detection_threshold_sweep.py \
    --pair_root="${PAIR_ROOT}" \
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
fi

echo "[SUCCESS] ${PAIR_ROOT}"
