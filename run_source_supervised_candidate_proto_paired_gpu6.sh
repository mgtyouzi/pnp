#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
GPU="${GPU:-6}"
DATASET="${DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
MEAN_STD="${MEAN_STD:-${DATASET}/mean_std.npy}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT}/experiments/source_supervised_candidate_proto_paired_${RUN_ID}_gpu${GPU}_bt1}"
BANK_DIR="${PAIR_ROOT}/offline_bank"
BANK="${BANK_DIR}/source_supervised_candidate_proto_bank.pth"
CONTROL_DIR="${PAIR_ROOT}/control_p2p"
PROTOTYPE_DIR="${PAIR_ROOT}/source_supervised_candidate_proto"
EVAL_DIR="${PAIR_ROOT}/source_eval"

cd "${PROJECT}"
mkdir -p "${BANK_DIR}" "${CONTROL_DIR}" "${PROTOTYPE_DIR}" "${EVAL_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

echo "[SSCP] physical_gpu=${GPU}"
echo "[SSCP] pair_root=${PAIR_ROOT}"
echo "[SSCP] dataset=${DATASET}"
echo "[SSCP] init=${INIT_CHECKPOINT}"

test -f "${MEAN_STD}"
test -f "${INIT_CHECKPOINT}"
python test_source_supervised_candidate_proto.py
python test_source_supervised_candidate_proto_wiring.py
python test_source_supervised_candidate_proto_bank_builder.py
python test_source_supervised_candidate_proto_decision.py

if [ ! -f "${BANK}" ]; then
  python build_source_supervised_candidate_proto_bank.py \
    --checkpoint "${INIT_CHECKPOINT}" \
    --dataset "${DATASET}" \
    --mean_std_path "${MEAN_STD}" \
    --output_dir "${BANK_DIR}" \
    --gpu 0 \
    --support_fraction 0.8 \
    --num_fg 4 \
    --num_bg 8 \
    --temperature 0.1 \
    --positive_radius 15 \
    --background_radius 30 \
    --max_hard_positive 32 \
    --max_random_positive 32 \
    --max_hard_background 32 \
    --max_random_background 32 \
    --kmeans_iterations 25 \
    --seed 0 \
    --num_workers 0
fi

COMMON_TRAIN_ARGS=(
  --batch_size=1
  --epochs=10
  --start_eval=999
  --checkpoint_interval=5
  --init_checkpoint="${INIT_CHECKPOINT}"
  --dataset="${DATASET}"
  --mean_std_path="${MEAN_STD}"
  --num_workers=0
  --debug_save_final_checkpoint=1
  --reset_rng_after_init=1
  --deterministic_training=1
  --lr=4e-5
  --min_lr=1e-6
  --warmup_epochs=0
  --weight_decay=1e-4
  --eos_coef=0.5
  --match_dis=15
  --seed=0
)

if [ ! -f "${CONTROL_DIR}/recent_model.pth" ]; then
  python train_p2p_no_empty_v2.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    --output_dir="${CONTROL_DIR}"
fi

PROTO_ARGS=(
  --proto_enable
  --proto_mode=source_supervised_candidate_proto
  --proto_bank_path="${BANK}"
  --proto_start_epoch=0
  --proto_num_fg=4
  --proto_num_bg=8
  --proto_fg_queue_size=8192
  --proto_bg_queue_size=16384
  --proto_temperature=0.1
  --proto_momentum=0.95
  --proto_dead_patience=2
  --proto_input_grad_scale=0.05
  --proto_max_hard_bg_per_image=32
  --proto_max_random_bg_per_image=32
  --proto_positive_radius=15
  --proto_background_radius=30
  --proto_inference_fusion=1
  --proto_loss_weight=1.0
  --proto_debug_interval=200
  --sscp_max_hard_positive_per_image=32
  --sscp_max_random_positive_per_image=32
  --sscp_proto_ce_weight=0.02
  --sscp_margin_weight=0.02
  --sscp_fused_ce_weight=0.05
  --sscp_margin=0.1
  --sscp_alpha_max=0.25
  --sscp_alpha_initial=0.001
  --sscp_confidence_threshold=0.05
  --sscp_fusion_clip=0.5
  --sscp_min_assignment_share=0.02
  --sscp_sinkhorn_epsilon=0.05
  --sscp_sinkhorn_iterations=3
)

if [ ! -f "${PROTOTYPE_DIR}/recent_model.pth" ]; then
  python train_p2p_no_empty_v2.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    "${PROTO_ARGS[@]}" \
    --output_dir="${PROTOTYPE_DIR}"
fi

run_eval() {
  local label="$1"
  local checkpoint="$2"
  local fusion="$3"
  local output="${EVAL_DIR}/${label}"
  if [ -f "${output}/summary.json" ]; then
    return
  fi
  local proto_eval_args=()
  if [ "${label}" != "control" ]; then
    proto_eval_args=("${PROTO_ARGS[@]}")
    proto_eval_args+=(--proto_inference_fusion="${fusion}")
  fi
  python diagnose_cross_domain.py \
    --checkpoint "${checkpoint}" \
    --case_name "${label}_paraffin_test" \
    --dataset "${DATASET}" \
    --output_dir "${output}" \
    --num_workers 0 \
    --batch_size 1 \
    --skip_empty_gt_in_eval \
    "${proto_eval_args[@]}"
}

run_eval control "${CONTROL_DIR}/recent_model.pth" 0
run_eval prototype_raw "${PROTOTYPE_DIR}/recent_model.pth" 0
run_eval prototype_fused "${PROTOTYPE_DIR}/recent_model.pth" 1

python decide_source_supervised_candidate_proto_gate.py \
  --control "${EVAL_DIR}/control" \
  --prototype_raw "${EVAL_DIR}/prototype_raw" \
  --prototype_fused "${EVAL_DIR}/prototype_fused" \
  --minimum_gain 0.002 \
  --output "${PAIR_ROOT}/decision.json"

printf '%s\n' "${PAIR_ROOT}" > "${PROJECT}/experiments/latest_source_supervised_candidate_proto.txt"
echo "[SUCCESS] ${PAIR_ROOT}"
