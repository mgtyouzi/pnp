#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
GPU="${GPU:-2}"
PAIR_ROOT="${PAIR_ROOT:-$(cat "${PROJECT}/experiments/latest_source_supervised_candidate_proto.txt")}"
CHECKPOINT="${CHECKPOINT:-${PAIR_ROOT}/source_supervised_candidate_proto/recent_model.pth}"
BANK="${BANK:-${PAIR_ROOT}/offline_bank/source_supervised_candidate_proto_bank.pth}"
DATASET="${DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-${PAIR_ROOT}/sscp_selective_rescue_${RUN_ID}_gpu${GPU}}"
MAX_SUPPORT_IMAGES="${MAX_SUPPORT_IMAGES:-0}"
MAX_SELECTION_IMAGES="${MAX_SELECTION_IMAGES:-0}"
MAX_TEST_IMAGES="${MAX_TEST_IMAGES:-0}"

cd "${PROJECT}"
mkdir -p "${OUTPUT}"
OUTPUT="$(realpath "${OUTPUT}")"
test -f "${CHECKPOINT}"
test -f "${BANK}"
test -f "${DATASET}/mean_std.npy"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

python test_sscp_selective_rescue.py
python test_sscp_selective_rescue_diagnostic.py
python test_diagnose_cross_domain_prototype_compat.py

python diagnose_sscp_selective_rescue.py \
  --checkpoint "${CHECKPOINT}" \
  --bank "${BANK}" \
  --dataset "${DATASET}" \
  --mean_std_path "${DATASET}/mean_std.npy" \
  --output_dir "${OUTPUT}" \
  --gpu 0 \
  --num_workers 0 \
  --num_fg 4 \
  --num_bg 8 \
  --uncertainty_grid "0.1,0.25,0.5,1.0" \
  --strength_grid "0.05,0.1,0.2,0.4" \
  --evidence_clip 3.0 \
  --minimum_promotion_precision 0.60 \
  --minimum_group_precision 0.50 \
  --minimum_predictions_per_group 10 \
  --minimum_slides_improved_fraction 0.60 \
  --promotion_budget_ratio 0.02 \
  --promotion_budget_min 1 \
  --promotion_budget_max 8 \
  --maximum_calibration_candidates_per_class 100000 \
  --max_support_images "${MAX_SUPPORT_IMAGES}" \
  --max_selection_images "${MAX_SELECTION_IMAGES}" \
  --max_test_images "${MAX_TEST_IMAGES}" \
  --dedup_interval 15 \
  --match_dis 15 \
  --near_radius 30 \
  --minimum_f1_gain 0.002

printf '%s\n' "${OUTPUT}" > "${PROJECT}/experiments/latest_sscp_selective_rescue.txt"
echo "[SUCCESS] ${OUTPUT}"
