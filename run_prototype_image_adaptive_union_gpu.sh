#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
GPU="${GPU:-7}"
DATASET="${DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
MEAN_STD="${MEAN_STD:-${DATASET}/mean_std.npy}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT}"
PAIR_ROOT="${PAIR_ROOT:-$(cat experiments/latest_prototype_local_soft_positive_v2.txt)}"
case "${PAIR_ROOT}" in
  /*) ;;
  *) PAIR_ROOT="${PROJECT}/${PAIR_ROOT}" ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-${PAIR_ROOT}/image_adaptive_union_${RUN_ID}_gpu${GPU}}"
PRIOR_DECISION="${PRIOR_DECISION:-${PAIR_ROOT}/raw_p2p_threshold_sweep/paired_threshold_decision.json}"

test -f "${PAIR_ROOT}/control_p2p/recent_model.pth"
test -f "${PAIR_ROOT}/prototype_local_soft_positive/recent_model.pth"
test -f "${PRIOR_DECISION}"
test -f "${MEAN_STD}"
mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${OUTPUT_DIR}" > experiments/latest_prototype_image_adaptive_union.txt

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[Image-adaptive-union] physical_gpu=${GPU}"
echo "[Image-adaptive-union] pair_root=${PAIR_ROOT}"
echo "[Image-adaptive-union] output=${OUTPUT_DIR}"
python test_prototype_image_adaptive_union.py
python test_prototype_image_adaptive_structure.py
python -m py_compile \
  prototype_image_adaptive_union_audit.py paired_detection_threshold_sweep.py \
  models/detr.py

python prototype_image_adaptive_union_audit.py \
  --pair_root="${PAIR_ROOT}" \
  --dataset="${DATASET}" \
  --mean_std_path="${MEAN_STD}" \
  --prototype_subdir=prototype_local_soft_positive \
  --prior_decision="${PRIOR_DECISION}" \
  --gpu=0 \
  --prototype_thresholds=0.56:0.76:0.02 \
  --margin_thresholds=-0.20:0.40:0.025 \
  --foreground_support_threshold=0.8 \
  --background_support_threshold=0.05 \
  --support_match_radius=8 \
  --num_foreground_prototypes=4 \
  --num_background_prototypes=4 \
  --max_foreground_supports=64 \
  --max_background_supports=256 \
  --kmeans_iterations=10 \
  --match_distance=15 \
  --dedup_interval=15 \
  --near_radius=30 \
  --minimum_gain=0.002 \
  --num_workers=0 \
  --output_dir="${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/console.log"

echo "[SUCCESS] ${OUTPUT_DIR}"
