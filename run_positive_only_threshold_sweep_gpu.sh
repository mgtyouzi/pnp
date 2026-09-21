#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
GPU="${GPU:-6}"
DATASET="${DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
MEAN_STD="${MEAN_STD:-${DATASET}/mean_std.npy}"
LATEST_FILE="${PROJECT}/experiments/latest_positive_only_train_proto.txt"

cd "${PROJECT}"
test -f "${LATEST_FILE}"
PAIR_ROOT="${PAIR_ROOT:-$(cat "${LATEST_FILE}")}"
case "${PAIR_ROOT}" in
  /*) ;;
  *) PAIR_ROOT="${PROJECT}/${PAIR_ROOT}" ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-${PAIR_ROOT}/raw_p2p_threshold_sweep}"

test -f "${PAIR_ROOT}/control_p2p/recent_model.pth"
test -f "${PAIR_ROOT}/positive_only_train_proto/recent_model.pth"
test -f "${MEAN_STD}"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[Threshold-sweep] physical_gpu=${GPU}"
echo "[Threshold-sweep] pair_root=${PAIR_ROOT}"
echo "[Threshold-sweep] output=${OUTPUT_DIR}"

python test_paired_detection_threshold_sweep.py
python -m py_compile paired_detection_threshold_sweep.py
python paired_detection_threshold_sweep.py \
  --pair_root="${PAIR_ROOT}" \
  --dataset="${DATASET}" \
  --mean_std_path="${MEAN_STD}" \
  --gpu=0 \
  --thresholds="${THRESHOLDS:-0.45:0.75:0.01}" \
  --match_distance="${MATCH_DISTANCE:-15}" \
  --dedup_interval="${DEDUP_INTERVAL:-15}" \
  --near_radius="${NEAR_RADIUS:-30}" \
  --minimum_gain="${MINIMUM_GAIN:-0.002}" \
  --num_workers="${NUM_WORKERS:-0}" \
  --output_dir="${OUTPUT_DIR}"

cat "${OUTPUT_DIR}/paired_threshold_decision.json"
