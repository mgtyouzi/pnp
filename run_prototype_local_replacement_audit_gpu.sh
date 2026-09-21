#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
GPU="${GPU:-7}"
DATASET="${DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
MEAN_STD="${MEAN_STD:-${DATASET}/mean_std.npy}"
MAX_IMAGES="${MAX_IMAGES:-200}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT}"
if [ -z "${PRR_ROOT:-}" ]; then
  PRR_ROOT="$(cat experiments/latest_prototype_reliability_rescue.txt)"
fi
case "${PRR_ROOT}" in
  /*) ;;
  *) PRR_ROOT="${PROJECT}/${PRR_ROOT}" ;;
esac
CHECKPOINT="${CHECKPOINT:-${PRR_ROOT}/prototype_reliability_rescue/recent_model.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${PRR_ROOT}/prototype_local_replacement_audit_${RUN_ID}_gpu${GPU}}"

test -f "${CHECKPOINT}"
test -f "${MEAN_STD}"
mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${OUTPUT_DIR}" > experiments/latest_prototype_local_replacement_audit.txt

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[Prototype-local-replacement] physical_gpu=${GPU}"
echo "[Prototype-local-replacement] expected_version=prototype_local_replacement_audit_v1_20260907"
echo "[Prototype-local-replacement] checkpoint=${CHECKPOINT}"
echo "[Prototype-local-replacement] output=${OUTPUT_DIR}"
python test_prototype_local_replacement_audit.py
python prototype_local_replacement_audit.py \
  --checkpoint="${CHECKPOINT}" \
  --dataset="${DATASET}" \
  --mean_std_path="${MEAN_STD}" \
  --output_dir="${OUTPUT_DIR}" \
  --gpu=0 \
  --max_images="${MAX_IMAGES}" \
  --match_radius=15 \
  --minimum_distance_improvement=3 \
  --minimum_similarity_improvement=0.1 \
  --minimum_candidate_score=0.2 \
  --maximum_score_drop=0.25 \
  --distance_utility_weight=1 \
  --similarity_utility_weight=1 \
  --score_utility_weight=0.5 \
  --num_workers=0

echo "[SUCCESS] ${OUTPUT_DIR}"
