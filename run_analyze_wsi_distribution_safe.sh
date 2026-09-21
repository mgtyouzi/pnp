#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
DATA_ROOT="${DATA_ROOT:-datasets}"
PARAFFIN_DATASET="${PARAFFIN_DATASET:-}"
FROZEN_DATASET="${FROZEN_DATASET:-}"

if [ -z "${PARAFFIN_DATASET}" ]; then
  PARAFFIN_DATASET=$'\347\237\263\350\234\241-2025'
fi
if [ -z "${FROZEN_DATASET}" ]; then
  FROZEN_DATASET=$'\345\206\260\345\206\273-2025'
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/result/wsi_distribution_${timestamp}}"
mkdir -p "${OUT_DIR}"

cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PARAFFIN_DATASET=${PARAFFIN_DATASET}"
echo "FROZEN_DATASET=${FROZEN_DATASET}"
echo "OUT_DIR=${OUT_DIR}"

python analyze_wsi_distribution.py \
  --project_root "${PROJECT_ROOT}" \
  --data_root "${DATA_ROOT}" \
  --datasets "${PARAFFIN_DATASET}" "${FROZEN_DATASET}" \
  --output_dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/console.log"

echo "Done. Output:"
echo "${OUT_DIR}"
