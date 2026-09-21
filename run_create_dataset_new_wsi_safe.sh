#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
SRC_DATA_ROOT="${SRC_DATA_ROOT:-datasets}"
DST_DATA_ROOT="${DST_DATA_ROOT:-dataset-new}"
MODE="${MODE:-symlink}"
PARAFFIN_DATASET="${PARAFFIN_DATASET:-}"
FROZEN_DATASET="${FROZEN_DATASET:-}"

if [ -z "${PARAFFIN_DATASET}" ]; then
  PARAFFIN_DATASET=$'\347\237\263\350\234\241-2025'
fi
if [ -z "${FROZEN_DATASET}" ]; then
  FROZEN_DATASET=$'\345\206\260\345\206\273-2025'
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/result/create_dataset_new_wsi_${timestamp}}"
mkdir -p "${OUT_DIR}"

cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SRC_DATA_ROOT=${SRC_DATA_ROOT}"
echo "DST_DATA_ROOT=${DST_DATA_ROOT}"
echo "MODE=${MODE}"
echo "PARAFFIN_DATASET=${PARAFFIN_DATASET}"
echo "FROZEN_DATASET=${FROZEN_DATASET}"
echo "OUT_DIR=${OUT_DIR}"
echo "Protocol: WSI-level train/val/test split; empty-GT patches are excluded from dataset-new."
echo "Destination folders will be overwritten only for:"
echo "  ${PROJECT_ROOT}/${DST_DATA_ROOT}/${PARAFFIN_DATASET}-wsi"
echo "  ${PROJECT_ROOT}/${DST_DATA_ROOT}/${FROZEN_DATASET}-wsi"

{
  python create_dataset_new_wsi.py \
    --project_root "${PROJECT_ROOT}" \
    --src_data_root "${SRC_DATA_ROOT}" \
    --dst_data_root "${DST_DATA_ROOT}" \
    --dataset "${PARAFFIN_DATASET}" \
    --output_name "${PARAFFIN_DATASET}-wsi" \
    --mode "${MODE}" \
    --exclude_empty \
    --overwrite

  python create_dataset_new_wsi.py \
    --project_root "${PROJECT_ROOT}" \
    --src_data_root "${SRC_DATA_ROOT}" \
    --dst_data_root "${DST_DATA_ROOT}" \
    --dataset "${FROZEN_DATASET}" \
    --output_name "${FROZEN_DATASET}-wsi" \
    --mode "${MODE}" \
    --exclude_empty \
    --overwrite
} 2>&1 | tee "${OUT_DIR}/console.log"

echo "Done. Output:"
echo "${OUT_DIR}"
