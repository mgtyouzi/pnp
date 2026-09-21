#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30400}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MATCH_DIS="${MATCH_DIS:-15}"
NEAR_RADIUS="${NEAR_RADIUS:-30}"
DEDUP_INTERVAL="${DEDUP_INTERVAL:-15}"
RUN_REVERSE="${RUN_REVERSE:-1}"

# Safety guard: GPU 7 must never be used in this container.
NORMALIZED_CUDA=",${CUDA_VISIBLE_DEVICES// /,},"
if [[ "${NORMALIZED_CUDA}" == *",7,"* ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} includes GPU 7. Refusing to run."
  exit 2
fi

if [ -z "${PARAFFIN_DATASET:-}" ]; then
  PARAFFIN_DATASET=$'\347\237\263\350\234\241-2025'
fi

if [ -z "${FROZEN_DATASET:-}" ]; then
  FROZEN_DATASET=$'\345\206\260\345\206\273-2025'
fi

CKPT_PAR="${CKPT_PAR:-${PROJECT_ROOT}/pth/0318/best_model.pth}"
CKPT_FROZEN="${CKPT_FROZEN:-${PROJECT_ROOT}/pth/0326_bd/best_model.pth}"

cd "${PROJECT_ROOT}"

export MASTER_ADDR
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments/error_diagnosis_${timestamp}}"
mkdir -p "${OUT_ROOT}"
OVERALL="${OUT_ROOT}/overall_summary.csv"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "PARAFFIN_DATASET=${PARAFFIN_DATASET}"
echo "FROZEN_DATASET=${FROZEN_DATASET}"
echo "CKPT_PAR=${CKPT_PAR}"
echo "CKPT_FROZEN=${CKPT_FROZEN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MATCH_DIS=${MATCH_DIS}"
echo "NEAR_RADIUS=${NEAR_RADIUS}"
echo "DEDUP_INTERVAL=${DEDUP_INTERVAL}"
echo "RUN_REVERSE=${RUN_REVERSE}"
echo "Protocol: evaluate all patches but compute metrics on non-empty-GT patches only; empty-GT patches are skipped from P/R/F1"

run_case() {
  local case_name="$1"
  local checkpoint="$2"
  local dataset="$3"
  local port="$4"
  local out_dir="${OUT_ROOT}/${case_name}"
  mkdir -p "${out_dir}"

  echo "===================================================================================================="
  echo "case=${case_name}"
  echo "checkpoint=${checkpoint}"
  echo "dataset=${dataset}"
  echo "out_dir=${out_dir}"
  echo "master_port=${port}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${port}" \
    diagnose_cross_domain.py \
    --checkpoint "${checkpoint}" \
    --case_name "${case_name}" \
    --dataset "${dataset}" \
    --output_dir "${out_dir}" \
    --num_classes=1 \
    --num_workers="${NUM_WORKERS}" \
    --match_dis="${MATCH_DIS}" \
    --near_radius="${NEAR_RADIUS}" \
    --dedup_interval="${DEDUP_INTERVAL}" \
    --skip_empty_gt_in_eval \
    2>&1 | tee "${out_dir}/console.log"

  if [ ! -f "${OVERALL}" ]; then
    cp "${out_dir}/summary.csv" "${OVERALL}"
  else
    tail -n +2 "${out_dir}/summary.csv" >> "${OVERALL}"
  fi
}

case_index=0
run_case "0318_paraffin_to_paraffin" "${CKPT_PAR}" "${PARAFFIN_DATASET}" "$((MASTER_PORT_BASE + case_index))"
case_index=$((case_index + 1))

run_case "0318_paraffin_to_frozen" "${CKPT_PAR}" "${FROZEN_DATASET}" "$((MASTER_PORT_BASE + case_index))"
case_index=$((case_index + 1))

run_case "0326bd_frozen_to_frozen" "${CKPT_FROZEN}" "${FROZEN_DATASET}" "$((MASTER_PORT_BASE + case_index))"
case_index=$((case_index + 1))

if [ "${RUN_REVERSE}" = "1" ]; then
  run_case "0326bd_frozen_to_paraffin" "${CKPT_FROZEN}" "${PARAFFIN_DATASET}" "$((MASTER_PORT_BASE + case_index))"
  case_index=$((case_index + 1))
fi

echo "Done. Overall summary:"
echo "${OVERALL}"
cat "${OVERALL}"
