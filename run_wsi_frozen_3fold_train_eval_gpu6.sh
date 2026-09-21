#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025}"
GPU_ID="${GPU_ID:-6}"
SRC_DATA_ROOT="${SRC_DATA_ROOT:-datasets}"
DST_DATA_ROOT="${DST_DATA_ROOT:-dataset-new}"
MODE="${MODE:-symlink}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-4e-5}"
EOS_COEF="${EOS_COEF:-0.5}"
MATCH_DIS="${MATCH_DIS:-15}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [ "${GPU_ID}" = "7" ]; then
  echo "ERROR: GPU 7 is forbidden. Refusing to run."
  exit 2
fi

FROZEN_SRC=$'\345\206\260\345\206\273-2025'

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/experiments/wsi_frozen_3fold_${timestamp}}"
mkdir -p "${RUN_ROOT}"

cd "${PROJECT_ROOT}"

if ! grep -q "_resolve_data_phase" dataset_zy_src.py; then
  echo "ERROR: dataset_zy_src.py does not contain _resolve_data_phase. Update code before running."
  exit 3
fi
if ! grep -q "assert_dataset_phase" train_p2p_no_empty_v2.py; then
  echo "ERROR: train_p2p_no_empty_v2.py does not contain assert_dataset_phase. Update code before running."
  exit 4
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "GPU_ID=${GPU_ID}"
echo "FROZEN_SRC=${FROZEN_SRC}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR}"
echo "EOS_COEF=${EOS_COEF}"
echo "MATCH_DIS=${MATCH_DIS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "Protocol: 3 repeated frozen WSI-level hold-out; val/test are paired by similar WSI density where possible."

set_fold_config() {
  case "$1" in
    fold1)
      TRAIN_WSI="2022-06-28_11_11_09.kfb,2022-06-28_11_12_06.kfb,2022-06-28_11_14_35.kfb,2022-06-28_11_15_54.kfb,2022-06-28_11_16_48.kfb,2022-06-28_11_18_14.kfb"
      VAL_WSI="2022-06-28_11_21_44.kfb"
      TEST_WSI="2022-06-28_11_20_39.kfb"
      ;;
    fold2)
      TRAIN_WSI="2022-06-28_11_11_09.kfb,2022-06-28_11_12_06.kfb,2022-06-28_11_14_35.kfb,2022-06-28_11_15_54.kfb,2022-06-28_11_16_48.kfb,2022-06-28_11_18_14.kfb"
      VAL_WSI="2022-06-28_11_20_39.kfb"
      TEST_WSI="2022-06-28_11_21_44.kfb"
      ;;
    fold3)
      TRAIN_WSI="2022-06-28_11_12_06.kfb,2022-06-28_11_14_35.kfb,2022-06-28_11_15_54.kfb,2022-06-28_11_16_48.kfb,2022-06-28_11_20_39.kfb,2022-06-28_11_21_44.kfb"
      VAL_WSI="2022-06-28_11_11_09.kfb"
      TEST_WSI="2022-06-28_11_18_14.kfb"
      ;;
    *)
      echo "ERROR: unknown fold: $1"
      exit 5
      ;;
  esac
}

run_create_fold_dataset() {
  fold="$1"
  set_fold_config "${fold}"
  dataset_name="${FROZEN_SRC}-wsi-${fold}"
  fold_dir="${RUN_ROOT}/${fold}"
  mkdir -p "${fold_dir}"

  echo ""
  echo "================ Create dataset ${fold} ================"
  echo "dataset_name=${dataset_name}"
  echo "train_wsi=${TRAIN_WSI}"
  echo "val_wsi=${VAL_WSI}"
  echo "test_wsi=${TEST_WSI}"

  python create_dataset_new_wsi.py \
    --project_root "${PROJECT_ROOT}" \
    --src_data_root "${SRC_DATA_ROOT}" \
    --dst_data_root "${DST_DATA_ROOT}" \
    --dataset "${FROZEN_SRC}" \
    --output_name "${dataset_name}" \
    --train_wsi "${TRAIN_WSI}" \
    --val_wsi "${VAL_WSI}" \
    --test_wsi "${TEST_WSI}" \
    --mode "${MODE}" \
    --exclude_empty \
    --overwrite \
    2>&1 | tee "${fold_dir}/create_dataset.log"

  cp "${PROJECT_ROOT}/${DST_DATA_ROOT}/${dataset_name}/split_summary.csv" "${fold_dir}/split_summary.csv"
  cp "${PROJECT_ROOT}/${DST_DATA_ROOT}/${dataset_name}/split_wsi.json" "${fold_dir}/split_wsi.json"
  cp "${PROJECT_ROOT}/${DST_DATA_ROOT}/${dataset_name}/mean_std_summary.json" "${fold_dir}/mean_std_summary.json"
}

run_train_fold() {
  fold="$1"
  dataset_name="${FROZEN_SRC}-wsi-${fold}"
  dataset_path="${PROJECT_ROOT}/${DST_DATA_ROOT}/${dataset_name}"
  fold_dir="${RUN_ROOT}/${fold}"
  train_dir="${fold_dir}/train"
  mkdir -p "${train_dir}"

  echo ""
  echo "================ Train ${fold} ================"
  echo "dataset_path=${dataset_path}"
  echo "train_dir=${train_dir}"

  python train_p2p_no_empty_v2.py \
    --dataset "${dataset_path}" \
    --output_dir "${train_dir}" \
    --num_classes=1 \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --lr="${LR}" \
    --eos_coef="${EOS_COEF}" \
    --start_eval=-1 \
    --num_workers="${NUM_WORKERS}" \
    --match_dis="${MATCH_DIS}" \
    --mean_std_path "${dataset_path}/mean_std.npy" \
    --eval_split val \
    --filter_empty_train \
    --skip_empty_eval \
    2>&1 | tee "${train_dir}/console.log"

  if ! grep -q "\[Dataset-test\] phase=val" "${train_dir}/console.log"; then
    echo "ERROR: ${fold} training did not use val split for evaluation."
    exit 6
  fi
}

run_eval_fold() {
  fold="$1"
  checkpoint_name="$2"
  checkpoint_path="$3"
  eval_name="$4"
  dataset_name="${FROZEN_SRC}-wsi-${fold}"
  dataset_path="${PROJECT_ROOT}/${DST_DATA_ROOT}/${dataset_name}"
  fold_dir="${RUN_ROOT}/${fold}"
  eval_dir="${fold_dir}/${eval_name}"
  mkdir -p "${eval_dir}"

  echo ""
  echo "================ Eval ${fold} ${checkpoint_name} ================"
  echo "dataset_path=${dataset_path}"
  echo "checkpoint_path=${checkpoint_path}"
  echo "eval_dir=${eval_dir}"

  if [ ! -f "${checkpoint_path}" ]; then
    echo "ERROR: checkpoint not found: ${checkpoint_path}"
    exit 7
  fi

  python eval_p2p_no_empty_wsi.py \
    --dataset "${dataset_path}" \
    --checkpoint "${checkpoint_path}" \
    --eval_output_dir "${eval_dir}" \
    --output_dir "${eval_dir}" \
    --num_classes=1 \
    --num_workers="${NUM_WORKERS}" \
    --match_dis="${MATCH_DIS}" \
    --mean_std_path "${dataset_path}/mean_std.npy" \
    --eval_split test \
    --skip_empty_eval \
    2>&1 | tee "${eval_dir}/console.log"

  if ! grep -q "\[Dataset-test\] phase=test" "${eval_dir}/console.log"; then
    echo "ERROR: ${fold} ${checkpoint_name} eval did not use test split."
    exit 8
  fi
}

FOLDS=(fold1 fold2 fold3)

echo ""
echo "================ Phase 1/4: create fold datasets ================"
for fold in "${FOLDS[@]}"; do
  run_create_fold_dataset "${fold}"
done

echo ""
echo "================ Phase 2/4: train all folds ================"
for fold in "${FOLDS[@]}"; do
  run_train_fold "${fold}"
done

echo ""
echo "================ Phase 3/4: evaluate all folds ================"
for fold in "${FOLDS[@]}"; do
  train_dir="${RUN_ROOT}/${fold}/train"
  run_eval_fold "${fold}" "best" "${train_dir}/best_model.pth" "eval_best_test"
  run_eval_fold "${fold}" "recent" "${train_dir}/recent_model.pth" "eval_recent_test"
done

echo ""
echo "================ Phase 4/4: summarize metrics ================"
python summarize_wsi_3fold_results.py \
  --run_root "${RUN_ROOT}" \
  --folds "${FOLDS[@]}" \
  2>&1 | tee "${RUN_ROOT}/summary.log"

echo ""
echo "Done. Run root:"
echo "${RUN_ROOT}"
echo "Metrics:"
echo "${RUN_ROOT}/wsi_3fold_metrics.csv"
echo "${RUN_ROOT}/wsi_3fold_aggregate.csv"
