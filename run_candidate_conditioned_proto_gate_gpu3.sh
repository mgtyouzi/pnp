#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
SOURCE_DATASET="${SOURCE_DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
TARGET_DATASET="${TARGET_DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/冰冻-2025}"
CHECKPOINT="${CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
GPU="${GPU:-3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiments/candidate_conditioned_proto_gate_0318_${STAMP}_gpu${GPU}}"
ARCHIVE_VERSION="candidate_conditioned_proto_archive_v2_20260829"
CHECKPOINT_SHA=$(python -c "import hashlib; p=r'''${CHECKPOINT}'''; h=hashlib.sha256(); f=open(p,'rb'); [h.update(b) for b in iter(lambda:f.read(1048576),b'')]; f.close(); print(h.hexdigest())")

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}"
printf '%s\n' "${ROOT}" > experiments/latest_candidate_conditioned_proto_gate.txt

extract_archive() {
  local dataset="$1"
  local phase="$2"
  local output="$3"
  if [ -s "${output}/candidate_archive.npz" ] && [ -s "${output}/candidate_archive_manifest.json" ]; then
    if python - "${output}/candidate_archive_manifest.json" "${dataset}" "${phase}" "${CHECKPOINT_SHA}" "${ARCHIVE_VERSION}" <<'PY'
import json
import os
import sys

manifest_path, dataset, phase, checkpoint_sha, archive_version = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
failures = []
if manifest.get("version") != archive_version:
    failures.append("version")
if manifest.get("checkpoint_sha256") != checkpoint_sha:
    failures.append("checkpoint_sha256")
if os.path.realpath(manifest.get("dataset", "")) != os.path.realpath(dataset):
    failures.append("dataset")
if manifest.get("phase") != phase:
    failures.append("phase")
if failures:
    print("archive identity mismatch: " + ",".join(failures), file=sys.stderr)
    raise SystemExit(2)
PY
    then
      echo "[Resume] validated archive: ${output}"
      return
    fi
    echo "[ERROR] refusing stale candidate archive: ${output}" >&2
    exit 2
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${GPU}" python extract_candidate_conditioned_proto_archive.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${dataset}" \
    --phase "${phase}" \
    --mean_std_path "${dataset}/mean_std.npy" \
    --output_dir "${output}" \
    --gpu 0 \
    --num_workers 0 \
    --dedup_interval 15 \
    --match_dis 15 \
    --near_radius 30 \
    2>&1 | tee "${output}/extract.log"
}

run_loso_audit() {
  local feature_mode="$1"
  local output="$2"
  mkdir -p "${output}"
  rm -f "${output}/candidate_conditioned_proto_bank.pth"
  CUDA_VISIBLE_DEVICES="${GPU}" python audit_candidate_conditioned_proto_loso.py \
    --archive "${ROOT}/source_train_archive/candidate_archive.npz" \
    --manifest "${ROOT}/source_train_archive/candidate_archive_manifest.json" \
    --output_dir "${output}" \
    --gpu 0 \
    --feature_mode "${feature_mode}" \
    --embedding_dim 32 \
    --foreground_prototypes 4 \
    --background_prototypes 4 \
    --temperature 0.15 \
    --epochs 40 \
    --patience 8 \
    --score_bins 5 \
    --max_per_class_per_bin 512 \
    2>&1 | tee "${output}/audit.log"
}

echo "[Stage 1/6] exact source-train candidate archive"
extract_archive "${SOURCE_DATASET}" train "${ROOT}/source_train_archive"

echo "[Stage 2/6] cls-only LOSO control on the same candidate archive"
run_loso_audit cls_only "${ROOT}/source_train_loso_cls_only"

echo "[Stage 3/6] contextual candidate prototype LOSO gate"
run_loso_audit cls_reg_context "${ROOT}/source_train_loso_context"
CONTEXT_DIR="${ROOT}/source_train_loso_context"
BANK="${CONTEXT_DIR}/candidate_conditioned_proto_bank.pth"

if [ ! -s "${BANK}" ]; then
  echo "[STOP] contextual source-train LOSO gate failed; no bank was exported."
  echo "[CONTROL] ${ROOT}/source_train_loso_cls_only/candidate_conditioned_proto_audit.json"
  echo "[RESULT] ${CONTEXT_DIR}/candidate_conditioned_proto_audit.json"
  exit 0
fi

echo "[Stage 4/6] independent source-test archive and fixed-bank gate"
extract_archive "${SOURCE_DATASET}" test "${ROOT}/source_test_archive"
mkdir -p "${ROOT}/source_test_fixed"
CUDA_VISIBLE_DEVICES="${GPU}" python score_candidate_conditioned_proto_bank.py \
  --archive "${ROOT}/source_test_archive/candidate_archive.npz" \
  --manifest "${ROOT}/source_test_archive/candidate_archive_manifest.json" \
  --bank "${BANK}" \
  --output_dir "${ROOT}/source_test_fixed" \
  --mode source_test_fixed \
  --gpu 0 \
  2>&1 | tee "${ROOT}/source_test_fixed/score.log"

SOURCE_STATUS=$(python -c "import json; print(json.load(open(r'${ROOT}/source_test_fixed/decision.json', encoding='utf-8'))['status'])")
if [ "${SOURCE_STATUS}" != "source_test_pass" ]; then
  echo "[STOP] independent source-test gate failed: ${SOURCE_STATUS}"
  echo "[RESULT] ${ROOT}/source_test_fixed/decision.json"
  exit 0
fi

echo "[Stage 5/6] target archive"
extract_archive "${TARGET_DATASET}" test "${ROOT}/target_test_archive"

echo "[Stage 6/6] frozen target diagnostic with source-fixed bank/calibration"
mkdir -p "${ROOT}/target_test_fixed"
CUDA_VISIBLE_DEVICES="${GPU}" python score_candidate_conditioned_proto_bank.py \
  --archive "${ROOT}/target_test_archive/candidate_archive.npz" \
  --manifest "${ROOT}/target_test_archive/candidate_archive_manifest.json" \
  --bank "${BANK}" \
  --output_dir "${ROOT}/target_test_fixed" \
  --mode target_fixed \
  --gpu 0 \
  2>&1 | tee "${ROOT}/target_test_fixed/score.log"

echo "[SUCCESS] candidate-conditioned prototype gate completed"
echo "[OUTPUT] ${ROOT}"
