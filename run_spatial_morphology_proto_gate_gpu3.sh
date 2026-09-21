#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
SOURCE_DATASET="${SOURCE_DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
CHECKPOINT="${CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
GPU="${GPU:-3}"
MAX_IMAGES="${MAX_IMAGES:-200}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiments/spatial_morphology_proto_gate_0318_${STAMP}_gpu${GPU}}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}/source_train_archive" "${ROOT}/source_train_loso"
printf '%s\n' "${ROOT}" > experiments/latest_spatial_morphology_proto_gate.txt

echo "[Stage 1/3] tensor, contract, and syntax preflight"
python test_spatial_morphology_proto_data.py
python test_spatial_morphology_proto_structure.py
python -m py_compile \
  spatial_morphology_proto_data.py \
  extract_spatial_morphology_proto_archive.py \
  audit_candidate_conditioned_proto_loso.py

echo "[Stage 2/3] extract independent P2/P3 local morphology from ${MAX_IMAGES} source-train images"
CUDA_VISIBLE_DEVICES="${GPU}" python extract_spatial_morphology_proto_archive.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${SOURCE_DATASET}" \
  --phase train \
  --mean_std_path "${SOURCE_DATASET}/mean_std.npy" \
  --output_dir "${ROOT}/source_train_archive" \
  --gpu 0 \
  --num_workers 0 \
  --max_images "${MAX_IMAGES}" \
  --dedup_interval 15 \
  --match_dis 15 \
  --near_radius 30 \
  --spatial_levels 0,1 \
  --spatial_grid_size 5 \
  --spatial_radius 12 \
  2>&1 | tee "${ROOT}/source_train_archive/extract.log"

echo "[Stage 3/3] slide-LOSO gate; no detector training is allowed in this script"
CUDA_VISIBLE_DEVICES="${GPU}" python audit_candidate_conditioned_proto_loso.py \
  --archive "${ROOT}/source_train_archive/spatial_morphology_archive.npz" \
  --manifest "${ROOT}/source_train_archive/spatial_morphology_manifest.json" \
  --output_dir "${ROOT}/source_train_loso" \
  --gpu 0 \
  --feature_mode spatial_morphology \
  --embedding_dim 32 \
  --foreground_prototypes 4 \
  --background_prototypes 4 \
  --temperature 0.15 \
  --epochs 40 \
  --patience 8 \
  --score_bins 5 \
  --max_per_class_per_bin 256 \
  2>&1 | tee "${ROOT}/source_train_loso/audit.log"

python - "${ROOT}/source_train_loso/candidate_conditioned_proto_audit.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    audit = json.load(handle)
gate = audit["gate"]
print("[Spatial-morphology-gate] pass=", gate["gate_pass"])
print(json.dumps(gate["metrics"], ensure_ascii=False, indent=2))
if not gate["gate_pass"]:
    print("[STOP] no detector training; failures=" + ",".join(gate["failures"]))
PY

echo "[RESULT] ${ROOT}"
