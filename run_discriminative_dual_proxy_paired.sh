#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
PARAFFIN_DATASET="${PARAFFIN_DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
FROZEN_DATASET="${FROZEN_DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/冰冻-2025}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
GATE_DIR="${GATE_DIR:?set GATE_DIR to one passed discriminative dual-proxy gate directory}"
GPU="${GPU:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"
SKIP_CONTROL="${SKIP_CONTROL:-0}"

cd "${PROJECT_ROOT}"
BANK="${GATE_DIR}/dual_proxy_bank.pth"
GATE_DECISION="${GATE_DIR}/decision.json"
for required in \
  "${PARAFFIN_DATASET}/mean_std.npy" \
  "${FROZEN_DATASET}/mean_std.npy" \
  "${BASELINE_CHECKPOINT}" \
  "${BANK}" \
  "${GATE_DECISION}"; do
  if [ ! -f "${required}" ]; then
    echo "[ERROR] required file not found: ${required}"
    exit 2
  fi
done

python - "${GATE_DECISION}" "${BANK}" "${BASELINE_CHECKPOINT}" <<'PY'
import hashlib
import json
import sys
import torch

decision = json.load(open(sys.argv[1], encoding="utf-8"))
if decision.get("gate_pass") is not True:
    raise SystemExit("[ERROR] gate_pass is not true; detector training is forbidden")
bank = torch.load(sys.argv[2], map_location="cpu")
digest = hashlib.sha256()
with open(sys.argv[3], "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
expected = str(bank.get("detector_init_sha256", ""))
if not expected or digest.hexdigest() != expected:
    raise SystemExit("[ERROR] baseline checkpoint does not match the gate bank")
print("[Gate] verified gate_pass=true")
print("[Gate] verified detector initialization SHA256")
PY

timestamp="$(date +%Y%m%d_%H%M%S)"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT_ROOT}/experiments/discriminative_dual_proxy_paired_${timestamp}_gpu${GPU}_bt${BATCH_SIZE}}"
CONTROL_DIR="${CONTROL_DIR:-${PAIR_ROOT}/control_p2p}"
PROXY_DIR="${PROXY_DIR:-${PAIR_ROOT}/dual_proxy_p2p}"
mkdir -p "${CONTROL_DIR}" "${PROXY_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[Pair] root=${PAIR_ROOT}"
echo "[Pair] physical_gpu=${GPU}, epochs=10, batch_size=${BATCH_SIZE}"
echo "[Pair] baseline=${BASELINE_CHECKPOINT}"
echo "[Pair] passed_bank=${BANK}"
echo "[Pair] invariant=same initialization/RNG/data/P2P settings; raw P2P inference only"

COMMON_ARGS=(
  train_p2p_no_empty_v2.py
  --dataset "${PARAFFIN_DATASET}"
  --mean_std_path "${PARAFFIN_DATASET}/mean_std.npy"
  --num_classes=1
  --epochs=10
  --batch_size="${BATCH_SIZE}"
  --lr=4e-5
  --min_lr=1e-6
  --weight_decay=1e-4
  --warmup_epochs=5
  --clip_max_norm=0.1
  --seed=0
  --reset_rng_after_init=1
  --deterministic_training=1
  --eos_coef=0.5
  --reg_loss_coef=0.002
  --cls_loss_coef=1.0
  --set_cost_point=0.1
  --set_cost_class=1.0
  --backbone=resnet50
  --position_embedding=sine
  --enc_layers=6
  --dim_feedforward=2048
  --hidden_dim=256
  --dropout=0.1
  --nheads=8
  --row=2
  --col=2
  --start_eval=8
  --checkpoint_interval=5
  --num_workers=0
  --match_dis=15
  --filter_empty_train
  --skip_empty_eval
  --init_checkpoint "${BASELINE_CHECKPOINT}"
)

PROXY_ARGS=(
  --proto_enable
  --proto_mode=discriminative_dual_proxy
  --proto_bank_path "${BANK}"
  --proto_start_epoch=0
  --proto_embedding_dim=64
  --proto_dual_num_fg=4
  --proto_dual_num_bg=4
  --proto_dual_fg_queue_size=8192
  --proto_dual_bg_queue_size=8192
  --proto_gt_max_support_per_image=64
  --proto_max_hard_bg_per_image=32
  --proto_max_random_bg_per_image=32
  --proto_positive_radius=15
  --proto_background_radius=30
  --proto_temperature=0.2
  --proto_momentum=0.99
  --proto_gt_projector_momentum=0.999
  --proto_dual_supcon_weight=0.1
  --proto_dual_separation_weight=1.0
  --proto_dual_separation_margin=0.1
  --proto_dual_balance_weight=0.1
  --proto_gt_min_assignment_share=0.05
  --proto_dead_patience=2
  --proto_sampling_seed=0
  --proto_gt_warmup_epochs=0
  --proto_loss_weight=0.02
  --proto_debug_interval=200
  --proto_debug_fail_fast=1
  --proto_inference_fusion=0
)

if [ "${EVAL_ONLY}" != "1" ]; then
  echo "[Stage 1/6] implementation preflight"
  python test_discriminative_dual_proxy_structure.py
  python test_discriminative_dual_proxy_wiring.py
  python test_discriminative_dual_proxy_gate.py
  python test_discriminative_dual_proxy_paired.py
  python test_discriminative_dual_proxy_run_script.py
  python test_discriminative_dual_proxy.py

  if [ "${SKIP_CONTROL}" = "1" ]; then
    echo "[Stage 2/6] reusing control=${CONTROL_DIR}/recent_model.pth"
  else
    echo "[Stage 2/6] paired control P2P"
    python "${COMMON_ARGS[@]}" --output_dir "${CONTROL_DIR}" \
      2>&1 | tee "${CONTROL_DIR}/console.log"
  fi
  test -f "${CONTROL_DIR}/recent_model.pth"

  echo "[Stage 3/6] discriminative dual-proxy P2P"
  python "${COMMON_ARGS[@]}" "${PROXY_ARGS[@]}" --output_dir "${PROXY_DIR}" \
    2>&1 | tee "${PROXY_DIR}/console.log"
else
  echo "[Stage 1-3/6] EVAL_ONLY=1; reusing ${PAIR_ROOT}"
fi
test -f "${CONTROL_DIR}/recent_model.pth"
test -f "${PROXY_DIR}/recent_model.pth"

EVAL_MODEL_ARGS=(
  --num_classes=1
  --batch_size=1
  --backbone=resnet50
  --position_embedding=sine
  --enc_layers=6
  --dim_feedforward=2048
  --hidden_dim=256
  --dropout=0.1
  --nheads=8
  --row=2
  --col=2
  --num_workers=0
  --match_dis=15
  --skip_empty_gt_in_eval
)

echo "[Stage 4/6] frozen target evaluation: control"
python diagnose_cross_domain.py \
  "${EVAL_MODEL_ARGS[@]}" \
  --dataset "${FROZEN_DATASET}" \
  --checkpoint "${CONTROL_DIR}/recent_model.pth" \
  --case_name dualproxy_control_e10_frozen \
  --output_dir "${CONTROL_DIR}/eval_frozen_e10_gpu${GPU}"

echo "[Stage 5/6] frozen target evaluation: dual proxy"
python diagnose_cross_domain.py \
  "${EVAL_MODEL_ARGS[@]}" \
  "${PROXY_ARGS[@]}" \
  --dataset "${FROZEN_DATASET}" \
  --checkpoint "${PROXY_DIR}/recent_model.pth" \
  --case_name dualproxy_p2p_e10_frozen \
  --output_dir "${PROXY_DIR}/eval_frozen_e10_gpu${GPU}" \
  --strict_load

echo "[Stage 6/6] source/target/mechanism decision"
python - "${PAIR_ROOT}" "${GPU}" <<'PY'
import csv
import json
import os
import sys
import torch

root, gpu = sys.argv[1:]
control_path = os.path.join(root, "control_p2p", f"eval_frozen_e10_gpu{gpu}", "summary.json")
proxy_path = os.path.join(root, "dual_proxy_p2p", f"eval_frozen_e10_gpu{gpu}", "summary.json")
control = json.load(open(control_path, encoding="utf-8"))
proxy = json.load(open(proxy_path, encoding="utf-8"))
rows = list(csv.DictReader(open(
    os.path.join(root, "dual_proxy_p2p", "prototype_epoch_metrics.csv"),
    encoding="utf-8",
)))
last = rows[-1] if rows else {}

def number(payload, key):
    value = payload.get(key)
    return float(value) if value not in (None, "") else float("nan")

def source_metrics(path):
    checkpoint = torch.load(path, map_location="cpu")
    metrics = checkpoint.get("metrics", {})
    detection = metrics.get("检测指标", metrics.get("分类指标", [0.0, 0.0, 0.0]))
    return {
        "precision": float(detection[0]),
        "recall": float(detection[1]),
        "f1": float(detection[2]),
        "mae": float(metrics.get("MAE", 0.0)),
    }

metric_keys = (
    "precision", "recall", "f1", "mae", "pred_gt_ratio",
    "fp_background_far", "fn_low_score_or_background",
)
mechanism_map = {
    "foreground_similarity_gap": "dualproxy_fg_similarity_gap",
    "background_similarity_gap": "dualproxy_bg_similarity_gap",
    "foreground_effective_prototypes": "foreground_effective_prototypes",
    "background_effective_prototypes": "background_effective_prototypes",
    "foreground_assignment_share_min": "foreground_assignment_share_min",
    "background_assignment_share_min": "background_assignment_share_min",
    "foreground_pairwise_similarity_max": "foreground_pairwise_similarity_max",
    "background_pairwise_similarity_max": "background_pairwise_similarity_max",
    "gradient_all_cosine": "gradient_all_cosine",
}
payload = {
    "control": {key: control.get(key) for key in metric_keys},
    "prototype": {key: proxy.get(key) for key in metric_keys},
    "source_control": source_metrics(os.path.join(root, "control_p2p", "recent_model.pth")),
    "source_prototype": source_metrics(os.path.join(root, "dual_proxy_p2p", "recent_model.pth")),
    "mechanism": {
        output_key: number(last, csv_key)
        for output_key, csv_key in mechanism_map.items()
    },
}
path = os.path.join(root, "paired_input.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
print("paired_input=", path)
PY

set +e
python decide_discriminative_dual_proxy_paired.py \
  --input "${PAIR_ROOT}/paired_input.json" \
  --output "${PAIR_ROOT}/paired_decision.json"
status=$?
set -e
if [ "${status}" -ne 0 ]; then
  echo "[STOP] paired detector gate failed; do not start a long run"
  echo "[RESULT] ${PAIR_ROOT}/paired_decision.json"
  exit "${status}"
fi

echo "[PASS] paired detector gate passed"
echo "[RESULT] ${PAIR_ROOT}/paired_decision.json"
