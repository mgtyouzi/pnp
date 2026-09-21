#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
GPU="${GPU:-6}"
DATASET="${DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
MEAN_STD="${MEAN_STD:-${DATASET}/mean_std.npy}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT}/experiments/positive_only_train_proto_paired_${RUN_ID}_gpu${GPU}_bt${BATCH_SIZE}}"
CONTROL_DIR="${CONTROL_DIR:-${PAIR_ROOT}/control_p2p}"
PROTOTYPE_DIR="${PAIR_ROOT}/positive_only_train_proto"
SKIP_CONTROL="${SKIP_CONTROL:-0}"

cd "${PROJECT}"
test -f "${MEAN_STD}"
test -f "${INIT_CHECKPOINT}"
mkdir -p "${CONTROL_DIR}" "${PROTOTYPE_DIR}"
printf '%s\n' "${PAIR_ROOT}" > experiments/latest_positive_only_train_proto.txt

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[POTP] physical_gpu=${GPU}"
echo "[POTP] pair_root=${PAIR_ROOT}"
echo "[POTP] initialization=${INIT_CHECKPOINT}"
echo "[POTP] contract=training-only positive prototypes; raw P2P inference"

python test_positive_only_train_proto.py
python test_positive_only_train_proto_wiring.py
python -m py_compile models/positive_only_train_proto.py models/detr.py train_p2p.py

COMMON_ARGS=(
  --dataset="${DATASET}"
  --mean_std_path="${MEAN_STD}"
  --batch_size="${BATCH_SIZE}"
  --epochs="${EPOCHS}"
  --start_eval=8
  --checkpoint_interval=5
  --init_checkpoint="${INIT_CHECKPOINT}"
  --num_workers=0
  --reset_rng_after_init=1
  --deterministic_training=1
  --lr=4e-5
  --min_lr=1e-6
  --warmup_epochs=0
  --weight_decay=1e-4
  --clip_max_norm=0.1
  --eos_coef=0.5
  --reg_loss_coef=0.002
  --cls_loss_coef=1.0
  --set_cost_point=0.1
  --set_cost_class=1.0
  --num_classes=1
  --backbone=resnet50
  --position_embedding=sine
  --enc_layers=6
  --dim_feedforward=2048
  --hidden_dim=256
  --dropout=0.1
  --nheads=8
  --row=2
  --col=2
  --match_dis=15
  --seed=0
  --filter_empty_train
  --skip_empty_eval
)

PROTO_ARGS=(
  --proto_enable
  --proto_mode=positive_only_train_proto
  --proto_start_epoch=0
  --proto_embedding_dim=64
  --proto_num_fg=4
  --proto_temperature=0.1
  --proto_loss_weight=1.0
  --proto_input_grad_scale=0.05
  --proto_background_radius=30
  --proto_max_hard_bg_per_image=32
  --proto_max_random_bg_per_image=16
  --proto_sampling_seed=0
  --proto_debug_interval=200
  --proto_debug_fail_fast=1
  --proto_inference_fusion=0
  --potp_warmup_epochs=2
  --potp_hidden_dim=128
  --potp_positive_margin=0.4
  --potp_negative_margin=0.2
  --potp_diversity_margin=0.5
  --potp_pair_weight=0.01
  --potp_positive_weight=0.01
  --potp_negative_weight=0.005
  --potp_balance_weight=0.001
  --potp_diversity_weight=0.001
)

if [ "${SKIP_CONTROL}" = "1" ]; then
  echo "[Stage 1/3] reusing control=${CONTROL_DIR}/recent_model.pth"
else
  echo "[Stage 1/3] paired raw P2P control"
  python train_p2p_no_empty_v2.py "${COMMON_ARGS[@]}" \
    --output_dir="${CONTROL_DIR}" 2>&1 | tee "${CONTROL_DIR}/console.log"
fi
test -f "${CONTROL_DIR}/recent_model.pth"

echo "[Stage 2/3] positive-only training prototype"
python train_p2p_no_empty_v2.py "${COMMON_ARGS[@]}" "${PROTO_ARGS[@]}" \
  --output_dir="${PROTOTYPE_DIR}" 2>&1 | tee "${PROTOTYPE_DIR}/console.log"
test -f "${PROTOTYPE_DIR}/recent_model.pth"

echo "[Stage 3/3] raw-P2P source gate"
python - "${PAIR_ROOT}" <<'PY'
import csv
import json
import math
import os
import sys

import torch

root = sys.argv[1]

def checkpoint_metrics(path):
    payload = torch.load(path, map_location="cpu")
    metrics = payload.get("metrics", {})
    values = metrics.get("检测指标", metrics.get("分类指标", [0.0, 0.0, 0.0]))
    return {
        "precision": float(values[0]),
        "recall": float(values[1]),
        "f1": float(values[2]),
        "mae": float(metrics.get("MAE", 0.0)),
    }

def number(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default

control = checkpoint_metrics(os.path.join(root, "control_p2p", "recent_model.pth"))
prototype = checkpoint_metrics(
    os.path.join(root, "positive_only_train_proto", "recent_model.pth")
)
metric_path = os.path.join(
    root, "positive_only_train_proto", "prototype_epoch_metrics.csv"
)
rows = list(csv.DictReader(open(metric_path, encoding="utf-8")))
last = rows[-1]

minimum_f1_gain = 0.002
maximum_precision_drop = 0.005
f1_gain = prototype["f1"] - control["f1"]
precision_drop = control["precision"] - prototype["precision"]
gradient_ratio = number(last, "gradient_all_proto_cls_ratio")
gradient_cosine = number(last, "gradient_all_cosine")
score_gap = number(last, "potp_score_gap")
effective = number(last, "potp_effective_prototypes")
minimum_share = number(last, "potp_assignment_share_min")

checks = {
    "f1_gain": f1_gain >= minimum_f1_gain,
    "precision_drop": precision_drop <= maximum_precision_drop,
    "positive_score_gap": score_gap > 0.0,
    "effective_prototypes": effective >= 2.5,
    "minimum_assignment_share": minimum_share >= 0.05,
    "gradient_ratio": math.isfinite(gradient_ratio) and 0.02 <= gradient_ratio <= 0.15,
    "gradient_cosine": math.isfinite(gradient_cosine) and gradient_cosine >= -0.10,
}
decision = {
    "gate_pass": all(checks.values()),
    "checks": checks,
    "thresholds": {
        "minimum_f1_gain": minimum_f1_gain,
        "maximum_precision_drop": maximum_precision_drop,
        "gradient_ratio": [0.02, 0.15],
        "minimum_effective_prototypes": 2.5,
        "minimum_assignment_share": 0.05,
    },
    "control": control,
    "prototype_raw_p2p": prototype,
    "delta": {"f1": f1_gain, "precision": -precision_drop},
    "mechanism": {
        "score_gap": score_gap,
        "effective_prototypes": effective,
        "minimum_assignment_share": minimum_share,
        "gradient_ratio": gradient_ratio,
        "gradient_cosine": gradient_cosine,
    },
    "next_action": "run_100_epochs" if all(checks.values()) else "stop_and_inspect_failed_checks",
}
path = os.path.join(root, "paired_decision.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(decision, handle, ensure_ascii=False, indent=2)
print(json.dumps(decision, ensure_ascii=False, indent=2))
print(f"[POTP] decision={path}")
PY

echo "[SUCCESS] ${PAIR_ROOT}"
