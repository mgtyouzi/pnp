#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
PARAFFIN_DATASET="${PARAFFIN_DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/石蜡-2025}"
FROZEN_DATASET="${FROZEN_DATASET:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets/冰冻-2025}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
GPU="${GPU:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-10}"
SKIP_CONTROL="${SKIP_CONTROL:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

cd "${PROJECT_ROOT}"
for required in \
  "${BASELINE_CHECKPOINT}" \
  "${PARAFFIN_DATASET}/mean_std.npy" \
  "${FROZEN_DATASET}/mean_std.npy"; do
  if [ ! -f "${required}" ]; then
    echo "[ERROR] required file not found: ${required}"
    exit 2
  fi
done

if ! grep -q 'GT_FOREGROUND_PROXY_VERSION = "gt_foreground_proxy_v1_20260829"' \
  models/gt_foreground_proxy.py; then
  echo "[ERROR] stale or incomplete GT foreground proxy implementation"
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT_ROOT}/experiments/gt_foreground_proxy_paired_${timestamp}_gpu${GPU}_bt${BATCH_SIZE}}"
CONTROL_DIR="${CONTROL_DIR:-${PAIR_ROOT}/control_p2p}"
PROXY_DIR="${PROXY_DIR:-${PAIR_ROOT}/gt_foreground_proxy}"
mkdir -p "${CONTROL_DIR}" "${PROXY_DIR}"
printf '%s\n' "${PAIR_ROOT}" > experiments/latest_gt_foreground_proxy_paired.txt

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[Pair] root=${PAIR_ROOT}"
echo "[Pair] physical_gpu=${GPU}, epochs=${EPOCHS}, batch_size=${BATCH_SIZE}"
echo "[Pair] initialization=${BASELINE_CHECKPOINT}"
echo "[Pair] invariant=identical checkpoint/RNG/data/P2P settings; no HBS; no inference fusion"

if [ "${EVAL_ONLY}" != "1" ]; then
  echo "[Stage 1/6] implementation preflight"
  python test_gt_foreground_proxy_structure.py
  python test_gt_foreground_proxy_wiring.py
  python test_gt_foreground_proxy_run_script.py
  python test_gt_foreground_proxy.py
  python test_p2p_prototype.py
else
  echo "[Stage 1-3/6] reusing completed training under ${PAIR_ROOT}"
fi

COMMON_ARGS=(
  train_p2p_no_empty_v2.py
  --dataset "${PARAFFIN_DATASET}"
  --mean_std_path "${PARAFFIN_DATASET}/mean_std.npy"
  --num_classes=1
  --epochs="${EPOCHS}"
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
  --proto_mode=gt_foreground_proxy
  --proto_start_epoch=0
  --proto_embedding_dim=64
  --proto_num_fg=4
  --proto_temperature=0.2
  --proto_loss_weight=0.02
  --proto_positive_radius=15
  --proto_background_radius=30
  --proto_max_hard_bg_per_image=16
  --proto_momentum=0.99
  --proto_dead_patience=2
  --proto_sampling_seed=0
  --proto_debug_interval=200
  --proto_debug_fail_fast=1
  --proto_inference_fusion=0
  --proto_gt_warmup_epochs=5
  --proto_gt_support_queue_size=8192
  --proto_gt_max_support_per_image=64
  --proto_gt_projector_momentum=0.999
  --proto_gt_background_margin=0.2
  --proto_gt_min_assignment_share=0.05
  --proto_gt_align_weight=1.0
  --proto_gt_support_weight=1.0
  --proto_gt_query_weight=1.0
  --proto_gt_background_weight=0.5
  --proto_gt_balance_weight=0.05
)

if [ "${EVAL_ONLY}" != "1" ]; then
  if [ "${SKIP_CONTROL}" = "1" ]; then
    echo "[Stage 2/6] reusing control=${CONTROL_DIR}/recent_model.pth"
  else
    echo "[Stage 2/6] paired control P2P"
    python "${COMMON_ARGS[@]}" --output_dir "${CONTROL_DIR}" \
      2>&1 | tee "${CONTROL_DIR}/console.log"
  fi
  test -f "${CONTROL_DIR}/recent_model.pth"

  echo "[Stage 3/6] GT-guided foreground multi-proxy P2P"
  python "${COMMON_ARGS[@]}" "${PROXY_ARGS[@]}" --output_dir "${PROXY_DIR}" \
    2>&1 | tee "${PROXY_DIR}/console.log"
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
  --case_name gtproxy_control_e10_frozen \
  --output_dir "${CONTROL_DIR}/eval_frozen_e10_gpu${GPU}"

echo "[Stage 5/6] frozen target evaluation: prototype"
python diagnose_cross_domain.py \
  "${EVAL_MODEL_ARGS[@]}" \
  "${PROXY_ARGS[@]}" \
  --dataset "${FROZEN_DATASET}" \
  --checkpoint "${PROXY_DIR}/recent_model.pth" \
  --case_name gtproxy_prototype_e10_frozen \
  --output_dir "${PROXY_DIR}/eval_frozen_e10_gpu${GPU}" \
  --strict_load

echo "[Stage 6/6] compact paired decision"
python - "${PAIR_ROOT}" "${GPU}" <<'PY'
import csv
import json
import os
import sys
import torch

root, gpu = sys.argv[1:]
control_path = os.path.join(root, "control_p2p", f"eval_frozen_e10_gpu{gpu}", "summary.json")
proxy_path = os.path.join(root, "gt_foreground_proxy", f"eval_frozen_e10_gpu{gpu}", "summary.json")
control = json.load(open(control_path, encoding="utf-8"))
proxy = json.load(open(proxy_path, encoding="utf-8"))
metrics_path = os.path.join(root, "gt_foreground_proxy", "prototype_epoch_metrics.csv")
rows = list(csv.DictReader(open(metrics_path, encoding="utf-8")))
last = rows[-1] if rows else {}

def number(payload, key):
    return float(payload.get(key, 0.0) or 0.0)

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

control_source = source_metrics(os.path.join(root, "control_p2p", "recent_model.pth"))
proxy_source = source_metrics(os.path.join(root, "gt_foreground_proxy", "recent_model.pth"))

decision = {
    "control": {key: control.get(key) for key in (
        "precision", "recall", "f1", "mae", "pred_gt_ratio",
        "fp_background_far", "fn_low_score_or_background",
    )},
    "prototype": {key: proxy.get(key) for key in (
        "precision", "recall", "f1", "mae", "pred_gt_ratio",
        "fp_background_far", "fn_low_score_or_background",
    )},
    "delta": {
        "f1": number(proxy, "f1") - number(control, "f1"),
        "bg_far_fp": number(proxy, "fp_background_far") - number(control, "fp_background_far"),
        "cls_fn": number(proxy, "fn_low_score_or_background") - number(control, "fn_low_score_or_background"),
    },
    "source_control": control_source,
    "source_prototype": proxy_source,
    "source_delta": {
        "f1": proxy_source["f1"] - control_source["f1"],
        "mae": proxy_source["mae"] - control_source["mae"],
    },
    "mechanism": {
        key: last.get(key) for key in (
            "gtproxy_reliable_query", "gtproxy_rejected_query",
            "gtproxy_hard_background", "gtproxy_ignored_near",
            "gtproxy_support_similarity", "gtproxy_query_similarity",
            "gtproxy_background_similarity", "gradient_all_proto_cls_ratio",
            "gradient_all_cosine", "foreground_assignment_share_min",
            "foreground_effective_prototypes", "foreground_pairwise_similarity_max",
        )
    },
}
decision["gate_pass"] = bool(
    decision["delta"]["f1"] >= 0.0
    and decision["source_delta"]["f1"] >= -0.005
    and number(last, "foreground_effective_prototypes") >= 3.0
    and number(last, "foreground_assignment_share_min") >= 0.05
)
decision["next_action"] = (
    "run_50_epoch_confirmation" if decision["gate_pass"]
    else "inspect_sampling_prototype_collapse_and_gradient_metrics"
)
path = os.path.join(root, "paired_decision.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(decision, handle, ensure_ascii=False, indent=2)
print(json.dumps(decision, ensure_ascii=False, indent=2))
print("decision=", path)
PY

echo "[SUCCESS] GT foreground proxy paired experiment completed: ${PAIR_ROOT}"
