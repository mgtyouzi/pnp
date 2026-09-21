import argparse
import csv
import json
import math
from pathlib import Path


def _number(row, key, default=float("nan")):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def evaluate_mechanism_rows(rows):
    rows = list(rows)
    failures = []
    if len(rows) < 3:
        return {
            "gate_pass": False,
            "failures": ["three_epoch_smoke_complete"],
            "next_action": "stop_before_full_training",
        }

    first, last = rows[0], rows[-1]
    initialization = max(
        _number(row, "pgrp_initialization_event", 0.0) for row in rows
    )
    if initialization < 0.5:
        failures.append("prototype_initialization")
    if _number(last, "pgrp_refresh_event", 0.0) < 0.5:
        failures.append("raw_gt_reprojection_refresh")
    if _number(last, "prototype_ready", 0.0) < 0.5:
        failures.append("prototype_ready")

    finite_fields = (
        "pgrp_loss_raw",
        "pgrp_proto_rank_gap",
        "pgrp_cls_rank_gap",
    )
    if not all(math.isfinite(_number(last, field)) for field in finite_fields):
        failures.append("finite_ranking_diagnostics")
    if _number(last, "pgrp_reliable_matched", 0.0) <= 0:
        failures.append("reliable_positive_sampling")
    if _number(last, "pgrp_rejected_matched", 0.0) <= 0:
        failures.append("inaccurate_match_rejection")
    if _number(last, "pgrp_gt_support", 0.0) <= 0:
        failures.append("exact_gt_support")
    hard_positive = _number(last, "pgrp_hard_positive", 0.0)
    hard_background = _number(last, "pgrp_hard_background", 0.0)
    if hard_positive <= 0 or abs(hard_positive - hard_background) > 0.5:
        failures.append("one_to_one_hard_pairs")
    if _number(last, "pgrp_effective_prototypes", 0.0) < 2.0:
        failures.append("effective_prototypes")
    if _number(last, "pgrp_assignment_share_min", 0.0) < 0.02:
        failures.append("prototype_assignment_share")
    if abs(_number(last, "pgrp_shared_gradient_scale") - 0.05) > 1e-6:
        failures.append("staged_shared_gradient")
    if abs(_number(first, "pgrp_shared_gradient_scale", 1.0)) > 1e-6:
        failures.append("warmup_gradient_block")

    metrics = {
        key: _number(last, key)
        for key in (
            "pgrp_loss_raw",
            "pgrp_reliable_matched",
            "pgrp_rejected_matched",
            "pgrp_hard_positive",
            "pgrp_hard_background",
            "pgrp_proto_rank_gap",
            "pgrp_cls_rank_gap",
            "pgrp_proto_violation_rate",
            "pgrp_cls_violation_rate",
            "pgrp_effective_prototypes",
            "pgrp_assignment_share_min",
            "pgrp_shared_gradient_scale",
        )
    }
    return {
        "gate_pass": not failures,
        "failures": failures,
        "epochs_seen": len(rows),
        "metrics": metrics,
        "thresholds": {
            "minimum_effective_prototypes": 2.0,
            "minimum_assignment_share": 0.02,
            "hard_pair_count_tolerance": 0.5,
            "expected_active_gradient_scale": 0.05,
        },
        "next_action": (
            "start_full_source_training"
            if not failures
            else "stop_before_full_training"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="PGRP three-epoch mechanism gate")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.metrics, newline="", encoding="utf-8") as handle:
        decision = evaluate_mechanism_rows(csv.DictReader(handle))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    if not decision["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
