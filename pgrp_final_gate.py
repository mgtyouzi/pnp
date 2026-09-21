import argparse
import csv
import json
import math
from pathlib import Path


def _number(mapping, key, default=float("nan")):
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _first_active_row(rows):
    for row in rows:
        if _number(row, "epoch", -1) >= 2 and math.isfinite(
            _number(row, "pgrp_proto_violation_rate")
        ):
            return row
    return rows[-1]


def evaluate_final_gate(epoch_rows, threshold_decision):
    rows = list(epoch_rows)
    failures = []
    if not rows:
        return {
            "gate_pass": False,
            "failures": ["prototype_epoch_metrics"],
            "next_action": "do_not_start_100_epoch_training",
        }
    first_active = _first_active_row(rows)
    last = rows[-1]

    f1_gain = _number(threshold_decision, "best_f1_gain")
    control_precision = _number(
        threshold_decision.get("control_best", {}), "precision"
    )
    prototype_precision = _number(
        threshold_decision.get("prototype_best", {}), "precision"
    )
    precision_delta = prototype_precision - control_precision
    if not bool(threshold_decision.get("integrity_pass", False)):
        failures.append("metric_integrity")
    if not math.isfinite(f1_gain) or f1_gain < 0.002:
        failures.append("best_f1_gain")
    if not math.isfinite(precision_delta) or precision_delta < -0.005:
        failures.append("precision_drop")

    proto_gap = _number(last, "pgrp_proto_rank_gap")
    cls_gap = _number(last, "pgrp_cls_rank_gap")
    if not math.isfinite(proto_gap) or proto_gap <= 0:
        failures.append("prototype_rank_gap")
    if not math.isfinite(cls_gap) or cls_gap <= 0:
        failures.append("classification_rank_gap")

    proto_violation_start = _number(
        first_active, "pgrp_proto_violation_rate"
    )
    proto_violation_end = _number(last, "pgrp_proto_violation_rate")
    cls_violation_start = _number(first_active, "pgrp_cls_violation_rate")
    cls_violation_end = _number(last, "pgrp_cls_violation_rate")
    if (
        not all(math.isfinite(x) for x in (
            proto_violation_start, proto_violation_end
        ))
        or proto_violation_end > proto_violation_start + 0.02
    ):
        failures.append("prototype_violation_trend")
    if (
        not all(math.isfinite(x) for x in (
            cls_violation_start, cls_violation_end
        ))
        or cls_violation_end > cls_violation_start + 0.02
    ):
        failures.append("classification_violation_trend")

    effective = _number(last, "pgrp_effective_prototypes")
    minimum_share = _number(last, "pgrp_assignment_share_min")
    gradient_ratio = _number(last, "gradient_all_proto_cls_ratio")
    gradient_cosine = _number(last, "gradient_all_cosine")
    if not math.isfinite(effective) or effective < 2.5:
        failures.append("effective_prototypes")
    if not math.isfinite(minimum_share) or minimum_share < 0.05:
        failures.append("prototype_assignment_share")
    if not math.isfinite(gradient_ratio) or not 0.01 <= gradient_ratio <= 0.10:
        failures.append("gradient_ratio")
    if not math.isfinite(gradient_cosine) or gradient_cosine < -0.10:
        failures.append("gradient_cosine")

    return {
        "gate_pass": not failures,
        "failures": failures,
        "metrics": {
            "best_f1_gain": f1_gain,
            "precision_delta": precision_delta,
            "prototype_rank_gap": proto_gap,
            "classification_rank_gap": cls_gap,
            "prototype_violation_start": proto_violation_start,
            "prototype_violation_end": proto_violation_end,
            "classification_violation_start": cls_violation_start,
            "classification_violation_end": cls_violation_end,
            "effective_prototypes": effective,
            "minimum_assignment_share": minimum_share,
            "gradient_ratio": gradient_ratio,
            "gradient_cosine": gradient_cosine,
        },
        "thresholds": {
            "minimum_f1_gain": 0.002,
            "maximum_precision_drop": 0.005,
            "minimum_effective_prototypes": 2.5,
            "minimum_assignment_share": 0.05,
            "gradient_ratio": [0.01, 0.10],
            "minimum_gradient_cosine": -0.10,
        },
        "next_action": (
            "eligible_for_100_epoch_confirmation"
            if not failures
            else "do_not_start_100_epoch_training"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="PGRP source-test final gate")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--threshold_decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.metrics, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with open(args.threshold_decision, encoding="utf-8") as handle:
        threshold_decision = json.load(handle)
    decision = evaluate_final_gate(rows, threshold_decision)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
