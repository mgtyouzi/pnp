import argparse
import csv
import json
import math
from pathlib import Path


def _number(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
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
        _number(row, "prr_initialization_event", 0.0) for row in rows
    )
    if initialization < 0.5:
        failures.append("prototype_initialization")
    if _number(last, "prr_refresh_event", 0.0) < 0.5:
        failures.append("raw_gt_reprojection_refresh")
    if _number(last, "prototype_ready", 0.0) < 0.5:
        failures.append("prototype_ready")

    finite_fields = (
        "prr_loss_raw",
        "prr_support_score_low",
        "prr_support_score_high",
    )
    if not all(math.isfinite(_number(last, key)) for key in finite_fields):
        failures.append("finite_reliability_diagnostics")
    low = _number(last, "prr_support_score_low")
    high = _number(last, "prr_support_score_high")
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        failures.append("support_quantile_calibration")

    if _number(last, "prr_gt_support", 0.0) <= 0:
        failures.append("exact_gt_support")
    if _number(last, "prr_reliable_matched", 0.0) <= 0:
        failures.append("reliable_positive_sampling")
    if _number(last, "prr_rejected_matched", 0.0) <= 0:
        failures.append("inaccurate_match_rejection")
    if _number(last, "prr_rescue_selected", 0.0) <= 0:
        failures.append("reliable_rescue_selection")

    metric_positive = _number(last, "prr_metric_positive", 0.0)
    hard_background = _number(last, "prr_hard_background", 0.0)
    if metric_positive <= 0 or abs(metric_positive - hard_background) > 0.5:
        failures.append("one_to_one_metric_pairs")

    selected_probability = _number(last, "prr_selected_probability")
    if (
        not math.isfinite(selected_probability)
        or selected_probability >= 0.55
    ):
        failures.append("low_score_rescue_contract")
    if _number(last, "prr_detector_background_updates", -1.0) != 0.0:
        failures.append("detector_background_gradient")
    detector_gradient_scale = _number(last, "prr_detector_gradient_scale")
    if (
        not math.isfinite(detector_gradient_scale)
        or abs(detector_gradient_scale - 0.005) > 1e-8
    ):
        failures.append("detector_gradient_scale")

    effective = _number(last, "prr_effective_prototypes")
    minimum_share = _number(last, "prr_assignment_share_min")
    pairwise_max = _number(last, "prr_foreground_pairwise_similarity_max")
    if not math.isfinite(effective) or effective < 2.5:
        failures.append("effective_prototypes")
    if not math.isfinite(minimum_share) or minimum_share < 0.04:
        failures.append("prototype_assignment_share")
    if not math.isfinite(pairwise_max) or pairwise_max >= 0.93:
        failures.append("prototype_diversity")

    gradient_ratio = _number(last, "gradient_all_proto_cls_ratio")
    gradient_cosine = _number(last, "gradient_all_cosine")
    if not math.isfinite(gradient_ratio) or not 0.02 <= gradient_ratio <= 0.08:
        failures.append("gradient_ratio")
    if not math.isfinite(gradient_cosine) or gradient_cosine < -0.10:
        failures.append("gradient_cosine")

    metric_keys = (
        "prr_loss_raw",
        "prr_gt_support",
        "prr_reliable_matched",
        "prr_rejected_matched",
        "prr_rescue_selected",
        "prr_metric_positive",
        "prr_hard_background",
        "prr_selected_probability",
        "prr_support_score_low",
        "prr_support_score_high",
        "prr_effective_prototypes",
        "prr_assignment_share_min",
        "prr_foreground_pairwise_similarity_max",
        "prr_detector_background_updates",
        "prr_detector_gradient_scale",
        "gradient_all_proto_cls_ratio",
        "gradient_all_cosine",
    )
    return {
        "gate_pass": not failures,
        "failures": failures,
        "epochs_seen": len(rows),
        "metrics": {key: _number(last, key) for key in metric_keys},
        "thresholds": {
            "maximum_rescue_cell_probability": 0.55,
            "minimum_effective_prototypes": 2.5,
            "minimum_assignment_share": 0.04,
            "maximum_pairwise_similarity": 0.93,
            "expected_detector_gradient_scale": 0.005,
            "gradient_ratio": [0.02, 0.08],
            "minimum_gradient_cosine": -0.10,
            "detector_background_updates": 0.0,
        },
        "next_action": (
            "start_ten_epoch_paired_source_training"
            if not failures
            else "stop_before_full_training"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="PRR three-epoch mechanism gate")
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
