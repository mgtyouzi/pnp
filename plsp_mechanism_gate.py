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
    if len(rows) < 5:
        return {
            "gate_pass": False,
            "failures": ["five_epoch_smoke_complete"],
            "next_action": "stop_before_full_training",
        }

    initialization = max(
        _number(row, "prr_initialization_event", 0.0) for row in rows
    )
    active_rows = [row for row in rows if _number(row, "plsp_active", 0.0) >= 0.5]
    last = rows[-1]
    if initialization < 0.5:
        failures.append("prototype_initialization")
    if _number(last, "prototype_ready", 0.0) < 0.5:
        failures.append("prototype_ready")
    if _number(last, "prr_refresh_event", 0.0) < 0.5:
        failures.append("prototype_refresh")
    if len(active_rows) < 2:
        failures.append("two_active_soft_positive_epochs")

    selected = _number(last, "plsp_selected", 0.0)
    if not math.isfinite(selected) or not 0 < selected <= 8:
        failures.append("local_soft_positive_selection")
    low_confidence_targets = _number(last, "plsp_low_confidence_targets", 0.0)
    if not math.isfinite(low_confidence_targets) or low_confidence_targets <= 0:
        failures.append("low_confidence_target_selection")
    funnel = [
        _number(last, "plsp_unassigned", 0.0),
        _number(last, "plsp_local_owner", 0.0),
        _number(last, "plsp_distance_pass", 0.0),
        _number(last, "plsp_similarity_pass", 0.0),
        selected,
    ]
    if any(not math.isfinite(value) for value in funnel) or any(
        left < right for left, right in zip(funnel, funnel[1:])
    ):
        failures.append("selection_funnel_monotonicity")

    first_active = active_rows[0] if active_rows else last
    initial_probability = _number(first_active, "plsp_selected_probability")
    if not math.isfinite(initial_probability) or initial_probability >= 0.5:
        failures.append("initially_suppressed_candidate_contract")
    matched_probability = _number(last, "plsp_selected_matched_probability")
    maximum_matched_probability = _number(
        last, "plsp_maximum_matched_probability"
    )
    if (
        not math.isfinite(matched_probability)
        or not math.isfinite(maximum_matched_probability)
        or abs(maximum_matched_probability - 0.56) > 1e-8
        or matched_probability >= maximum_matched_probability
    ):
        failures.append("low_confidence_matched_target_contract")
    selected_distance = _number(last, "plsp_selected_distance")
    distance_improvement = _number(last, "plsp_distance_improvement")
    similarity_improvement = _number(last, "plsp_similarity_improvement")
    if not math.isfinite(selected_distance) or selected_distance > 15.0 + 1e-6:
        failures.append("local_radius_contract")
    if (
        not math.isfinite(distance_improvement)
        or distance_improvement < 3.0 - 1e-6
    ):
        failures.append("distance_improvement_contract")
    if (
        not math.isfinite(similarity_improvement)
        or similarity_improvement < 0.1 - 1e-6
    ):
        failures.append("similarity_improvement_contract")
    if abs(_number(last, "plsp_positive_weight") - 0.1) > 1e-8:
        failures.append("weak_positive_weight")
    if _number(last, "plsp_matcher_changes", -1.0) != 0.0:
        failures.append("matcher_immutability")
    if _number(last, "plsp_regression_updates", -1.0) != 0.0:
        failures.append("regression_immutability")

    effective = _number(last, "prr_effective_prototypes")
    minimum_share = _number(last, "prr_assignment_share_min")
    pairwise_max = _number(last, "prr_foreground_pairwise_similarity_max")
    if not math.isfinite(effective) or effective < 2.5:
        failures.append("effective_prototypes")
    if not math.isfinite(minimum_share) or minimum_share < 0.04:
        failures.append("prototype_assignment_share")
    if not math.isfinite(pairwise_max) or pairwise_max >= 0.93:
        failures.append("prototype_diversity")

    metrics = {
        key: _number(last, key)
        for key in (
            "plsp_selected",
            "plsp_low_confidence_targets",
            "plsp_unassigned",
            "plsp_local_owner",
            "plsp_distance_pass",
            "plsp_similarity_pass",
            "plsp_selected_probability",
            "plsp_selected_matched_probability",
            "plsp_maximum_matched_probability",
            "plsp_selected_distance",
            "plsp_distance_improvement",
            "plsp_similarity_improvement",
            "plsp_positive_weight",
            "plsp_matcher_changes",
            "plsp_regression_updates",
            "prr_effective_prototypes",
            "prr_assignment_share_min",
            "prr_foreground_pairwise_similarity_max",
        )
    }
    metrics["initial_selected_probability"] = initial_probability
    return {
        "gate_pass": not failures,
        "failures": failures,
        "epochs_seen": len(rows),
        "active_epochs_seen": len(active_rows),
        "metrics": metrics,
        "thresholds": {
            "selection_count_per_image": [0.0, 8.0],
            "maximum_initial_probability": 0.5,
            "maximum_matched_probability": 0.56,
            "maximum_local_radius": 15.0,
            "minimum_distance_improvement": 3.0,
            "minimum_similarity_improvement": 0.1,
            "weak_positive_weight": 0.1,
            "minimum_effective_prototypes": 2.5,
            "minimum_assignment_share": 0.04,
            "maximum_pairwise_similarity": 0.93,
        },
        "next_action": (
            "start_ten_epoch_paired_source_training"
            if not failures
            else "stop_before_full_training"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="PLSP five-epoch mechanism gate")
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
