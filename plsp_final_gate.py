import argparse
import csv
import json
import math
from pathlib import Path

from plsp_mechanism_gate import evaluate_mechanism_rows


def _number(mapping, key, default=float("nan")):
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def evaluate_final_gate(epoch_rows, threshold_decision):
    rows = list(epoch_rows)
    mechanism = evaluate_mechanism_rows(rows)
    failures = ["mechanism:" + name for name in mechanism.get("failures", [])]
    control = threshold_decision.get("control_best", {})
    prototype = threshold_decision.get("prototype_best", {})
    f1_gain = _number(threshold_decision, "best_f1_gain")
    precision_delta = _number(prototype, "precision") - _number(
        control, "precision"
    )
    background_fp_delta = _number(prototype, "fp_background_far") - _number(
        control, "fp_background_far"
    )
    classification_fn_delta = _number(
        prototype, "fn_low_score_or_background", _number(prototype, "fn")
    ) - _number(control, "fn_low_score_or_background", _number(control, "fn"))
    background_fp_allowance = max(
        500.0, 0.005 * _number(control, "fp_background_far", 0.0)
    )
    transitions = threshold_decision.get("target_transitions", {})
    rescued_fn = _number(transitions, "rescued_fn")
    newly_lost_tp = _number(transitions, "newly_lost_tp")
    net_rescue = _number(transitions, "net_rescue")

    if not bool(threshold_decision.get("integrity_pass", False)):
        failures.append("metric_integrity")
    if not math.isfinite(f1_gain) or f1_gain < 0.002:
        failures.append("best_f1_gain")
    if not math.isfinite(precision_delta) or precision_delta < -0.003:
        failures.append("precision_drop")
    if (
        not math.isfinite(background_fp_delta)
        or background_fp_delta > background_fp_allowance
    ):
        failures.append("background_far_fp")
    if not math.isfinite(classification_fn_delta) or classification_fn_delta >= 0:
        failures.append("low_score_false_negative_rescue")
    if (
        not math.isfinite(rescued_fn)
        or not math.isfinite(newly_lost_tp)
        or not math.isfinite(net_rescue)
        or rescued_fn <= newly_lost_tp
        or net_rescue <= 0
    ):
        failures.append("net_target_rescue")

    return {
        "gate_pass": not failures,
        "failures": failures,
        "mechanism_gate_pass": mechanism.get("gate_pass", False),
        "metrics": {
            "best_f1_gain": f1_gain,
            "precision_delta": precision_delta,
            "background_far_fp_delta": background_fp_delta,
            "low_score_or_background_fn_delta": classification_fn_delta,
            "rescued_fn": rescued_fn,
            "newly_lost_tp": newly_lost_tp,
            "net_target_rescue": net_rescue,
        },
        "thresholds": {
            "minimum_f1_gain": 0.002,
            "maximum_precision_drop": 0.003,
            "maximum_background_far_fp_increase": background_fp_allowance,
            "require_low_score_or_background_fn_reduction": True,
            "require_rescued_fn_greater_than_newly_lost_tp": True,
        },
        "next_action": (
            "eligible_for_100_epoch_confirmation"
            if not failures
            else "do_not_start_100_epoch_training"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="PLSP source-test final gate")
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
    if not decision["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
