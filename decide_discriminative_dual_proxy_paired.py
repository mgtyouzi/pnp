import argparse
import json
import math
from pathlib import Path


def _number(payload, key):
    value = payload.get(key)
    return float(value) if value is not None else float("nan")


def evaluate_paired(payload):
    control = payload["control"]
    prototype = payload["prototype"]
    source_control = payload["source_control"]
    source_prototype = payload["source_prototype"]
    mechanism = payload["mechanism"]
    failures = []

    if _number(prototype, "f1") < _number(control, "f1"):
        failures.append("frozen_f1")
    if _number(prototype, "fp_background_far") > _number(
        control, "fp_background_far"
    ):
        failures.append("frozen_background_far_fp")
    if _number(source_prototype, "f1") - _number(source_control, "f1") < -0.005:
        failures.append("source_f1")

    minimum_checks = {
        "foreground_similarity_gap": 0.05,
        "background_similarity_gap": 0.05,
        "foreground_effective_prototypes": 3.0,
        "background_effective_prototypes": 3.0,
        "foreground_assignment_share_min": 0.05,
        "background_assignment_share_min": 0.05,
    }
    for key, threshold in minimum_checks.items():
        if _number(mechanism, key) < threshold:
            failures.append(key)
    for key in (
        "foreground_pairwise_similarity_max",
        "background_pairwise_similarity_max",
    ):
        if _number(mechanism, key) >= 0.90:
            failures.append(key)
    gradient_cosine = _number(mechanism, "gradient_all_cosine")
    if not math.isfinite(gradient_cosine):
        failures.append("gradient_all_cosine")

    decision = dict(payload)
    decision["delta"] = {
        "f1": _number(prototype, "f1") - _number(control, "f1"),
        "background_far_fp": _number(prototype, "fp_background_far")
        - _number(control, "fp_background_far"),
        "source_f1": _number(source_prototype, "f1")
        - _number(source_control, "f1"),
    }
    decision["gate_pass"] = not failures
    decision["failures"] = failures
    decision["next_action"] = (
        "consider_fifty_epoch_confirmation"
        if not failures
        else "stop_and_inspect_dual_proxy_mechanism"
    )
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    decision = evaluate_paired(payload)
    Path(args.output).write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    raise SystemExit(0 if decision["gate_pass"] else 2)


if __name__ == "__main__":
    main()
