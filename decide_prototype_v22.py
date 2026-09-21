import argparse
import json
import os
from typing import Dict


def read_json(path: str) -> Dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_decision_from_payloads(
    mechanism: Dict, paired: Dict, offline: Dict, mode: str
) -> Dict:
    if mode not in {"rapid", "faithful"}:
        raise ValueError("mode must be rapid or faithful")
    comparison = offline.get("configured_feature_space_comparison", {})
    projected = comparison.get("projected", {})
    deltas = comparison.get("projected_minus_raw", {})
    training_batch_fraction = float(
        mechanism.get("mean_training_batch_fraction", 0.0)
    )

    gates = {
        "prototype_mechanism_ready": bool(mechanism.get("long_run_ready")),
        "paired_attribution_valid": bool(paired.get("attribution_valid")),
        "paired_optimization_ready": bool(
            paired.get("paired_optimization_ready")
        ),
        "configured_raw_projected_comparison_valid": bool(
            comparison.get("valid")
        ),
        "projected_balanced_accuracy_at_least_0p65": (
            float(projected.get("balanced_accuracy", -1.0)) >= 0.65
        ),
        "projected_foreground_recall_at_least_0p60": (
            float(projected.get("foreground_recall", -1.0)) >= 0.60
        ),
        "projected_hard_background_recall_at_least_0p60": (
            float(projected.get("hard_background_recall", -1.0)) >= 0.60
        ),
        "projected_balanced_accuracy_not_worse_than_raw": (
            float(deltas.get("balanced_accuracy", -1.0)) >= -0.005
        ),
        "faithful_updates_use_full_epoch": (
            mode == "rapid" or training_batch_fraction >= 0.999
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    all_pass = not failed
    ready_for_long_run = bool(all_pass and mode == "faithful")
    if not all_pass:
        next_action = "stop_and_fix_failed_mechanism"
    elif mode == "rapid":
        next_action = "run_faithful_paired_validation"
    else:
        next_action = "run_long_training"

    return {
        "mode": mode,
        "all_gates_pass": all_pass,
        "ready_for_long_run": ready_for_long_run,
        "next_action": next_action,
        "failed_gates": failed,
        "gates": gates,
        "mechanism_gates": mechanism.get("mechanism_gates", {}),
        "optimization_gates": paired.get("optimization_gates", {}),
        "configured_feature_space_comparison": comparison,
    }


def build_decision(
    mechanism_path: str, paired_path: str, offline_path: str, mode: str
) -> Dict:
    return build_decision_from_payloads(
        read_json(mechanism_path),
        read_json(paired_path),
        read_json(offline_path),
        mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine Prototype-v2.2 mechanism, attribution, and source-only gates."
    )
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--offline", required=True)
    parser.add_argument("--mode", choices=("rapid", "faithful"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="Prototype-v2.2")
    args = parser.parse_args()

    decision = build_decision(
        args.mechanism, args.paired, args.offline, args.mode
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)

    markdown_path = os.path.splitext(args.output)[0] + ".md"
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {args.label} decision\n\n")
        handle.write(f"- Mode: `{decision['mode']}`\n")
        handle.write(f"- All gates pass: `{decision['all_gates_pass']}`\n")
        handle.write(
            f"- Ready for long run: `{decision['ready_for_long_run']}`\n"
        )
        handle.write(f"- Next action: `{decision['next_action']}`\n\n")
        handle.write("| gate | pass |\n|---|---:|\n")
        for name, passed in decision["gates"].items():
            handle.write(f"| {name} | {passed} |\n")

    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print(f"[SUCCESS] decision={args.output}", flush=True)


if __name__ == "__main__":
    main()
