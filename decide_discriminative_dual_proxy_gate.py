import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict


THRESHOLDS = {
    "prototype_macro_auc": (">=", 0.75),
    "prototype_min_slide_auc": (">=", 0.55),
    "foreground_similarity_gap": (">=", 0.05),
    "background_similarity_gap": (">=", 0.05),
    "foreground_effective_prototypes": (">=", 3.0),
    "background_effective_prototypes": (">=", 3.0),
    "foreground_pairwise_similarity_max": ("<", 0.90),
    "background_pairwise_similarity_max": ("<", 0.90),
    "cross_bank_similarity_max": ("<", 0.90),
    "foreground_assignment_share_min": (">=", 0.05),
    "background_assignment_share_min": (">=", 0.05),
}


def evaluate_gate(metrics: Dict) -> Dict:
    failures = []
    for key, (operator, threshold) in THRESHOLDS.items():
        value = float(metrics.get(key, float("nan")))
        passed = value >= threshold if operator == ">=" else value < threshold
        if not passed:
            failures.append(key)
    if metrics.get("optimizer_projector_only") is not True:
        failures.append("optimizer_projector_only")
    return {
        "gate_pass": not failures,
        "failures": failures,
        "metrics": metrics,
        "thresholds": {
            key: {"operator": operator, "value": value}
            for key, (operator, value) in THRESHOLDS.items()
        },
        "next_action": (
            "run_ten_epoch_paired_detector_gate"
            if not failures
            else "stop_before_detector_training"
        ),
    }


def export_passed_bank(
    checkpoint_path: Path, bank_path: Path, decision: Dict
) -> bool:
    if not decision.get("gate_pass", False):
        return False
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model_state = checkpoint.get("model", checkpoint)
    prefix = "prototype_head."
    state = {
        key[len(prefix):]: value
        for key, value in model_state.items()
        if key.startswith(prefix)
    }
    required = {"foreground_prototypes", "background_prototypes", "prototype_ready"}
    if not required.issubset(state):
        raise RuntimeError(
            "checkpoint does not contain a complete discriminative dual-proxy bank"
        )
    if "sampling_step" in state:
        state["sampling_step"] = state["sampling_step"].new_zeros(())
    checkpoint_args = checkpoint.get("args", {})
    detector_init_checkpoint = str(checkpoint_args.get("init_checkpoint", ""))
    if not detector_init_checkpoint or not Path(detector_init_checkpoint).is_file():
        raise RuntimeError(
            "gate checkpoint does not identify an accessible detector init_checkpoint"
        )
    digest = hashlib.sha256()
    with open(detector_init_checkpoint, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    payload = {
        "implementation_version": "discriminative_dual_proxy_v2_20260831",
        "checkpoint": str(checkpoint_path),
        "prototype_state": state,
        "gate": decision,
        "detector_init_checkpoint": detector_init_checkpoint,
        "detector_init_sha256": digest.hexdigest(),
    }
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(bank_path))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bank_output", required=True)
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    decision = evaluate_gate(metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    exported = export_passed_bank(
        Path(args.checkpoint), Path(args.bank_output), decision
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("bank_exported=", exported)
    raise SystemExit(0 if decision["gate_pass"] else 2)


if __name__ == "__main__":
    main()
