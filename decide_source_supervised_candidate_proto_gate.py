import argparse
import json
from pathlib import Path


def _load_summary(path):
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _metrics(summary):
    keys = (
        "precision",
        "recall",
        "f1",
        "pred_gt_ratio",
        "fp_background_far",
        "fn_low_score_or_background",
    )
    return {key: float(summary.get(key, 0.0)) for key in keys}


def decide(control_path, raw_path, fused_path, minimum_gain=0.002):
    control = _metrics(_load_summary(control_path))
    raw = _metrics(_load_summary(raw_path))
    fused = _metrics(_load_summary(fused_path))
    candidates = {
        "prototype_raw": raw,
        "prototype_fused": fused,
    }
    best_mode = max(candidates, key=lambda name: candidates[name]["f1"])
    best = candidates[best_mode]
    gain = best["f1"] - control["f1"]
    stable_counting = 0.7 <= best["pred_gt_ratio"] <= 1.5
    gate_pass = bool(gain >= minimum_gain and stable_counting)
    return {
        "gate_pass": gate_pass,
        "best_mode": best_mode,
        "minimum_f1_gain": float(minimum_gain),
        "control": control,
        "prototype_raw": raw,
        "prototype_fused": fused,
        "delta": {
            "best_f1": gain,
            "raw_f1": raw["f1"] - control["f1"],
            "fused_f1": fused["f1"] - control["f1"],
            "fusion_only_f1": fused["f1"] - raw["f1"],
            "best_bg_far_fp": best["fp_background_far"]
            - control["fp_background_far"],
            "best_cls_fn": best["fn_low_score_or_background"]
            - control["fn_low_score_or_background"],
        },
        "next_action": (
            "run_100_epoch_source_supervised_candidate_proto"
            if gate_pass
            else "stop_before_100_epoch_training"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--prototype_raw", required=True)
    parser.add_argument("--prototype_fused", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum_gain", default=0.002, type=float)
    args = parser.parse_args()
    result = decide(
        args.control,
        args.prototype_raw,
        args.prototype_fused,
        minimum_gain=args.minimum_gain,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
