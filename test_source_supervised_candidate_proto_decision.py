from unittest.mock import patch

from decide_source_supervised_candidate_proto_gate import decide


def summary(f1, precision=0.7, recall=0.75):
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pred_gt_ratio": 1.1,
        "fp_background_far": 100,
        "fn_low_score_or_background": 80,
    }


def main():
    values = {"control": summary(0.70), "raw": summary(0.701), "fused": summary(0.704)}
    with patch(
        "decide_source_supervised_candidate_proto_gate._load_summary",
        side_effect=lambda path: values[path],
    ):
        result = decide("control", "raw", "fused", minimum_gain=0.002)
        assert result["gate_pass"]
        assert result["best_mode"] == "prototype_fused"
        values["weak"] = summary(0.699)
        result = decide("control", "weak", "weak", minimum_gain=0.002)
        assert not result["gate_pass"]
        assert result["next_action"] == "stop_before_100_epoch_training"
    print("Source-supervised candidate prototype decision tests passed")


if __name__ == "__main__":
    main()
