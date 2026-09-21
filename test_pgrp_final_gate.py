from pgrp_final_gate import evaluate_final_gate


def healthy_epoch_rows():
    rows = []
    for epoch in range(10):
        rows.append({
            "epoch": str(epoch),
            "pgrp_proto_rank_gap": "0.10",
            "pgrp_cls_rank_gap": "0.22",
            "pgrp_proto_violation_rate": str(0.70 - 0.03 * epoch),
            "pgrp_cls_violation_rate": str(0.65 - 0.02 * epoch),
            "pgrp_effective_prototypes": "3.2",
            "pgrp_assignment_share_min": "0.10",
            "gradient_all_proto_cls_ratio": "0.05",
            "gradient_all_cosine": "0.01",
        })
    return rows


def healthy_threshold_decision():
    return {
        "best_f1_gain": 0.003,
        "control_best": {"precision": 0.75},
        "prototype_best": {"precision": 0.748},
        "integrity_pass": True,
    }


def test_final_gate_requires_both_detection_and_mechanism_gain():
    decision = evaluate_final_gate(
        healthy_epoch_rows(), healthy_threshold_decision()
    )
    assert decision["gate_pass"] is True
    assert decision["next_action"] == "eligible_for_100_epoch_confirmation"


def test_final_gate_rejects_tiny_f1_gain_and_excess_gradient():
    rows = healthy_epoch_rows()
    rows[-1]["gradient_all_proto_cls_ratio"] = "0.2"
    threshold = healthy_threshold_decision()
    threshold["best_f1_gain"] = 0.0001
    decision = evaluate_final_gate(rows, threshold)
    assert decision["gate_pass"] is False
    assert "best_f1_gain" in decision["failures"]
    assert "gradient_ratio" in decision["failures"]
    assert decision["next_action"] == "do_not_start_100_epoch_training"


def main():
    test_final_gate_requires_both_detection_and_mechanism_gain()
    test_final_gate_rejects_tiny_f1_gain_and_excess_gradient()
    print("PGRP final gate tests passed")


if __name__ == "__main__":
    main()
