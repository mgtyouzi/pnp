from plsp_final_gate import evaluate_final_gate
from plsp_mechanism_gate import evaluate_mechanism_rows


def mechanism_rows():
    rows = []
    for epoch in range(5):
        rows.append(
            {
                "epoch": str(epoch),
                "prototype_ready": "1" if epoch else "0",
                "prr_initialization_event": "1" if epoch == 0 else "0",
                "prr_refresh_event": "1" if epoch else "0",
                "prr_effective_prototypes": "3.4" if epoch else "0",
                "prr_assignment_share_min": "0.10" if epoch else "0",
                "prr_foreground_pairwise_similarity_max": "0.82",
                "plsp_active": "1" if epoch >= 2 else "0",
                "plsp_selected": "8" if epoch >= 2 else "0",
                "plsp_low_confidence_targets": "12" if epoch >= 2 else "0",
                "plsp_unassigned": "1000" if epoch >= 2 else "0",
                "plsp_local_owner": "40" if epoch >= 2 else "0",
                "plsp_distance_pass": "20" if epoch >= 2 else "0",
                "plsp_similarity_pass": "8" if epoch >= 2 else "0",
                "plsp_selected_probability": "0.08" if epoch >= 2 else "nan",
                "plsp_selected_matched_probability": "0.40" if epoch >= 2 else "nan",
                "plsp_maximum_matched_probability": "0.56",
                "plsp_selected_distance": "5.0" if epoch >= 2 else "nan",
                "plsp_distance_improvement": "4.0" if epoch >= 2 else "nan",
                "plsp_similarity_improvement": "0.2" if epoch >= 2 else "nan",
                "plsp_positive_weight": "0.1",
                "plsp_matcher_changes": "0",
                "plsp_regression_updates": "0",
            }
        )
    return rows


def test_mechanism_gate_accepts_local_weak_positive_contract():
    decision = evaluate_mechanism_rows(mechanism_rows())
    assert decision["gate_pass"]


def test_mechanism_gate_rejects_no_selection_or_matcher_changes():
    rows = mechanism_rows()
    rows[-1]["plsp_selected"] = "0"
    decision = evaluate_mechanism_rows(rows)
    assert "local_soft_positive_selection" in decision["failures"]

    rows = mechanism_rows()
    rows[-1]["plsp_matcher_changes"] = "1"
    decision = evaluate_mechanism_rows(rows)
    assert "matcher_immutability" in decision["failures"]


def test_final_gate_requires_detection_gain_and_false_negative_reduction():
    threshold = {
        "integrity_pass": True,
        "best_f1_gain": 0.0025,
        "target_transitions": {
            "rescued_fn": 600,
            "newly_lost_tp": 100,
            "net_rescue": 500,
        },
        "control_best": {
            "precision": 0.74,
            "fn_low_score_or_background": 24000,
            "fp_background_far": 33000,
        },
        "prototype_best": {
            "precision": 0.739,
            "fn_low_score_or_background": 23500,
            "fp_background_far": 33100,
        },
    }
    assert evaluate_final_gate(mechanism_rows(), threshold)["gate_pass"]

    threshold["prototype_best"]["fn_low_score_or_background"] = 24500
    decision = evaluate_final_gate(mechanism_rows(), threshold)
    assert not decision["gate_pass"]
    assert "low_score_false_negative_rescue" in decision["failures"]


def test_final_gate_requires_more_rescued_gt_than_newly_lost_gt():
    threshold = {
        "integrity_pass": True,
        "best_f1_gain": 0.0025,
        "target_transitions": {
            "rescued_fn": 100,
            "newly_lost_tp": 120,
            "net_rescue": -20,
        },
        "control_best": {
            "precision": 0.74,
            "fn_low_score_or_background": 24000,
            "fp_background_far": 33000,
        },
        "prototype_best": {
            "precision": 0.739,
            "fn_low_score_or_background": 23500,
            "fp_background_far": 33100,
        },
    }
    decision = evaluate_final_gate(mechanism_rows(), threshold)
    assert not decision["gate_pass"]
    assert "net_target_rescue" in decision["failures"]


def main():
    test_mechanism_gate_accepts_local_weak_positive_contract()
    test_mechanism_gate_rejects_no_selection_or_matcher_changes()
    test_final_gate_requires_detection_gain_and_false_negative_reduction()
    test_final_gate_requires_more_rescued_gt_than_newly_lost_gt()
    print("Prototype local soft-positive gate tests passed")


if __name__ == "__main__":
    main()
