from prr_final_gate import evaluate_final_gate
from prr_mechanism_gate import evaluate_mechanism_rows


def mechanism_rows():
    rows = []
    for epoch in range(3):
        rows.append(
            {
                "epoch": str(epoch),
                "prototype_ready": "1" if epoch else "0",
                "prr_initialization_event": "1" if epoch == 0 else "0",
                "prr_refresh_event": "1" if epoch == 2 else "0",
                "prr_loss_raw": "0.01",
                "prr_reliable_matched": "170",
                "prr_rejected_matched": "5",
                "prr_rescue_selected": "12" if epoch == 2 else "0",
                "prr_metric_positive": "32",
                "prr_hard_background": "32",
                "prr_gt_support": "180",
                "prr_selected_probability": "0.31" if epoch == 2 else "nan",
                "prr_support_score_low": "0.40" if epoch else "nan",
                "prr_support_score_high": "0.75" if epoch else "nan",
                "prr_effective_prototypes": "3.4" if epoch else "0",
                "prr_assignment_share_min": "0.10" if epoch else "0",
                "prr_foreground_pairwise_similarity_max": "0.82",
                "prr_detector_background_updates": "0",
                "prr_detector_gradient_scale": "0.005",
                "gradient_all_proto_cls_ratio": "0.04" if epoch == 2 else "nan",
                "gradient_all_cosine": "0.5" if epoch == 2 else "nan",
            }
        )
    return rows


def test_mechanism_gate_accepts_the_planned_contract():
    decision = evaluate_mechanism_rows(mechanism_rows())
    assert decision["gate_pass"]


def test_mechanism_gate_rejects_any_background_detector_update():
    rows = mechanism_rows()
    rows[-1]["prr_detector_background_updates"] = "1"
    decision = evaluate_mechanism_rows(rows)
    assert not decision["gate_pass"]
    assert "detector_background_gradient" in decision["failures"]


def test_mechanism_gate_rejects_rescue_above_the_low_score_budget():
    rows = mechanism_rows()
    rows[-1]["prr_selected_probability"] = "0.60"
    decision = evaluate_mechanism_rows(rows)
    assert not decision["gate_pass"]
    assert "low_score_rescue_contract" in decision["failures"]


def test_mechanism_gate_rejects_the_old_excessive_gradient_scale():
    rows = mechanism_rows()
    rows[-1]["prr_detector_gradient_scale"] = "0.025"
    decision = evaluate_mechanism_rows(rows)
    assert not decision["gate_pass"]
    assert "detector_gradient_scale" in decision["failures"]


def test_final_gate_requires_detection_gain_and_precision_protection():
    threshold = {
        "integrity_pass": True,
        "best_f1_gain": 0.0025,
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
    assert decision["gate_pass"]

    threshold["prototype_best"]["precision"] = 0.735
    decision = evaluate_final_gate(mechanism_rows(), threshold)
    assert not decision["gate_pass"]
    assert "precision_drop" in decision["failures"]


def test_final_gate_keeps_the_initialization_history_and_accepts_total_fn():
    rows = mechanism_rows()
    for epoch in range(3, 10):
        row = dict(rows[-1])
        row["epoch"] = str(epoch)
        row["prr_initialization_event"] = "0"
        rows.append(row)
    threshold = {
        "integrity_pass": True,
        "best_f1_gain": 0.003,
        "control_best": {
            "precision": 0.74,
            "fn": 24000,
            "fp_background_far": 33000,
        },
        "prototype_best": {
            "precision": 0.739,
            "fn": 23500,
            "fp_background_far": 33100,
        },
    }
    decision = evaluate_final_gate(rows, threshold)
    assert decision["gate_pass"]


def main():
    test_mechanism_gate_accepts_the_planned_contract()
    test_mechanism_gate_rejects_any_background_detector_update()
    test_mechanism_gate_rejects_rescue_above_the_low_score_budget()
    test_mechanism_gate_rejects_the_old_excessive_gradient_scale()
    test_final_gate_requires_detection_gain_and_precision_protection()
    test_final_gate_keeps_the_initialization_history_and_accepts_total_fn()
    print("Prototype reliability rescue gate tests passed")


if __name__ == "__main__":
    main()
