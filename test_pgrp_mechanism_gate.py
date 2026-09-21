from pgrp_mechanism_gate import evaluate_mechanism_rows


def healthy_rows():
    base = {
        "pgrp_loss_raw": "0.02",
        "pgrp_reliable_matched": "40",
        "pgrp_rejected_matched": "2",
        "pgrp_hard_positive": "10",
        "pgrp_hard_background": "10",
        "pgrp_gt_support": "45",
        "pgrp_proto_rank_gap": "0.12",
        "pgrp_cls_rank_gap": "0.25",
        "pgrp_effective_prototypes": "3.1",
        "pgrp_assignment_share_min": "0.08",
        "prototype_ready": "1",
        "pgrp_shared_gradient_scale": "0.05",
        "pgrp_initialization_event": "0",
        "pgrp_refresh_event": "1",
    }
    rows = []
    for epoch in range(3):
        row = dict(base, epoch=str(epoch))
        if epoch == 0:
            row["prototype_ready"] = "0"
            row["pgrp_initialization_event"] = "1"
            row["pgrp_refresh_event"] = "0"
            row["pgrp_shared_gradient_scale"] = "0"
        elif epoch == 1:
            row["pgrp_shared_gradient_scale"] = "0"
        rows.append(row)
    return rows


def test_gate_accepts_initialized_balanced_ranking_mechanism():
    decision = evaluate_mechanism_rows(healthy_rows())
    assert decision["gate_pass"] is True
    assert decision["failures"] == []


def test_gate_rejects_unpaired_background_and_collapsed_prototypes():
    rows = healthy_rows()
    rows[-1]["pgrp_hard_background"] = "6"
    rows[-1]["pgrp_effective_prototypes"] = "1.2"
    decision = evaluate_mechanism_rows(rows)
    assert decision["gate_pass"] is False
    assert "one_to_one_hard_pairs" in decision["failures"]
    assert "effective_prototypes" in decision["failures"]


def main():
    test_gate_accepts_initialized_balanced_ranking_mechanism()
    test_gate_rejects_unpaired_background_and_collapsed_prototypes()
    print("PGRP mechanism gate tests passed")


if __name__ == "__main__":
    main()
