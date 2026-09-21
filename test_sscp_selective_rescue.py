import numpy as np

from sscp_selective_rescue import (
    MINIMUM_MEANINGFUL_F1_GAIN,
    audit_dataset_split,
    calculate_promotion_budget,
    decide_fixed_rescue,
    fit_margin_calibration,
    fit_precision_constrained_calibration,
    prototype_bank_state_sha256,
    round_robin_group_indices,
    select_rescue_configuration,
    select_precision_constrained_configuration,
    selective_positive_rescue,
    validate_experiment_contract,
)


def test_full_group_selection_is_round_robin_instead_of_block_order():
    groups = ["a", "a", "a", "b", "b", "c"]
    assert round_robin_group_indices(groups, maximum=0) == [0, 3, 5, 1, 4, 2]
    assert round_robin_group_indices(groups, maximum=4) == [0, 3, 5, 1]


def test_negative_prototype_margins_are_calibrated_by_rank_not_zero():
    labels = np.asarray([1, 1, 1, 0, 0, 0])
    margins = np.asarray([-0.12, -0.10, -0.08, -0.50, -0.45, -0.40])
    result = fit_margin_calibration(labels, margins)
    assert result["auc"] == 1.0
    assert -0.40 <= result["threshold"] <= -0.12
    assert result["scale"] > 0


def test_rescue_is_one_way_and_limited_to_uncertain_raw_background():
    raw = np.asarray([
        [-0.05, 0.05],  # uncertain background, should be rescued
        [-1.00, 1.00],  # confident background, must stay unchanged
        [0.10, -0.10],  # existing cell, must stay unchanged
        [-0.05, 0.05],  # uncertain but prototype-negative
    ])
    proto = np.asarray([
        [-0.10, -0.40],
        [-0.10, -0.40],
        [-0.10, -0.40],
        [-0.50, -0.10],
    ])
    fused, diagnostics = selective_positive_rescue(
        raw,
        proto,
        threshold=0.0,
        scale=0.25,
        uncertainty=0.25,
        strength=0.5,
        evidence_clip=3.0,
    )
    assert np.argmax(fused[0]) == 0
    assert np.array_equal(fused[1], raw[1])
    assert np.array_equal(fused[2], raw[2])
    assert np.array_equal(fused[3], raw[3])
    assert diagnostics["eligible"] == 1
    assert diagnostics["promoted"] == 1
    assert diagnostics["cell_to_background"] == 0


def test_rescue_budget_keeps_only_strongest_prototype_evidence():
    raw = np.asarray([[-0.05, 0.05]] * 4)
    proto = np.asarray([
        [0.9, 0.0],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.1, 0.0],
    ])
    fused, diagnostics = selective_positive_rescue(
        raw,
        proto,
        threshold=0.0,
        scale=0.2,
        uncertainty=0.25,
        strength=1.0,
        max_promotions=2,
    )
    assert np.argmax(fused, axis=-1).tolist() == [0, 1, 0, 1]
    assert diagnostics["eligible_before_budget"] == 4
    assert diagnostics["eligible"] == 2
    assert diagnostics["budget"] == 2


def test_promotion_budget_scales_with_raw_cells_and_is_bounded():
    assert calculate_promotion_budget(217, ratio=0.02, minimum=1, maximum=8) == 5
    assert calculate_promotion_budget(0, ratio=0.02, minimum=1, maximum=8) == 1
    assert calculate_promotion_budget(1000, ratio=0.02, minimum=1, maximum=8) == 8


def test_precision_constrained_calibration_prefers_coverage_within_constraints():
    labels = np.asarray([1, 1, 0, 0, 1, 0])
    scores = np.asarray([0.90, 0.80, 0.85, 0.10, 0.95, 0.20])
    groups = np.asarray(["a", "a", "a", "a", "b", "b"])
    result = fit_precision_constrained_calibration(
        labels,
        scores,
        groups,
        minimum_precision=0.60,
        minimum_group_precision=0.50,
        minimum_predictions_per_group=1,
    )
    assert np.isclose(result["threshold"], 0.80)
    assert result["precision"] >= 0.60
    assert result["minimum_group_precision"] >= 0.50
    assert result["true_positive_count"] == 3


def test_configuration_is_selected_only_from_calibration_metrics():
    rows = [
        {"uncertainty": 0.0, "strength": 0.0, "f1": 0.70,
         "precision": 0.70, "recall": 0.70, "background_to_cell": 0},
        {"uncertainty": 0.25, "strength": 0.10, "f1": 0.705,
         "precision": 0.698, "recall": 0.712, "background_to_cell": 10},
        {"uncertainty": 0.50, "strength": 0.40, "f1": 0.704,
         "precision": 0.680, "recall": 0.730, "background_to_cell": 100},
    ]
    selected = select_rescue_configuration(rows)
    assert selected["uncertainty"] == 0.25
    assert selected["strength"] == 0.10
    assert np.isclose(selected["calibration_f1_gain"], 0.005)


def test_precision_constrained_selector_rejects_fp_heavy_rescue():
    rows = [
        {"uncertainty": 0.0, "strength": 0.0, "f1": 0.700,
         "macro_f1": 0.700, "precision": 0.70, "recall": 0.70,
         "background_to_cell": 0, "promotion_precision": 1.0,
         "promoted_positive_candidate": 0, "promoted_far_background": 0,
         "slides_improved_fraction": 0.0},
        {"uncertainty": 0.25, "strength": 0.10, "f1": 0.706,
         "macro_f1": 0.706, "precision": 0.69, "recall": 0.72,
         "background_to_cell": 100, "promotion_precision": 0.35,
         "promoted_positive_candidate": 35, "promoted_far_background": 60,
         "slides_improved_fraction": 0.8},
        {"uncertainty": 0.10, "strength": 0.05, "f1": 0.704,
         "macro_f1": 0.705, "precision": 0.70, "recall": 0.71,
         "background_to_cell": 20, "promotion_precision": 0.70,
         "promoted_positive_candidate": 14, "promoted_far_background": 4,
         "slides_improved_fraction": 0.75},
    ]
    selected = select_precision_constrained_configuration(
        rows,
        minimum_promotion_precision=0.60,
        minimum_slides_improved_fraction=0.60,
    )
    assert selected["uncertainty"] == 0.10
    assert selected["status"] == "selected"


def test_fixed_test_gate_requires_meaningful_f1_gain():
    baseline = {"f1": 0.756, "precision": 0.71, "recall": 0.81,
                "pred_gt_ratio": 1.14, "fp_background_far": 41000,
                "fn_low_score_or_background": 24000}
    weak = dict(baseline, f1=0.7565)
    strong = dict(baseline, f1=0.759, pred_gt_ratio=1.16)
    assert not decide_fixed_rescue(baseline, weak)["gate_pass"]
    assert decide_fixed_rescue(baseline, strong)["gate_pass"]


def test_fixed_gate_rejects_rescue_with_more_far_background_than_positive():
    baseline = {"f1": 0.756, "precision": 0.71, "recall": 0.81,
                "pred_gt_ratio": 1.14, "fp_background_far": 41000,
                "fn_low_score_or_background": 24000}
    fixed = dict(
        baseline,
        f1=0.760,
        promoted_positive_candidate=30,
        promoted_far_background=40,
        background_to_cell=80,
        promotion_precision=0.50,
    )
    result = decide_fixed_rescue(
        baseline, fixed, minimum_promotion_precision=0.60
    )
    assert not result["gate_pass"]
    assert not result["promotion_gate_pass"]


def test_fixed_gate_threshold_cannot_be_weakened_from_cli():
    baseline = {"f1": 0.756, "precision": 0.71, "recall": 0.81,
                "pred_gt_ratio": 1.14, "fp_background_far": 41000,
                "fn_low_score_or_background": 24000}
    try:
        decide_fixed_rescue(
            baseline, dict(baseline, f1=0.7561), minimum_f1_gain=0.0
        )
    except ValueError as error:
        assert str(MINIMUM_MEANINGFUL_F1_GAIN) in str(error)
    else:
        raise AssertionError("the meaningful F1 gate was silently weakened")


def test_experiment_contract_binds_checkpoint_bank_and_temperature():
    bank = {
        "implementation_version": "test-source-prototype-v1",
        "bank_id": "bank-a",
        "foreground_prototypes": np.zeros((4, 256)),
        "background_prototypes": np.zeros((8, 256)),
    }
    training_args = {
        "proto_mode": "source_supervised_candidate_proto",
        "proto_bank_path": "/old/location/bank.pth",
        "proto_temperature": 0.2,
        "proto_num_fg": 4,
        "proto_num_bg": 8,
        "proto_fg_queue_size": 8192,
        "proto_bg_queue_size": 16384,
    }
    initialization = {
        "prototype_enabled": True,
        "prototype_mode": "source_supervised_candidate_proto",
        "prototype_bank_id": "bank-a",
        "prototype_bank_path": "/old/location/bank.pth",
        "prototype_fixed_state_sha256": prototype_bank_state_sha256(bank),
    }
    contract = validate_experiment_contract(training_args, initialization, bank)
    assert contract["prototype_temperature"] == 0.2
    assert contract["prototype_bank_id"] == "bank-a"
    assert contract["foreground_queue_size"] == 8192

    bad_initialization = dict(initialization, prototype_bank_id="bank-b")
    try:
        validate_experiment_contract(training_args, bad_initialization, bank)
    except RuntimeError as error:
        assert "bank_id" in str(error)
    else:
        raise AssertionError("mismatched checkpoint and bank were accepted")

    bad_hash = dict(initialization, prototype_fixed_state_sha256="wrong")
    try:
        validate_experiment_contract(training_args, bad_hash, bank)
    except RuntimeError as error:
        assert "state hash" in str(error)
    else:
        raise AssertionError("modified prototype tensors were accepted")


def test_dataset_split_audit_separates_exact_overlap_from_slide_overlap():
    train = ["slide_a_1.jpg", "slide_b_1.jpg"]
    test = ["slide_a_2.jpg", "slide_c_1.jpg"]
    audit = audit_dataset_split(
        train,
        test,
        group_fn=lambda value: value.rsplit("_", 1)[0],
    )
    assert audit["exact_file_overlap_count"] == 0
    assert audit["slide_group_overlap_count"] == 1
    assert not audit["source_test_group_disjoint"]

    try:
        audit_dataset_split(train, ["slide_a_1.jpg"], group_fn=lambda value: value)
    except RuntimeError as error:
        assert "exact file overlap" in str(error)
    else:
        raise AssertionError("exact train/test leakage was accepted")


def main():
    test_full_group_selection_is_round_robin_instead_of_block_order()
    test_negative_prototype_margins_are_calibrated_by_rank_not_zero()
    test_rescue_is_one_way_and_limited_to_uncertain_raw_background()
    test_rescue_budget_keeps_only_strongest_prototype_evidence()
    test_promotion_budget_scales_with_raw_cells_and_is_bounded()
    test_precision_constrained_calibration_prefers_coverage_within_constraints()
    test_configuration_is_selected_only_from_calibration_metrics()
    test_precision_constrained_selector_rejects_fp_heavy_rescue()
    test_fixed_test_gate_requires_meaningful_f1_gain()
    test_fixed_gate_rejects_rescue_with_more_far_background_than_positive()
    test_fixed_gate_threshold_cannot_be_weakened_from_cli()
    test_experiment_contract_binds_checkpoint_bank_and_temperature()
    test_dataset_split_audit_separates_exact_overlap_from_slide_overlap()
    print("SSCP selective rescue tests passed")


if __name__ == "__main__":
    main()
