import numpy as np

from frozen_proto_rescoring import (
    apply_candidate_joint_calibration,
    candidate_separation_report,
    decide_suppression_target,
    decide_rescoring,
    evaluate_image,
    fit_candidate_joint_calibration,
    fuse_binary_logits,
    label_candidate_diagnostics,
    parse_alphas,
    parse_quantiles,
    load_mean_std,
    resolve_csv_fieldnames,
    select_source_suppression,
    suppress_raw_cell_logits,
    summarize_distribution,
    summarize_counts,
    thresholds_from_raw_cell_quantiles,
    validate_suppression_calibration,
    validate_checkpoint_load_report,
    validate_candidate_audit_calibration,
)


def test_candidate_diagnostics_reproduce_p2p_match_categories():
    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [20.0, 0.0], [100.0, 0.0], [200.0, 200.0]]
    )
    raw = np.array(
        [[4.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    )
    proto = np.array(
        [[3.0, 0.0], [1.0, 0.0], [0.5, 0.0], [-1.0, 1.0], [-2.0, 2.0]]
    )
    gt = np.array([[0.0, 0.0], [200.0, 200.0]])

    rows = label_candidate_diagnostics(
        points,
        raw,
        proto,
        gt,
        dedup_interval=1.0,
        match_dis=5.0,
        near_radius=30.0,
    )

    assert [row["category"] for row in rows] == [
        "tp",
        "duplicate_fp",
        "near_miss_fp",
        "background_far_fp",
        "low_score_fn",
    ]
    assert [row["candidate_index"] for row in rows] == [0, 1, 2, 3, 4]
    assert rows[0]["raw_margin"] == 4.0
    assert rows[3]["prototype_margin"] == -2.0
    assert rows[4]["raw_class"] == 1


def test_candidate_diagnostics_respect_localization_fn_precedence():
    points = np.array([[10.0, 0.0], [20.0, 0.0]])
    raw = np.array([[2.0, 0.0], [0.0, 2.0]])
    proto = np.array([[1.0, 0.0], [-1.0, 1.0]])
    gt = np.array([[20.0, 0.0]])

    rows = label_candidate_diagnostics(
        points,
        raw,
        proto,
        gt,
        dedup_interval=1.0,
        match_dis=5.0,
        near_radius=30.0,
    )

    assert [row["category"] for row in rows] == ["near_miss_fp"]


def test_source_joint_calibration_uses_prototype_when_raw_margin_is_uninformative():
    rows = [
        {"category": "tp", "raw_margin": 1.0, "prototype_margin": 3.0},
        {"category": "tp", "raw_margin": 2.0, "prototype_margin": 2.0},
        {"category": "background_far_fp", "raw_margin": 1.0, "prototype_margin": -3.0},
        {"category": "background_far_fp", "raw_margin": 2.0, "prototype_margin": -2.0},
    ]
    calibration = fit_candidate_joint_calibration(rows, beta_grid=[0.0, 0.5, 1.0])
    scores = apply_candidate_joint_calibration(rows, calibration)
    report = candidate_separation_report(
        [{**row, "joint_score": float(score), "slide_group": "slide-a"}
         for row, score in zip(rows, scores)],
        score_names=("raw_margin", "prototype_margin", "joint_score"),
    )

    assert calibration["beta"] > 0.0
    assert calibration["source_auc"] == 1.0
    joint = next(
        row for row in report
        if row["comparison"] == "tp_vs_background_far_fp"
        and row["score"] == "joint_score"
    )
    assert joint["global_auc"] == 1.0


def test_candidate_calibration_rejects_detector_or_bank_mismatch():
    payload = {
        "version": "frozen_proto_candidate_separation_v1_20260828",
        "checkpoint_sha256": "checkpoint-a",
        "bank_id": "bank-a",
        "bank_state_sha256": "state-a",
    }
    validate_candidate_audit_calibration(
        payload,
        checkpoint_sha256="checkpoint-a",
        bank_id="bank-a",
        bank_state_sha256="state-a",
    )
    for key, value in (
        ("checkpoint_sha256", "checkpoint-b"),
        ("bank_id", "bank-b"),
        ("bank_state_sha256", "state-b"),
    ):
        changed = dict(payload)
        changed[key] = value
        try:
            validate_candidate_audit_calibration(
                changed,
                checkpoint_sha256="checkpoint-a",
                bank_id="bank-a",
                bank_state_sha256="state-a",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"candidate calibration should reject {key}")


def test_candidate_diagnostic_loads_fixed_calibration_before_target_extraction():
    import ast
    from pathlib import Path

    path = Path("diagnose_frozen_proto_candidate_separation.py")
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node.func.id
        for node in ast.walk(main_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "_load_fixed_calibration" in calls
    assert "_extract_records" in calls
    main_source = source[source.index("def main()") :]
    assert main_source.index("_load_fixed_calibration") < main_source.index(
        "_extract_records"
    )
    target_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_target_fixed"
    )
    target_calls = {
        node.func.id
        for node in ast.walk(target_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "fit_candidate_joint_calibration" not in target_calls


def test_alpha_zero_is_exact_identity():
    raw = np.array([[2.0, -1.0], [0.25, 0.75]], dtype=np.float64)
    proto = np.array([[3.0, -2.0], [-4.0, 1.0]], dtype=np.float64)
    fused = fuse_binary_logits(raw, proto, alpha=0.0, clip=2.0)
    assert np.array_equal(fused, raw)


def test_fusion_changes_only_binary_margin_and_preserves_logit_mean():
    raw = np.array([[1.0, 0.0]], dtype=np.float64)
    proto = np.array([[4.0, -1.0]], dtype=np.float64)
    fused = fuse_binary_logits(raw, proto, alpha=0.5, clip=2.0)
    assert np.allclose(fused.mean(axis=-1), raw.mean(axis=-1))
    assert np.allclose(fused[:, 0] - fused[:, 1], [2.0])


def test_prototype_rescoring_can_remove_far_false_positive():
    points = np.array([[0.0, 0.0], [50.0, 50.0]], dtype=np.float64)
    gt = np.array([[0.0, 0.0]], dtype=np.float64)
    raw = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    proto = np.array([[2.0, 0.0], [-4.0, 2.0]], dtype=np.float64)

    baseline = evaluate_image(
        points, raw, gt, dedup_interval=15.0, match_dis=15.0, near_radius=30.0
    )
    fused = evaluate_image(
        points,
        fuse_binary_logits(raw, proto, alpha=1.0, clip=4.0),
        gt,
        dedup_interval=15.0,
        match_dis=15.0,
        near_radius=30.0,
    )

    assert baseline["tp"] == 1
    assert baseline["fp_background_far"] == 1
    assert fused["tp"] == 1
    assert fused["fp_background_far"] == 0
    assert summarize_counts(fused)["f1"] == 1.0


def test_suppressing_true_candidate_is_counted_as_classification_fn():
    points = np.array([[0.0, 0.0]], dtype=np.float64)
    gt = np.array([[0.0, 0.0]], dtype=np.float64)
    logits = np.array([[-1.0, 1.0]], dtype=np.float64)
    result = evaluate_image(
        points, logits, gt, dedup_interval=15.0, match_dis=15.0, near_radius=30.0
    )
    assert result["tp"] == 0
    assert result["fn"] == 1
    assert result["fn_low_score_or_background"] == 1


def test_parse_alphas_requires_identity_control_and_deduplicates():
    assert parse_alphas("0.2,0,0.1,0.2") == [0.0, 0.1, 0.2]
    try:
        parse_alphas("0.1,0.2")
    except ValueError as error:
        assert "alpha=0" in str(error)
    else:
        raise AssertionError("missing alpha=0 should fail")


def test_decision_requires_f1_gain_and_controlled_fn_tradeoff():
    rows = [
        {"alpha": 0.0, "f1": 0.600, "fp_background_far": 1000, "fn_low_score_or_background": 500, "gt": 10000},
        {"alpha": 0.1, "f1": 0.606, "fp_background_far": 850, "fn_low_score_or_background": 550, "gt": 10000},
        {"alpha": 0.2, "f1": 0.604, "fp_background_far": 700, "fn_low_score_or_background": 900, "gt": 10000},
    ]
    decision = decide_rescoring(rows)
    assert decision["status"] == "promising"
    assert decision["best_alpha"] == 0.1
    assert abs(decision["f1_gain"] - 0.006) < 1e-12


def test_mean_std_loader_unpacks_two_vectors(tmp_path=None):
    import io

    payload = io.BytesIO()
    np.save(payload, np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]))
    payload.seek(0)
    mean, std = load_mean_std(payload)
    assert np.allclose(mean, [0.1, 0.2, 0.3])
    assert np.allclose(std, [1.1, 1.2, 1.3])


def test_checkpoint_validation_allows_only_prototype_head_missing_keys():
    validate_checkpoint_load_report(
        {
            "skipped_count": 0,
            "shape_mismatches": {},
            "unexpected_checkpoint_keys": [],
            "missing_model_keys": ["prototype_head.prototypes"],
            "unexpected_model_keys": [],
        }
    )
    invalid = {
        "skipped_count": 1,
        "shape_mismatches": {"cls_head.weight": {}},
        "unexpected_checkpoint_keys": [],
        "missing_model_keys": ["cls_head.weight"],
        "unexpected_model_keys": [],
    }
    try:
        validate_checkpoint_load_report(invalid)
    except RuntimeError as error:
        assert "detector checkpoint" in str(error)
    else:
        raise AssertionError("partial detector load should fail")


def test_empty_csv_uses_explicit_fieldnames():
    fields = ["alpha", "image", "gt"]
    assert resolve_csv_fieldnames([], fields) == fields
    try:
        resolve_csv_fieldnames([], None)
    except ValueError as error:
        assert "fieldnames" in str(error)
    else:
        raise AssertionError("empty CSV without explicit fields should fail")


def test_csv_fieldnames_include_keys_added_after_first_row():
    rows = [{"alpha": 0.0}, {"alpha": 0.1, "threshold_quantile": 0.2}]
    assert resolve_csv_fieldnames(rows) == ["alpha", "threshold_quantile"]


def test_diagnostic_declares_test_phase_and_strict_loading_contract():
    from pathlib import Path

    source = Path("diagnose_frozen_proto_rescoring.py").read_text(encoding="utf-8")
    assert 'parser.set_defaults(phase="test")' in source
    assert "load_mean_std" in source
    assert "validate_checkpoint_load_report" in source


def test_one_way_alpha_zero_is_exact_identity():
    raw = np.array([[2.0, -1.0], [-0.5, 0.75]], dtype=np.float64)
    proto = np.array([[-2.0, 2.0], [3.0, -3.0]], dtype=np.float64)
    fused = suppress_raw_cell_logits(
        raw, proto, alpha=0.0, threshold=0.5, clip=2.0
    )
    assert np.array_equal(fused, raw)


def test_one_way_suppression_never_promotes_raw_background():
    raw = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    proto = np.array([[-2.0, 2.0], [4.0, -4.0]], dtype=np.float64)
    fused = suppress_raw_cell_logits(
        raw, proto, alpha=1.0, threshold=0.0, clip=2.0
    )
    assert np.argmax(fused[0]) == 1
    assert np.array_equal(fused[1], raw[1])
    assert np.argmax(fused[1]) == 1


def test_one_way_suppression_preserves_supported_raw_cell():
    raw = np.array([[1.0, 0.0]], dtype=np.float64)
    proto = np.array([[2.0, 0.0]], dtype=np.float64)
    fused = suppress_raw_cell_logits(
        raw, proto, alpha=0.5, threshold=1.0, clip=2.0
    )
    assert np.array_equal(fused, raw)


def test_threshold_quantiles_use_only_raw_cell_candidates():
    raw = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    proto = np.array([[-1.0, 1.0], [50.0, -50.0], [1.0, -1.0]])
    thresholds = thresholds_from_raw_cell_quantiles(
        raw, proto, quantiles=[0.0, 0.5, 1.0]
    )
    assert np.allclose(thresholds, [-2.0, 0.0, 2.0])


def test_source_selection_prefers_fp_reduction_within_safety_constraints():
    rows = [
        {"alpha": 0.0, "threshold": None, "f1": 0.6000, "recall": 0.7500, "fp_background_far": 1000},
        {"alpha": 0.05, "threshold": -1.0, "f1": 0.5995, "recall": 0.7470, "fp_background_far": 900},
        {"alpha": 0.10, "threshold": 0.0, "f1": 0.5980, "recall": 0.7480, "fp_background_far": 800},
        {"alpha": 0.20, "threshold": 1.0, "f1": 0.6005, "recall": 0.7440, "fp_background_far": 700},
    ]
    selection = select_source_suppression(
        rows, max_f1_drop=0.001, max_recall_drop=0.005
    )
    assert selection["status"] == "selected"
    assert selection["alpha"] == 0.05
    assert selection["threshold"] == -1.0
    assert selection["background_far_reduction"] == 100


def test_parse_quantiles_validates_probability_range():
    assert parse_quantiles("0.5,0.1,0.5") == [0.1, 0.5]
    try:
        parse_quantiles("-0.1,0.5")
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:
        raise AssertionError("negative quantile should fail")


def test_target_suppression_decision_requires_fp_and_f1_gain():
    rows = [
        {
            "label": "identity",
            "f1": 0.600,
            "recall": 0.750,
            "fp_background_far": 1000,
            "fn_low_score_or_background": 500,
            "gt": 10000,
            "background_to_cell": 0,
        },
        {
            "label": "source_fixed",
            "f1": 0.605,
            "recall": 0.748,
            "fp_background_far": 900,
            "fn_low_score_or_background": 550,
            "gt": 10000,
            "background_to_cell": 0,
        },
    ]
    decision = decide_suppression_target(rows)
    assert decision["status"] == "promising"
    assert abs(decision["f1_gain"] - 0.005) < 1e-12
    assert abs(decision["background_far_reduction_fraction"] - 0.1) < 1e-12


def test_calibration_validation_rejects_identity_mismatch():
    payload = {
        "version": "frozen_proto_one_way_suppression_v1_20260828",
        "selection": {"status": "selected", "alpha": 0.1, "threshold": -1.0},
        "checkpoint_sha256": "checkpoint-a",
        "bank_id": "bank-a",
        "bank_state_sha256": "state-a",
        "fusion_clip": 2.0,
    }
    validate_suppression_calibration(
        payload,
        checkpoint_sha256="checkpoint-a",
        bank_id="bank-a",
        bank_state_sha256="state-a",
    )
    try:
        validate_suppression_calibration(
            payload,
            checkpoint_sha256="checkpoint-b",
            bank_id="bank-a",
            bank_state_sha256="state-a",
        )
    except RuntimeError as error:
        assert "checkpoint" in str(error)
    else:
        raise AssertionError("checkpoint mismatch should fail")


def test_one_way_diagnostic_exposes_separate_source_and_target_modes():
    import ast
    from pathlib import Path

    path = Path("diagnose_frozen_proto_suppression.py")
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert 'choices=("source_calibrate", "target_fixed")' in source
    assert "validate_suppression_calibration" in source
    assert "suppress_raw_cell_logits" in source
    assert source.count("fieldnames=SUPPRESSION_SAMPLE_FIELDS") == 2
    main_source = source[source.index("def main()") :]
    assert main_source.index("_load_target_calibration") < main_source.index(
        "_extract_records"
    )
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert isinstance(functions["_load_model"].body[-1], ast.Return)


def test_one_way_diagnostic_passes_calibration_only_to_target_runner():
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path("diagnose_frozen_proto_suppression.py").read_text(encoding="utf-8")
    )
    main_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = {
        node.func.id: node
        for node in ast.walk(main_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    source_call = calls["_run_source_calibration"]
    target_call = calls["_run_target_fixed"]
    assert len(source_call.args) == 6
    assert len(target_call.args) == 7
    assert isinstance(target_call.args[-1], ast.Name)
    assert target_call.args[-1].id == "calibration"


def test_distribution_summary_is_defined_for_empty_and_nonempty_values():
    empty = summarize_distribution(np.array([], dtype=np.float64))
    assert empty == {"count": 0}
    summary = summarize_distribution(np.array([-2.0, 0.0, 2.0]))
    assert summary["count"] == 3
    assert summary["p50"] == 0.0
    assert summary["mean"] == 0.0


def main():
    test_candidate_diagnostics_reproduce_p2p_match_categories()
    test_candidate_diagnostics_respect_localization_fn_precedence()
    test_source_joint_calibration_uses_prototype_when_raw_margin_is_uninformative()
    test_candidate_calibration_rejects_detector_or_bank_mismatch()
    test_candidate_diagnostic_loads_fixed_calibration_before_target_extraction()
    test_alpha_zero_is_exact_identity()
    test_fusion_changes_only_binary_margin_and_preserves_logit_mean()
    test_prototype_rescoring_can_remove_far_false_positive()
    test_suppressing_true_candidate_is_counted_as_classification_fn()
    test_parse_alphas_requires_identity_control_and_deduplicates()
    test_decision_requires_f1_gain_and_controlled_fn_tradeoff()
    test_mean_std_loader_unpacks_two_vectors()
    test_checkpoint_validation_allows_only_prototype_head_missing_keys()
    test_empty_csv_uses_explicit_fieldnames()
    test_csv_fieldnames_include_keys_added_after_first_row()
    test_diagnostic_declares_test_phase_and_strict_loading_contract()
    test_one_way_alpha_zero_is_exact_identity()
    test_one_way_suppression_never_promotes_raw_background()
    test_one_way_suppression_preserves_supported_raw_cell()
    test_threshold_quantiles_use_only_raw_cell_candidates()
    test_source_selection_prefers_fp_reduction_within_safety_constraints()
    test_parse_quantiles_validates_probability_range()
    test_target_suppression_decision_requires_fp_and_f1_gain()
    test_calibration_validation_rejects_identity_mismatch()
    test_one_way_diagnostic_exposes_separate_source_and_target_modes()
    test_one_way_diagnostic_passes_calibration_only_to_target_runner()
    test_distribution_summary_is_defined_for_empty_and_nonempty_values()
    print("Frozen prototype rescoring tests passed")


if __name__ == "__main__":
    main()
