import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_diagnostic_has_disjoint_calibration_selection_and_test_stages():
    path = ROOT / "diagnose_sscp_selective_rescue.py"
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert "support_groups" in source
    assert "calibration_groups" in source
    assert "fit_precision_constrained_calibration" in source
    assert "select_precision_constrained_configuration" in source
    assert "no_precision_supported_uncertainty" in source
    assert "decide_fixed_rescue" in source
    assert "promoted_positive_candidate" in source
    assert "promoted_far_background" in source
    assert "test labels never select rescue parameters" in source
    assert "validate_experiment_contract" in source
    assert "audit_dataset_split" in source
    assert "unsupported_uncertainties" in source
    assert "minimum_promotion_precision" in source
    assert "promotion_budget_ratio" in source
    assert "slides_improved_fraction" in source
    assert "select_precision_constrained_configuration" in source


def test_diagnostic_parser_exposes_only_bounded_prespecified_sweeps():
    source = (
        ROOT / "diagnose_sscp_selective_rescue.py"
    ).read_text(encoding="utf-8")
    strings = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "--uncertainty_grid",
        "--strength_grid",
        "--evidence_clip",
        "--minimum_f1_gain",
        "--maximum_calibration_candidates_per_class",
        "--minimum_promotion_precision",
        "--minimum_group_precision",
        "--promotion_budget_ratio",
        "--promotion_budget_min",
        "--promotion_budget_max",
    }.issubset(strings)


def test_run_script_is_diagnostic_only():
    source = (
        ROOT / "run_sscp_selective_rescue_gpu.sh"
    ).read_text(encoding="utf-8")
    assert "diagnose_sscp_selective_rescue.py" in source
    assert "train_p2p" not in source
    assert 'GPU="${GPU:-2}"' in source
    assert "MAX_SUPPORT_IMAGES" in source
    assert "MAX_SELECTION_IMAGES" in source
    assert "MAX_TEST_IMAGES" in source
    assert "--temperature 0.1" not in source
    assert 'OUTPUT="$(realpath "${OUTPUT}")"' in source


def main():
    test_diagnostic_has_disjoint_calibration_selection_and_test_stages()
    test_diagnostic_parser_exposes_only_bounded_prespecified_sweeps()
    test_run_script_is_diagnostic_only()
    print("SSCP selective rescue diagnostic tests passed")


if __name__ == "__main__":
    main()
