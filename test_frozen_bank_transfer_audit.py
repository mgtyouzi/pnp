import ast
from pathlib import Path

import numpy as np

from audit_frozen_supervised_bank_transfer import (
    binary_roc_auc,
    decide_transfer,
    summarize_separation,
)
from prototype_audit_sampling import stratified_round_robin_indices


ROOT = Path(__file__).resolve().parent


def test_binary_roc_auc_handles_perfect_order_and_ties():
    assert binary_roc_auc(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert binary_roc_auc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5


def test_separation_reports_macro_and_worst_group_auc():
    candidate_type = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    group_index = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.9, 0.8, 0.2, 0.1, 0.9, 0.4, 0.6, 0.1])

    result = summarize_separation(
        score,
        candidate_type,
        group_index,
        positive_type=0,
        negative_type=1,
    )

    assert result["global_auc"] == 0.9375
    assert result["macro_group_auc"] == 0.875
    assert result["min_group_auc"] == 0.75
    assert result["valid_groups"] == 2


def test_transfer_decision_has_pass_weak_and_fail_branches():
    passed = decide_transfer(0.72, 0.63, 0.94, 0.03)
    weak = decide_transfer(0.66, 0.56, 0.91, 0.01)
    failed = decide_transfer(0.58, 0.49, 0.88, -0.02)

    assert passed["status"] == "pass"
    assert weak["status"] == "weak"
    assert failed["status"] == "fail"


def test_candidate_extractor_exposes_and_forwards_phase():
    build_source = (ROOT / "build_frozen_teacher_fg_bank.py").read_text(encoding="utf-8")
    extract_source = (ROOT / "extract_frozen_teacher_candidate_audit.py").read_text(
        encoding="utf-8"
    )
    build_tree = ast.parse(build_source)
    extract_tree = ast.parse(extract_source)

    deterministic = next(
        node
        for node in build_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "deterministic_dataset"
    )
    assert [arg.arg for arg in deterministic.args.args][-1] == "phase"
    assert isinstance(deterministic.args.defaults[-1], ast.Constant)
    assert deterministic.args.defaults[-1].value == "train"

    parser_has_phase = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--phase"
        for node in ast.walk(build_tree)
    )
    assert parser_has_phase

    forwards_phase = any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "deterministic_dataset"
        and any(
            keyword.arg == "phase"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "phase"
            for keyword in node.keywords
        )
        for node in ast.walk(extract_tree)
    )
    assert forwards_phase


def test_stratified_sampling_round_robins_slide_groups():
    files = ["a_1.jpg", "a_2.jpg", "a_3.jpg", "b_1.jpg", "b_2.jpg", "c_1.jpg"]
    selected = stratified_round_robin_indices(
        files, 4, key=lambda name: name.split("_", 1)[0]
    )
    assert selected == [0, 3, 5, 1]


def main():
    test_binary_roc_auc_handles_perfect_order_and_ties()
    test_separation_reports_macro_and_worst_group_auc()
    test_transfer_decision_has_pass_weak_and_fail_branches()
    test_candidate_extractor_exposes_and_forwards_phase()
    test_stratified_sampling_round_robins_slide_groups()
    print("Frozen supervised bank transfer audit tests passed")


if __name__ == "__main__":
    main()
