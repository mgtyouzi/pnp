from pathlib import Path

import numpy as np

from prototype_guided_assignment_audit import (
    PROTOTYPE_GUIDED_ASSIGNMENT_AUDIT_VERSION,
    aggregate_assignment_metrics,
    build_target_switch_records,
    build_guided_cost,
    compare_target_assignments,
    evaluate_assignment_gate,
    get_parser,
    parse_lambdas,
    solve_target_assignment,
)


def test_parse_lambdas_is_sorted_unique_and_positive():
    assert PROTOTYPE_GUIDED_ASSIGNMENT_AUDIT_VERSION == (
        "prototype_guided_assignment_audit_v2_20260907"
    )
    assert parse_lambdas("0.10,0.02,0.10,0.05") == [0.02, 0.05, 0.1]


def test_guided_cost_uses_similarity_only_inside_local_radius():
    baseline = np.asarray([[1.0], [1.0], [1.0]])
    distances = np.asarray([[5.0], [20.0], [40.0]])
    similarities = np.asarray([[0.8], [0.2], [0.9]])
    guided = build_guided_cost(
        baseline,
        distances,
        similarities,
        prototype_weight=0.2,
        local_radius=30.0,
    )
    assert np.allclose(guided[:, 0], [1.04, 1.16, 1.20])


def test_assignment_comparison_is_target_aligned_and_counts_rescue():
    baseline_cost = np.asarray(
        [
            [0.10, 8.00],
            [8.00, 0.10],
            [0.20, 8.00],
        ]
    )
    guided_cost = np.asarray(
        [
            [0.30, 8.00],
            [8.00, 0.10],
            [0.05, 8.00],
        ]
    )
    scores = np.asarray([0.40, 0.80, 0.70])
    distances = np.asarray(
        [
            [5.0, 50.0],
            [50.0, 4.0],
            [5.5, 50.0],
        ]
    )
    similarities = np.asarray(
        [
            [0.20, 0.10],
            [0.10, 0.80],
            [0.90, 0.10],
        ]
    )
    baseline = solve_target_assignment(baseline_cost)
    guided = solve_target_assignment(guided_cost)
    result = compare_target_assignments(
        baseline,
        guided,
        scores,
        distances,
        similarities,
        score_threshold=0.5,
        match_radius=15.0,
        distance_tolerance=1.0,
    )
    assert baseline.tolist() == [0, 1]
    assert guided.tolist() == [2, 1]
    assert result["switch_count"] == 1
    assert result["score_improved_switch_count"] == 1
    assert result["low_to_high_count"] == 1
    assert np.isclose(result["switched_score_delta_sum"], 0.30)
    assert np.isclose(result["switched_distance_delta_sum"], 0.5)
    assert np.isclose(result["switched_similarity_delta_sum"], 0.70)


def test_reliable_switch_can_have_lower_current_score_but_better_geometry():
    records = build_target_switch_records(
        baseline_assignment=np.asarray([0]),
        guided_assignment=np.asarray([1]),
        candidate_scores=np.asarray([0.40, 0.25]),
        point_distances=np.asarray([[6.0], [1.5]]),
        prototype_similarities=np.asarray([[0.20], [0.75]]),
        match_radius=15.0,
        minimum_distance_improvement=3.0,
        minimum_similarity_improvement=0.1,
        minimum_candidate_score=0.2,
        maximum_score_drop=0.25,
    )
    assert len(records) == 1
    assert records[0]["score_delta"] < 0
    assert records[0]["distance_improvement"] == 4.5
    assert records[0]["reliable_switch"] is True


def test_gate_requires_enough_reliable_switches_not_higher_current_scores():
    per_image = [
        {
            "lambda": 0.05,
            "images": 1,
            "targets": 1000,
            "switch_count": 10,
            "score_improved_switch_count": 0,
            "local_switch_count": 10,
            "distance_safe_switch_count": 10,
            "reliable_switch_count": 10,
            "nonlocal_switch_count": 0,
            "baseline_low_score_count": 20,
            "low_to_high_count": 0,
            "switched_score_delta_sum": -1.5,
            "switched_distance_delta_sum": -45.0,
            "switched_similarity_delta_sum": 5.0,
            "reliable_score_delta_sum": -1.5,
            "reliable_distance_improvement_sum": 45.0,
            "reliable_similarity_delta_sum": 5.0,
            "baseline_within_match_count": 980,
            "guided_within_match_count": 990,
        }
    ]
    rows = aggregate_assignment_metrics(per_image)
    decision = evaluate_assignment_gate(
        rows, thresholds={"minimum_reliable_switch_count": 10}
    )
    assert decision["gate_pass"] is True
    assert decision["selected_lambda"] == 0.05

    rows[0]["reliable_switch_share"] = 0.5
    decision = evaluate_assignment_gate(
        rows, thresholds={"minimum_reliable_switch_count": 10}
    )
    assert decision["gate_pass"] is False
    assert "reliable_switch_share" in decision["failures"]


def test_run_script_is_a_read_only_single_gpu_audit():
    root = Path(__file__).resolve().parent
    script = (root / "run_prototype_guided_assignment_audit_gpu.sh").read_text(
        encoding="utf-8"
    )
    assert 'CUDA_VISIBLE_DEVICES="${GPU}"' in script
    assert "prototype_guided_assignment_audit.py" in script
    assert "--gpu=0" in script
    assert "--checkpoint=" in script
    assert "train_p2p_no_empty_v2.py" not in script
    assert "torchrun" not in script
    assert 'LAMBDAS="${LAMBDAS:-0.2,0.5,1.0,2.0}"' in script
    assert 'LOCAL_RADIUS="${LOCAL_RADIUS:-15}"' in script


def test_revised_audit_defaults_match_the_training_assignment_question():
    args = get_parser().parse_args(
        [
            "--checkpoint", "model.pth",
            "--dataset", "dataset",
            "--mean_std_path", "mean_std.npy",
            "--output_dir", "output",
        ]
    )
    assert parse_lambdas(args.lambdas) == [0.2, 0.5, 1.0, 2.0]
    assert args.local_radius == 15.0
    assert args.minimum_distance_improvement == 3.0
    assert args.minimum_similarity_improvement == 0.1
    assert args.minimum_candidate_score == 0.2
    assert args.maximum_score_drop == 0.25


def main():
    test_parse_lambdas_is_sorted_unique_and_positive()
    test_guided_cost_uses_similarity_only_inside_local_radius()
    test_assignment_comparison_is_target_aligned_and_counts_rescue()
    test_reliable_switch_can_have_lower_current_score_but_better_geometry()
    test_gate_requires_enough_reliable_switches_not_higher_current_scores()
    test_run_script_is_a_read_only_single_gpu_audit()
    test_revised_audit_defaults_match_the_training_assignment_question()
    print("Prototype-guided assignment audit tests passed")


if __name__ == "__main__":
    main()
