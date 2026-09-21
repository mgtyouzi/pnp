import numpy as np
from pathlib import Path

from paired_detection_threshold_sweep import (
    compare_detection_transitions,
    compare_sweeps,
    deduplicate_cell_candidates,
    evaluate_baseline_preserving_union,
    evaluate_thresholds,
    get_parser,
    parse_thresholds,
)


def test_threshold_parser_includes_default_operating_point():
    values = parse_thresholds("0.45:0.55:0.05")
    assert np.allclose(values, [0.45, 0.50, 0.55])
    assert 0.5 in values


def test_cell_score_deduplication_keeps_highest_scoring_candidate():
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [20.0, 20.0]])
    scores = np.asarray([0.60, 0.90, 0.70])
    kept_points, kept_scores = deduplicate_cell_candidates(points, scores, 5.0)
    assert np.allclose(kept_points, [[1.0, 0.0], [20.0, 20.0]])
    assert np.allclose(kept_scores, [0.90, 0.70])


def test_threshold_sweep_exposes_precision_recall_tradeoff():
    records = [
        {
            "points": np.asarray([[0.0, 0.0], [30.0, 30.0]]),
            "scores": np.asarray([0.80, 0.55]),
            "gt_points": np.asarray([[0.0, 0.0]]),
        }
    ]
    rows = evaluate_thresholds(
        records,
        thresholds=[0.50, 0.70],
        match_distance=2.0,
        dedup_interval=2.0,
        near_radius=5.0,
    )
    assert rows[0]["recall"] == 1.0
    assert rows[0]["precision"] == 0.5
    assert rows[1]["recall"] == 1.0
    assert rows[1]["precision"] == 1.0
    assert rows[1]["f1"] > rows[0]["f1"]


def test_comparison_distinguishes_calibration_recovery_from_no_gain():
    control = [
        {"threshold": 0.50, "precision": 0.80, "recall": 0.80, "f1": 0.80},
        {"threshold": 0.60, "precision": 0.85, "recall": 0.70, "f1": 0.7677},
    ]
    shifted_but_better = [
        {"threshold": 0.50, "precision": 0.70, "recall": 0.85, "f1": 0.7677},
        {"threshold": 0.60, "precision": 0.84, "recall": 0.78, "f1": 0.8089},
    ]
    decision = compare_sweeps(control, shifted_but_better, minimum_gain=0.002)
    assert decision["status"] == "calibration_shift_with_recoverable_gain"
    assert decision["prototype_best_threshold"] == 0.60

    no_gain = [
        {"threshold": 0.50, "precision": 0.70, "recall": 0.84, "f1": 0.7636},
        {"threshold": 0.60, "precision": 0.80, "recall": 0.78, "f1": 0.7899},
    ]
    decision = compare_sweeps(control, no_gain, minimum_gain=0.002)
    assert decision["status"] == "no_recoverable_ranking_gain"


def test_transition_audit_tracks_rescued_and_newly_lost_gt_identity():
    control = [
        {
            "points": np.asarray([[0.0, 0.0], [100.0, 0.0]]),
            "scores": np.asarray([0.9, 0.4]),
            "gt_points": np.asarray([[0.0, 0.0], [100.0, 0.0]]),
        }
    ]
    prototype = [
        {
            "points": np.asarray([[0.0, 0.0], [100.0, 0.0]]),
            "scores": np.asarray([0.9, 0.8]),
            "gt_points": np.asarray([[0.0, 0.0], [100.0, 0.0]]),
        }
    ]
    result = compare_detection_transitions(
        control, prototype, 0.5, 0.5, match_distance=5.0, dedup_interval=5.0
    )
    assert result["rescued_fn"] == 1
    assert result["newly_lost_tp"] == 0
    assert result["net_rescue"] == 1

    reversed_result = compare_detection_transitions(
        prototype, control, 0.5, 0.5, match_distance=5.0, dedup_interval=5.0
    )
    assert reversed_result["rescued_fn"] == 0
    assert reversed_result["newly_lost_tp"] == 1
    assert reversed_result["net_rescue"] == -1


def test_baseline_preserving_union_only_adds_nonduplicate_prototype_detections():
    control = [
        {
            "points": np.asarray([[0.0, 0.0], [60.0, 60.0]]),
            "scores": np.asarray([0.9, 0.8]),
            "gt_points": np.asarray([[0.0, 0.0], [30.0, 0.0]]),
        }
    ]
    prototype = [
        {
            "points": np.asarray(
                [[1.0, 0.0], [30.0, 0.0], [100.0, 100.0]]
            ),
            "scores": np.asarray([0.95, 0.75, 0.70]),
            "gt_points": np.asarray([[0.0, 0.0], [30.0, 0.0]]),
        }
    ]
    rows = evaluate_baseline_preserving_union(
        control,
        prototype,
        control_threshold=0.5,
        addition_thresholds=[0.72, 0.65],
        match_distance=5.0,
        dedup_interval=5.0,
        near_radius=10.0,
    )

    strict, loose = rows
    assert strict["baseline_tp"] == 1
    assert strict["added_pred"] == 1
    assert strict["added_tp"] == 1
    assert strict["added_fp"] == 0
    assert strict["newly_lost_tp"] == 0
    assert strict["final_tp"] == 2
    assert loose["added_pred"] == 2
    assert loose["added_tp"] == 1
    assert loose["added_fp"] == 1
    assert loose["added_background_far_fp"] == 1


def test_run_script_maps_the_selected_physical_gpu_to_cuda_zero():
    script = (
        Path(__file__).resolve().parent
        / "run_positive_only_threshold_sweep_gpu.sh"
    ).read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES="${GPU}"' in script
    assert "--gpu=0" in script
    assert "paired_detection_threshold_sweep.py" in script
    assert "latest_positive_only_train_proto.txt" in script


def test_prototype_checkpoint_subdirectory_is_configurable():
    args = get_parser().parse_args([
        "--pair_root", "pair",
        "--dataset", "dataset",
        "--mean_std_path", "mean_std.npy",
        "--prototype_subdir", "prototype_guided_ranking",
    ])
    assert args.prototype_subdir == "prototype_guided_ranking"


def main():
    test_threshold_parser_includes_default_operating_point()
    test_cell_score_deduplication_keeps_highest_scoring_candidate()
    test_threshold_sweep_exposes_precision_recall_tradeoff()
    test_comparison_distinguishes_calibration_recovery_from_no_gain()
    test_transition_audit_tracks_rescued_and_newly_lost_gt_identity()
    test_baseline_preserving_union_only_adds_nonduplicate_prototype_detections()
    test_run_script_maps_the_selected_physical_gpu_to_cuda_zero()
    test_prototype_checkpoint_subdirectory_is_configurable()
    print("Paired detection threshold sweep tests passed")


if __name__ == "__main__":
    main()
