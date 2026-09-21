from pathlib import Path

import numpy as np

from prototype_local_replacement_audit import (
    PROTOTYPE_LOCAL_REPLACEMENT_AUDIT_VERSION,
    analyze_local_replacement_candidates,
    build_local_replacement_proposals,
    evaluate_local_replacement_gate,
    select_conflict_free_replacements,
    summarize_local_replacements,
)


def test_only_unassigned_candidates_owned_by_nearest_target_are_proposed():
    baseline = np.asarray([0, 1])
    scores = np.asarray([0.40, 0.70, 0.25, 0.80])
    distances = np.asarray(
        [
            [6.0, 30.0],
            [30.0, 4.0],
            [1.5, 20.0],
            [8.0, 9.0],
        ]
    )
    similarities = np.asarray(
        [
            [0.20, 0.10],
            [0.10, 0.70],
            [0.75, 0.10],
            [0.90, 0.90],
        ]
    )
    proposals, funnel = analyze_local_replacement_candidates(
        baseline,
        scores,
        distances,
        similarities,
        match_radius=15.0,
        minimum_distance_improvement=3.0,
        minimum_similarity_improvement=0.1,
        minimum_candidate_score=0.2,
        maximum_score_drop=0.25,
    )
    assert [item["candidate_index"] for item in proposals] == [2]
    assert proposals[0]["target_index"] == 0
    assert proposals[0]["score_delta"] < 0
    assert proposals[0]["distance_improvement"] == 4.5
    assert funnel == {
        "unassigned_candidates": 2,
        "nearest_target_local": 2,
        "distance_improvement_pass": 1,
        "similarity_improvement_pass": 1,
        "candidate_score_floor_pass": 1,
        "score_drop_pass": 1,
        "eligible_proposals": 1,
    }
    assert len(build_local_replacement_proposals(
        baseline, scores, distances, similarities
    )) == 1


def test_conflict_resolution_keeps_one_highest_utility_candidate_per_target():
    proposals = [
        {"target_index": 0, "candidate_index": 4, "utility": 0.8},
        {"target_index": 0, "candidate_index": 5, "utility": 1.1},
        {"target_index": 1, "candidate_index": 6, "utility": 0.9},
    ]
    selected = select_conflict_free_replacements(proposals)
    assert [(item["target_index"], item["candidate_index"]) for item in selected] == [
        (0, 5),
        (1, 6),
    ]


def test_summary_and_gate_accept_a_sufficient_safe_replacement_set():
    selected = [
        {
            "target_index": index,
            "candidate_index": index + 1000,
            "score_delta": -0.02,
            "distance_improvement": 5.0,
            "similarity_delta": 0.3,
            "baseline_score": 0.4,
            "candidate_score": 0.38,
            "baseline_distance": 7.0,
            "candidate_distance": 2.0,
            "baseline_similarity": 0.2,
            "candidate_similarity": 0.5,
            "utility": 0.7,
        }
        for index in range(10)
    ]
    summary = summarize_local_replacements(
        selected, targets=1000, unassigned_candidates=9000
    )
    decision = evaluate_local_replacement_gate(
        summary, thresholds={"minimum_replacement_count": 10}
    )
    assert decision["gate_pass"] is True
    assert summary["replacement_rate"] == 0.01
    assert summary["mean_score_delta"] < 0

    summary["mean_distance_improvement"] = 2.0
    decision = evaluate_local_replacement_gate(
        summary, thresholds={"minimum_replacement_count": 10}
    )
    assert decision["gate_pass"] is False
    assert "mean_distance_improvement" in decision["failures"]


def test_launcher_is_versioned_single_gpu_and_read_only():
    assert PROTOTYPE_LOCAL_REPLACEMENT_AUDIT_VERSION == (
        "prototype_local_replacement_audit_v1_20260907"
    )
    script = (
        Path(__file__).resolve().parent
        / "run_prototype_local_replacement_audit_gpu.sh"
    ).read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES="${GPU}"' in script
    assert "prototype_local_replacement_audit.py" in script
    assert "--gpu=0" in script
    assert "train_p2p_no_empty_v2.py" not in script
    assert "torchrun" not in script


def main():
    test_only_unassigned_candidates_owned_by_nearest_target_are_proposed()
    test_conflict_resolution_keeps_one_highest_utility_candidate_per_target()
    test_summary_and_gate_accept_a_sufficient_safe_replacement_set()
    test_launcher_is_versioned_single_gpu_and_read_only()
    print("Prototype local replacement audit tests passed")


if __name__ == "__main__":
    main()
