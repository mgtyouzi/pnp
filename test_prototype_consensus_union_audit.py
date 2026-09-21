import numpy as np

from prototype_consensus_union_audit import (
    evaluate_consensus_gated_union,
    local_control_support,
)


def test_local_control_support_uses_highest_score_inside_each_radius():
    query = np.asarray([[10.0, 10.0]])
    points = np.asarray([[11.0, 10.0], [14.0, 10.0], [30.0, 30.0]])
    scores = np.asarray([0.30, 0.45, 0.90])

    support = local_control_support(query, points, scores, [2.0, 5.0])

    assert np.allclose(support, [[0.30, 0.45]])


def test_consensus_gate_retains_supported_rescue_and_rejects_unsupported_fp():
    control = [
        {
            "points": np.asarray(
                [[0.0, 0.0], [60.0, 60.0], [30.0, 1.0]]
            ),
            "scores": np.asarray([0.90, 0.80, 0.40]),
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

    rows = evaluate_consensus_gated_union(
        control,
        prototype,
        control_threshold=0.5,
        prototype_thresholds=[0.65],
        support_floors=[0.35, 0.45],
        agreement_radii=[3.0],
        match_distance=5.0,
        dedup_interval=5.0,
        near_radius=10.0,
    )

    accepted, rejected = rows
    assert accepted["baseline_tp"] == 1
    assert accepted["added_pred"] == 1
    assert accepted["added_tp"] == 1
    assert accepted["added_fp"] == 0
    assert accepted["newly_lost_tp"] == 0
    assert accepted["final_tp"] == 2
    assert rejected["added_pred"] == 0
    assert rejected["final_tp"] == 1


def test_consensus_gate_does_not_readd_a_prototype_duplicate():
    control = [
        {
            "points": np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            "scores": np.asarray([0.90, 0.40]),
            "gt_points": np.asarray([[0.0, 0.0]]),
        }
    ]
    prototype = [
        {
            "points": np.asarray([[2.0, 0.0]]),
            "scores": np.asarray([0.95]),
            "gt_points": np.asarray([[0.0, 0.0]]),
        }
    ]
    row = evaluate_consensus_gated_union(
        control,
        prototype,
        control_threshold=0.5,
        prototype_thresholds=[0.5],
        support_floors=[0.1],
        agreement_radii=[3.0],
        match_distance=5.0,
        dedup_interval=5.0,
        near_radius=10.0,
    )[0]
    assert row["added_pred"] == 0
    assert row["newly_lost_tp"] == 0


def main():
    test_local_control_support_uses_highest_score_inside_each_radius()
    test_consensus_gate_retains_supported_rescue_and_rejects_unsupported_fp()
    test_consensus_gate_does_not_readd_a_prototype_duplicate()
    print("Prototype consensus union audit tests passed")


if __name__ == "__main__":
    main()
