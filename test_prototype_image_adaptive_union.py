import numpy as np

from prototype_image_adaptive_union_audit import (
    _bounded_background_indices,
    binary_auc,
    choose_next_action,
    evaluate_image_adaptive_prototype_union,
    spherical_kmeans,
)


def test_spherical_kmeans_preserves_two_separated_modes():
    values = np.asarray(
        [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]],
        dtype=np.float32,
    )
    centers = spherical_kmeans(values, num_prototypes=2, iterations=10)
    similarities = centers @ np.asarray([[1.0, 0.0], [0.0, 1.0]]).T
    assert similarities.max(axis=0).min() > 0.99


def test_binary_auc_reports_margin_separation_direction():
    labels = np.asarray([1, 1, 0, 0])
    assert binary_auc(labels, np.asarray([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert binary_auc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) == 0.0


def test_next_action_distinguishes_support_representation_and_precision_failures():
    assert choose_next_action(0.003, 0.002, 0.95, 0.7).startswith("integrate_")
    assert choose_next_action(0.0, 0.002, 0.5, 0.7).startswith("insufficient_")
    assert choose_next_action(0.0, 0.002, 0.95, 0.5).startswith("prototype_margin_")
    assert choose_next_action(0.0, 0.002, 0.95, 0.7).startswith("margin_separates_")


def test_background_support_excludes_low_score_candidates_near_detections():
    points = np.asarray([[1.0, 0.0], [50.0, 50.0]])
    scores = np.asarray([0.01, 0.01])
    indices = _bounded_background_indices(
        points,
        scores,
        threshold=0.05,
        maximum_supports=16,
        exclusion_points=np.asarray([[0.0, 0.0]]),
        exclusion_radius=30.0,
    )
    assert indices.tolist() == [1]


def test_image_adaptive_prototypes_keep_cell_like_addition_and_reject_background():
    control = [
        {
            "points": np.asarray([[0.0, 0.0], [30.0, 1.0]]),
            "scores": np.asarray([0.90, 0.30]),
            "gt_points": np.asarray([[0.0, 0.0], [30.0, 0.0]]),
        }
    ]
    prototype = [
        {
            "points": np.asarray(
                [[0.0, 0.0], [30.0, 0.0], [100.0, 100.0], [70.0, 70.0]]
            ),
            "scores": np.asarray([0.90, 0.70, 0.70, 0.01]),
            "embeddings": np.asarray(
                [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
            "gt_points": np.asarray([[0.0, 0.0], [30.0, 0.0]]),
        }
    ]

    rows = evaluate_image_adaptive_prototype_union(
        control,
        prototype,
        control_threshold=0.56,
        prototype_thresholds=[0.65],
        margin_thresholds=[0.2],
        foreground_support_threshold=0.8,
        background_support_threshold=0.05,
        support_match_radius=5.0,
        num_foreground_prototypes=1,
        num_background_prototypes=1,
        match_distance=5.0,
        dedup_interval=5.0,
        near_radius=10.0,
    )

    row = rows[0]
    assert row["images_with_ready_local_prototypes"] == 1
    assert row["added_pred"] == 1
    assert row["added_tp"] == 1
    assert row["added_fp"] == 0
    assert row["newly_lost_tp"] == 0
    assert row["final_tp"] == 2
    assert row["mean_added_margin"] > 0.2
    assert row["candidate_margin_auc"] == 1.0


def test_local_bank_is_not_ready_when_support_count_is_below_requested_k():
    control = [
        {
            "points": np.asarray([[0.0, 0.0]]),
            "scores": np.asarray([0.90]),
            "gt_points": np.asarray([[0.0, 0.0], [30.0, 0.0]]),
        }
    ]
    prototype = [
        {
            "points": np.asarray([[0.0, 0.0], [30.0, 0.0], [70.0, 70.0]]),
            "scores": np.asarray([0.90, 0.70, 0.01]),
            "embeddings": np.asarray(
                [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]], dtype=np.float32
            ),
            "gt_points": np.asarray([[0.0, 0.0], [30.0, 0.0]]),
        }
    ]
    row = evaluate_image_adaptive_prototype_union(
        control,
        prototype,
        control_threshold=0.56,
        prototype_thresholds=[0.65],
        margin_thresholds=[0.2],
        foreground_support_threshold=0.8,
        background_support_threshold=0.05,
        support_match_radius=5.0,
        num_foreground_prototypes=2,
        num_background_prototypes=2,
        match_distance=5.0,
        dedup_interval=5.0,
        near_radius=10.0,
    )[0]
    assert row["images_with_ready_local_prototypes"] == 0
    assert row["images_with_insufficient_foreground_support"] == 1
    assert row["images_with_insufficient_background_support"] == 1
    assert row["added_pred"] == 0


def test_query_candidate_cannot_become_foreground_support_without_baseline_confirmation():
    control = [
        {
            "points": np.asarray([[0.0, 0.0]]),
            "scores": np.asarray([0.90]),
            "gt_points": np.asarray([[0.0, 0.0]]),
        }
    ]
    prototype = [
        {
            "points": np.asarray(
                [[0.0, 0.0], [50.0, 50.0], [70.0, 70.0]]
            ),
            "scores": np.asarray([0.90, 0.99, 0.01]),
            "embeddings": np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
            ),
            "gt_points": np.asarray([[0.0, 0.0]]),
        }
    ]
    row = evaluate_image_adaptive_prototype_union(
        control,
        prototype,
        control_threshold=0.56,
        prototype_thresholds=[0.90],
        margin_thresholds=[0.2],
        foreground_support_threshold=0.8,
        background_support_threshold=0.05,
        support_match_radius=5.0,
        num_foreground_prototypes=2,
        num_background_prototypes=2,
        match_distance=5.0,
        dedup_interval=5.0,
        near_radius=10.0,
    )[0]
    assert row["foreground_support_count"] == 1
    assert row["added_pred"] == 0


def main():
    test_spherical_kmeans_preserves_two_separated_modes()
    test_binary_auc_reports_margin_separation_direction()
    test_next_action_distinguishes_support_representation_and_precision_failures()
    test_background_support_excludes_low_score_candidates_near_detections()
    test_image_adaptive_prototypes_keep_cell_like_addition_and_reject_background()
    test_local_bank_is_not_ready_when_support_count_is_below_requested_k()
    test_query_candidate_cannot_become_foreground_support_without_baseline_confirmation()
    print("Prototype image-adaptive union tests passed")


if __name__ == "__main__":
    main()
