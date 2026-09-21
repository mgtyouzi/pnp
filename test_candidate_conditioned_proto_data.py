import numpy as np

from candidate_conditioned_proto_data import (
    CATEGORY_BACKGROUND_FAR_FP,
    CATEGORY_NEAR_MISS_FP,
    CATEGORY_TP,
    build_candidate_feature_matrix,
    build_loso_image_split,
    candidate_conditioned_gate,
    group_balanced_sample_weights,
    score_matched_group_sample,
)


def test_loso_split_has_no_group_or_image_leakage():
    groups = np.repeat(np.arange(3), 8)
    images = np.repeat(np.arange(12), 2)
    split = build_loso_image_split(
        groups,
        images,
        held_group=1,
        calibration_fraction=0.25,
        seed=7,
    )

    assert set(groups[split.test_indices]) == {1}
    assert 1 not in set(groups[split.train_indices])
    assert 1 not in set(groups[split.calibration_indices])
    assert not set(images[split.train_indices]).intersection(
        set(images[split.calibration_indices])
    )
    assert not set(images[split.test_indices]).intersection(
        set(images[split.train_indices]) | set(images[split.calibration_indices])
    )


def test_score_matched_sampling_balances_each_group_and_score_bin():
    categories = np.asarray(
        [
            CATEGORY_TP,
            CATEGORY_TP,
            CATEGORY_BACKGROUND_FAR_FP,
            CATEGORY_BACKGROUND_FAR_FP,
            CATEGORY_NEAR_MISS_FP,
        ]
        * 2,
        dtype=np.int64,
    )
    groups = np.repeat([0, 1], 5)
    margins = np.asarray([0.1, 0.9, 0.2, 0.8, 0.5] * 2, dtype=np.float64)

    selected, audit = score_matched_group_sample(
        np.arange(len(categories)),
        categories,
        groups,
        margins,
        score_bins=2,
        max_per_class_per_bin=8,
        seed=3,
    )

    assert CATEGORY_NEAR_MISS_FP not in set(categories[selected])
    for row in audit:
        assert row["selected_tp"] == row["selected_background_far_fp"]
    assert sum(row["selected_tp"] for row in audit) > 0


def test_score_matched_sampling_is_deterministic():
    categories = np.asarray(
        [CATEGORY_TP] * 20 + [CATEGORY_BACKGROUND_FAR_FP] * 20,
        dtype=np.int64,
    )
    groups = np.zeros(40, dtype=np.int64)
    margins = np.linspace(-2.0, 2.0, 40)
    args = (
        np.arange(40),
        categories,
        groups,
        margins,
    )
    first, _ = score_matched_group_sample(
        *args, score_bins=4, max_per_class_per_bin=3, seed=11
    )
    second, _ = score_matched_group_sample(
        *args, score_bins=4, max_per_class_per_bin=3, seed=11
    )
    assert np.array_equal(first, second)


def test_group_balanced_weights_give_each_slide_equal_total_weight():
    groups = np.asarray([0, 0, 1, 1, 1, 1, 2], dtype=np.int64)
    weights = group_balanced_sample_weights(groups, np.arange(len(groups)))
    totals = [float(weights[groups == group].sum()) for group in range(3)]
    assert np.allclose(totals, totals[0])
    assert np.isclose(weights.mean(), 1.0)


def test_context_feature_matrix_adds_regression_attention_and_offset_information():
    data = {
        "features": np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
        "reg_features": np.asarray([[0.0, 5.0], [6.0, 8.0]], dtype=np.float32),
        "cls_attn": np.asarray([[0.25, 0.75], [0.6, 0.4]], dtype=np.float32),
        "reg_attn": np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        "reg_offset": np.asarray([[64.0, -64.0], [16.0, 0.0]], dtype=np.float32),
    }
    cls_only, cls_source = build_candidate_feature_matrix(data, "cls_only")
    context, context_source = build_candidate_feature_matrix(
        data, "cls_reg_context"
    )

    assert cls_only.shape == (2, 2)
    assert context.shape == (2, 10)
    assert cls_source == "p2p_cls_features_before_final_classifier"
    assert context_source == "p2p_cls_reg_attention_offset_context"
    assert np.allclose(np.linalg.norm(context[:, :2], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(context[:, 2:4], axis=1), 1.0)
    assert np.all(np.abs(context[:, -2:]) <= 2.0)


def test_gate_requires_complementary_and_slide_stable_signal():
    passing = candidate_conditioned_gate(
        prototype_macro_auc=0.68,
        prototype_min_slide_auc=0.58,
        joint_macro_auc=0.84,
        raw_macro_auc=0.81,
        conditional_macro_auc=0.63,
        joint_near_miss_macro_auc=0.76,
        raw_near_miss_macro_auc=0.75,
    )
    assert passing["gate_pass"] is True

    no_complement = candidate_conditioned_gate(
        prototype_macro_auc=0.68,
        prototype_min_slide_auc=0.58,
        joint_macro_auc=0.815,
        raw_macro_auc=0.81,
        conditional_macro_auc=0.63,
        joint_near_miss_macro_auc=0.75,
        raw_near_miss_macro_auc=0.75,
    )
    assert no_complement["gate_pass"] is False
    assert "joint_gain" in no_complement["failures"]

    slide_biased = candidate_conditioned_gate(
        prototype_macro_auc=0.72,
        prototype_min_slide_auc=0.49,
        joint_macro_auc=0.85,
        raw_macro_auc=0.81,
        conditional_macro_auc=0.64,
        joint_near_miss_macro_auc=0.76,
        raw_near_miss_macro_auc=0.75,
    )
    assert slide_biased["gate_pass"] is False
    assert "prototype_min_slide_auc" in slide_biased["failures"]


def main():
    test_loso_split_has_no_group_or_image_leakage()
    test_score_matched_sampling_balances_each_group_and_score_bin()
    test_score_matched_sampling_is_deterministic()
    test_group_balanced_weights_give_each_slide_equal_total_weight()
    test_context_feature_matrix_adds_regression_attention_and_offset_information()
    test_gate_requires_complementary_and_slide_stable_signal()
    print("Candidate-conditioned prototype data tests passed")


if __name__ == "__main__":
    main()
