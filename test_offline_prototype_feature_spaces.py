import numpy as np
import torch

from offline_validate_p2p_prototypes import compare_feature_spaces, evaluate_k


def metric_row(space, balanced_accuracy, foreground_recall, hard_recall):
    return {
        "feature_space": space,
        "k_fg": 4,
        "k_bg": 8,
        "balanced_accuracy": balanced_accuracy,
        "auc": 0.80,
        "foreground_recall": foreground_recall,
        "hard_background_recall": hard_recall,
        "random_background_recall": 0.90,
        "center_cross_similarity_max": 0.75,
    }


def test_configured_raw_projected_comparison_uses_same_k_pair():
    rows = [
        metric_row("raw", 0.68, 0.70, 0.66),
        metric_row("projected", 0.72, 0.73, 0.71),
        {**metric_row("projected", 0.99, 0.99, 0.99), "k_fg": 8},
    ]

    comparison = compare_feature_spaces(rows, 4, 8)

    assert comparison["valid"]
    assert comparison["projected"]["balanced_accuracy"] == 0.72
    assert abs(
        comparison["projected_minus_raw"]["balanced_accuracy"] - 0.04
    ) < 1e-12


def test_comparison_is_invalid_when_projected_space_is_missing():
    comparison = compare_feature_spaces(
        [metric_row("raw", 0.68, 0.70, 0.66)], 4, 8
    )

    assert not comparison["valid"]


def test_hard_only_offline_evaluation_allows_empty_random_background():
    foreground = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    hard_background = np.asarray([[-1.0, 0.0], [-0.9, -0.1]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    payload = {
        "foreground": foreground,
        "hard_background": hard_background,
        "random_background": empty,
        "background": hard_background,
    }

    metrics = evaluate_k(
        payload,
        payload,
        foreground_clusters=1,
        background_clusters=1,
        iterations=3,
        device=torch.device("cpu"),
    )

    assert metrics["foreground_recall"] == 1.0
    assert metrics["hard_background_recall"] == 1.0
    assert metrics["random_background_recall"] is None


def main():
    test_configured_raw_projected_comparison_uses_same_k_pair()
    test_comparison_is_invalid_when_projected_space_is_missing()
    test_hard_only_offline_evaluation_allows_empty_random_background()
    print("Offline prototype feature-space tests passed")


if __name__ == "__main__":
    main()
