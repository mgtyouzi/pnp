import argparse

import numpy as np
import torch

import diagnose_supervised_metric_prototypes as metric_proto


def synthetic_grouped_candidates(seed=7):
    random = np.random.RandomState(seed)
    features = []
    candidate_type = []
    group_index = []
    image_index = []
    image_counter = 0
    for group in range(4):
        group_shift = random.normal(scale=0.15, size=(6,))
        for _ in range(4):
            for class_id, center in enumerate(
                (
                    np.asarray([1.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
                    np.asarray([-0.4, 0.8, 0.0, 0.0, 0.0, 0.0]),
                    np.asarray([-1.0, -0.4, 0.0, 0.0, 0.0, 0.0]),
                )
            ):
                values = center + group_shift + random.normal(scale=0.18, size=(24, 6))
                features.append(values.astype(np.float32))
                candidate_type.append(np.full(24, class_id, dtype=np.uint8))
                group_index.append(np.full(24, group, dtype=np.int16))
                image_index.append(np.full(24, image_counter, dtype=np.int32))
            image_counter += 1
    return {
        "features": np.concatenate(features),
        "candidate_type": np.concatenate(candidate_type),
        "group_index": np.concatenate(group_index),
        "image_index": np.concatenate(image_index),
        "teacher_score": random.uniform(size=sum(map(len, candidate_type))).astype(
            np.float32
        ),
    }


def test_group_split_has_no_image_or_group_leakage():
    data = synthetic_grouped_candidates()
    split = metric_proto.build_fold_split(
        data["candidate_type"],
        data["group_index"],
        data["image_index"],
        held_group=2,
        calibration_fraction=0.25,
        seed=3,
    )
    assert set(data["group_index"][split.train_indices]) == {0, 1, 3}
    assert set(data["group_index"][split.calibration_indices]) == {0, 1, 3}
    assert set(data["group_index"][split.test_indices]) == {2}
    assert not set(data["image_index"][split.train_indices]).intersection(
        set(data["image_index"][split.calibration_indices])
    )


def test_supervised_metric_prototypes_learn_hard_negative_boundary():
    torch.manual_seed(0)
    data = synthetic_grouped_candidates()
    split = metric_proto.build_fold_split(
        data["candidate_type"],
        data["group_index"],
        data["image_index"],
        held_group=3,
        calibration_fraction=0.25,
        seed=0,
    )
    args = argparse.Namespace(
        embedding_dim=4,
        foreground_prototypes=2,
        hard_background_prototypes=2,
        random_background_prototypes=1,
        temperature=0.15,
        learning_rate=0.02,
        weight_decay=1e-4,
        epochs=20,
        batch_size=128,
        hard_negative_weight=2.0,
        diversity_weight=0.02,
        patience=6,
        seed=0,
    )
    model, history, audit = metric_proto.fit_supervised_prototypes(
        data["features"],
        data["candidate_type"],
        split.train_indices,
        split.calibration_indices,
        args,
        torch.device("cpu"),
    )
    scores, assignments = metric_proto.score_supervised_prototypes(
        model, data["features"][split.test_indices], batch_size=256
    )
    test_types = data["candidate_type"][split.test_indices]
    hard = metric_proto.evaluate_scores(
        scores[test_types == metric_proto.TYPE_POSITIVE],
        scores[test_types == metric_proto.TYPE_HARD_NEGATIVE],
    )
    assert hard["auc"] > 0.90
    assert history
    assert audit["selected_epoch"] >= 0
    assert assignments.shape == test_types.shape


def test_gate_rejects_good_random_but_weak_hard_separation():
    decision = metric_proto.make_gate_decision(
        hard_auc=0.58,
        hard_min_auc=0.56,
        random_auc=0.99,
        teacher_hard_auc=0.50,
        ridge_hard_auc=0.64,
    )
    assert not decision["gate_pass"]
    assert "macro_hard_auc" in decision["failures"]


def main():
    test_group_split_has_no_image_or_group_leakage()
    test_supervised_metric_prototypes_learn_hard_negative_boundary()
    test_gate_rejects_good_random_but_weak_hard_separation()
    print("Supervised metric prototype tests passed")


if __name__ == "__main__":
    main()
