import torch

from models.prototype_guided_ranking import PrototypeGuidedRanking


def make_head(**overrides):
    values = dict(
        feat_dim=8,
        hidden_dim=6,
        embedding_dim=4,
        num_prototypes=2,
        warmup_epochs=2,
        prototype_temperature=0.1,
        ranking_temperature=0.1,
        prototype_margin=0.1,
        classification_margin=0.2,
        prototype_rank_weight=0.005,
        classification_rank_weight=0.01,
        metric_weight=0.005,
        center_weight=0.005,
        balance_weight=0.001,
        diversity_weight=0.001,
        input_gradient_scale=0.05,
        positive_radius=3.0,
        background_radius=6.0,
        hard_positive_fraction=0.25,
        max_pairs_per_image=2,
        max_random_background_per_image=1,
        support_reservoir_size=16,
        kmeans_iterations=5,
        sampling_seed=3,
    )
    values.update(overrides)
    return PrototypeGuidedRanking(**values)


def logits_from_cell_probabilities(probabilities):
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    return torch.stack(
        [probabilities.clamp_min(1e-6).log(),
         (1.0 - probabilities).clamp_min(1e-6).log()],
        dim=-1,
    )[None]


def test_forward_preserves_detector_logits_and_exposes_raw_features():
    head = make_head()
    candidates = torch.randn(1, 5, 8)
    supports = torch.randn(1, 2, 8)
    logits = torch.randn(1, 5, 2)
    outputs = head(candidates, logits, supports)
    assert torch.equal(outputs["fused_logits"], logits)
    assert outputs["candidate_raw_features"] is candidates
    assert outputs["support_raw_features"] is supports
    assert outputs["embeddings"].shape == (1, 5, 4)
    assert outputs["support_embeddings"].shape == (1, 2, 4)


def test_sampling_rejects_inaccurate_matches_and_keeps_three_distance_zones():
    head = make_head(hard_positive_fraction=1.0, max_pairs_per_image=4)
    points = torch.tensor([[[1.0, 0.0], [14.0, 0.0], [15.0, 0.0],
                            [45.0, 0.0], [50.0, 0.0], [70.0, 0.0]]])
    logits = logits_from_cell_probabilities([0.2, 0.1, 0.8, 0.9, 0.7, 0.6])
    targets = {"gt_points": [torch.tensor([[0.0, 0.0], [10.0, 0.0]])]}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    sampled = head.sample_candidates(
        points, logits, torch.randn(1, 6, 4), targets, indices
    )
    assert sampled["reliable_positive_indices"][0].tolist() == [0]
    assert sampled["rejected_matched_indices"][0].tolist() == [1]
    assert sampled["ignored_near_indices"][0].tolist() == [2]
    assert sampled["eligible_background_indices"][0].tolist() == [3, 4, 5]
    assert sampled["hard_background_indices"][0].numel() == 1


def test_hard_positive_and_hard_background_counts_are_one_to_one():
    head = make_head(hard_positive_fraction=0.25, max_pairs_per_image=8)
    head.prototype_ready.fill_(1)
    points = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
                            [20.0, 0.0], [30.0, 0.0], [40.0, 0.0], [50.0, 0.0]]])
    logits = logits_from_cell_probabilities([0.9, 0.2, 0.7, 0.4, 0.8, 0.6, 0.5, 0.3])
    targets = {"gt_points": [points[0, :4].clone()]}
    indices = [(torch.arange(4), torch.arange(4))]
    embeddings = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0], [-0.9, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0],
        [-0.8, 0.2, 0.0, 0.0], [-0.7, 0.3, 0.0, 0.0],
    ]])
    sampled = head.sample_candidates(points, logits, embeddings, targets, indices)
    assert sampled["reliable_positive_count"][0] == 4
    assert sampled["hard_positive_indices"][0].tolist() == [1]
    assert sampled["hard_background_indices"][0].numel() == 1


def test_raw_gt_features_are_cached_and_projected_only_at_epoch_end():
    head = make_head(num_prototypes=2, support_reservoir_size=8)
    candidates = torch.randn(1, 2, 8)
    supports = torch.randn(1, 3, 8)
    logits = torch.zeros(1, 2, 2)
    points = torch.tensor([[[0.0, 0.0], [20.0, 0.0]]])
    targets = {"gt_points": [torch.tensor([[0.0, 0.0]])]}
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    outputs = head(candidates, logits, supports)
    loss, diagnostics = head.compute_loss(
        outputs["embeddings"], outputs["support_embeddings"],
        outputs["candidate_raw_features"], outputs["support_raw_features"],
        torch.tensor([[True, True, True]]), points, logits, targets, indices,
    )
    assert torch.isfinite(loss)
    assert diagnostics["pgrp_gt_support"] == 3.0
    assert head.support_reservoir.shape == (8, 8)
    assert head.support_count.item() == 3
    assert head.prototype_ready.item() == 0
    state = head.finalize_epoch()
    assert head.prototype_ready.item() == 1
    assert state["pgrp_initialization_event"] == 1.0


def test_ranking_loss_routes_shared_gradient_only_after_warmup():
    head = make_head(warmup_epochs=2, hard_positive_fraction=1.0)
    head.prototype_ready.fill_(1)
    candidates = torch.randn(1, 4, 8, requires_grad=True)
    supports = torch.randn(1, 1, 8, requires_grad=True)
    raw_logits = logits_from_cell_probabilities([0.2, 0.9, 0.8, 0.7]).requires_grad_()
    points = torch.tensor([[[0.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]])
    targets = {"gt_points": [torch.tensor([[0.0, 0.0]])]}
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    head.set_epoch(1)
    outputs = head(candidates, raw_logits, supports)
    loss, _ = head.compute_loss(
        outputs["embeddings"], outputs["support_embeddings"],
        outputs["candidate_raw_features"], outputs["support_raw_features"],
        torch.tensor([[True]]), points, raw_logits, targets, indices,
    )
    loss.backward()
    assert candidates.grad is not None and torch.count_nonzero(candidates.grad) == 0
    assert supports.grad is None or torch.count_nonzero(supports.grad) == 0
    assert raw_logits.grad is not None and torch.count_nonzero(raw_logits.grad) == 0
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in head.projector.parameters()
        if parameter.grad is not None
    ) > 0

    head.zero_grad(set_to_none=True)
    candidates.grad = None
    raw_logits.grad = None
    head.set_epoch(2)
    outputs = head(candidates, raw_logits, supports)
    loss, diagnostics = head.compute_loss(
        outputs["embeddings"], outputs["support_embeddings"],
        outputs["candidate_raw_features"], outputs["support_raw_features"],
        torch.tensor([[True]]), points, raw_logits, targets, indices,
    )
    loss.backward()
    assert diagnostics["pgrp_shared_gradient_scale"] == 0.05
    assert candidates.grad is not None and torch.count_nonzero(candidates.grad) > 0
    assert raw_logits.grad is not None and torch.count_nonzero(raw_logits.grad) > 0
    assert supports.grad is None or torch.count_nonzero(supports.grad) == 0


def main():
    test_forward_preserves_detector_logits_and_exposes_raw_features()
    test_sampling_rejects_inaccurate_matches_and_keeps_three_distance_zones()
    test_hard_positive_and_hard_background_counts_are_one_to_one()
    test_raw_gt_features_are_cached_and_projected_only_at_epoch_end()
    test_ranking_loss_routes_shared_gradient_only_after_warmup()
    print("Prototype-guided ranking tests passed")


if __name__ == "__main__":
    main()
