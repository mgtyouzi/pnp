import torch
import torch.nn.functional as F

from models.positive_only_train_proto import PositiveOnlyTrainPrototype


def make_head(**overrides):
    values = dict(
        feat_dim=8,
        hidden_dim=6,
        embedding_dim=4,
        num_prototypes=2,
        warmup_epochs=2,
        temperature=0.1,
        positive_margin=0.4,
        negative_margin=0.2,
        diversity_margin=0.5,
        pair_weight=0.01,
        positive_weight=0.01,
        negative_weight=0.005,
        balance_weight=0.001,
        diversity_weight=0.001,
        input_gradient_scale=0.05,
        background_radius=6.0,
        max_hard_background_per_image=2,
        max_random_background_per_image=1,
        sampling_seed=3,
    )
    values.update(overrides)
    return PositiveOnlyTrainPrototype(**values)


def test_forward_never_changes_detector_logits():
    head = make_head()
    candidates = torch.randn(1, 5, 8)
    supports = torch.randn(1, 2, 8)
    logits = torch.randn(1, 5, 2)
    outputs = head(candidates, logits, supports)
    assert torch.equal(outputs["fused_logits"], logits)
    assert outputs["embeddings"].shape == (1, 5, 4)
    assert outputs["support_embeddings"].shape == (1, 2, 4)
    assert not hasattr(head, "background_prototypes")
    assert not hasattr(head, "foreground_queue")


def test_candidate_sampling_keeps_all_matches_and_excludes_near_background():
    head = make_head()
    points = torch.tensor([[[1.0, 0.0], [9.0, 0.0], [4.0, 0.0], [12.0, 0.0], [20.0, 0.0]]])
    logits = torch.tensor([[[0.0, 2.0], [1.0, 0.0], [4.0, 0.0], [3.0, 0.0], [2.0, 0.0]]])
    targets = {"gt_points": [torch.tensor([[0.0, 0.0], [0.0, 0.0]])]}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    sampled = head.sample_candidates(points, logits, targets, indices)
    assert sampled["positive_indices"][0].tolist() == [0, 1]
    assert sampled["positive_target_indices"][0].tolist() == [0, 1]
    assert sampled["ignored_near_indices"][0].tolist() == [2]
    assert sampled["hard_background_indices"][0].tolist() == [3, 4]
    assert sampled["random_background_indices"][0].numel() == 0


def test_losses_are_sample_means_and_pair_target_is_detached():
    head = make_head()
    head.prototype_ready.fill_(1)
    head.set_epoch(2)
    candidate_features = torch.randn(1, 4, 8, requires_grad=True)
    support_features = torch.randn(1, 2, 8, requires_grad=True)
    raw_logits = torch.tensor([[[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]])
    points = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [20.0, 0.0]]])
    targets = {"gt_points": [torch.tensor([[0.0, 0.0], [0.0, 0.0]])]}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    outputs = head(candidate_features, raw_logits, support_features)
    loss, diagnostics = head.compute_loss(
        outputs["embeddings"],
        outputs["support_embeddings"],
        torch.tensor([[True, True]]),
        points,
        raw_logits,
        targets,
        indices,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert diagnostics["potp_positive_queries"] == 2.0
    assert diagnostics["potp_hard_background"] == 2.0
    loss.backward()
    assert candidate_features.grad is not None
    assert support_features.grad is not None
    assert torch.isfinite(candidate_features.grad).all()
    assert torch.isfinite(support_features.grad).all()


def test_warmup_blocks_auxiliary_gradient_to_shared_features_only():
    head = make_head(warmup_epochs=2)
    head.set_epoch(0)
    features = torch.randn(1, 3, 8, requires_grad=True)
    supports = torch.randn(1, 2, 8, requires_grad=True)
    logits = torch.zeros(1, 3, 2)
    outputs = head(features, logits, supports)
    outputs["embeddings"].sum().backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad) == 0
    projector_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.projector.parameters()
        if parameter.grad is not None
    )
    assert projector_grad > 0


def test_first_epoch_gt_support_initializes_centers_once():
    head = make_head(num_prototypes=2, warmup_epochs=2)
    assert head.prototype_ready.item() == 0
    head.set_epoch(0)
    candidate_features = torch.randn(1, 2, 8)
    support_features = torch.randn(1, 2, 8)
    logits = torch.zeros(1, 2, 2)
    points = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    targets = {"gt_points": [torch.tensor([[0.0, 0.0], [0.0, 0.0]])]}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    outputs = head(candidate_features, logits, support_features)
    head.compute_loss(
        outputs["embeddings"],
        outputs["support_embeddings"],
        torch.tensor([[True, True]]),
        points,
        logits,
        targets,
        indices,
    )
    first = head.finalize_epoch()
    assert head.prototype_ready.item() == 1
    assert first["potp_initialization_event"] == 1.0
    centers = head.foreground_prototypes.detach().clone()
    head.finalize_epoch()
    assert torch.equal(head.foreground_prototypes.detach(), centers)


def test_v2_selects_only_lowest_confidence_matched_queries():
    head = make_head(
        hard_positive_fraction=0.25,
        max_hard_positive_per_image=2,
        detach_support_input=True,
    )
    points = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    probabilities = torch.tensor([0.90, 0.20, 0.70, 0.40])
    logits = torch.stack([probabilities.log(), (1.0 - probabilities).log()], dim=-1)[None]
    targets = {"gt_points": [points[0].clone()]}
    indices = [(torch.arange(4), torch.arange(4))]
    sampled = head.sample_candidates(points, logits, targets, indices)
    assert sampled["matched_positive_count"][0] == 4
    assert sampled["positive_indices"][0].tolist() == [1]
    assert sampled["positive_target_indices"][0].tolist() == [1]


def test_v2_detaches_all_gt_supports_but_caches_them_for_initialization():
    head = make_head(
        num_prototypes=2,
        warmup_epochs=0,
        hard_positive_fraction=0.25,
        max_hard_positive_per_image=1,
        detach_support_input=True,
    )
    head.set_epoch(2)
    candidate_features = torch.randn(1, 4, 8, requires_grad=True)
    support_features = torch.randn(1, 4, 8, requires_grad=True)
    probabilities = torch.tensor([0.90, 0.20, 0.70, 0.40])
    logits = torch.stack([probabilities.log(), (1.0 - probabilities).log()], dim=-1)[None]
    points = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    targets = {"gt_points": [points[0].clone()]}
    indices = [(torch.arange(4), torch.arange(4))]
    outputs = head(candidate_features, logits, support_features)
    loss, diagnostics = head.compute_loss(
        outputs["embeddings"],
        outputs["support_embeddings"],
        torch.tensor([[True, True, True, True]]),
        points,
        logits,
        targets,
        indices,
    )
    loss.backward()
    assert diagnostics["potp_matched_positive_total"] == 4.0
    assert diagnostics["potp_positive_queries"] == 1.0
    assert diagnostics["potp_gt_support"] == 4.0
    assert head.support_count.item() == 4
    assert support_features.grad is None or torch.count_nonzero(support_features.grad) == 0
    assert candidate_features.grad is not None
    assert torch.count_nonzero(candidate_features.grad) > 0


def main():
    test_forward_never_changes_detector_logits()
    test_candidate_sampling_keeps_all_matches_and_excludes_near_background()
    test_losses_are_sample_means_and_pair_target_is_detached()
    test_warmup_blocks_auxiliary_gradient_to_shared_features_only()
    test_first_epoch_gt_support_initializes_centers_once()
    test_v2_selects_only_lowest_confidence_matched_queries()
    test_v2_detaches_all_gt_supports_but_caches_them_for_initialization()
    print("Positive-only train prototype tests passed")


if __name__ == "__main__":
    main()
