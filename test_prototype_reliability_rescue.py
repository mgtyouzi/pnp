import torch

from models.prototype_reliability_rescue import PrototypeReliabilityRescue


def make_head(**overrides):
    values = dict(
        feat_dim=8,
        hidden_dim=6,
        embedding_dim=4,
        num_prototypes=2,
        warmup_epochs=2,
        prototype_temperature=0.1,
        target_margin=0.2,
        margin_temperature=0.2,
        rescue_weight=0.005,
        logit_gradient_scale=0.025,
        prototype_rank_weight=0.005,
        metric_weight=0.005,
        center_weight=0.005,
        balance_weight=0.001,
        diversity_weight=0.001,
        positive_radius=3.0,
        background_radius=6.0,
        maximum_cell_probability=0.55,
        max_rescue_per_image=2,
        max_random_background_per_image=1,
        support_reservoir_size=16,
        kmeans_iterations=5,
        sampling_seed=3,
    )
    values.update(overrides)
    return PrototypeReliabilityRescue(**values)


def logits_from_cell_probabilities(probabilities):
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    return torch.stack(
        [
            probabilities.clamp_min(1e-6).log(),
            (1.0 - probabilities).clamp_min(1e-6).log(),
        ],
        dim=-1,
    )[None]


def test_forward_preserves_detector_logits():
    head = make_head()
    candidates = torch.randn(1, 4, 8)
    supports = torch.randn(1, 2, 8)
    logits = torch.randn(1, 4, 2)
    outputs = head(candidates, logits, supports)
    assert outputs["fused_logits"] is logits
    assert outputs["candidate_raw_features"] is candidates
    assert outputs["support_raw_features"] is supports


def test_sampling_only_selects_reliable_matched_low_score_candidates():
    head = make_head(max_rescue_per_image=4)
    head.prototype_ready.fill_(1)
    head.support_score_low.fill_(-2.0)
    head.support_score_high.fill_(1.0)
    points = torch.tensor(
        [[[1.0, 0.0], [14.0, 0.0], [15.0, 0.0], [40.0, 0.0]]]
    )
    logits = logits_from_cell_probabilities([0.2, 0.3, 0.9, 0.8])
    targets = {"gt_points": [torch.tensor([[0.0, 0.0], [10.0, 0.0]])]}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    embeddings = torch.nn.functional.normalize(torch.randn(1, 4, 4), dim=-1)

    warmup_sampled = head.sample_candidates(
        points, logits, embeddings, targets, indices
    )
    assert warmup_sampled["rescue_indices"][0].numel() == 0

    head.set_epoch(head.warmup_epochs)
    sampled = head.sample_candidates(points, logits, embeddings, targets, indices)
    assert sampled["reliable_positive_indices"][0].tolist() == [0]
    assert sampled["rejected_matched_indices"][0].tolist() == [1]
    assert sampled["rescue_indices"][0].tolist() == [0]
    assert sampled["ignored_near_indices"][0].tolist() == [2]
    assert sampled["eligible_background_indices"][0].tolist() == [3]


def test_rescue_loss_has_no_background_logit_gradient():
    head = make_head(warmup_epochs=0, max_rescue_per_image=2)
    head.prototype_ready.fill_(1)
    head.support_score_low.fill_(-1.0)
    head.support_score_high.fill_(1.0)
    candidates = torch.randn(1, 3, 8, requires_grad=True)
    supports = torch.randn(1, 1, 8)
    logits = logits_from_cell_probabilities([0.2, 0.9, 0.8]).requires_grad_()
    points = torch.tensor([[[0.0, 0.0], [20.0, 0.0], [40.0, 0.0]]])
    targets = {"gt_points": [torch.tensor([[0.0, 0.0]])]}
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    outputs = head(candidates, logits, supports)
    loss, diagnostics = head.compute_loss(
        outputs["embeddings"],
        outputs["support_embeddings"],
        outputs["candidate_raw_features"],
        outputs["support_raw_features"],
        torch.tensor([[True]]),
        points,
        logits,
        targets,
        indices,
    )
    loss.backward()
    assert diagnostics["prr_detector_background_updates"] == 0.0
    assert diagnostics["prr_gt_support"] == 1.0
    assert torch.count_nonzero(logits.grad[0, 0]) > 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0


def test_reliability_weights_are_detached_from_projector():
    head = make_head(warmup_epochs=0)
    embeddings = torch.nn.functional.normalize(
        torch.randn(3, 4, requires_grad=True), dim=-1
    )
    probability = torch.tensor([0.2, 0.3, 0.4])
    weights, _ = head.reliability_weights(embeddings, probability)
    assert not weights.requires_grad


def test_finalize_epoch_calibrates_support_quantiles():
    head = make_head(num_prototypes=2, support_reservoir_size=8)
    head._cache_raw_support(torch.randn(8, 8))
    state = head.finalize_epoch()
    assert head.prototype_ready.item() == 1
    assert state["prr_initialization_event"] == 1.0
    assert torch.isfinite(head.support_score_low)
    assert torch.isfinite(head.support_score_high)
    assert head.support_score_high.item() > head.support_score_low.item()


def main():
    test_forward_preserves_detector_logits()
    test_sampling_only_selects_reliable_matched_low_score_candidates()
    test_rescue_loss_has_no_background_logit_gradient()
    test_reliability_weights_are_detached_from_projector()
    test_finalize_epoch_calibrates_support_quantiles()
    print("Prototype reliability rescue tests passed")


if __name__ == "__main__":
    main()
