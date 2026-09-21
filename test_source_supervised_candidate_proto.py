import torch
import torch.nn.functional as F

from models.source_supervised_candidate_proto import (
    SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION,
    SourceSupervisedCandidatePrototype,
)


def make_head(**overrides):
    values = dict(
        feat_dim=4,
        num_foreground_prototypes=2,
        num_background_prototypes=3,
        foreground_queue_size=16,
        background_queue_size=16,
        max_hard_positive_per_image=1,
        max_random_positive_per_image=1,
        max_hard_background_per_image=1,
        max_random_background_per_image=1,
        positive_radius=3.0,
        background_radius=6.0,
        prototype_temperature=0.1,
        prototype_margin=0.1,
        prototype_momentum=0.5,
        alpha_max=0.25,
        confidence_threshold=0.05,
        fusion_clip=0.5,
        sampling_seed=7,
    )
    values.update(overrides)
    return SourceSupervisedCandidatePrototype(**values)


def test_version_and_no_projector_contract():
    head = make_head()
    assert SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION
    assert not hasattr(head, "projector")
    assert head.foreground_prototypes.shape == (2, 4)
    assert head.background_prototypes.shape == (3, 4)


def test_logmeanexp_removes_prototype_count_bias():
    head = make_head()
    direction = F.normalize(torch.tensor([1.0, 2.0, 0.0, 0.0]), dim=0)
    with torch.no_grad():
        head.foreground_prototypes.copy_(direction.repeat(2, 1))
        head.background_prototypes.copy_(direction.repeat(3, 1))
        head.prototype_ready.fill_(1)
    embedding = direction.reshape(1, 1, -1)
    foreground, background = head.class_scores(embedding)
    assert torch.allclose(foreground, background, atol=1e-6)


def test_unready_bank_leaves_logits_unchanged():
    head = make_head()
    features = torch.randn(1, 5, 4)
    raw_logits = torch.randn(1, 5, 2)
    outputs = head(features, raw_logits)
    assert torch.equal(outputs["fused_logits"], raw_logits)


def test_confident_prototype_margin_moves_logits_in_expected_direction():
    head = make_head(alpha_initial=0.2)
    with torch.no_grad():
        head.foreground_prototypes.copy_(F.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]
        ]), dim=-1))
        head.background_prototypes.copy_(F.normalize(torch.tensor([
            [0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.8, 0.2, 0.0]
        ]), dim=-1))
        head.prototype_ready.fill_(1)
    raw = torch.zeros(1, 2, 2)
    features = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    fused = head(features, raw)["fused_logits"]
    assert fused[0, 0, 0] > raw[0, 0, 0]
    assert fused[0, 1, 0] < raw[0, 1, 0]


def test_fused_auxiliary_loss_does_not_bypass_feature_gradient_scaling():
    head = make_head(alpha_initial=0.2)
    with torch.no_grad():
        head.foreground_prototypes.copy_(F.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]
        ]), dim=-1))
        head.background_prototypes.copy_(F.normalize(torch.tensor([
            [0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.8, 0.2, 0.0]
        ]), dim=-1))
        head.prototype_ready.fill_(1)
    raw = torch.zeros(1, 2, 2, requires_grad=True)
    features = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    fused = head(features, raw)["fused_logits"]
    F.cross_entropy(fused.reshape(-1, 2), torch.tensor([0, 1])).backward()
    assert raw.grad is None or torch.count_nonzero(raw.grad) == 0
    assert features.grad is not None and torch.count_nonzero(features.grad) > 0


def test_sampling_uses_hard_and_random_positive_and_background_groups():
    head = make_head()
    points = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [9.0, 0.0], [4.0, 0.0], [12.0, 0.0], [20.0, 0.0]]])
    raw_logits = torch.tensor([[[0.0, 3.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [4.0, 0.0], [5.0, 0.0]]])
    targets = {"gt_points": [torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])]}
    indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]
    sampled = head.sample_candidates(points, raw_logits, targets, indices)
    assert sampled["hard_positive_indices"][0].tolist() == [0]
    assert sampled["random_positive_indices"][0].tolist() == [1]
    assert sampled["rejected_match_indices"][0].tolist() == [2]
    assert sampled["ignored_near_indices"][0].tolist() == [3]
    assert sampled["hard_background_indices"][0].tolist() == [5]
    assert sampled["random_background_indices"][0].tolist() == [4]


def test_balanced_candidate_loss_is_invariant_to_positive_duplication():
    head = make_head(num_foreground_prototypes=1, num_background_prototypes=1)
    with torch.no_grad():
        head.foreground_prototypes.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        head.background_prototypes.copy_(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
        head.prototype_ready.fill_(1)
    positive = F.normalize(torch.tensor([[0.9, 0.1, 0.0, 0.0]]), dim=-1)
    background = F.normalize(torch.tensor([[0.1, 0.9, 0.0, 0.0]]), dim=-1)
    loss_a, _ = head.candidate_losses(positive, background, None, None)
    loss_b, diagnostics = head.candidate_losses(positive.repeat(9, 1), background, None, None)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)
    assert diagnostics["sscp_balanced_count"] == 1


def main():
    test_version_and_no_projector_contract()
    test_logmeanexp_removes_prototype_count_bias()
    test_unready_bank_leaves_logits_unchanged()
    test_confident_prototype_margin_moves_logits_in_expected_direction()
    test_fused_auxiliary_loss_does_not_bypass_feature_gradient_scaling()
    test_sampling_uses_hard_and_random_positive_and_background_groups()
    test_balanced_candidate_loss_is_invariant_to_positive_duplication()
    print("Source-supervised candidate prototype tests passed")


if __name__ == "__main__":
    main()
