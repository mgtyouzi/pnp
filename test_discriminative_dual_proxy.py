import torch
import torch.nn.functional as F

from models.discriminative_dual_proxy import DiscriminativeDualProxy


def make_head(**overrides):
    values = dict(
        feat_dim=8,
        embedding_dim=4,
        num_fg=2,
        num_bg=2,
        warmup_epochs=1,
        foreground_queue_size=16,
        background_queue_size=16,
        max_foreground_per_image=2,
        max_hard_background_per_image=1,
        max_random_background_per_image=1,
        prototype_momentum=0.5,
        projector_momentum=0.9,
        positive_radius=3.0,
        background_radius=6.0,
        temperature=0.2,
        supcon_weight=0.5,
        separation_weight=0.5,
        separation_margin=0.1,
        balance_weight=0.05,
        minimum_assignment_share=0.1,
        sampling_seed=7,
    )
    values.update(overrides)
    return DiscriminativeDualProxy(**values)


def make_projected(head):
    query_features = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]]
    )
    support_features = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]]
    )
    raw_logits = torch.tensor(
        [[[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [4.0, 0.0], [5.0, 0.0]]]
    )
    return raw_logits, head(query_features, raw_logits, support_features)


def test_sampling_contract_excludes_near_and_far_matched_candidates():
    head = make_head()
    raw_logits, outputs = make_projected(head)
    points = torch.tensor(
        [[[1.0, 0.0], [9.0, 0.0], [4.0, 0.0], [12.0, 0.0], [20.0, 0.0]]]
    )
    targets = {"gt_points": [torch.tensor([[0.0, 0.0], [0.0, 0.0]])]}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    sampled = head.sample_candidates(points, raw_logits, targets, indices)

    assert sampled["reliable_indices"][0].tolist() == [0]
    assert sampled["rejected_match_indices"][0].tolist() == [1]
    assert sampled["ignored_near_indices"][0].tolist() == [2]
    assert sorted(sampled["background_indices"][0].tolist()) == [3, 4]
    assert sampled["hard_background_indices"][0].tolist() == [4]
    assert sampled["random_background_indices"][0].tolist() == [3]
    assert not set(sampled["background_indices"][0].tolist()).intersection({0, 1, 2})
    assert torch.equal(outputs["fused_logits"], raw_logits)


def test_class_reservoirs_are_independent_and_bounded():
    head = make_head(foreground_queue_size=3, background_queue_size=2)
    head.begin_epoch()
    foreground = F.normalize(torch.arange(1, 21).reshape(5, 4).float(), dim=-1)
    background = F.normalize(torch.arange(21, 41).reshape(5, 4).float(), dim=-1)
    head._cache_foreground(foreground)
    head._cache_background(background)
    assert head.foreground_queue_count.item() == 3
    assert head.background_queue_count.item() == 2
    assert head.foreground_seen_count.item() == 5
    assert head.background_seen_count.item() == 5
    assert not torch.equal(head._local_foreground()[:2], head._local_background())


def test_foreground_reservoir_contains_gt_support_and_reliable_queries():
    head = make_head(warmup_epochs=0, max_foreground_per_image=4)
    raw_logits, outputs = make_projected(head)
    points = torch.tensor(
        [[[1.0, 0.0], [9.0, 0.0], [4.0, 0.0], [12.0, 0.0], [20.0, 0.0]]]
    )
    targets = {"gt_points": [torch.tensor([[0.0, 0.0], [0.0, 0.0]])]}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    head.begin_epoch()
    head.compute_loss_and_cache(
        outputs["embeddings"],
        outputs["momentum_embeddings"],
        outputs["support_embeddings"],
        outputs["momentum_support_embeddings"],
        torch.ones(1, 2, dtype=torch.bool),
        points,
        raw_logits,
        targets,
        indices,
    )
    assert head.foreground_queue_count.item() == 3
    assert head.background_queue_count.item() == 2


def test_separable_geometry_has_positive_class_gaps_and_finite_gradient():
    head = make_head(warmup_epochs=0)
    with torch.no_grad():
        head.foreground_prototypes.copy_(F.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]
        ]), dim=-1))
        head.background_prototypes.copy_(F.normalize(torch.tensor([
            [0.0, 1.0, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0]
        ]), dim=-1))
        head.prototype_ready.fill_(1)
    foreground = F.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]), dim=-1).requires_grad_(True)
    background = F.normalize(torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0]]), dim=-1).requires_grad_(True)
    loss, diagnostics = head.compute_metric_loss(foreground, background)
    loss.backward()
    assert torch.isfinite(loss)
    assert foreground.grad is not None
    assert background.grad is not None
    assert diagnostics["dualproxy_fg_similarity_gap"] > 0
    assert diagnostics["dualproxy_bg_similarity_gap"] > 0
    assert diagnostics["dualproxy_metric_foreground"] == 2
    assert diagnostics["dualproxy_metric_background"] == 2
    assert 0.0 <= diagnostics["dualproxy_separation_fraction"] <= 1.0


def test_metric_loss_is_invariant_to_foreground_duplication():
    head = make_head(
        warmup_epochs=0,
        num_fg=1,
        num_bg=1,
        supcon_weight=0.0,
        separation_weight=0.0,
        balance_weight=0.0,
    )
    with torch.no_grad():
        head.foreground_prototypes.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        head.background_prototypes.copy_(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
        head.prototype_ready.fill_(1)
    foreground = F.normalize(
        torch.tensor([[0.9, 0.1, 0.0, 0.0]]), dim=-1
    )
    background = F.normalize(
        torch.tensor([[0.8, 0.2, 0.0, 0.0]]), dim=-1
    )
    balanced_loss, _ = head.compute_metric_loss(foreground, background)
    duplicated_loss, diagnostics = head.compute_metric_loss(
        foreground.repeat(11, 1), background
    )
    assert torch.allclose(balanced_loss, duplicated_loss, atol=1e-6)
    assert diagnostics["dualproxy_metric_foreground"] == 1
    assert diagnostics["dualproxy_metric_background"] == 1


def test_separation_term_changes_projector_gradient_for_ambiguous_embeddings():
    base = make_head(warmup_epochs=0, supcon_weight=0.0, balance_weight=0.0)
    with torch.no_grad():
        base.foreground_prototypes.copy_(F.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]
        ]), dim=-1))
        base.background_prototypes.copy_(F.normalize(torch.tensor([
            [0.7, 0.7, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0]
        ]), dim=-1))
        base.prototype_ready.fill_(1)
    no_separation = make_head(
        warmup_epochs=0, supcon_weight=0.0, balance_weight=0.0,
        separation_weight=0.0,
    )
    no_separation.load_state_dict(base.state_dict())
    with_separation = make_head(
        warmup_epochs=0, supcon_weight=0.0, balance_weight=0.0,
        separation_weight=1.0,
    )
    with_separation.load_state_dict(base.state_dict())
    foreground_a = F.normalize(torch.tensor([[0.7, 0.7, 0.0, 0.0]]), dim=-1).requires_grad_(True)
    background_a = F.normalize(torch.tensor([[0.9, 0.1, 0.0, 0.0]]), dim=-1).requires_grad_(True)
    foreground_b = foreground_a.detach().clone().requires_grad_(True)
    background_b = background_a.detach().clone().requires_grad_(True)
    loss_a, _ = no_separation.compute_metric_loss(foreground_a, background_a)
    loss_b, diagnostics = with_separation.compute_metric_loss(foreground_b, background_b)
    loss_a.backward()
    loss_b.backward()
    assert diagnostics["dualproxy_loss_separation"] > 0
    assert not torch.allclose(foreground_a.grad, foreground_b.grad)
    assert not torch.allclose(background_a.grad, background_b.grad)


def test_finalize_initializes_both_banks_and_reports_geometry():
    head = make_head(warmup_epochs=1)
    head.begin_epoch()
    head._cache_foreground(F.normalize(torch.tensor([
        [1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.9, 0.1],
    ]), dim=-1))
    head._cache_background(F.normalize(torch.tensor([
        [0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 0.9],
    ]), dim=-1))
    state = head.finalize_epoch()
    assert head.prototype_ready.item() == 1
    assert state["prototype_initialization_event"] == 1
    assert state["foreground_effective_prototypes"] >= 1
    assert state["background_effective_prototypes"] >= 1
    assert "cross_bank_similarity_max" in state


def test_passed_bank_payload_restores_projector_and_both_proxy_banks():
    source = make_head(warmup_epochs=0)
    with torch.no_grad():
        source.projector[1].weight.fill_(0.125)
        source.foreground_prototypes.copy_(F.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
        ]), dim=-1))
        source.background_prototypes.copy_(F.normalize(torch.tensor([
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        ]), dim=-1))
        source.prototype_ready.fill_(1)
    payload = {
        "implementation_version": "discriminative_dual_proxy_v2_20260831",
        "checkpoint": "/tmp/gate.pth",
        "prototype_state": source.state_dict(),
        "gate": {"gate_pass": True},
    }

    restored = make_head(warmup_epochs=0)
    metadata = restored.load_bank_payload(payload)

    assert metadata["gate_pass"] is True
    assert metadata["checkpoint"] == "/tmp/gate.pth"
    assert restored.prototype_ready.item() == 1
    assert torch.equal(restored.foreground_prototypes, source.foreground_prototypes)
    assert torch.equal(restored.background_prototypes, source.background_prototypes)
    assert torch.equal(restored.projector[1].weight, source.projector[1].weight)


def test_rejected_or_unready_bank_payload_is_not_loaded():
    head = make_head(warmup_epochs=0)
    payload = {
        "implementation_version": "discriminative_dual_proxy_v2_20260831",
        "prototype_state": head.state_dict(),
        "gate": {"gate_pass": False},
    }
    try:
        head.load_bank_payload(payload)
    except RuntimeError as error:
        assert "did not pass" in str(error)
    else:
        raise AssertionError("a rejected bank must not be loaded")


def main():
    test_sampling_contract_excludes_near_and_far_matched_candidates()
    test_class_reservoirs_are_independent_and_bounded()
    test_foreground_reservoir_contains_gt_support_and_reliable_queries()
    test_separable_geometry_has_positive_class_gaps_and_finite_gradient()
    test_metric_loss_is_invariant_to_foreground_duplication()
    test_separation_term_changes_projector_gradient_for_ambiguous_embeddings()
    test_finalize_initializes_both_banks_and_reports_geometry()
    test_passed_bank_payload_restores_projector_and_both_proxy_banks()
    test_rejected_or_unready_bank_payload_is_not_loaded()
    print("Discriminative dual-proxy tensor tests passed")


if __name__ == "__main__":
    main()
