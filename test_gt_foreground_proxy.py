from types import SimpleNamespace

import torch

from models.gt_foreground_proxy import GTForegroundProxy


def make_head(**overrides):
    values = dict(
        feat_dim=8,
        embedding_dim=4,
        num_prototypes=2,
        warmup_epochs=1,
        support_queue_size=32,
        max_support_per_image=2,
        prototype_momentum=0.5,
        projector_momentum=0.9,
        query_radius=3.0,
        background_radius=6.0,
        max_hard_background_per_image=2,
        background_margin=0.2,
        minimum_assignment_share=0.1,
        dead_prototype_patience=2,
        temperature=0.2,
        align_weight=1.0,
        support_weight=1.0,
        query_weight=1.0,
        background_weight=1.0,
        balance_weight=0.05,
        sampling_seed=7,
    )
    values.update(overrides)
    return GTForegroundProxy(**values)


def make_batch(head):
    query_features = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    support_features = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    raw_logits = torch.tensor(
        [[[3.0, 0.0], [2.0, 0.0], [4.0, 0.0], [1.0, 0.0]]]
    )
    projected = head(query_features, raw_logits, support_features)
    return query_features, support_features, raw_logits, projected


def test_forward_never_changes_detector_logits():
    head = make_head()
    _, _, raw_logits, outputs = make_batch(head)
    assert torch.equal(outputs["fused_logits"], raw_logits)
    assert outputs["embeddings"].shape == (1, 4, 4)
    assert outputs["support_embeddings"].shape == (1, 2, 4)
    assert outputs["momentum_support_embeddings"].shape == (1, 2, 4)
    assert outputs["support_embeddings"].requires_grad
    assert not outputs["momentum_support_embeddings"].requires_grad


def test_candidate_contract_separates_reliable_near_and_far_samples():
    head = make_head()
    _, _, raw_logits, outputs = make_batch(head)
    points = torch.tensor([[[1.0, 0.0], [5.0, 0.0], [12.0, 0.0], [20.0, 0.0]]])
    anchors = points.clone()
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0], [5.0, 0.0]])],
        "gt_labels": [torch.zeros(2, dtype=torch.long)],
    }
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    support_mask = torch.tensor([[True, True]])

    _, diagnostics = head.compute_loss_and_cache(
        outputs["embeddings"],
        outputs["support_embeddings"],
        outputs["momentum_support_embeddings"],
        support_mask,
        points,
        raw_logits,
        targets,
        indices,
        anchor_points=anchors,
    )

    assert diagnostics["gtproxy_reliable_query"] == 2
    assert diagnostics["gtproxy_rejected_query"] == 0
    assert diagnostics["gtproxy_hard_background"] == 2
    assert diagnostics["gtproxy_ignored_near"] == 0


def test_far_matched_query_is_rejected_and_not_relabelled_as_background():
    head = make_head()
    _, _, raw_logits, outputs = make_batch(head)
    points = torch.tensor([[[8.0, 0.0], [5.0, 0.0], [12.0, 0.0], [20.0, 0.0]]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0], [5.0, 0.0]])],
        "gt_labels": [torch.zeros(2, dtype=torch.long)],
    }
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    _, diagnostics = head.compute_loss_and_cache(
        outputs["embeddings"],
        outputs["support_embeddings"],
        outputs["momentum_support_embeddings"],
        torch.tensor([[True, True]]),
        points,
        raw_logits,
        targets,
        indices,
        anchor_points=points,
    )

    assert diagnostics["gtproxy_reliable_query"] == 1
    assert diagnostics["gtproxy_rejected_query"] == 1
    assert diagnostics["gtproxy_hard_background"] == 2


def test_warmup_initializes_foreground_prototypes_from_gt_support_only():
    head = make_head(warmup_epochs=1)
    _, _, raw_logits, outputs = make_batch(head)
    points = torch.tensor([[[1.0, 0.0], [5.0, 0.0], [12.0, 0.0], [20.0, 0.0]]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0], [5.0, 0.0]])],
        "gt_labels": [torch.zeros(2, dtype=torch.long)],
    }
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    head.begin_epoch()
    loss, diagnostics = head.compute_loss_and_cache(
        outputs["embeddings"],
        outputs["support_embeddings"],
        outputs["momentum_support_embeddings"],
        torch.tensor([[True, True]]),
        points,
        raw_logits,
        targets,
        indices,
        anchor_points=points,
    )
    assert torch.isfinite(loss)
    assert diagnostics["prototype_ready"] == 0
    state = head.finalize_epoch()

    assert state["prototype_initialization_event"] == 1
    assert head.prototype_ready.item() == 1
    assert head.prototypes.shape == (2, 4)
    assert torch.allclose(head.prototypes.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_initialized_loss_updates_momentum_projector_after_optimizer_step():
    head = make_head(warmup_epochs=0)
    with torch.no_grad():
        head.prototypes.copy_(torch.nn.functional.normalize(torch.eye(2, 4), dim=-1))
        head.prototype_ready.fill_(1)
        for parameter in head.projector.parameters():
            parameter.add_(0.1)
    before = [value.clone() for value in head.momentum_projector.parameters()]
    diagnostics = head.maybe_refresh_prototypes()
    after = list(head.momentum_projector.parameters())
    assert diagnostics["prototype_refresh_event"] == 0
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_assignment_balance_has_a_student_gradient():
    head = make_head(
        warmup_epochs=0,
        align_weight=0.0,
        support_weight=0.0,
        query_weight=0.0,
        background_weight=0.0,
        balance_weight=1.0,
    )
    with torch.no_grad():
        head.prototypes.copy_(torch.nn.functional.normalize(torch.eye(2, 4), dim=-1))
        head.prototype_ready.fill_(1)
    _, _, raw_logits, outputs = make_batch(head)
    points = torch.tensor([[[1.0, 0.0], [5.0, 0.0], [12.0, 0.0], [20.0, 0.0]]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0], [5.0, 0.0]])],
        "gt_labels": [torch.zeros(2, dtype=torch.long)],
    }
    loss, _ = head.compute_loss_and_cache(
        outputs["embeddings"],
        outputs["support_embeddings"],
        outputs["momentum_support_embeddings"],
        torch.tensor([[True, True]]),
        points,
        raw_logits,
        targets,
        [(torch.tensor([0, 1]), torch.tensor([0, 1]))],
    )
    loss.backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.projector.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0


def test_support_bank_uses_a_bounded_epoch_reservoir():
    head = make_head(support_queue_size=4, max_support_per_image=2)
    head.begin_epoch()
    first = torch.nn.functional.normalize(torch.arange(1, 33).reshape(8, 4).float(), dim=-1)
    head._cache_support(first)
    assert head.support_seen_count.item() == 8
    assert head.support_queue_count.item() == 4
    expected_keys = head._support_priorities_for_indices(
        torch.arange(8), epoch=0
    ).float()
    selected = torch.topk(expected_keys, 4, sorted=False).indices
    expected = first[selected]
    actual = head._local_support()
    actual_order = torch.argsort(head.support_queue_priorities[:4])
    expected_order = torch.argsort(expected_keys[selected])
    assert torch.allclose(actual[actual_order], expected[expected_order])


def test_dead_foreground_prototype_is_reinitialized_from_gt_support():
    head = make_head(warmup_epochs=0, dead_prototype_patience=2)
    with torch.no_grad():
        head.prototypes.copy_(torch.nn.functional.normalize(torch.eye(2, 4), dim=-1))
        head.prototype_ready.fill_(1)
        head.dead_prototype_age[1] = 1
    head.begin_epoch()
    support = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1)
    head._cache_support(support)
    diagnostics = head.finalize_epoch()
    assert diagnostics["prototype_reinitialization_count"] == 1
    assert head.dead_prototype_age[1].item() == 0


def main():
    test_forward_never_changes_detector_logits()
    test_candidate_contract_separates_reliable_near_and_far_samples()
    test_far_matched_query_is_rejected_and_not_relabelled_as_background()
    test_warmup_initializes_foreground_prototypes_from_gt_support_only()
    test_initialized_loss_updates_momentum_projector_after_optimizer_step()
    test_assignment_balance_has_a_student_gradient()
    test_support_bank_uses_a_bounded_epoch_reservoir()
    test_dead_foreground_prototype_is_reinitialized_from_gt_support()
    print("GT foreground proxy tests passed")


if __name__ == "__main__":
    main()
