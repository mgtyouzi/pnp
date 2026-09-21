from types import SimpleNamespace

import torch

from models.frozen_teacher_fg_prototype import FrozenTeacherForegroundPrototype


def build_head(**overrides):
    kwargs = {
        "feat_dim": 4,
        "num_prototypes": 2,
        "negative_bank_size": 8,
        "negative_sample_size": 4,
        "temperature": 0.1,
        "positive_radius": 10.0,
        "background_radius": 30.0,
        "max_positive_per_image": 8,
        "max_hard_negative_per_image": 2,
        "max_random_negative_per_image": 2,
        "sampling_seed": 7,
    }
    kwargs.update(overrides)
    return FrozenTeacherForegroundPrototype(**kwargs)


def test_module_has_foreground_prototypes_but_no_background_prototypes_or_projector():
    head = build_head()

    assert hasattr(head, "fg_prototypes")
    assert not hasattr(head, "bg_prototypes")
    assert not hasattr(head, "projector")
    assert head.prototype_ready.item() == 0


def test_loading_bank_normalizes_and_marks_fixed_state_ready():
    head = build_head()
    prototypes = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    negatives = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0],
        ]
    )

    head.load_fixed_bank(prototypes, negatives, bank_id="unit-test-bank")

    assert torch.allclose(head.fg_prototypes.norm(dim=1), torch.ones(2))
    assert torch.allclose(head.negative_bank[:3].norm(dim=1), torch.ones(3))
    assert head.negative_count.item() == 3
    assert head.prototype_ready.item() == 1
    assert head.bank_id == "unit-test-bank"
    assert all(not parameter.requires_grad for parameter in head.parameters())


def test_candidate_selection_never_uses_score_to_accept_positive():
    head = build_head()
    predicted = torch.tensor([[[5.0, 5.0], [11.0, 5.0], [80.0, 80.0], [90.0, 90.0]]])
    anchors = predicted.clone()
    logits = torch.tensor([[[0.0, 10.0], [10.0, 0.0], [9.0, 0.0], [1.0, 0.0]]])
    targets = {"gt_points": [torch.tensor([[5.0, 5.0], [30.0, 5.0]])]}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    selection = head.select_candidates(
        predicted, logits, targets, indices, anchor_points=anchors
    )

    assert selection.positive_mask.tolist() == [[True, False, False, False]]
    assert selection.rejected_positive_mask.tolist() == [[False, True, False, False]]
    assert not torch.any(selection.positive_mask & selection.negative_mask)
    assert selection.hard_negative_mask.sum().item() == 2
    assert selection.random_negative_mask.sum().item() == 0


def test_positive_only_loss_uses_fixed_bank_without_mutating_it():
    head = build_head()
    head.load_fixed_bank(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
            ]
        ),
        bank_id="fixed",
    )
    before_prototypes = head.fg_prototypes.clone()
    before_negatives = head.negative_bank.clone()
    features = torch.tensor(
        [[[0.8, 0.2, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    points = torch.tensor([[[5.0, 5.0], [20.0, 5.0], [80.0, 80.0]]])
    logits = torch.tensor([[[0.0, 5.0], [5.0, 0.0], [9.0, 0.0]]])
    targets = {"gt_points": [torch.tensor([[5.0, 5.0], [20.0, 5.0]])]}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    loss, diagnostics = head.compute_loss(
        features, points, logits, targets, indices, anchor_points=points
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert features.grad is not None
    assert features.grad[0, :2].abs().sum().item() > 0
    assert features.grad[0, 2].abs().sum().item() == 0
    assert torch.equal(before_prototypes, head.fg_prototypes)
    assert torch.equal(before_negatives, head.negative_bank)
    assert diagnostics["ft_proto_positive"] == 2.0
    assert diagnostics["ft_proto_online_negative_used_for_bank"] == 0.0


def test_epoch_hooks_cannot_update_fixed_prototypes():
    head = build_head()
    head.load_fixed_bank(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[-1.0, 0.0, 0.0, 0.0]]),
        bank_id="fixed",
    )
    before = head.fixed_state_sha256()

    head.begin_epoch()
    finalize = head.finalize_epoch()
    refresh = head.maybe_refresh_prototypes()

    assert head.fixed_state_sha256() == before
    assert finalize["ft_proto_fixed_state_unchanged"] == 1.0
    assert refresh["prototype_refresh_event"] == 0


def main():
    test_module_has_foreground_prototypes_but_no_background_prototypes_or_projector()
    test_loading_bank_normalizes_and_marks_fixed_state_ready()
    test_candidate_selection_never_uses_score_to_accept_positive()
    test_positive_only_loss_uses_fixed_bank_without_mutating_it()
    test_epoch_hooks_cannot_update_fixed_prototypes()
    print("Frozen-teacher foreground prototype tests passed")


if __name__ == "__main__":
    main()
