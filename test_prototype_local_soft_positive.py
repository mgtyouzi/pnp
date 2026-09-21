import torch
import torch.nn.functional as F

from models.prototype_local_soft_positive import PrototypeLocalSoftPositive
from prototype_soft_positive_loss import (
    cross_entropy_with_local_soft_positives,
)


def make_head(**kwargs):
    defaults = dict(
        feat_dim=4,
        hidden_dim=4,
        embedding_dim=2,
        num_prototypes=2,
        warmup_epochs=0,
        local_radius=15.0,
        minimum_distance_improvement=3.0,
        minimum_similarity_improvement=0.1,
        maximum_matched_probability=0.56,
        max_soft_positive_per_image=4,
        soft_positive_weight=0.1,
        support_reservoir_size=16,
    )
    defaults.update(kwargs)
    head = PrototypeLocalSoftPositive(**defaults)
    with torch.no_grad():
        head.foreground_prototypes.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        )
        head.prototype_ready.fill_(1)
    head.set_epoch(0)
    return head


def test_selects_low_score_candidate_that_is_closer_and_more_prototypical():
    head = make_head()
    points = torch.tensor([[[8.0, 0.0], [3.0, 0.0], [40.0, 0.0]]])
    embeddings = F.normalize(
        torch.tensor([[[0.60, 0.80], [1.0, 0.0], [0.0, 1.0]]]), dim=-1
    )
    raw_logits = torch.tensor([[[0.0, 2.0], [-5.0, 5.0], [0.0, 2.0]]])
    support_embeddings = torch.tensor([[[1.0, 0.0]]])
    support_valid_mask = torch.tensor([[True]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0])],
        "gt_nums": [1],
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    selected, diagnostics = head.select_local_soft_positives(
        points,
        raw_logits,
        embeddings,
        support_embeddings,
        support_valid_mask,
        targets,
        indices,
    )

    assert selected[0].tolist() == [1]
    assert diagnostics["plsp_selected"] == 1.0
    assert diagnostics["plsp_selected_matched_probability"] < 0.56
    assert diagnostics["plsp_selected_probability"] < 0.001
    assert diagnostics["plsp_distance_improvement"] == 5.0
    assert diagnostics["plsp_similarity_improvement"] > 0.1


def test_selection_requires_nearest_target_ownership_and_one_candidate_per_gt():
    head = make_head()
    points = torch.tensor(
        [[[8.0, 0.0], [22.0, 0.0], [3.0, 0.0], [4.0, 0.0], [18.0, 0.0]]]
    )
    embeddings = F.normalize(
        torch.tensor(
            [[[0.60, 0.80], [0.60, 0.80], [1.0, 0.0], [0.95, 0.05], [1.0, 0.0]]]
        ),
        dim=-1,
    )
    raw_logits = torch.zeros(1, 5, 2)
    support_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    support_valid_mask = torch.tensor([[True, True]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0], [30.0, 0.0]])],
        "gt_labels": [torch.tensor([0, 0])],
        "gt_nums": [2],
    }
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    selected, diagnostics = head.select_local_soft_positives(
        points,
        raw_logits,
        embeddings,
        support_embeddings,
        support_valid_mask,
        targets,
        indices,
    )

    assert selected[0].tolist() == [2]
    assert diagnostics["plsp_selected"] == 1.0


def test_high_confidence_original_match_does_not_trigger_soft_positive():
    head = make_head()
    points = torch.tensor([[[8.0, 0.0], [3.0, 0.0]]])
    embeddings = F.normalize(
        torch.tensor([[[0.60, 0.80], [1.0, 0.0]]]), dim=-1
    )
    raw_logits = torch.tensor([[[2.0, 0.0], [-5.0, 5.0]]])
    support_embeddings = torch.tensor([[[1.0, 0.0]]])
    support_valid_mask = torch.tensor([[True]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0])],
        "gt_nums": [1],
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    selected, diagnostics = head.select_local_soft_positives(
        points,
        raw_logits,
        embeddings,
        support_embeddings,
        support_valid_mask,
        targets,
        indices,
    )

    assert selected[0].numel() == 0
    assert diagnostics["plsp_low_confidence_targets"] == 0.0


def test_soft_positive_loss_is_exact_baseline_when_no_candidate_is_selected():
    logits = torch.tensor([[[0.2, -0.1], [-0.3, 0.4]]], requires_grad=True)
    targets = torch.tensor([[0, 1]])
    class_weight = torch.tensor([1.0, 0.5])

    expected = F.cross_entropy(logits.transpose(1, 2), targets, class_weight)
    actual = cross_entropy_with_local_soft_positives(
        logits, targets, class_weight, [torch.empty(0, dtype=torch.long)], 0.1
    )

    assert torch.equal(actual, expected)


def test_soft_positive_replaces_background_gradient_instead_of_adding_conflict():
    baseline_logits = torch.zeros(1, 1, 2, requires_grad=True)
    targets = torch.tensor([[1]])
    class_weight = torch.tensor([1.0, 0.5])
    baseline = F.cross_entropy(
        baseline_logits.transpose(1, 2), targets, class_weight
    )
    baseline.backward()

    soft_logits = torch.zeros(1, 1, 2, requires_grad=True)
    soft = cross_entropy_with_local_soft_positives(
        soft_logits, targets, class_weight, [torch.tensor([0])], 0.1
    )
    soft.backward()

    assert baseline_logits.grad[0, 0, 0].item() > 0
    assert soft_logits.grad[0, 0, 0].item() < 0


def main():
    test_selects_low_score_candidate_that_is_closer_and_more_prototypical()
    test_selection_requires_nearest_target_ownership_and_one_candidate_per_gt()
    test_high_confidence_original_match_does_not_trigger_soft_positive()
    test_soft_positive_loss_is_exact_baseline_when_no_candidate_is_selected()
    test_soft_positive_replaces_background_gradient_instead_of_adding_conflict()
    print("Prototype local soft-positive tests passed")


if __name__ == "__main__":
    main()
