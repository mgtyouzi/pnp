import torch

from models.prototype_metric_adapter import (
    PrototypeMetricAdapter,
    compute_image_proto_e1_loss,
    prototype_margin_loss,
    select_proto_candidates,
)


def test_candidate_selection_uses_distance_and_score_strata():
    gt = torch.tensor([[0.0, 0.0], [100.0, 0.0]])
    points = torch.tensor([
        [1.0, 0.0],
        [10.0, 0.0],
        [100.0, 1.0],
        [50.0, 0.0],
        [500.0, 0.0],
        [600.0, 0.0],
    ])
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.7, 0.6])
    src = torch.tensor([0, 1, 2])
    tgt = torch.tensor([0, 0, 1])

    selected = select_proto_candidates(
        points,
        scores,
        gt,
        src,
        tgt,
        max_positive=2,
        max_negative=2,
        positive_radius=15.0,
        negative_radius=30.0,
    )

    assert selected["positive_indices"].tolist() == [0, 1]
    assert selected["negative_indices"].tolist() == [3, 4]


def test_positive_sampling_covers_score_range():
    gt = torch.tensor([[0.0, 0.0]])
    points = torch.tensor([[float(i), 0.0] for i in range(8)])
    scores = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95])

    selected = select_proto_candidates(
        points,
        scores,
        gt,
        torch.arange(8),
        torch.zeros(8, dtype=torch.long),
        max_positive=4,
        max_negative=0,
        positive_radius=15.0,
        negative_radius=30.0,
    )

    selected_scores = scores[selected["positive_indices"]]
    assert selected_scores.min().item() == scores.min().item()
    assert selected_scores.max().item() == scores.max().item()
    assert selected_scores.numel() == 4


def test_margin_loss_is_balanced_and_backpropagates_to_all_inputs():
    adapter = PrototypeMetricAdapter(8, 6, 4)
    features = torch.randn(5, 8, requires_grad=True)
    output = adapter(features)
    pos_margin = output["margin"][:2]
    neg_margin = output["margin"][2:]
    loss = prototype_margin_loss(pos_margin, neg_margin, margin=0.1, beta=16.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert features.grad is not None and features.grad.abs().sum().item() > 0
    assert adapter.projector[0].weight.grad is not None
    assert adapter.fg_proxy.grad is not None
    assert adapter.bg_proxy.grad is not None


def test_margin_targets_have_the_expected_sign():
    positive = prototype_margin_loss(torch.tensor([0.5]), torch.empty(0), 0.1, 16.0)
    weak_positive = prototype_margin_loss(torch.tensor([-0.5]), torch.empty(0), 0.1, 16.0)
    negative = prototype_margin_loss(torch.empty(0), torch.tensor([-0.5]), 0.1, 16.0)
    weak_negative = prototype_margin_loss(torch.empty(0), torch.tensor([0.5]), 0.1, 16.0)

    assert positive.item() < weak_positive.item()
    assert negative.item() < weak_negative.item()


def test_training_loss_keeps_gradient_to_selected_detector_features():
    adapter = PrototypeMetricAdapter(8, 6, 4)
    features = torch.randn(4, 8, requires_grad=True)
    logits = torch.tensor([
        [3.0, 0.0], [2.0, 0.0], [4.0, 0.0], [1.0, 0.0]
    ])
    points = torch.tensor([
        [1.0, 0.0], [3.0, 0.0], [100.0, 100.0], [120.0, 100.0]
    ])
    gt = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
    loss, diagnostics = compute_image_proto_e1_loss(
        features,
        logits,
        points,
        gt,
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        adapter,
    )
    loss.backward()

    assert diagnostics["num_positive"] == 2.0
    assert diagnostics["num_negative"] == 2.0
    assert features.grad is not None and features.grad.abs().sum().item() > 0
    assert adapter.projector[0].weight.grad.abs().sum().item() > 0
    assert adapter.fg_proxy.grad.abs().sum().item() > 0
    assert adapter.bg_proxy.grad.abs().sum().item() > 0


def test_cuda_points_accept_cpu_hungarian_indices():
    if not torch.cuda.is_available():
        return
    points = torch.tensor(
        [[1.0, 0.0], [100.0, 0.0], [200.0, 0.0]], device="cuda"
    )
    scores = torch.tensor([0.8, 0.9, 0.7], device="cuda")
    gt = torch.tensor([[0.0, 0.0]], device="cuda")
    selected = select_proto_candidates(
        points,
        scores,
        gt,
        torch.tensor([0]),
        torch.tensor([0]),
    )
    assert selected["positive_indices"].device.type == "cuda"
    assert selected["negative_indices"].tolist() == [1, 2]


def main():
    test_candidate_selection_uses_distance_and_score_strata()
    test_positive_sampling_covers_score_range()
    test_margin_loss_is_balanced_and_backpropagates_to_all_inputs()
    test_margin_targets_have_the_expected_sign()
    test_training_loss_keeps_gradient_to_selected_detector_features()
    test_cuda_points_accept_cpu_hungarian_indices()
    print("Prototype metric adapter tests passed")


if __name__ == "__main__":
    main()
