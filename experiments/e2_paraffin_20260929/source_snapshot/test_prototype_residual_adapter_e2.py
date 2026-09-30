from types import SimpleNamespace

import torch
from torch import nn

from models.detr import DETR
from models.prototype_residual_adapter_e2 import (
    E2PrototypeResidualAdapter,
    compute_e2_image_loss,
    select_e2_candidates,
)
from matcher import HungarianMatcher


def test_residual_is_bounded_and_does_not_backpropagate_to_detector_features():
    adapter = E2PrototypeResidualAdapter(in_dim=8, hidden_dim=4, embed_dim=3)
    features = torch.randn(1, 5, 8, requires_grad=True)
    logits = torch.randn(1, 5, 2, requires_grad=True)
    result = adapter(features, logits, residual_enabled=True)
    assert result["cls_logits"].shape == logits.shape
    assert result["residual"].abs().max().item() <= adapter.r_max
    torch.nn.functional.cross_entropy(
        result["calibration_logits"].reshape(-1, 2),
        torch.zeros(5, dtype=torch.long),
    ).backward()
    assert features.grad is None
    assert logits.grad is None
    assert adapter.projector[0].weight.grad is not None


def test_disabled_residual_is_exact_base_logit_bypass():
    adapter = E2PrototypeResidualAdapter(in_dim=8, hidden_dim=4, embed_dim=3)
    logits = torch.randn(2, 5, 2)
    result = adapter(torch.randn(2, 5, 8), logits, residual_enabled=False)
    assert torch.equal(result["cls_logits"], logits)
    assert torch.equal(result["calibration_logits"], logits)
    assert torch.count_nonzero(result["residual"]).item() == 0


def test_candidate_sampling_respects_match_distance_and_spans_positive_scores():
    points = torch.tensor([[0., 0.], [5., 0.], [20., 0.], [80., 0.], [90., 0.]])
    scores = torch.tensor([0.10, 0.45, 0.30, 0.80, 0.60])
    gt = torch.tensor([[0., 0.]])
    selected = select_e2_candidates(
        points, scores, gt,
        matched_src=torch.tensor([0, 1]),
        matched_tgt=torch.tensor([0, 0]),
        max_positive=2, max_negative=2,
        positive_radius=15., negative_radius=30., negative_score_floor=0.2,
    )
    assert set(selected["positive_indices"].tolist()) == {0, 1}
    assert set(selected["negative_indices"].tolist()) == {3, 4}
    assert not set(selected["negative_indices"].tolist()).intersection({0, 1})


def test_matcher_uses_base_logits_not_residual_logits():
    matcher = HungarianMatcher(cost_point=0.0, cost_class=1.0)
    base = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    adjusted = torch.tensor([[[0.0, 5.0], [5.0, 0.0]]])
    outputs = {
        "pnt_coords": torch.zeros(1, 2, 2),
        "cls_logits_base": base,
        "cls_logits": adjusted,
    }
    targets = {
        "gt_nums": [1],
        "gt_points": [torch.zeros(1, 2)],
        "gt_labels": [torch.zeros(1, dtype=torch.long)],
    }
    src, _ = matcher(outputs, targets)[0]
    assert src.tolist() == [0]


def test_residual_phase_flag_survives_checkpoint_round_trip():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.neck = SimpleNamespace(num_outs=2)

    def make_model():
        return DETR(
            Backbone(), hidden_dim=8, num_classes=1, row=2, col=2,
            use_proto_e2=True, e2_hidden_dim=4, e2_embed_dim=3,
        )

    source = make_model()
    source.set_e2_residual_enabled(False)
    restored = make_model()
    restored.load_state_dict(source.state_dict())
    assert not restored.e2_residual_enabled


def test_image_loss_slices_the_requested_batch_item():
    adapter = E2PrototypeResidualAdapter(8, 4, 3)
    features = torch.randn(1, 5, 8, requires_grad=True)
    base_logits = torch.tensor([[[3., 0.], [2., 0.], [1., 0.], [2., 0.], [1., 0.]]], requires_grad=True)
    e2 = adapter(features, base_logits)
    outputs = {"cls_features": features, "cls_logits_base": base_logits,
               "e2_margin": e2["margin"], "e2_calibration_logits": e2["calibration_logits"]}
    args = SimpleNamespace(e2_max_positive=2, e2_max_negative=2,
        e2_positive_radius=15., e2_negative_radius=30., e2_negative_score_floor=.1,
        e2_metric_margin=.1, e2_metric_beta=16., e2_metric_loss_weight=.1)
    loss, metrics = compute_e2_image_loss(outputs,
        torch.tensor([[0., 0.], [5., 0.], [20., 0.], [80., 0.], [100., 0.]]),
        torch.tensor([[0., 0.]]), torch.tensor([0, 1]), torch.tensor([0, 0]),
        adapter, args, True, True, batch_index=0)
    assert metrics["positive"] == 2 and metrics["negative"] == 2
    assert torch.isfinite(loss)

def test_disabled_warmup_returns_complete_diagnostics():
    adapter = E2PrototypeResidualAdapter(8, 4, 3)
    features = torch.randn(1, 5, 8)
    logits = torch.randn(1, 5, 2)
    e2 = adapter(features, logits, residual_enabled=False)
    outputs = {
        "cls_features": features,
        "cls_logits_base": logits,
        "e2_margin": e2["margin"],
        "e2_calibration_logits": e2["calibration_logits"],
    }
    loss, metrics = compute_e2_image_loss(
        outputs, torch.zeros(5, 2), torch.zeros(0, 2),
        torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
        adapter, SimpleNamespace(), False, False, batch_index=0,
    )
    assert torch.isfinite(loss)
    assert {"positive", "negative", "geometry", "calibration",
            "margin_gap", "alpha", "fg_bg_cosine"}.issubset(metrics)
    assert metrics["alpha"] == adapter.alpha.item()


if __name__ == "__main__":
    test_residual_is_bounded_and_does_not_backpropagate_to_detector_features()
    test_disabled_residual_is_exact_base_logit_bypass()
    test_candidate_sampling_respects_match_distance_and_spans_positive_scores()
    test_matcher_uses_base_logits_not_residual_logits()
    test_residual_phase_flag_survives_checkpoint_round_trip()
    test_image_loss_slices_the_requested_batch_item()
    test_disabled_warmup_returns_complete_diagnostics()
    print("E2 prototype residual adapter tests passed")
