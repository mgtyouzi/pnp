import tempfile
from pathlib import Path

import torch

from models.frozen_supervised_metric_prototype import (
    FrozenSupervisedMetricPrototype,
)


def make_bank(path: Path) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    projector_weight = torch.randn(4, 8, generator=generator)
    projector_bias = torch.randn(4, generator=generator)
    prototypes = torch.randn(5, 4, generator=generator)
    torch.save(
        {
            "projector_weight": projector_weight,
            "projector_bias": projector_bias,
            "prototypes": prototypes,
            "prototype_counts": (2, 2, 1),
            "temperature": 0.15,
            "bank_id": "unit-test-bank",
            "metadata": {
                "version": "frozen_supervised_metric_proto_v1_20260816",
                "gate_pass": True,
                "feature_source": "p2p_cls_features_before_final_classifier",
                "checkpoint_sha256": "teacher-sha",
            },
        },
        str(path),
    )


def build_module(path: Path) -> FrozenSupervisedMetricPrototype:
    module = FrozenSupervisedMetricPrototype(
        feat_dim=8,
        embedding_dim=4,
        prototype_counts=(2, 2, 1),
        temperature=0.15,
        positive_radius=10.0,
        background_radius=30.0,
        max_positive_per_image=4,
        max_hard_negative_per_image=2,
        max_random_negative_per_image=2,
        hard_negative_weight=2.0,
        sampling_seed=0,
    )
    module.load_bank_file(str(path))
    return module


def test_frozen_bank_preserves_input_gradient() -> None:
    with tempfile.TemporaryDirectory() as directory:
        bank = Path(directory) / "bank.pth"
        make_bank(bank)
        module = build_module(bank)
        features = torch.randn(2, 5, 8, requires_grad=True)
        raw_logits = torch.randn(2, 5, 2)

        outputs = module(features, raw_logits, apply_fusion=False)
        outputs["prototype_logits"].sum().backward()

        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        assert features.grad.abs().sum() > 0
        assert all(not parameter.requires_grad for parameter in module.parameters())


def test_fixed_state_stays_unchanged_across_epoch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        bank = Path(directory) / "bank.pth"
        make_bank(bank)
        module = build_module(bank)
        before = module.fixed_state_sha256()
        module.begin_epoch()
        summary = module.finalize_epoch()

        assert module.fixed_state_sha256() == before
        assert summary["supervised_proto_fixed_state_unchanged"] == 1.0
        assert summary["prototype_epoch_refresh_events"] == 0.0


def test_three_candidate_sources_produce_finite_loss() -> None:
    with tempfile.TemporaryDirectory() as directory:
        bank = Path(directory) / "bank.pth"
        make_bank(bank)
        module = build_module(bank)
        features = torch.randn(1, 8, 8, requires_grad=True)
        raw_logits = torch.tensor(
            [[[4.0, -1.0], [3.0, -1.0], [2.5, -1.0], [2.0, -1.0],
              [1.5, -1.0], [1.0, -1.0], [0.5, -1.0], [0.0, -1.0]]]
        )
        points = torch.tensor(
            [[[1.0, 1.0], [120.0, 120.0], [80.0, 0.0], [100.0, 0.0],
              [140.0, 0.0], [160.0, 0.0], [180.0, 0.0], [200.0, 0.0]]]
        )
        targets = {
            "gt_points": [torch.tensor([[0.0, 0.0], [100.0, 100.0]])]
        }
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        embedded = module(features, raw_logits, apply_fusion=False)["embeddings"]

        loss, diagnostics = module.compute_loss(
            embedded, points, raw_logits, targets, indices
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert diagnostics["supervised_proto_positive"] == 1.0
        assert diagnostics["supervised_proto_rejected_positive"] == 1.0
        assert diagnostics["supervised_proto_hard_negative"] == 2.0
        assert diagnostics["supervised_proto_random_negative"] == 2.0
        assert features.grad is not None and features.grad.abs().sum() > 0


def test_bank_metadata_is_strictly_validated() -> None:
    with tempfile.TemporaryDirectory() as directory:
        bank = Path(directory) / "bank.pth"
        make_bank(bank)
        payload = torch.load(str(bank), map_location="cpu")
        payload["metadata"]["gate_pass"] = False
        torch.save(payload, str(bank))
        module = FrozenSupervisedMetricPrototype(
            feat_dim=8,
            embedding_dim=4,
            prototype_counts=(2, 2, 1),
            temperature=0.15,
        )
        try:
            module.load_bank_file(str(bank))
        except RuntimeError as error:
            assert "gate" in str(error)
        else:
            raise AssertionError("a failed offline gate must reject the bank")


def main() -> None:
    test_frozen_bank_preserves_input_gradient()
    test_fixed_state_stays_unchanged_across_epoch()
    test_three_candidate_sources_produce_finite_loss()
    test_bank_metadata_is_strictly_validated()
    print("Frozen supervised metric prototype tests passed")


if __name__ == "__main__":
    main()
