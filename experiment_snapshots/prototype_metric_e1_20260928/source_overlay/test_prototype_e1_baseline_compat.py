import os
from types import SimpleNamespace

from models.detr import build_model
from prototype_e1_checkpoint import load_baseline_checkpoint


def main():
    args = SimpleNamespace(
        backbone="resnet50",
        hidden_dim=256,
        num_classes=1,
        row=2,
        col=2,
        use_proto_e1=True,
        proto_hidden_dim=128,
        proto_embed_dim=64,
        position_embedding="sine",
        dropout=0.1,
        nheads=8,
        dim_feedforward=2048,
        enc_layers=6,
        pre_norm=False,
    )
    model = build_model(args)
    checkpoint_path = os.environ.get("BASELINE_CHECKPOINT")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise SystemExit("Set BASELINE_CHECKPOINT to the compatible baseline checkpoint")
    report = load_baseline_checkpoint(
        model,
        checkpoint_path,
    )
    assert report["loaded_tensors"] > 0
    assert report["missing_new_parameters"]
    assert all(
        key.startswith("proto_metric_adapter.")
        for key in report["missing_new_parameters"]
    )
    print(
        "Prototype E1 real baseline compatibility passed: "
        f"loaded={report['loaded_tensors']}, "
        f"new_adapter={len(report['missing_new_parameters'])}"
    )


if __name__ == "__main__":
    main()
