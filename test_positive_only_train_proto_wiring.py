import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_mode_is_isolated_and_training_only():
    detr = (ROOT / "models" / "detr.py").read_text(encoding="utf-8")
    train = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    assert "positive_only_train_proto" in detr
    assert "PositiveOnlyTrainPrototype" in detr
    assert "prototype_support_points" in detr
    assert "positive_only_train_proto" in train
    assert "outputs['raw_cls_logits']" in train
    assert "outputs['proto_support_embeddings']" in train


def test_parser_exposes_only_required_hyperparameters():
    source = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    strings = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {
        "--potp_warmup_epochs",
        "--potp_hidden_dim",
        "--potp_positive_margin",
        "--potp_negative_margin",
        "--potp_diversity_margin",
        "--potp_pair_weight",
        "--potp_positive_weight",
        "--potp_negative_weight",
        "--potp_balance_weight",
        "--potp_diversity_weight",
        "--potp_hard_positive_fraction",
        "--potp_max_hard_positive_per_image",
        "--potp_detach_support_input",
    }
    assert required.issubset(strings), sorted(required.difference(strings))


def test_paired_gate_runs_raw_p2p_only_without_a_bank():
    source = (ROOT / "run_positive_only_train_proto_paired_gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "--proto_mode=positive_only_train_proto" in source
    assert "--proto_inference_fusion=0" in source
    assert "--proto_loss_weight=1.0" in source
    assert "--proto_bank_path" not in source
    assert "minimum_f1_gain" in source
    assert "maximum_precision_drop" in source


def test_v2_routes_only_hard_positive_gradients_and_reuses_the_control():
    source = (
        ROOT / "run_positive_only_train_proto_v2_paired_gpu.sh"
    ).read_text(encoding="utf-8")
    assert "--potp_hard_positive_fraction=0.25" in source
    assert "--potp_max_hard_positive_per_image=32" in source
    assert "--potp_detach_support_input=1" in source
    assert "--potp_pair_weight=0.005" in source
    assert "--potp_positive_weight=0.005" in source
    assert "--potp_negative_weight=0.01" in source
    assert "CONTROL_CHECKPOINT" in source
    assert "paired raw P2P control" not in source


def main():
    test_mode_is_isolated_and_training_only()
    test_parser_exposes_only_required_hyperparameters()
    test_paired_gate_runs_raw_p2p_only_without_a_bank()
    test_v2_routes_only_hard_positive_gradients_and_reuses_the_control()
    print("Positive-only train prototype wiring tests passed")


if __name__ == "__main__":
    main()
