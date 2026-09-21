import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_pgrp_is_an_isolated_training_only_mode():
    detr = (ROOT / "models" / "detr.py").read_text(encoding="utf-8")
    train = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    assert "prototype_guided_ranking" in detr
    assert "PrototypeGuidedRanking" in detr
    assert "prototype_guided_ranking" in train
    assert "outputs['raw_cls_logits']" in train
    assert "outputs['proto_candidate_raw_features']" in train
    assert "outputs['proto_support_raw_features']" in train


def test_pgrp_parser_exposes_the_fixed_mechanism_parameters():
    source = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    strings = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {
        "--pgrp_hidden_dim",
        "--pgrp_warmup_epochs",
        "--pgrp_ranking_temperature",
        "--pgrp_proto_margin",
        "--pgrp_cls_margin",
        "--pgrp_proto_rank_weight",
        "--pgrp_cls_rank_weight",
        "--pgrp_metric_weight",
        "--pgrp_center_weight",
        "--pgrp_balance_weight",
        "--pgrp_diversity_weight",
        "--pgrp_hard_positive_fraction",
        "--pgrp_max_pairs_per_image",
    }
    assert required.issubset(strings), sorted(required.difference(strings))


def test_pgrp_run_script_reuses_control_and_never_fuses_at_inference():
    source = (ROOT / "run_prototype_guided_ranking_gate_gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "--proto_mode=prototype_guided_ranking" in source
    assert "--proto_inference_fusion=0" in source
    assert "--proto_positive_radius=15" in source
    assert "--proto_background_radius=30" in source
    assert "--pgrp_hard_positive_fraction=0.25" in source
    assert "--pgrp_max_pairs_per_image=32" in source
    assert "CONTROL_CHECKPOINT" in source
    assert "DEBUG_MAX_TRAIN_BATCHES" in source


def main():
    test_pgrp_is_an_isolated_training_only_mode()
    test_pgrp_parser_exposes_the_fixed_mechanism_parameters()
    test_pgrp_run_script_reuses_control_and_never_fuses_at_inference()
    print("Prototype-guided ranking wiring tests passed")


if __name__ == "__main__":
    main()
