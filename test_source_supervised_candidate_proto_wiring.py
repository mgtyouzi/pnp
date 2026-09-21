import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_new_mode_is_isolated_and_uses_raw_matching_during_training():
    detr = (ROOT / "models" / "detr.py").read_text(encoding="utf-8")
    train = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    assert "source_supervised_candidate_proto" in detr
    assert "SourceSupervisedCandidatePrototype" in detr
    assert "proto_fused_logits" in detr
    assert "source_supervised_candidate_proto" in train
    assert "outputs['raw_cls_logits']" in train
    assert "outputs['proto_fused_logits']" in train
    assert "sscp_loss_raw" in train


def test_parser_exposes_complete_sscp_contract():
    source = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    strings = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {
        "--sscp_proto_ce_weight",
        "--sscp_margin_weight",
        "--sscp_fused_ce_weight",
        "--sscp_alpha_max",
        "--sscp_alpha_initial",
        "--sscp_confidence_threshold",
        "--sscp_fusion_clip",
        "--sscp_min_assignment_share",
    }
    assert required.issubset(strings), sorted(required.difference(strings))


def main():
    test_new_mode_is_isolated_and_uses_raw_matching_during_training()
    test_parser_exposes_complete_sscp_contract()
    print("Source-supervised candidate prototype wiring tests passed")


if __name__ == "__main__":
    main()
