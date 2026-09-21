import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_prr_is_an_isolated_training_only_mode():
    module = ROOT / "models" / "prototype_reliability_rescue.py"
    assert module.is_file()
    source = module.read_text(encoding="utf-8")
    assert "class PrototypeReliabilityRescue" in source
    assert '"fused_logits": raw_logits' in source
    assert '"prr_gt_support"' in source

    detr = (ROOT / "models" / "detr.py").read_text(encoding="utf-8")
    train = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    assert "prototype_reliability_rescue" in detr
    assert "PrototypeReliabilityRescue" in detr
    assert "prototype_reliability_rescue" in train


def test_prr_parser_exposes_only_the_planned_controls():
    source = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    strings = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {
        "--prr_hidden_dim",
        "--prr_warmup_epochs",
        "--prr_support_low_quantile",
        "--prr_support_high_quantile",
        "--prr_max_cell_probability",
        "--prr_target_margin",
        "--prr_margin_temperature",
        "--prr_rescue_weight",
        "--prr_logit_gradient_scale",
        "--prr_proto_rank_weight",
        "--prr_metric_weight",
        "--prr_center_weight",
        "--prr_balance_weight",
        "--prr_diversity_weight",
        "--prr_max_rescue_per_image",
    }
    assert required.issubset(strings), sorted(required.difference(strings))


def test_prr_does_not_modify_the_base_criterion_contract():
    source = (ROOT / "loss.py").read_text(encoding="utf-8")
    assert "prototype_reliability_rescue" not in source
    assert "prr_" not in source


def test_prr_run_script_has_staged_gates_and_no_inference_fusion():
    path = ROOT / "run_prototype_reliability_rescue_gate_gpu.sh"
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert "--proto_mode=prototype_reliability_rescue" in source
    assert "--proto_inference_fusion=0" in source
    assert "--proto_positive_radius=15" in source
    assert "--proto_background_radius=30" in source
    assert "--prr_logit_gradient_scale=0.005" in source
    assert "--prr_rescue_weight=0.005" in source
    assert "DEBUG_MAX_TRAIN_BATCHES" in source
    assert "paired_detection_threshold_sweep.py" in source


def main():
    test_prr_is_an_isolated_training_only_mode()
    test_prr_parser_exposes_only_the_planned_controls()
    test_prr_does_not_modify_the_base_criterion_contract()
    test_prr_run_script_has_staged_gates_and_no_inference_fusion()
    print("Prototype reliability rescue structure tests passed")


if __name__ == "__main__":
    main()
