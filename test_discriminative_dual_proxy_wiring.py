import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DETR = ROOT / "models" / "detr.py"
TRAIN = ROOT / "train_p2p.py"


def _source(path):
    return path.read_text(encoding="utf-8")


def _strings(path):
    return {
        node.value
        for node in ast.walk(ast.parse(_source(path)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_model_wires_dual_proxy_without_inference_fusion():
    source = _source(DETR)
    strings = _strings(DETR)
    assert "discriminative_dual_proxy" in strings
    assert "DiscriminativeDualProxy" in source
    assert "prototype inference fusion is disabled for discriminative_dual_proxy" in source
    assert "proto_momentum_embeddings" in strings
    assert "proto_support_embeddings" in strings
    assert "extract_features(features, support_points" in source
    assert "load_bank_file" in source
    assert "bank_path" in source


def test_parser_and_training_have_dedicated_dual_proxy_contract():
    source = _source(TRAIN)
    strings = _strings(TRAIN)
    required_args = {
        "--proto_dual_num_fg",
        "--proto_dual_num_bg",
        "--proto_dual_fg_queue_size",
        "--proto_dual_bg_queue_size",
        "--proto_dual_supcon_weight",
        "--proto_dual_separation_weight",
        "--proto_dual_separation_margin",
        "--proto_dual_balance_weight",
        "--proto_max_random_bg_per_image",
        "--proto_freeze_detector",
    }
    assert required_args.issubset(strings), sorted(required_args.difference(strings))
    assert "discriminative_dual_proxy" in strings
    assert "proto_momentum_embeddings" in strings
    assert "dualproxy_loss_raw" in strings
    assert "[Dual-Proxy-health]" in source
    assert "_freeze_detector_for_proxy_gate" in source
    assert "_apply_projector_only_train_mode" in source
    assert "base_model.eval()" in source
    assert "base_model.prototype_head.projector.train()" in source
    assert "optimizer_projector_only" in source


def test_ddp_hash_guard_only_protects_frozen_banks_after_parameter_broadcast():
    source = _source(TRAIN)
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "_should_protect_fixed_prototype_bank" in functions

    helper_source = ast.get_source_segment(
        source, functions["_should_protect_fixed_prototype_bank"]
    )
    assert "frozen_teacher_fg" in helper_source
    assert "frozen_supervised_metric" in helper_source
    assert "discriminative_dual_proxy" not in helper_source

    ddp_position = source.index("model = DistributedDataParallel(")
    hash_position = source.index("fixed_bank_hash_before_load =")
    assert hash_position > ddp_position
    assert "_should_protect_fixed_prototype_bank(args)" in source


def test_dual_proxy_module_validates_and_loads_a_passed_bank():
    source = _source(ROOT / "models" / "discriminative_dual_proxy.py")
    strings = _strings(ROOT / "models" / "discriminative_dual_proxy.py")
    assert "load_bank_payload" in source
    assert "load_bank_file" in source
    assert "prototype_state" in strings
    assert "implementation_version" in strings
    assert "loaded dual-proxy bank is not ready" in strings
    assert "momentum_reliable" in source
    assert "foreground_gap_tensor" in source
    assert "background_gap_tensor" in source
    assert "self.separation_margin - foreground_gap_tensor" in source
    assert "max_hard_background_per_image" in source
    assert "max_random_background_per_image" in source
    assert "dualproxy_random_background" in source


def main():
    test_model_wires_dual_proxy_without_inference_fusion()
    test_parser_and_training_have_dedicated_dual_proxy_contract()
    test_ddp_hash_guard_only_protects_frozen_banks_after_parameter_broadcast()
    test_dual_proxy_module_validates_and_loads_a_passed_bank()
    print("Discriminative dual-proxy wiring tests passed")


if __name__ == "__main__":
    main()
