import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DETR = ROOT / "models" / "detr.py"
TRAIN = ROOT / "train_p2p.py"


def _source(path):
    return path.read_text(encoding="utf-8")


def _string_literals(path):
    tree = ast.parse(_source(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_detr_wires_gt_support_into_the_shared_six_level_feature_extractor():
    source = _source(DETR)
    strings = _string_literals(DETR)
    tree = ast.parse(source)
    detr = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DETR"
    )
    forward = next(
        node for node in detr.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    args = {argument.arg for argument in forward.args.args}
    assert "prototype_support_points" in args
    assert "gt_foreground_proxy" in strings
    assert "GTForegroundProxy" in source
    assert "extract_features(features, support_points" in source
    assert "proto_support_embeddings" in strings
    assert "proto_momentum_support_embeddings" in strings
    assert "proto_support_valid_mask" in strings


def test_training_supplies_gt_points_and_dispatches_the_dedicated_loss_contract():
    source = _source(TRAIN)
    strings = _string_literals(TRAIN)
    assert "gt_foreground_proxy" in strings
    assert "prototype_support_points" in source
    assert "proto_support_embeddings" in strings
    assert "proto_momentum_support_embeddings" in strings
    assert "proto_support_valid_mask" in strings
    required_args = {
        "--proto_gt_warmup_epochs",
        "--proto_gt_support_queue_size",
        "--proto_gt_projector_momentum",
        "--proto_gt_background_margin",
        "--proto_gt_balance_weight",
    }
    assert required_args.issubset(strings), sorted(required_args.difference(strings))


def test_new_mode_cannot_modify_inference_logits():
    source = _source(DETR)
    assert "prototype inference fusion is disabled for gt_foreground_proxy" in source


def main():
    test_detr_wires_gt_support_into_the_shared_six_level_feature_extractor()
    test_training_supplies_gt_points_and_dispatches_the_dedicated_loss_contract()
    test_new_mode_cannot_modify_inference_logits()
    print("GT foreground proxy wiring tests passed")


if __name__ == "__main__":
    main()
