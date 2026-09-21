import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_mode_is_training_only_and_does_not_change_matcher_or_regression():
    module = ROOT / "models" / "prototype_local_soft_positive.py"
    assert module.is_file()
    source = module.read_text(encoding="utf-8")
    assert "class PrototypeLocalSoftPositive" in source
    assert "select_local_soft_positives" in source

    detr = (ROOT / "models" / "detr.py").read_text(encoding="utf-8")
    train = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    loss = (ROOT / "loss.py").read_text(encoding="utf-8")
    assert "prototype_local_soft_positive" in detr
    assert "prototype_local_soft_positive" in train
    assert "prototype_soft_positive_indices" in train
    assert "prototype_soft_positive_indices" in loss
    assert "loss_reg(" not in source
    assert "criterion.matcher" not in source


def test_parser_exposes_only_local_selection_and_weak_label_controls():
    source = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    strings = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {
        "--plsp_local_radius",
        "--plsp_min_distance_improvement",
        "--plsp_min_similarity_improvement",
        "--plsp_max_matched_probability",
        "--plsp_max_per_image",
        "--plsp_positive_weight",
    }
    assert required.issubset(strings), sorted(required.difference(strings))


def test_run_script_is_staged_and_keeps_inference_and_matcher_unchanged():
    path = ROOT / "run_prototype_local_soft_positive_gate_gpu.sh"
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert "--proto_mode=prototype_local_soft_positive" in source
    assert "--proto_inference_fusion=0" in source
    assert "--plsp_local_radius=15" in source
    assert "--plsp_min_distance_improvement=3" in source
    assert "--plsp_min_similarity_improvement=0.1" in source
    assert "--plsp_max_matched_probability=0.56" in source
    assert "--plsp_max_per_image=8" in source
    assert "--plsp_positive_weight=0.1" in source
    assert "DEBUG_MAX_TRAIN_BATCHES" in source
    assert "plsp_mechanism_gate.py" in source
    assert "paired_detection_threshold_sweep.py" in source
    assert "plsp_final_gate.py" in source


def main():
    test_mode_is_training_only_and_does_not_change_matcher_or_regression()
    test_parser_exposes_only_local_selection_and_weak_label_controls()
    test_run_script_is_staged_and_keeps_inference_and_matcher_unchanged()
    print("Prototype local soft-positive structure tests passed")


if __name__ == "__main__":
    main()
