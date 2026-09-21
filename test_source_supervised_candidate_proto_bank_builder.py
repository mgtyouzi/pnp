import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_builder_is_source_only_native_feature_and_two_class():
    path = ROOT / "build_source_supervised_candidate_proto_bank.py"
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert "source_supervised_candidate_proto" in source
    assert "foreground_prototypes" in source
    assert "background_prototypes" in source
    assert "cls_features" in source
    assert "projector" not in source.lower()
    assert "frozen" not in source.lower()


def test_builder_keeps_slide_disjoint_calibration_and_exports_audit():
    source = (
        ROOT / "build_source_supervised_candidate_proto_bank.py"
    ).read_text(encoding="utf-8")
    strings = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "support_groups" in strings
    assert "calibration_groups" in strings
    assert "prototype_offline_audit.json" in strings
    assert "source_supervised_candidate_proto_bank.pth" in strings
    assert "foreground_slides" in strings
    assert "background_slides" in strings


def main():
    test_builder_is_source_only_native_feature_and_two_class()
    test_builder_keeps_slide_disjoint_calibration_and_exports_audit()
    print("Source-supervised candidate prototype bank builder tests passed")


if __name__ == "__main__":
    main()
