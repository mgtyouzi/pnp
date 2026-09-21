import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "models" / "frozen_teacher_fg_prototype.py"


def parsed_module():
    assert MODULE.is_file(), "frozen-teacher module has not been implemented"
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def test_module_exposes_only_foreground_prototype_class():
    tree = parsed_module()
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    assert "FrozenTeacherForegroundPrototype" in classes
    source = MODULE.read_text(encoding="utf-8")
    assert "bg_prototypes" not in source
    assert "self.projector" not in source


def test_module_has_fixed_bank_and_positive_only_loss_interfaces():
    tree = parsed_module()
    prototype_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FrozenTeacherForegroundPrototype"
    )
    methods = {
        node.name for node in prototype_class.body if isinstance(node, ast.FunctionDef)
    }

    assert "load_fixed_bank" in methods
    assert "select_candidates" in methods
    assert "compute_loss" in methods
    assert "fixed_state_sha256" in methods
    assert "maybe_refresh_prototypes" in methods


def test_failed_offline_gate_cannot_be_loaded_for_training():
    source = MODULE.read_text(encoding="utf-8")

    assert 'metadata.get("gate_pass") is not True' in source


def test_training_dispatches_frozen_mode_away_from_legacy_cache_updates():
    source = (ROOT / "train_p2p.py").read_text(encoding="utf-8")

    frozen_branch = source.index(
        "if str(getattr(args, 'proto_mode', 'legacy_online')) == 'frozen_teacher_fg':"
    )
    legacy_branch = source.index("else:", frozen_branch)
    frozen_source = source[frozen_branch:legacy_branch]
    assert ".compute_loss(" in frozen_source
    assert "compute_loss_and_cache" not in frozen_source
    assert (
        "if prototype_active and str(getattr(args, 'proto_mode', "
        "'legacy_online')) != 'frozen_teacher_fg':"
    ) in source


def test_runtime_diagnostics_cover_selection_separation_and_cluster_use():
    source = MODULE.read_text(encoding="utf-8")
    required = (
        "ft_proto_positive_acceptance_rate",
        "ft_proto_positive_vs_fixed_negative_accuracy",
        "ft_proto_similarity_gap",
        "ft_proto_online_hard_similarity",
        "ft_proto_effective_foreground_prototypes",
    )
    for key in required:
        assert key in source


def test_detr_keeps_legacy_and_frozen_teacher_modes_separate():
    source = (ROOT / "models" / "detr.py").read_text(encoding="utf-8")

    assert "proto_mode" in source
    assert "legacy_online" in source
    assert "frozen_teacher_fg" in source


def main():
    test_module_exposes_only_foreground_prototype_class()
    test_module_has_fixed_bank_and_positive_only_loss_interfaces()
    test_failed_offline_gate_cannot_be_loaded_for_training()
    test_training_dispatches_frozen_mode_away_from_legacy_cache_updates()
    test_runtime_diagnostics_cover_selection_separation_and_cluster_use()
    test_detr_keeps_legacy_and_frozen_teacher_modes_separate()
    print("Frozen-teacher prototype structure tests passed")


if __name__ == "__main__":
    main()
