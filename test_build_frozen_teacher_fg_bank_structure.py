import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "build_frozen_teacher_fg_bank.py"
RECALIBRATE_SCRIPT = ROOT / "recalibrate_frozen_teacher_fg_bank.py"
GEOMETRY_SCRIPT = ROOT / "diagnose_frozen_teacher_proto_geometry.py"


def test_builder_is_independent_from_legacy_online_prototypes():
    assert SCRIPT.is_file(), "frozen-teacher bank builder has not been implemented"
    source = SCRIPT.read_text(encoding="utf-8")

    assert "CandidatePrototypeBank" not in source
    assert "background_centers" not in source
    assert "self.projector" not in source
    assert "foreground_prototypes" in source
    assert "negative_bank" in source


def test_builder_exposes_split_cluster_audit_and_gate_stages():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert "split_support_calibration" in functions
    assert "spherical_kmeans" in functions
    assert "extract_teacher_features" in functions
    assert "evaluate_separation" in functions
    assert "build_bank_payload" in functions


def test_builder_uses_strict_identity_preserving_reads_and_group_split():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "class StrictTeacherDataset" in source
    assert "def slide_group_id" in source
    assert "replacement_samples" not in source
    assert '"support_groups"' in source
    assert '"calibration_groups"' in source


def test_training_requires_the_same_teacher_checkpoint_feature_space():
    source = (ROOT / "train_p2p.py").read_text(encoding="utf-8")

    assert "validate_frozen_teacher_initialization" in source
    assert "checkpoint_sha256" in source
    assert "teacher checkpoint hash" in source
    assert "fixed prototype bank changed while loading checkpoint" in source


def test_failed_gate_stops_pipeline_and_smoke_samples_both_splits():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "offline foreground prototype gate failed" in source
    assert 'counts["support_images"]' in source
    assert 'counts["calibration_images"]' in source


def test_kmeans_initialization_is_torch_1_10_device_safe():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "torch.Generator(device=values.device)" not in source
    assert "first = int(seed) % int(values.shape[0])" in source


def test_offline_gate_matches_the_training_protonce_energy():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "negative_bank" in source
    assert "positive_energy" in source
    assert "negative_energy" in source
    assert '"gate_score": "protonce_relative_energy"' in source


def test_failed_extraction_can_be_recalibrated_without_teacher_inference():
    assert RECALIBRATE_SCRIPT.is_file(), "bank recalibration script is missing"
    source = RECALIBRATE_SCRIPT.read_text(encoding="utf-8")

    assert "prototype_calibration_features.npz" in source
    assert "evaluate_separation" in source
    assert "build_frozen_teacher_fg_bank" in source
    assert "output_bank" in source
    assert "torch.save" in source
    assert "build_model" not in source


def test_geometry_diagnosis_separates_feature_and_objective_failures():
    assert GEOMETRY_SCRIPT.is_file(), "offline geometry diagnosis is missing"
    source = GEOMETRY_SCRIPT.read_text(encoding="utf-8")

    assert "foreground_similarity" in source
    assert "nearest_bank_margin" in source
    assert "protonce_relative_energy" in source
    assert "ridge_linear_probe" in source
    assert "hard_negative" in source
    assert "random_negative" in source
    assert "feature_not_linearly_separable" in source
    assert "prototype_objective_mismatch" in source


def main():
    test_builder_is_independent_from_legacy_online_prototypes()
    test_builder_exposes_split_cluster_audit_and_gate_stages()
    test_builder_uses_strict_identity_preserving_reads_and_group_split()
    test_training_requires_the_same_teacher_checkpoint_feature_space()
    test_failed_gate_stops_pipeline_and_smoke_samples_both_splits()
    test_kmeans_initialization_is_torch_1_10_device_safe()
    test_offline_gate_matches_the_training_protonce_energy()
    test_failed_extraction_can_be_recalibrated_without_teacher_inference()
    test_geometry_diagnosis_separates_feature_and_objective_failures()
    print("Frozen-teacher bank builder structure tests passed")


if __name__ == "__main__":
    main()
