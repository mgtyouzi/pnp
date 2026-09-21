from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXTRACTOR = ROOT / "extract_frozen_teacher_candidate_audit.py"
DIAGNOSIS = ROOT / "diagnose_class_conditional_prototypes.py"


def test_extractor_preserves_candidate_identity_and_teacher_scores():
    assert EXTRACTOR.is_file()
    source = EXTRACTOR.read_text(encoding="utf-8")

    for token in (
        "candidate_type",
        "teacher_score",
        "matched_distance",
        "nearest_gt_distance",
        "image_index",
        "group_index",
        "image_manifest.json",
        "candidate_features.npz",
    ):
        assert token in source
    assert "split_support_calibration" not in source


def test_diagnosis_uses_group_loso_and_three_class_prototypes():
    assert DIAGNOSIS.is_file()
    source = DIAGNOSIS.read_text(encoding="utf-8")

    for token in (
        "leave_one_group_out",
        "group_balanced_subset",
        "foreground_prototypes",
        "hard_background_prototypes",
        "random_background_prototypes",
        "teacher_score",
        "hard_negative",
        "random_negative",
        "macro_hard_auc",
        "hard_auc_gain_over_teacher",
    ):
        assert token in source


def main():
    test_extractor_preserves_candidate_identity_and_teacher_scores()
    test_diagnosis_uses_group_loso_and_three_class_prototypes()
    print("Class-conditional prototype offline structure tests passed")


if __name__ == "__main__":
    main()
