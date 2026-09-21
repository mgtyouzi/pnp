from pathlib import Path

from decide_discriminative_dual_proxy_gate import evaluate_gate


def passing_metrics():
    return {
        "prototype_macro_auc": 0.80,
        "prototype_min_slide_auc": 0.65,
        "foreground_similarity_gap": 0.08,
        "background_similarity_gap": 0.07,
        "foreground_effective_prototypes": 3.5,
        "background_effective_prototypes": 3.4,
        "foreground_pairwise_similarity_max": 0.82,
        "background_pairwise_similarity_max": 0.84,
        "cross_bank_similarity_max": 0.82,
        "foreground_assignment_share_min": 0.10,
        "background_assignment_share_min": 0.09,
        "optimizer_projector_only": True,
    }


def test_all_required_thresholds_pass():
    decision = evaluate_gate(passing_metrics())
    assert decision["gate_pass"]
    assert decision["failures"] == []


def test_each_failed_threshold_blocks_detector_training():
    cases = {
        "prototype_macro_auc": 0.74,
        "prototype_min_slide_auc": 0.54,
        "foreground_similarity_gap": 0.049,
        "background_similarity_gap": 0.049,
        "foreground_effective_prototypes": 2.9,
        "background_effective_prototypes": 2.9,
        "foreground_pairwise_similarity_max": 0.91,
        "background_pairwise_similarity_max": 0.91,
        "cross_bank_similarity_max": 0.91,
        "foreground_assignment_share_min": 0.049,
        "background_assignment_share_min": 0.049,
        "optimizer_projector_only": False,
    }
    for key, value in cases.items():
        metrics = passing_metrics()
        metrics[key] = value
        decision = evaluate_gate(metrics)
        assert not decision["gate_pass"], key
        assert key in decision["failures"], (key, decision["failures"])


def test_failed_gate_does_not_export_a_bank():
    from decide_discriminative_dual_proxy_gate import export_passed_bank

    root = Path(__file__).resolve().parent
    checkpoint = root / "checkpoint_that_must_not_be_loaded.pth"
    bank = root / "bank_that_must_not_be_exported.pth"
    assert not checkpoint.exists()
    assert not bank.exists()
    exported = export_passed_bank(
        checkpoint, bank, evaluate_gate({**passing_metrics(), "prototype_macro_auc": 0.5})
    )
    assert not exported
    assert not bank.exists()


def test_exported_bank_records_detector_initialization_identity():
    source = (Path(__file__).resolve().parent / "decide_discriminative_dual_proxy_gate.py").read_text(
        encoding="utf-8"
    )
    assert "detector_init_checkpoint" in source
    assert "detector_init_sha256" in source
    assert "init_checkpoint" in source


def test_gate_evaluator_accepts_explicit_mean_std_path():
    source = (
        Path(__file__).resolve().parent / "evaluate_discriminative_dual_proxy_gate.py"
    ).read_text(encoding="utf-8")
    assert '"--mean_std_path"' in source
    assert "explicit mean_std.npy used by the gate dataset" in source


def main():
    test_all_required_thresholds_pass()
    test_each_failed_threshold_blocks_detector_training()
    test_failed_gate_does_not_export_a_bank()
    test_exported_bank_records_detector_initialization_identity()
    test_gate_evaluator_accepts_explicit_mean_std_path()
    print("Discriminative dual-proxy gate tests passed")


if __name__ == "__main__":
    main()
