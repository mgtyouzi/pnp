from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_diagnosis_treats_prototype_distances_as_optional():
    source = (ROOT / "diagnose_cross_domain.py").read_text(encoding="utf-8")
    assert 'if "proto_distances" in outputs' in source
    assert 'raw_debug["proto_distances_in"] is not None' in source


def test_eval_loader_resizes_only_training_queue_buffers():
    source = (ROOT / "diagnose_cross_domain.py").read_text(encoding="utf-8")
    assert "_resize_eval_only_prototype_buffers" in source
    for name in (
        "foreground_queue",
        "background_queue",
        "foreground_queue_priorities",
        "background_queue_priorities",
    ):
        assert name in source


def test_sscp_exposes_distance_diagnostics():
    source = (
        ROOT / "models" / "source_supervised_candidate_proto.py"
    ).read_text(encoding="utf-8")
    assert source.count('"prototype_distances"') >= 2


def test_sscp_queue_counts_use_current_buffer_names():
    source = (ROOT / "diagnose_cross_domain.py").read_text(encoding="utf-8")
    assert '"foreground_queue_count"' in source
    assert '"background_queue_count"' in source


def main():
    test_diagnosis_treats_prototype_distances_as_optional()
    test_eval_loader_resizes_only_training_queue_buffers()
    test_sscp_exposes_distance_diagnostics()
    test_sscp_queue_counts_use_current_buffer_names()
    print("Cross-domain prototype compatibility tests passed")


if __name__ == "__main__":
    main()
