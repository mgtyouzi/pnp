from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_extractor_samples_only_labeled_candidates_from_raw_fpn_levels():
    path = ROOT / "extract_spatial_morphology_proto_archive.py"
    source = path.read_text(encoding="utf-8")
    assert "sample_spatial_fpn_patches" in source
    assert "filtered_points[candidate_indices]" in source
    assert "build_spatial_morphology_descriptor" in source
    assert '"morphology_features"' in source
    assert '"features"' not in source


def test_extractor_manifest_records_independent_spatial_contract():
    source = (ROOT / "extract_spatial_morphology_proto_archive.py").read_text(
        encoding="utf-8"
    )
    assert "SPATIAL_MORPHOLOGY_ARCHIVE_VERSION" in source
    assert "SPATIAL_MORPHOLOGY_FEATURE_SOURCE" in source
    assert '"level_indices"' in source
    assert '"grid_size"' in source
    assert '"radius"' in source


def test_smoke_runner_limits_images_and_never_starts_detector_training():
    source = (ROOT / "run_spatial_morphology_proto_gate_gpu3.sh").read_text(
        encoding="utf-8"
    )
    assert 'MAX_IMAGES="${MAX_IMAGES:-200}"' in source
    assert "extract_spatial_morphology_proto_archive.py" in source
    assert "--feature_mode spatial_morphology" in source
    assert "train_p2p" not in source
    assert "100ep" not in source


def main():
    test_extractor_samples_only_labeled_candidates_from_raw_fpn_levels()
    test_extractor_manifest_records_independent_spatial_contract()
    test_smoke_runner_limits_images_and_never_starts_detector_training()
    print("Spatial morphology prototype structure tests passed")


if __name__ == "__main__":
    main()
