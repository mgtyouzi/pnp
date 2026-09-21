import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from spatial_morphology_proto_data import (
    SPATIAL_MORPHOLOGY_FEATURE_SOURCE,
    build_spatial_morphology_descriptor,
    image_space_grid_offsets,
    sample_spatial_fpn_patches,
    validate_spatial_archive_contract,
)
from candidate_conditioned_proto_data import build_candidate_feature_matrix


def test_image_space_grid_is_centered_and_uses_requested_radius():
    offsets = image_space_grid_offsets(grid_size=5, radius=12.0)
    assert offsets.shape == (25, 2)
    assert np.allclose(offsets[12], [0.0, 0.0])
    assert np.isclose(np.abs(offsets).max(), 12.0)
    assert np.allclose(offsets.mean(axis=0), 0.0)


def test_sampling_uses_the_same_image_coordinates_across_fpn_levels():
    if torch is None:
        return
    image_height = image_width = 64
    points = torch.tensor([[[24.0, 40.0]]])
    features = []
    for stride in (2, 4):
        height = image_height // stride
        width = image_width // stride
        yy, xx = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        image_x = xx.float() * stride
        image_y = yy.float() * stride
        features.append(torch.stack([image_x, image_y], dim=0).unsqueeze(0))

    patches, valid = sample_spatial_fpn_patches(
        features,
        points,
        image_size=(image_height, image_width),
        strides=(2, 4),
        level_indices=(0, 1),
        grid_size=5,
        radius=0.0,
    )

    assert patches.shape == (1, 1, 2, 5, 5, 2)
    assert valid.all()
    center = patches[0, 0, :, 2, 2]
    assert torch.allclose(center[:, 0], torch.tensor([24.0, 24.0]), atol=1e-4)
    assert torch.allclose(center[:, 1], torch.tensor([40.0, 40.0]), atol=1e-4)


def test_descriptor_separates_center_ring_and_asymmetry_without_nan():
    if torch is None:
        return
    patches = torch.zeros((1, 2, 5, 5, 3), dtype=torch.float32)
    valid = torch.ones((1, 2, 5, 5), dtype=torch.bool)
    patches[:, :, 1:4, 1:4, 0] = 2.0
    patches[:, :, :, :, 1] = 1.0
    patches[:, :, :2, :2, 2] = 3.0

    descriptor, diagnostics = build_spatial_morphology_descriptor(patches, valid)

    assert descriptor.shape == (1, 2 * 3 * 3 + 2 * 8)
    assert torch.isfinite(descriptor).all()
    assert diagnostics["minimum_valid_fraction"].item() == 1.0
    assert not torch.allclose(descriptor, torch.zeros_like(descriptor))


def test_invalid_border_samples_are_excluded_from_weighted_regions():
    if torch is None:
        return
    patches = torch.ones((1, 1, 5, 5, 2), dtype=torch.float32)
    valid = torch.ones((1, 1, 5, 5), dtype=torch.bool)
    valid[:, :, 0, :] = False
    patches[:, :, 0, :, :] = 1000.0

    descriptor, diagnostics = build_spatial_morphology_descriptor(patches, valid)

    assert torch.isfinite(descriptor).all()
    assert descriptor.abs().max().item() < 100.0
    assert np.isclose(diagnostics["minimum_valid_fraction"].item(), 0.8)


def test_archive_contract_rejects_candidate_head_feature_source():
    valid_manifest = {
        "version": "spatial_morphology_proto_archive_v1_20260829",
        "phase": "train",
        "feature_source": SPATIAL_MORPHOLOGY_FEATURE_SOURCE,
        "spatial_sampling": {
            "level_indices": [0, 1],
            "strides": [2, 4],
            "grid_size": 5,
            "radius": 12.0,
        },
    }
    validate_spatial_archive_contract(valid_manifest)

    invalid = dict(valid_manifest)
    invalid["feature_source"] = "p2p_cls_features_before_final_classifier"
    try:
        validate_spatial_archive_contract(invalid)
    except RuntimeError as error:
        assert "independent spatial" in str(error)
    else:
        raise AssertionError("candidate-head feature source should be rejected")


def test_generic_loso_audit_accepts_only_the_spatial_descriptor_array():
    data = {
        "morphology_features": np.asarray(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float16
        )
    }
    features, source = build_candidate_feature_matrix(
        data, "spatial_morphology"
    )
    assert features.dtype == np.float32
    assert features.shape == (2, 3)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)
    assert source == SPATIAL_MORPHOLOGY_FEATURE_SOURCE


def main():
    test_image_space_grid_is_centered_and_uses_requested_radius()
    test_sampling_uses_the_same_image_coordinates_across_fpn_levels()
    test_descriptor_separates_center_ring_and_asymmetry_without_nan()
    test_invalid_border_samples_are_excluded_from_weighted_regions()
    test_archive_contract_rejects_candidate_head_feature_source()
    test_generic_loso_audit_accepts_only_the_spatial_descriptor_array()
    print("Spatial morphology prototype data tests passed")


if __name__ == "__main__":
    main()
