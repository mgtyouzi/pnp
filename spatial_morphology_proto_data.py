from typing import Dict, Sequence, Tuple

import numpy as np


SPATIAL_MORPHOLOGY_DATA_VERSION = "spatial_morphology_proto_data_v1_20260829"
SPATIAL_MORPHOLOGY_ARCHIVE_VERSION = (
    "spatial_morphology_proto_archive_v1_20260829"
)
SPATIAL_MORPHOLOGY_FEATURE_SOURCE = (
    "p2p_fpn_p2_p3_local_spatial_center_ring_contrast"
)


def image_space_grid_offsets(grid_size: int, radius: float) -> np.ndarray:
    if grid_size < 3 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer >= 3")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    axis = np.linspace(-float(radius), float(radius), int(grid_size), dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack([xx, yy], axis=-1).reshape(-1, 2)


def sample_spatial_fpn_patches(
    features,
    points,
    *,
    image_size: Tuple[int, int],
    strides: Sequence[int],
    level_indices: Sequence[int] = (0, 1),
    grid_size: int = 5,
    radius: float = 12.0,
    align_corners: bool = True,
):
    import torch
    import torch.nn.functional as F

    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, N, 2]")
    if len(features) != len(strides):
        raise ValueError("features and strides must have the same length")
    if not level_indices:
        raise ValueError("at least one FPN level is required")
    if min(level_indices) < 0 or max(level_indices) >= len(features):
        raise IndexError("FPN level index is out of range")

    image_height, image_width = map(int, image_size)
    offsets = torch.as_tensor(
        image_space_grid_offsets(grid_size, radius),
        dtype=points.dtype,
        device=points.device,
    )
    sample_points = points.unsqueeze(2) + offsets.view(1, 1, -1, 2)
    valid = (
        (sample_points[..., 0] >= 0)
        & (sample_points[..., 0] <= image_width - 1)
        & (sample_points[..., 1] >= 0)
        & (sample_points[..., 1] <= image_height - 1)
    )

    level_patches = []
    for level_index in level_indices:
        feature = features[level_index]
        if feature.shape[0] != points.shape[0]:
            raise ValueError("feature and point batch sizes differ")
        feature_height, feature_width = feature.shape[-2:]
        stride = float(strides[level_index])
        feature_points = sample_points / stride
        if align_corners:
            denominator = torch.tensor(
                [max(feature_width - 1, 1), max(feature_height - 1, 1)],
                dtype=points.dtype,
                device=points.device,
            )
        else:
            denominator = torch.tensor(
                [feature_width, feature_height],
                dtype=points.dtype,
                device=points.device,
            )
            feature_points = feature_points + 0.5
        grid = 2.0 * feature_points / denominator - 1.0
        grid = grid.reshape(points.shape[0], -1, 1, 2)
        sampled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        sampled = sampled.reshape(
            points.shape[0],
            points.shape[1],
            grid_size,
            grid_size,
            feature.shape[1],
        )
        level_patches.append(sampled)
    patches = torch.stack(level_patches, dim=2)
    valid = valid.reshape(
        points.shape[0], points.shape[1], grid_size, grid_size
    )
    valid = valid.unsqueeze(2).expand(-1, -1, len(level_indices), -1, -1)
    return patches, valid


def _weighted_region_mean(values, valid, region):
    import torch

    weights = (valid & region).to(values.dtype)
    denominator = weights.sum(dim=(-2, -1), keepdim=False).clamp_min(1.0)
    numerator = (values * weights.unsqueeze(-1)).sum(dim=(-3, -2))
    return numerator / denominator.unsqueeze(-1)


def build_spatial_morphology_descriptor(patches, valid):
    import torch
    import torch.nn.functional as F

    if patches.ndim != 5:
        raise ValueError("patches must have shape [N, L, G, G, C]")
    if valid.shape != patches.shape[:-1]:
        raise ValueError("valid mask must align with spatial patches")
    if patches.shape[-3] != patches.shape[-2] or patches.shape[-3] % 2 == 0:
        raise ValueError("spatial patch must be an odd square")

    grid_size = patches.shape[-3]
    center_index = grid_size // 2
    spatial_shape = (1, 1, grid_size, grid_size)
    center_region = torch.zeros(spatial_shape, dtype=torch.bool, device=patches.device)
    center_region[
        ...,
        center_index - 1 : center_index + 2,
        center_index - 1 : center_index + 2,
    ] = True
    ring_region = ~center_region

    center = _weighted_region_mean(patches, valid, center_region)
    ring = _weighted_region_mean(patches, valid, ring_region)
    contrast = center - ring
    vector_descriptor = torch.cat(
        [
            F.normalize(center, dim=-1, eps=1e-6),
            F.normalize(ring, dim=-1, eps=1e-6),
            F.normalize(contrast, dim=-1, eps=1e-6),
        ],
        dim=-1,
    )

    valid_float = valid.to(patches.dtype)
    global_mean = _weighted_region_mean(
        patches,
        valid,
        torch.ones(spatial_shape, dtype=torch.bool, device=patches.device),
    )
    residual = patches - global_mean.unsqueeze(-2).unsqueeze(-2)
    variance = (
        residual.square() * valid_float.unsqueeze(-1)
    ).sum(dim=(-3, -2, -1)) / (
        valid_float.sum(dim=(-2, -1)).clamp_min(1.0) * patches.shape[-1]
    )

    quadrant_vectors = []
    for y_slice, x_slice in (
        (slice(0, center_index), slice(0, center_index)),
        (slice(0, center_index), slice(center_index + 1, grid_size)),
        (slice(center_index + 1, grid_size), slice(0, center_index)),
        (
            slice(center_index + 1, grid_size),
            slice(center_index + 1, grid_size),
        ),
    ):
        region = torch.zeros(
            spatial_shape, dtype=torch.bool, device=patches.device
        )
        region[..., y_slice, x_slice] = True
        quadrant_vectors.append(_weighted_region_mean(patches, valid, region))
    quadrants = torch.stack(quadrant_vectors, dim=-2)
    quadrant_distance = (quadrants - center.unsqueeze(-2)).norm(dim=-1)

    center_norm = center.norm(dim=-1)
    ring_norm = ring.norm(dim=-1)
    contrast_norm = contrast.norm(dim=-1)
    center_ring_cosine = F.cosine_similarity(center, ring, dim=-1, eps=1e-6)
    valid_fraction = valid_float.mean(dim=(-2, -1))
    scalar_descriptor = torch.stack(
        [
            torch.log1p(center_norm),
            torch.log1p(ring_norm),
            torch.log1p(contrast_norm),
            center_ring_cosine,
            torch.log1p(variance),
            torch.log1p(quadrant_distance.mean(dim=-1)),
            torch.log1p(quadrant_distance.std(dim=-1, unbiased=False)),
            valid_fraction,
        ],
        dim=-1,
    )
    descriptor = torch.cat(
        [vector_descriptor.flatten(1), scalar_descriptor.flatten(1)], dim=-1
    )
    if not torch.isfinite(descriptor).all():
        raise RuntimeError("spatial morphology descriptor contains non-finite values")
    diagnostics = {
        "minimum_valid_fraction": valid_fraction.min(),
        "mean_valid_fraction": valid_fraction.mean(),
        "mean_center_ring_cosine": center_ring_cosine.mean(),
        "mean_contrast_norm": contrast_norm.mean(),
    }
    return descriptor, diagnostics


def validate_spatial_archive_contract(manifest: Dict[str, object]) -> None:
    if manifest.get("version") != SPATIAL_MORPHOLOGY_ARCHIVE_VERSION:
        raise RuntimeError("spatial morphology archive version mismatch")
    if manifest.get("feature_source") != SPATIAL_MORPHOLOGY_FEATURE_SOURCE:
        raise RuntimeError(
            "archive is not built from an independent spatial morphology feature"
        )
    sampling = manifest.get("spatial_sampling")
    if not isinstance(sampling, dict):
        raise RuntimeError("spatial morphology sampling metadata is missing")
    required = {"level_indices", "strides", "grid_size", "radius"}
    if not required.issubset(sampling):
        raise RuntimeError("spatial morphology sampling metadata is incomplete")
    if manifest.get("phase") != "train":
        raise RuntimeError("spatial morphology LOSO archive must use source train")
