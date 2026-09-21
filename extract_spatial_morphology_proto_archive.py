import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from build_frozen_teacher_fg_bank import (
    deterministic_dataset,
    file_sha256,
    get_args_parser,
    slide_group_id,
)
from candidate_conditioned_proto_data import CATEGORY_NAME_TO_ID
from extract_candidate_conditioned_proto_archive import _validate_exact_counts
from extract_frozen_teacher_candidate_audit import strict_teacher_collate
from frozen_proto_rescoring import evaluate_image, label_candidate_diagnostics
from models.detr import build_model
from prototype_audit_sampling import stratified_round_robin_indices
from spatial_morphology_proto_data import (
    SPATIAL_MORPHOLOGY_ARCHIVE_VERSION,
    SPATIAL_MORPHOLOGY_DATA_VERSION,
    SPATIAL_MORPHOLOGY_FEATURE_SOURCE,
    build_spatial_morphology_descriptor,
    sample_spatial_fpn_patches,
)
from utils import load_model_weights


def get_parser() -> argparse.ArgumentParser:
    parser = get_args_parser()
    parser.description = (
        "Extract exact P2P candidate labels and independent P2/P3 local morphology."
    )
    parser.add_argument("--dedup_interval", default=15.0, type=float)
    parser.add_argument("--match_dis", default=15.0, type=float)
    parser.add_argument("--near_radius", default=30.0, type=float)
    parser.add_argument("--spatial_levels", default="0,1")
    parser.add_argument("--spatial_grid_size", default=5, type=int)
    parser.add_argument("--spatial_radius", default=12.0, type=float)
    parser.set_defaults(phase="train")
    return parser


def _parse_levels(value: str, available: int) -> Sequence[int]:
    levels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not levels:
        raise ValueError("spatial_levels cannot be empty")
    if len(set(levels)) != len(levels):
        raise ValueError("spatial_levels cannot contain duplicates")
    if min(levels) < 0 or max(levels) >= available:
        raise ValueError(f"spatial_levels must be in [0, {available - 1}]")
    return levels


def _stack(parts: List[np.ndarray], width: int = None, dtype=np.float32):
    if not parts:
        shape = (0, width) if width is not None else (0,)
        return np.zeros(shape, dtype=dtype)
    return np.concatenate(parts, axis=0).astype(dtype, copy=False)


@torch.inference_mode()
def extract_archive(args, model, dataset, selected_indices, device, levels):
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=strict_teacher_collate,
    )
    descriptor_width = len(levels) * (3 * args.hidden_dim + 8)
    morphology_parts = []
    valid_fraction_parts = []
    contrast_norm_parts = []
    category_parts = []
    raw_margin_parts = []
    raw_probability_parts = []
    raw_class_parts = []
    image_index_parts = []
    group_index_parts = []
    candidate_index_parts = []
    gt_index_parts = []
    point_parts = []
    group_to_index = {}
    counts = Counter()
    image_manifest = []

    for local_index, (images, points, labels, lengths, image_paths) in enumerate(loader):
        del labels, lengths
        image_path = image_paths[0]
        image_name = os.path.basename(image_path)
        group_name = slide_group_id(image_name)
        group_id = group_to_index.setdefault(group_name, len(group_to_index))
        images = images.to(device, non_blocking=True)
        anchors = model.get_aps(images)
        pyramid = model.backbone(images)
        reg_features, cls_features, _, _ = model.extract_features(pyramid, anchors)
        predicted_offsets = model.reg_head(reg_features)
        predicted_points = predicted_offsets + anchors
        raw_logits = model.cls_head(cls_features)
        height, width = images.shape[-2:]
        in_border = (
            (predicted_points[0, :, 0] >= 0)
            & (predicted_points[0, :, 0] < width)
            & (predicted_points[0, :, 1] >= 0)
            & (predicted_points[0, :, 1] < height)
        )
        filtered_points = predicted_points[0, in_border]
        filtered_logits = raw_logits[0, in_border]
        gt_points = points[0][points[0] != -1].reshape(-1, 2).to(device)
        points_np = filtered_points.detach().cpu().numpy().astype(np.float32)
        logits_np = filtered_logits.detach().cpu().numpy().astype(np.float32)
        gt_np = gt_points.detach().cpu().numpy().astype(np.float32)
        rows = label_candidate_diagnostics(
            points_np,
            logits_np,
            np.zeros_like(logits_np, dtype=np.float32),
            gt_np,
            dedup_interval=args.dedup_interval,
            match_dis=args.match_dis,
            near_radius=args.near_radius,
        )
        expected = evaluate_image(
            points_np,
            logits_np,
            gt_np,
            dedup_interval=args.dedup_interval,
            match_dis=args.match_dis,
            near_radius=args.near_radius,
        )
        _validate_exact_counts(rows, expected, image_name)

        local_counts = Counter(row["category"] for row in rows)
        image_row = {
            "dataset_index": int(selected_indices[local_index]),
            "image_index": local_index,
            "image": image_name,
            "group": group_name,
            "group_index": group_id,
            "gt_points": int(len(gt_np)),
            "candidate_counts": dict(local_counts),
        }
        if rows:
            candidate_indices = np.asarray(
                [int(row["candidate_index"]) for row in rows], dtype=np.int64
            )
            if candidate_indices.min() < 0 or candidate_indices.max() >= len(
                filtered_points
            ):
                raise RuntimeError(f"candidate point index is invalid for {image_name}")
            selected_points = filtered_points[candidate_indices].unsqueeze(0)
            patches, valid = sample_spatial_fpn_patches(
                pyramid,
                selected_points,
                image_size=(height, width),
                strides=model.strides,
                level_indices=levels,
                grid_size=args.spatial_grid_size,
                radius=args.spatial_radius,
            )
            descriptor, diagnostics = build_spatial_morphology_descriptor(
                patches[0].float(), valid[0]
            )
            if descriptor.shape != (len(rows), descriptor_width):
                raise RuntimeError(
                    f"unexpected morphology descriptor shape for {image_name}: "
                    f"{tuple(descriptor.shape)}"
                )
            morphology_parts.append(
                descriptor.detach().cpu().numpy().astype(np.float16)
            )
            candidate_valid = valid[0].float().mean(dim=(-2, -1)).min(dim=1).values
            valid_fraction_parts.append(candidate_valid.cpu().numpy())
            contrast_norm_parts.append(
                np.full(
                    len(rows),
                    float(diagnostics["mean_contrast_norm"].cpu().item()),
                    dtype=np.float32,
                )
            )
            category_parts.append(
                np.asarray(
                    [CATEGORY_NAME_TO_ID[row["category"]] for row in rows],
                    dtype=np.uint8,
                )
            )
            raw_margin_parts.append(
                np.asarray([row["raw_margin"] for row in rows], dtype=np.float32)
            )
            raw_probability_parts.append(
                np.asarray(
                    [row["raw_cell_probability"] for row in rows], dtype=np.float32
                )
            )
            raw_class_parts.append(
                np.asarray([row["raw_class"] for row in rows], dtype=np.uint8)
            )
            image_index_parts.append(np.full(len(rows), local_index, dtype=np.int32))
            group_index_parts.append(np.full(len(rows), group_id, dtype=np.int16))
            candidate_index_parts.append(candidate_indices.astype(np.int32))
            gt_index_parts.append(
                np.asarray(
                    [int(row.get("gt_index", -1)) for row in rows], dtype=np.int32
                )
            )
            point_parts.append(points_np[candidate_indices])
            counts.update(local_counts)
            image_row.update(
                {
                    "minimum_valid_fraction": float(candidate_valid.min().item()),
                    "mean_valid_fraction": float(candidate_valid.mean().item()),
                    "mean_contrast_norm": float(
                        diagnostics["mean_contrast_norm"].cpu().item()
                    ),
                }
            )
        image_manifest.append(image_row)
        if (local_index + 1) % 25 == 0:
            print(
                f"[Spatial-morphology-archive] extracted={local_index + 1}/"
                f"{len(loader)}",
                flush=True,
            )

    arrays = {
        "morphology_features": _stack(
            morphology_parts, descriptor_width, dtype=np.float16
        ),
        "spatial_valid_fraction": _stack(valid_fraction_parts),
        "spatial_contrast_norm": _stack(contrast_norm_parts),
        "category": _stack(category_parts, dtype=np.uint8),
        "raw_margin": _stack(raw_margin_parts),
        "raw_cell_probability": _stack(raw_probability_parts),
        "raw_class": _stack(raw_class_parts, dtype=np.uint8),
        "image_index": _stack(image_index_parts, dtype=np.int32),
        "group_index": _stack(group_index_parts, dtype=np.int16),
        "candidate_index": _stack(candidate_index_parts, dtype=np.int32),
        "gt_index": _stack(gt_index_parts, dtype=np.int32),
        "predicted_point": _stack(point_parts, 2),
    }
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"spatial morphology arrays are misaligned: {lengths}")
    if len(arrays["morphology_features"]) and not np.isfinite(
        arrays["morphology_features"]
    ).all():
        raise RuntimeError("spatial morphology archive contains non-finite values")
    return arrays, image_manifest, group_to_index, counts, descriptor_width


def main() -> None:
    args = get_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original P2P implementation")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    mean_std_path = args.mean_std_path or os.path.join(args.dataset, "mean_std.npy")
    mean, std = np.load(mean_std_path)
    dataset = deterministic_dataset(args.dataset, mean, std, phase=args.phase)
    selected_indices = (
        stratified_round_robin_indices(
            dataset.files,
            args.max_images,
            key=lambda path: slide_group_id(os.path.basename(path)),
        )
        if args.max_images > 0
        else list(range(len(dataset)))
    )

    model = build_model(args).to(device)
    checkpoint, load_result = load_model_weights(args.checkpoint, model)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "detector checkpoint mismatch: "
            f"missing={list(load_result.missing_keys)}, "
            f"unexpected={list(load_result.unexpected_keys)}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    levels = _parse_levels(args.spatial_levels, model.num_levels)

    arrays, image_manifest, group_to_index, counts, descriptor_width = extract_archive(
        args, model, dataset, selected_indices, device, levels
    )
    archive_path = os.path.join(args.output_dir, "spatial_morphology_archive.npz")
    np.savez_compressed(archive_path, **arrays)
    manifest: Dict[str, object] = {
        "version": SPATIAL_MORPHOLOGY_ARCHIVE_VERSION,
        "data_contract_version": SPATIAL_MORPHOLOGY_DATA_VERSION,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": (
            checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        ),
        "dataset": os.path.abspath(args.dataset),
        "phase": args.phase,
        "feature_source": SPATIAL_MORPHOLOGY_FEATURE_SOURCE,
        "available_feature_sources": {
            "spatial_morphology": SPATIAL_MORPHOLOGY_FEATURE_SOURCE
        },
        "spatial_sampling": {
            "level_indices": list(levels),
            "strides": [int(model.strides[index]) for index in levels],
            "grid_size": int(args.spatial_grid_size),
            "radius": float(args.spatial_radius),
            "descriptor": "normalized_center_ring_contrast_plus_8_shape_scalars",
            "descriptor_width": int(descriptor_width),
            "border_policy": "valid_weighted_regions_plus_valid_fraction",
        },
        "candidate_protocol": {
            "border_filter": True,
            "raw_cell_only_for_fp": True,
            "dedup_interval": args.dedup_interval,
            "match_dis": args.match_dis,
            "near_radius": args.near_radius,
        },
        "category": CATEGORY_NAME_TO_ID,
        "group_to_index": group_to_index,
        "counts": dict(counts),
        "sampling": {
            "strategy": "all" if args.max_images <= 0 else "slide_group_round_robin",
            "requested_max_images": int(args.max_images),
            "selected_images": len(selected_indices),
        },
        "images": image_manifest,
    }
    with open(
        os.path.join(args.output_dir, "spatial_morphology_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[Spatial-morphology-archive] candidates={len(arrays['category'])}")
    print(f"[Spatial-morphology-archive] descriptor_width={descriptor_width}")
    print(f"[Spatial-morphology-archive] counts={dict(counts)}")
    print(f"[Spatial-morphology-archive] output={archive_path}")


if __name__ == "__main__":
    main()
