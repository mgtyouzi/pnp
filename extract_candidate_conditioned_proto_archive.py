import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from build_frozen_teacher_fg_bank import (
    deterministic_dataset,
    file_sha256,
    get_args_parser,
    slide_group_id,
)
from candidate_conditioned_proto_data import (
    CANDIDATE_CONDITIONED_DATA_VERSION,
    CATEGORY_NAME_TO_ID,
)
from extract_frozen_teacher_candidate_audit import strict_teacher_collate
from frozen_proto_rescoring import evaluate_image, label_candidate_diagnostics
from models.detr import build_model
from prototype_audit_sampling import stratified_round_robin_indices
from utils import load_model_weights


ARCHIVE_VERSION = "candidate_conditioned_proto_archive_v2_20260829"


def get_parser() -> argparse.ArgumentParser:
    parser = get_args_parser()
    parser.description = (
        "Extract exact post-P2P TP and false-positive candidate features."
    )
    parser.add_argument("--dedup_interval", default=15.0, type=float)
    parser.add_argument("--match_dis", default=15.0, type=float)
    parser.add_argument("--near_radius", default=30.0, type=float)
    parser.set_defaults(phase="train")
    return parser


def _stack(parts: List[np.ndarray], width: int = None, dtype=np.float32):
    if not parts:
        shape = (0, width) if width is not None else (0,)
        return np.zeros(shape, dtype=dtype)
    return np.concatenate(parts, axis=0).astype(dtype, copy=False)


def _validate_exact_counts(rows, expected: Dict[str, float], image_name: str) -> None:
    fp_categories = {
        "duplicate_fp",
        "near_miss_fp",
        "background_far_fp",
        "empty_image_fp",
    }
    observed = {
        "tp": sum(row["category"] == "tp" for row in rows),
        "fp": sum(row["category"] in fp_categories for row in rows),
        "low_fn": sum(row["category"] == "low_score_fn" for row in rows),
    }
    wanted = {
        "tp": int(expected["tp"]),
        "fp": int(expected["fp"]),
        "low_fn": int(expected["fn_low_score_or_background"]),
    }
    if observed != wanted:
        raise RuntimeError(
            f"candidate labels diverged for {image_name}: "
            f"observed={observed}, expected={wanted}"
        )


@torch.inference_mode()
def extract_archive(args, model, dataset, selected_indices, device):
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=strict_teacher_collate,
    )
    feature_parts = []
    reg_feature_parts = []
    cls_attn_parts = []
    reg_attn_parts = []
    reg_offset_parts = []
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
        reg_features, cls_features, reg_attn, cls_attn = model.extract_features(pyramid, anchors)
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
        filtered_features = F.normalize(
            cls_features[0, in_border].float(), dim=-1, eps=1e-6
        )
        filtered_reg_features = F.normalize(
            reg_features[0, in_border].float(), dim=-1, eps=1e-6
        )
        filtered_cls_attn = cls_attn[0, in_border].squeeze(-1).float()
        filtered_reg_attn = reg_attn[0, in_border].squeeze(-1).float()
        filtered_reg_offset = predicted_offsets[0, in_border].float()
        gt_points = points[0][points[0] != -1].reshape(-1, 2).to(device)
        points_np = filtered_points.detach().cpu().numpy().astype(np.float32)
        logits_np = filtered_logits.detach().cpu().numpy().astype(np.float32)
        gt_np = gt_points.detach().cpu().numpy().astype(np.float32)
        dummy_proto = np.zeros_like(logits_np, dtype=np.float32)
        rows = label_candidate_diagnostics(
            points_np,
            logits_np,
            dummy_proto,
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
        image_manifest.append(
            {
                "dataset_index": int(selected_indices[local_index]),
                "image_index": local_index,
                "image": image_name,
                "group": group_name,
                "group_index": group_id,
                "gt_points": int(len(gt_np)),
                "candidate_counts": dict(local_counts),
            }
        )
        if rows:
            candidate_indices = np.asarray(
                [int(row["candidate_index"]) for row in rows], dtype=np.int64
            )
            if candidate_indices.min() < 0 or candidate_indices.max() >= len(
                filtered_features
            ):
                raise RuntimeError(f"candidate feature index is invalid for {image_name}")
            selected_features = (
                filtered_features[candidate_indices].detach().cpu().numpy()
            )
            feature_parts.append(selected_features)
            reg_feature_parts.append(
                filtered_reg_features[candidate_indices].detach().cpu().numpy()
            )
            cls_attn_parts.append(
                filtered_cls_attn[candidate_indices].detach().cpu().numpy()
            )
            reg_attn_parts.append(
                filtered_reg_attn[candidate_indices].detach().cpu().numpy()
            )
            reg_offset_parts.append(
                filtered_reg_offset[candidate_indices].detach().cpu().numpy()
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
                    [row["raw_cell_probability"] for row in rows],
                    dtype=np.float32,
                )
            )
            raw_class_parts.append(
                np.asarray([row["raw_class"] for row in rows], dtype=np.uint8)
            )
            image_index_parts.append(
                np.full(len(rows), local_index, dtype=np.int32)
            )
            group_index_parts.append(np.full(len(rows), group_id, dtype=np.int16))
            candidate_index_parts.append(candidate_indices.astype(np.int32))
            gt_index_parts.append(
                np.asarray(
                    [int(row.get("gt_index", -1)) for row in rows],
                    dtype=np.int32,
                )
            )
            point_parts.append(points_np[candidate_indices])
            counts.update(local_counts)
        if (local_index + 1) % 25 == 0:
            print(
                f"[Candidate-conditioned-archive] extracted={local_index + 1}/"
                f"{len(loader)}",
                flush=True,
            )

    arrays = {
        "features": _stack(feature_parts, args.hidden_dim),
        "reg_features": _stack(reg_feature_parts, args.hidden_dim),
        "cls_attn": _stack(cls_attn_parts, model.num_levels),
        "reg_attn": _stack(reg_attn_parts, model.num_levels),
        "reg_offset": _stack(reg_offset_parts, 2),
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
        raise RuntimeError(f"candidate archive arrays are misaligned: {lengths}")
    return arrays, image_manifest, group_to_index, counts


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
    if args.max_images > 0:
        selected_indices = stratified_round_robin_indices(
            dataset.files,
            args.max_images,
            key=lambda path: slide_group_id(os.path.basename(path)),
        )
    else:
        selected_indices = list(range(len(dataset)))

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

    arrays, image_manifest, group_to_index, counts = extract_archive(
        args, model, dataset, selected_indices, device
    )
    archive_path = os.path.join(args.output_dir, "candidate_archive.npz")
    np.savez_compressed(archive_path, **arrays)
    manifest = {
        "version": ARCHIVE_VERSION,
        "data_contract_version": CANDIDATE_CONDITIONED_DATA_VERSION,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": (
            checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        ),
        "dataset": os.path.abspath(args.dataset),
        "phase": args.phase,
        "feature_source": "p2p_cls_features_before_final_classifier",
        "available_feature_sources": {
            "cls_only": "p2p_cls_features_before_final_classifier",
            "cls_reg_context": "p2p_cls_reg_attention_offset_context",
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
        os.path.join(args.output_dir, "candidate_archive_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[Candidate-conditioned-archive] candidates={len(arrays['features'])}")
    print(f"[Candidate-conditioned-archive] counts={dict(counts)}")
    print(f"[Candidate-conditioned-archive] output={archive_path}")


if __name__ == "__main__":
    main()
