import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from build_frozen_teacher_fg_bank import (
    deterministic_dataset,
    file_sha256,
    get_args_parser,
    slide_group_id,
)
from matcher import build_matcher
from models.detr import build_model
from models.frozen_teacher_fg_prototype import FrozenTeacherForegroundPrototype
from prototype_audit_sampling import stratified_round_robin_indices
from utils import load_model_weights


CANDIDATE_TYPES = {"positive": 0, "hard_negative": 1, "random_negative": 2}
AUDIT_VERSION = "frozen_teacher_candidate_audit_v1_20260816"


def strict_teacher_collate(batch):
    images, points, labels, names = zip(*batch)
    lengths = [len(value) for value in labels]
    points = pad_sequence(points, batch_first=True, padding_value=-1).reshape(
        len(batch), -1
    )
    labels = pad_sequence(labels, batch_first=True, padding_value=-1).reshape(
        len(batch), -1
    )
    return torch.stack(images), points.float(), labels.long(), lengths, list(names)


def stack_tensor(items, width=None, dtype=np.float32):
    if not items:
        shape = (0, width) if width is not None else (0,)
        return np.zeros(shape, dtype=dtype)
    values = torch.cat(items, dim=0).numpy()
    return values.astype(dtype, copy=False)


@torch.inference_mode()
def extract_candidates(model, matcher, selector, loader, device, maximum_images):
    feature_parts = []
    type_parts = []
    score_parts = []
    matched_distance_parts = []
    nearest_distance_parts = []
    image_index_parts = []
    group_index_parts = []
    point_parts = []
    anchor_parts = []
    image_manifest = []
    group_to_index = {}
    total_candidates = Counter()

    for image_index, (images, points, labels, lengths, image_paths) in enumerate(loader):
        if maximum_images > 0 and image_index >= maximum_images:
            break
        if len(image_paths) != 1:
            raise RuntimeError("candidate extraction requires batch_size=1")
        image_name = os.path.basename(image_paths[0])
        group_name = slide_group_id(image_name)
        group_index = group_to_index.setdefault(group_name, len(group_to_index))
        manifest_row = {
            "image_index": image_index,
            "image": image_name,
            "group_index": group_index,
            "group": group_name,
            "gt_points": int(lengths[0]),
        }
        if int(lengths[0]) == 0:
            manifest_row["candidate_counts"] = {name: 0 for name in CANDIDATE_TYPES}
            image_manifest.append(manifest_row)
            continue

        images = images.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = {
            "gt_nums": lengths,
            "gt_points": [
                point_seq[point_seq != -1].reshape(-1, 2) for point_seq in points
            ],
            "gt_labels": [label_seq[label_seq != -1] for label_seq in labels],
        }
        anchors = model.get_aps(images)
        pyramid = model.backbone(images)
        reg_features, cls_features, _, _ = model.extract_features(pyramid, anchors)
        predicted_points = model.reg_head(reg_features) + anchors
        raw_logits = model.cls_head(cls_features)
        indices = matcher(
            {"pnt_coords": predicted_points, "cls_logits": raw_logits}, targets
        )
        selection = selector.select_candidates(
            predicted_points,
            raw_logits,
            targets,
            indices,
            anchor_points=anchors,
        )
        normalized = F.normalize(cls_features, dim=-1, eps=1e-6)
        foreground_scores = raw_logits.softmax(dim=-1)[..., 0]
        positive = torch.where(selection.positive_mask[0])[0]
        if positive.numel() > selector.max_positive_per_image:
            distance = selection.matched_distance[0, positive]
            positive = positive[
                torch.argsort(distance)[: selector.max_positive_per_image]
            ]
        selected_groups = {
            "positive": positive,
            "hard_negative": torch.where(selection.hard_negative_mask[0])[0],
            "random_negative": torch.where(selection.random_negative_mask[0])[0],
        }
        manifest_row["candidate_counts"] = {
            name: int(selected.numel()) for name, selected in selected_groups.items()
        }
        image_manifest.append(manifest_row)

        for candidate_name, selected in selected_groups.items():
            if selected.numel() == 0:
                continue
            count = int(selected.numel())
            feature_parts.append(normalized[0, selected].detach().cpu())
            type_parts.append(
                torch.full((count,), CANDIDATE_TYPES[candidate_name], dtype=torch.uint8)
            )
            score_parts.append(foreground_scores[0, selected].detach().cpu())
            matched_distance_parts.append(
                selection.matched_distance[0, selected].detach().cpu()
            )
            nearest_distance_parts.append(
                selection.nearest_gt_distance[0, selected].detach().cpu()
            )
            image_index_parts.append(
                torch.full((count,), image_index, dtype=torch.int32)
            )
            group_index_parts.append(
                torch.full((count,), group_index, dtype=torch.int16)
            )
            point_parts.append(predicted_points[0, selected].detach().cpu())
            anchor_parts.append(anchors[0, selected].detach().cpu())
            total_candidates[candidate_name] += count

        if (image_index + 1) % 50 == 0:
            print(
                f"[Candidate-audit] processed={image_index + 1}/"
                f"{maximum_images or len(loader.dataset)}",
                flush=True,
            )

    arrays = {
        "features": stack_tensor(feature_parts, selector.feat_dim),
        "candidate_type": stack_tensor(type_parts, dtype=np.uint8),
        "teacher_score": stack_tensor(score_parts),
        "matched_distance": stack_tensor(matched_distance_parts),
        "nearest_gt_distance": stack_tensor(nearest_distance_parts),
        "image_index": stack_tensor(image_index_parts, dtype=np.int32),
        "group_index": stack_tensor(group_index_parts, dtype=np.int16),
        "predicted_point": stack_tensor(point_parts, 2),
        "anchor_point": stack_tensor(anchor_parts, 2),
    }
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"candidate audit arrays are misaligned: {lengths}")
    return arrays, image_manifest, group_to_index, total_candidates


def main():
    args = get_args_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original P2P implementation")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    mean_std_path = args.mean_std_path or os.path.join(args.dataset, "mean_std.npy")
    mean, std = np.load(mean_std_path)
    dataset = deterministic_dataset(
        args.dataset,
        mean,
        std,
        phase=args.phase,
    )
    requested_max_images = int(args.max_images)
    if requested_max_images > 0:
        selected = stratified_round_robin_indices(
            dataset.files,
            requested_max_images,
            key=lambda path: slide_group_id(os.path.basename(path)),
        )
        dataset.data = [dataset.data[index] for index in selected]
        dataset.files = [dataset.files[index] for index in selected]
        print(
            f"[Candidate-audit] stratified_images={len(selected)}, "
            f"requested={requested_max_images}",
            flush=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=strict_teacher_collate,
    )
    model = build_model(args).to(device)
    checkpoint, load_result = load_model_weights(args.checkpoint, model)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "teacher checkpoint mismatch: "
            f"missing={list(load_result.missing_keys)}, "
            f"unexpected={list(load_result.unexpected_keys)}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    matcher = build_matcher(args)
    selector = FrozenTeacherForegroundPrototype(
        feat_dim=args.hidden_dim,
        num_prototypes=args.num_prototypes,
        negative_bank_size=args.negative_bank_size,
        negative_sample_size=min(args.negative_sample_size, args.negative_bank_size),
        positive_radius=args.positive_radius,
        background_radius=args.background_radius,
        max_positive_per_image=args.max_positive_per_image,
        max_hard_negative_per_image=args.max_hard_negative_per_image,
        max_random_negative_per_image=args.max_random_negative_per_image,
        sampling_seed=args.seed,
    ).to(device)
    arrays, image_manifest, group_to_index, counts = extract_candidates(
        model, matcher, selector, loader, device, 0
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "candidate_features.npz"), **arrays
    )
    manifest = {
        "version": AUDIT_VERSION,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": (
            checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        ),
        "dataset": os.path.abspath(args.dataset),
        "phase": args.phase,
        "sampling": {
            "strategy": "slide_group_round_robin",
            "requested_max_images": requested_max_images,
            "selected_images": len(dataset),
        },
        "feature_source": "p2p_cls_features_before_final_classifier",
        "candidate_type": CANDIDATE_TYPES,
        "group_to_index": group_to_index,
        "counts": dict(counts),
        "images": image_manifest,
        "selection": {
            "positive_radius": args.positive_radius,
            "background_radius": args.background_radius,
            "max_positive_per_image": args.max_positive_per_image,
            "max_hard_negative_per_image": args.max_hard_negative_per_image,
            "max_random_negative_per_image": args.max_random_negative_per_image,
        },
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
    }
    with open(
        os.path.join(args.output_dir, "image_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[Candidate-audit] candidates={len(arrays['features'])}")
    print(f"[Candidate-audit] groups={len(group_to_index)}")
    print(f"[Candidate-audit] output={args.output_dir}")


if __name__ == "__main__":
    main()
