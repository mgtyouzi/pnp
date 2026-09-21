import argparse
import hashlib
import json
import os
import random
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset_zy_src import DataFolder
from matcher import build_matcher
from models.detr import build_model
from models.frozen_teacher_fg_prototype import (
    FROZEN_TEACHER_PROTOTYPE_VERSION,
    FrozenTeacherForegroundPrototype,
)
from transforms import Preprocessing
from utils import load_model_weights


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fixed foreground prototypes from a frozen P2P teacher."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", choices=("train", "test"), default="train")
    parser.add_argument("--mean_std_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--support_fraction", default=0.8, type=float)
    parser.add_argument("--max_images", default=0, type=int)
    parser.add_argument("--num_prototypes", default=4, type=int)
    parser.add_argument("--negative_bank_size", default=8192, type=int)
    parser.add_argument("--negative_sample_size", default=256, type=int)
    parser.add_argument("--negative_energy_blocks", default=4, type=int)
    parser.add_argument("--negative_sampling_seed", default=0, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--positive_radius", default=10.0, type=float)
    parser.add_argument("--background_radius", default=30.0, type=float)
    parser.add_argument("--max_positive_per_image", default=64, type=int)
    parser.add_argument("--max_hard_negative_per_image", default=16, type=int)
    parser.add_argument("--max_random_negative_per_image", default=16, type=int)
    parser.add_argument("--kmeans_iterations", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--gate_auc", default=0.70, type=float)
    parser.add_argument("--gate_balanced_accuracy", default=0.65, type=float)
    parser.add_argument("--gate_min_cluster_share", default=0.05, type=float)

    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--position_embedding", default="sine")
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--row", default=2, type=int)
    parser.add_argument("--col", default=2, type=int)
    parser.add_argument("--set_cost_point", default=0.1, type=float)
    parser.add_argument("--set_cost_class", default=1.0, type=float)
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--pre_norm", action="store_true")
    parser.set_defaults(proto_enable=False)
    return parser


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def split_support_calibration(
    files: Sequence[str], support_fraction: float, seed: int
) -> Tuple[Set[str], Set[str]]:
    if not 0.0 < support_fraction < 1.0:
        raise ValueError("support_fraction must be in (0, 1)")
    names = sorted(os.path.basename(path) for path in files)
    grouped = {}
    for name in names:
        grouped.setdefault(slide_group_id(name), []).append(name)
    group_names = sorted(grouped)
    if len(group_names) < 2:
        raise ValueError("at least two slide groups are required for calibration")
    random.Random(seed).shuffle(group_names)
    boundary = int(round(len(group_names) * support_fraction))
    boundary = min(max(boundary, 1), len(group_names) - 1)
    support_groups = set(group_names[:boundary])
    calibration_groups = set(group_names[boundary:])
    support = {
        name for group in support_groups for name in grouped[group]
    }
    calibration = {
        name for group in calibration_groups for name in grouped[group]
    }
    if support & calibration or support | calibration != set(names):
        raise RuntimeError("support/calibration split is not a partition")
    return support, calibration


def slide_group_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if ".kfb_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


class StrictTeacherDataset(DataFolder):
    """Read the requested patch exactly; never replace it with a neighbour."""

    def __getitem__(self, index: int):
        image_path = self.files[index]
        raw_sample = self.read_data(self.data[index], image_path)
        image, points, labels = self.data_transform(raw_sample)
        if image.shape[1] != 1080 or image.shape[2] != 1920:
            raise ValueError(
                f"unexpected teacher patch size for {image_path}: {tuple(image.shape)}"
            )
        return image, points, labels, image_path


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


def deterministic_dataset(
    dataset_root: str,
    mean: np.ndarray,
    std: np.ndarray,
    phase: str = "train",
) -> StrictTeacherDataset:
    dataset = StrictTeacherDataset(
        dataset_root,
        num_classes=1,
        phase=phase,
        data_transform=Preprocessing(mean, std),
    )
    paired = sorted(zip(dataset.data, dataset.files), key=lambda item: item[1])
    dataset.data = [item[0] for item in paired]
    dataset.files = [item[1] for item in paired]
    return dataset


def _empty_feature(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float32)


def _stack(items: List[torch.Tensor], width: int) -> np.ndarray:
    if not items:
        return _empty_feature(width)
    return torch.cat(items, dim=0).numpy().astype(np.float32, copy=False)


@torch.inference_mode()
def extract_teacher_features(
    model: torch.nn.Module,
    matcher,
    selector: FrozenTeacherForegroundPrototype,
    loader: DataLoader,
    support_files: Set[str],
    calibration_files: Set[str],
    maximum_images: int,
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    buckets = {
        split: {"positive": [], "hard_negative": [], "random_negative": []}
        for split in ("support", "calibration")
    }
    score_buckets = {
        split: {"positive": [], "hard_negative": [], "random_negative": []}
        for split in ("support", "calibration")
    }
    counts = {
        "images": 0,
        "support_images": 0,
        "calibration_images": 0,
        "empty_images": 0,
        "outside_split": 0,
    }
    if maximum_images > 0:
        support_quota = max(1, maximum_images // 2)
        calibration_quota = max(1, maximum_images - support_quota)
    else:
        support_quota = calibration_quota = 0

    for images, points, labels, lengths, image_paths in loader:
        if maximum_images > 0 and (
            counts["support_images"] >= support_quota
            and counts["calibration_images"] >= calibration_quota
        ):
            break
        if len(image_paths) != 1:
            raise RuntimeError("teacher extraction requires batch_size=1")
        name = os.path.basename(image_paths[0])
        if name in support_files:
            split = "support"
        elif name in calibration_files:
            split = "calibration"
        else:
            counts["outside_split"] += 1
            continue
        split_count_key = f"{split}_images"
        split_quota = support_quota if split == "support" else calibration_quota
        if maximum_images > 0 and counts[split_count_key] >= split_quota:
            continue
        counts["images"] += 1
        counts[split_count_key] += 1
        if int(lengths[0]) == 0:
            counts["empty_images"] += 1
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
        for batch_idx in range(images.shape[0]):
            positive = torch.where(selection.positive_mask[batch_idx])[0]
            if positive.numel() > selector.max_positive_per_image:
                distance = selection.matched_distance[batch_idx, positive]
                positive = positive[
                    torch.argsort(distance)[: selector.max_positive_per_image]
                ]
            index_groups = {
                "positive": positive,
                "hard_negative": torch.where(
                    selection.hard_negative_mask[batch_idx]
                )[0],
                "random_negative": torch.where(
                    selection.random_negative_mask[batch_idx]
                )[0],
            }
            for group, selected in index_groups.items():
                if selected.numel() == 0:
                    continue
                buckets[split][group].append(
                    normalized[batch_idx, selected].detach().cpu()
                )
                score_buckets[split][group].append(
                    foreground_scores[batch_idx, selected].detach().cpu()
                )
        if counts["images"] % 50 == 0:
            print(
                f"[Teacher-bank] processed={counts['images']}/"
                f"{maximum_images or len(loader.dataset)}",
                flush=True,
            )

    result = {"counts": counts}
    for split in ("support", "calibration"):
        result[split] = {}
        for group in ("positive", "hard_negative", "random_negative"):
            result[split][group] = _stack(buckets[split][group], selector.feat_dim)
            result[split][f"{group}_raw_score"] = (
                torch.cat(score_buckets[split][group]).numpy().astype(np.float32)
                if score_buckets[split][group]
                else np.zeros((0,), dtype=np.float32)
            )
    return result


def spherical_kmeans(
    values: torch.Tensor, clusters: int, iterations: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = F.normalize(values.float(), dim=-1, eps=1e-6)
    if values.shape[0] < clusters:
        raise ValueError(
            f"cannot fit {clusters} prototypes from {values.shape[0]} positive samples"
        )
    first = int(seed) % int(values.shape[0])
    selected = [first]
    nearest_distance = 1.0 - values.matmul(values[first : first + 1].t()).squeeze(1)
    for _ in range(1, clusters):
        next_index = int(torch.argmax(nearest_distance).item())
        selected.append(next_index)
        distance = 1.0 - values.matmul(values[next_index : next_index + 1].t()).squeeze(1)
        nearest_distance = torch.minimum(nearest_distance, distance)
    centers = values[selected].clone()
    assignment = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    for _ in range(max(int(iterations), 1)):
        assignment = values.matmul(centers.t()).argmax(dim=1)
        updated = []
        nearest_similarity = values.matmul(centers.t()).max(dim=1).values
        for cluster_idx in range(clusters):
            members = values[assignment == cluster_idx]
            if members.numel() == 0:
                replacement = values[torch.argmin(nearest_similarity)]
                updated.append(replacement)
            else:
                updated.append(F.normalize(members.mean(dim=0), dim=0, eps=1e-6))
        new_centers = torch.stack(updated)
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    assignment = values.matmul(centers.t()).argmax(dim=1)
    return centers, assignment


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels == 1) / positives
    false_positive = np.cumsum(sorted_labels == 0) / negatives
    true_positive = np.concatenate([[0.0], true_positive, [1.0]])
    false_positive = np.concatenate([[0.0], false_positive, [1.0]])
    return float(np.trapz(true_positive, false_positive))


def evaluate_separation(
    prototypes: torch.Tensor,
    negative_bank: np.ndarray,
    positive: np.ndarray,
    hard_negative: np.ndarray,
    random_negative: np.ndarray,
    temperature: float = 0.1,
    negative_sample_size: int = 256,
    negative_energy_blocks: int = 4,
    negative_sampling_seed: int = 0,
) -> Dict[str, float]:
    if len(positive) == 0 or len(hard_negative) == 0:
        raise RuntimeError("calibration requires positive and hard-negative features")
    if len(negative_bank) == 0:
        raise RuntimeError("ProtoNCE calibration requires a fixed negative bank")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if negative_sample_size <= 0 or negative_energy_blocks <= 0:
        raise ValueError("negative energy sampling parameters must be positive")

    negatives = np.concatenate([hard_negative, random_negative], axis=0)
    device = prototypes.device
    prototypes = F.normalize(prototypes.float(), dim=-1, eps=1e-6)
    fixed_negative_bank = F.normalize(
        torch.from_numpy(negative_bank).float().to(device), dim=-1, eps=1e-6
    )
    sample_size = min(int(negative_sample_size), len(fixed_negative_bank))
    block_count = min(int(negative_energy_blocks), len(fixed_negative_bank))
    block_offsets = torch.remainder(
        int(negative_sampling_seed)
        + torch.arange(block_count, device=device) * sample_size,
        len(fixed_negative_bank),
    )

    def score_values(values: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        relative_scores = []
        foreground_scores = []
        for start in range(0, len(values), 4096):
            query = F.normalize(
                torch.from_numpy(values[start : start + 4096]).float().to(device),
                dim=-1,
                eps=1e-6,
            )
            foreground_logits = query.matmul(prototypes.t())
            foreground_scores.append(foreground_logits.max(dim=1).values.cpu())
            positive_energy = torch.logsumexp(
                foreground_logits / temperature, dim=1
            )
            block_scores = []
            for offset in block_offsets:
                indices = (
                    torch.arange(sample_size, device=device) + int(offset.item())
                ) % len(fixed_negative_bank)
                negative_energy = torch.logsumexp(
                    query.matmul(fixed_negative_bank[indices].t()) / temperature,
                    dim=1,
                )
                block_scores.append(positive_energy - negative_energy)
            relative_scores.append(torch.stack(block_scores).mean(dim=0).cpu())
        return torch.cat(relative_scores), torch.cat(foreground_scores)

    positive_energy_score, positive_score = score_values(positive)
    negative_energy_score, negative_score = score_values(negatives)
    hard_energy_score = negative_energy_score[: len(hard_negative)]
    random_energy_score = negative_energy_score[len(hard_negative) :]
    hard_score = negative_score[: len(hard_negative)]
    random_score = negative_score[len(hard_negative) :]
    scores = torch.cat([positive_energy_score, negative_energy_score]).numpy()
    labels = np.concatenate(
        [
            np.ones(len(positive_energy_score), dtype=np.int64),
            np.zeros(len(negative_energy_score), dtype=np.int64),
        ]
    )
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 501)))
    best = {"balanced_accuracy": -1.0, "threshold": 0.0, "tpr": 0.0, "tnr": 0.0}
    for threshold in thresholds:
        prediction = scores >= threshold
        tpr = float(prediction[labels == 1].mean())
        tnr = float((~prediction[labels == 0]).mean())
        balanced = 0.5 * (tpr + tnr)
        if balanced > best["balanced_accuracy"]:
            best = {
                "balanced_accuracy": balanced,
                "threshold": float(threshold),
                "tpr": tpr,
                "tnr": tnr,
            }
    similarity_scores = torch.cat([positive_score, negative_score]).numpy()
    metrics = {
        "gate_score": "protonce_relative_energy",
        "auc": _roc_auc(labels, scores),
        **best,
        "foreground_similarity_auc": _roc_auc(labels, similarity_scores),
        "positive_energy_mean": float(positive_energy_score.mean().item()),
        "positive_energy_p10": float(torch.quantile(positive_energy_score, 0.1).item()),
        "hard_negative_energy_mean": float(hard_energy_score.mean().item()),
        "hard_negative_energy_p90": float(torch.quantile(hard_energy_score, 0.9).item()),
        "random_negative_energy_mean": (
            float(random_energy_score.mean().item())
            if random_energy_score.numel()
            else None
        ),
        "positive_similarity_mean": float(positive_score.mean().item()),
        "positive_similarity_p10": float(torch.quantile(positive_score, 0.1).item()),
        "hard_negative_similarity_mean": float(hard_score.mean().item()),
        "hard_negative_similarity_p90": float(torch.quantile(hard_score, 0.9).item()),
        "random_negative_similarity_mean": (
            float(random_score.mean().item()) if random_score.numel() else None
        ),
        "temperature": float(temperature),
        "negative_sample_size": int(sample_size),
        "negative_energy_blocks": int(block_count),
        "negative_sampling_seed": int(negative_sampling_seed),
    }
    for fpr_limit in (0.10, 0.20):
        feasible_tpr = []
        for threshold in thresholds:
            prediction = scores >= threshold
            fpr = float(prediction[labels == 0].mean())
            if fpr <= fpr_limit:
                feasible_tpr.append(float(prediction[labels == 1].mean()))
        metrics[f"tpr_at_fpr_{fpr_limit:.2f}"] = max(feasible_tpr or [0.0])
    return metrics


def _deterministic_rows(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(values) <= count:
        return values
    generator = np.random.RandomState(seed)
    return values[generator.choice(len(values), size=count, replace=False)]


def build_bank_payload(
    prototypes: torch.Tensor,
    negative_bank: np.ndarray,
    bank_id: str,
    metadata: Dict,
) -> Dict:
    return {
        "foreground_prototypes": prototypes.detach().cpu(),
        "negative_bank": torch.from_numpy(negative_bank).float(),
        "bank_id": bank_id,
        "metadata": metadata,
    }


def main() -> None:
    args = get_args_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original P2P implementation")
    if args.num_classes != 1:
        raise ValueError("frozen-teacher prototypes support one foreground class")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    mean_std_path = args.mean_std_path or os.path.join(args.dataset, "mean_std.npy")
    if not os.path.isfile(mean_std_path):
        raise FileNotFoundError(f"mean/std not found: {mean_std_path}")
    mean, std = np.load(mean_std_path)
    dataset = deterministic_dataset(args.dataset, mean, std)
    support_files, calibration_files = split_support_calibration(
        dataset.files, args.support_fraction, args.seed
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
            "teacher checkpoint is not an exact architectural match: "
            f"missing={list(load_result.missing_keys)}, "
            f"unexpected={list(load_result.unexpected_keys)}"
        )
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    matcher = build_matcher(args)
    selector = FrozenTeacherForegroundPrototype(
        feat_dim=args.hidden_dim,
        num_prototypes=args.num_prototypes,
        negative_bank_size=args.negative_bank_size,
        negative_sample_size=min(256, args.negative_bank_size),
        positive_radius=args.positive_radius,
        background_radius=args.background_radius,
        max_positive_per_image=args.max_positive_per_image,
        max_hard_negative_per_image=args.max_hard_negative_per_image,
        max_random_negative_per_image=args.max_random_negative_per_image,
        sampling_seed=args.seed,
    ).to(device)
    features = extract_teacher_features(
        model,
        matcher,
        selector,
        loader,
        support_files,
        calibration_files,
        args.max_images,
        device,
    )
    support_positive = features["support"]["positive"]
    if len(support_positive) < args.num_prototypes:
        raise RuntimeError("insufficient support positives for foreground prototypes")
    prototypes, assignment = spherical_kmeans(
        torch.from_numpy(support_positive).to(device),
        args.num_prototypes,
        args.kmeans_iterations,
        args.seed,
    )
    cluster_counts = torch.bincount(
        assignment, minlength=args.num_prototypes
    ).float()
    cluster_shares = (cluster_counts / cluster_counts.sum()).cpu().tolist()
    support_hard = features["support"]["hard_negative"]
    support_random = features["support"]["random_negative"]
    hard_count = min(len(support_hard), args.negative_bank_size // 2)
    random_count = min(len(support_random), args.negative_bank_size - hard_count)
    fixed_negatives = np.concatenate(
        [
            _deterministic_rows(support_hard, hard_count, args.seed + 11),
            _deterministic_rows(support_random, random_count, args.seed + 29),
        ],
        axis=0,
    )
    if len(fixed_negatives) < 1:
        raise RuntimeError("support split did not produce fixed negatives")
    fixed_negatives = _deterministic_rows(
        fixed_negatives, args.negative_bank_size, args.seed + 47
    )
    metrics = evaluate_separation(
        prototypes,
        fixed_negatives,
        features["calibration"]["positive"],
        features["calibration"]["hard_negative"],
        features["calibration"]["random_negative"],
        temperature=args.temperature,
        negative_sample_size=args.negative_sample_size,
        negative_energy_blocks=args.negative_energy_blocks,
        negative_sampling_seed=args.negative_sampling_seed,
    )

    checkpoint_hash = file_sha256(args.checkpoint)
    identity = {
        "checkpoint_sha256": checkpoint_hash,
        "support_files": sorted(support_files),
        "num_prototypes": args.num_prototypes,
        "positive_radius": args.positive_radius,
        "background_radius": args.background_radius,
        "seed": args.seed,
    }
    bank_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    gate_pass = bool(
        metrics["auc"] >= args.gate_auc
        and metrics["balanced_accuracy"] >= args.gate_balanced_accuracy
        and min(cluster_shares) >= args.gate_min_cluster_share
    )
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    prototype_cosine = prototypes.matmul(prototypes.t()).detach().cpu().numpy()
    audit = {
        "version": FROZEN_TEACHER_PROTOTYPE_VERSION,
        "bank_id": bank_id,
        "gate_pass": gate_pass,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_epoch,
        "dataset": os.path.abspath(args.dataset),
        "feature_source": "p2p_cls_features_before_final_classifier",
        "gate_score": "protonce_relative_energy",
        "teacher_frozen": True,
        "teacher_trainable_parameters": int(
            sum(parameter.requires_grad for parameter in model.parameters())
        ),
        "support_fraction": args.support_fraction,
        "support_images": len(support_files),
        "calibration_images": len(calibration_files),
        "support_groups": len({slide_group_id(name) for name in support_files}),
        "calibration_groups": len(
            {slide_group_id(name) for name in calibration_files}
        ),
        "support_positive": len(support_positive),
        "support_hard_negative": len(support_hard),
        "support_random_negative": len(support_random),
        "calibration_positive": len(features["calibration"]["positive"]),
        "calibration_hard_negative": len(
            features["calibration"]["hard_negative"]
        ),
        "calibration_random_negative": len(
            features["calibration"]["random_negative"]
        ),
        "fixed_negative_bank": len(fixed_negatives),
        "fixed_negative_hard": hard_count,
        "fixed_negative_random": random_count,
        "cluster_shares": cluster_shares,
        "prototype_pairwise_cosine": prototype_cosine.tolist(),
        "metrics": metrics,
        "gates": {
            "auc_min": args.gate_auc,
            "balanced_accuracy_min": args.gate_balanced_accuracy,
            "cluster_share_min": args.gate_min_cluster_share,
        },
        "sample_policy": {
            "positive": (
                "hungarian_match_and_distance_le_"
                f"{args.positive_radius:g}_without_score_filter"
            ),
            "hard_negative": "far_unmatched_top_teacher_cell_score_negative_only",
            "random_negative": "far_unmatched_deterministic_random_negative_only",
            "online_negative_can_update_bank": False,
        },
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
    }
    payload = build_bank_payload(
        prototypes,
        fixed_negatives,
        bank_id,
        {
            "version": FROZEN_TEACHER_PROTOTYPE_VERSION,
            "gate_pass": gate_pass,
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_epoch": checkpoint_epoch,
            "dataset": os.path.abspath(args.dataset),
            "feature_source": audit["feature_source"],
            "support_fraction": args.support_fraction,
            "support_images": len(support_files),
            "calibration_images": len(calibration_files),
            "metrics": metrics,
            "cluster_shares": cluster_shares,
            "gate_score": "protonce_relative_energy",
            "gate_temperature": args.temperature,
            "gate_negative_sample_size": args.negative_sample_size,
            "gate_negative_energy_blocks": args.negative_energy_blocks,
            "gate_negative_sampling_seed": args.negative_sampling_seed,
        },
    )
    torch.save(payload, os.path.join(args.output_dir, "frozen_teacher_fg_bank.pth"))
    with open(
        os.path.join(args.output_dir, "prototype_offline_audit.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        os.path.join(args.output_dir, "prototype_calibration_features.npz"),
        support_positive=support_positive,
        calibration_positive=features["calibration"]["positive"],
        calibration_hard_negative=features["calibration"]["hard_negative"],
        calibration_random_negative=features["calibration"]["random_negative"],
    )
    print(
        f"[Teacher-bank] gate_pass={gate_pass}, AUC={metrics['auc']:.6f}, "
        f"BA={metrics['balanced_accuracy']:.6f}, min_share={min(cluster_shares):.6f}",
        flush=True,
    )
    print(f"[Teacher-bank] output={args.output_dir}", flush=True)
    if not gate_pass:
        raise RuntimeError(
            "offline foreground prototype gate failed; inspect "
            "prototype_offline_audit.json before any training"
        )


if __name__ == "__main__":
    main()
