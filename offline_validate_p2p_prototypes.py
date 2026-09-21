import argparse
import csv
import json
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_zy_src import DataFolder
from matcher import build_matcher
from models.detr import build_model
from models.p2p_prototype import CandidatePrototypeBank
from transforms import Preprocessing
from utils import collate_fn_pad, load_model_weights


def parse_k_values(value: str) -> List[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("k_values must contain positive integers")
    return values


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit foreground/background prototypes on deterministic paraffin train "
            "features and validate them on the independent paraffin test split."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mean_std_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fit_images", default=300, type=int)
    parser.add_argument("--eval_images", default=200, type=int)
    parser.add_argument("--fg_k_values", default="1,2,4,8", type=parse_k_values)
    parser.add_argument("--bg_k_values", default="1,2,4,8", type=parse_k_values)
    parser.add_argument("--background_radius", default=30.0, type=float)
    parser.add_argument("--max_pos_per_image", default=64, type=int)
    parser.add_argument("--max_bg_per_image", default=32, type=int)
    parser.add_argument("--max_hard_bg_per_image", default=16, type=int)
    parser.add_argument("--max_random_bg_per_image", default=16, type=int)
    parser.add_argument("--kmeans_iterations", default=20, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_feature_bank", default=1, type=int, choices=(0, 1))
    parser.add_argument(
        "--include_projected_space", default=0, type=int, choices=(0, 1)
    )
    parser.add_argument("--proto_embedding_dim", default=128, type=int)
    parser.add_argument("--proto_num_fg", default=4, type=int)
    parser.add_argument("--proto_num_bg", default=8, type=int)

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


def deterministic_dataset(
    dataset_root: str,
    phase: str,
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
) -> DataFolder:
    dataset = DataFolder(
        dataset_root,
        num_classes=1,
        phase=phase,
        data_transform=Preprocessing(mean, std),
    )
    pairs = sorted(zip(dataset.data, dataset.files), key=lambda item: item[1])
    random.Random(seed).shuffle(pairs)
    dataset.data = [item[0] for item in pairs]
    dataset.files = [item[1] for item in pairs]
    return dataset


def quantile_positive_indices(
    indices: torch.Tensor, foreground_scores: torch.Tensor, maximum: int
) -> torch.Tensor:
    if maximum <= 0 or indices.numel() <= maximum:
        return indices
    sorted_indices = indices[torch.argsort(foreground_scores[indices])]
    positions = torch.linspace(
        0,
        sorted_indices.numel() - 1,
        steps=maximum,
        device=indices.device,
    ).round().long()
    return sorted_indices[positions]


@torch.inference_mode()
def extract_support_features(
    model: torch.nn.Module,
    matcher,
    selector: CandidatePrototypeBank,
    loader: DataLoader,
    maximum_images: int,
    device: torch.device,
    split_name: str,
    projector=None,
) -> Dict[str, np.ndarray]:
    foreground_features: List[torch.Tensor] = []
    hard_background_features: List[torch.Tensor] = []
    random_background_features: List[torch.Tensor] = []
    projected_foreground_features: List[torch.Tensor] = []
    projected_hard_background_features: List[torch.Tensor] = []
    projected_random_background_features: List[torch.Tensor] = []
    foreground_scores: List[torch.Tensor] = []
    hard_background_scores: List[torch.Tensor] = []
    random_background_scores: List[torch.Tensor] = []
    processed = 0
    skipped_empty = 0

    for images, points, labels, lengths in loader:
        if maximum_images > 0 and processed >= maximum_images:
            break
        if int(lengths[0]) == 0:
            skipped_empty += 1
            continue

        images = images.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = {
            "gt_nums": lengths,
            "gt_points": [
                points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points
            ],
            "gt_labels": [label_seq[label_seq != -1] for label_seq in labels],
        }

        anchors = model.get_aps(images)
        pyramid_features = model.backbone(images)
        reg_features, cls_features, _, _ = model.extract_features(
            pyramid_features, anchors
        )
        predicted_points = model.reg_head(reg_features) + anchors
        raw_logits = model.cls_head(cls_features)
        outputs = {"pnt_coords": predicted_points, "cls_logits": raw_logits}
        indices = matcher(outputs, targets)
        selection = selector.select_candidates(
            predicted_points,
            raw_logits,
            targets,
            indices,
            anchor_points=anchors,
        )

        probability = raw_logits.softmax(dim=-1)[..., 0]
        normalized = F.normalize(cls_features, dim=-1, eps=1e-6)
        projected = projector(cls_features) if projector is not None else None
        for batch_idx in range(images.shape[0]):
            positive_indices = torch.where(selection.positive_mask[batch_idx])[0]
            positive_indices = quantile_positive_indices(
                positive_indices,
                probability[batch_idx],
                selector.max_positive_per_image,
            )
            hard_indices = torch.where(
                selection.hard_background_mask[batch_idx]
            )[0]
            random_indices = torch.where(
                selection.random_background_mask[batch_idx]
            )[0]
            if positive_indices.numel() > 0:
                foreground_features.append(
                    normalized[batch_idx, positive_indices].detach().cpu()
                )
                foreground_scores.append(
                    probability[batch_idx, positive_indices].detach().cpu()
                )
                if projected is not None:
                    projected_foreground_features.append(
                        projected[batch_idx, positive_indices].detach().cpu()
                    )
            if hard_indices.numel() > 0:
                hard_background_features.append(
                    normalized[batch_idx, hard_indices].detach().cpu()
                )
                hard_background_scores.append(
                    probability[batch_idx, hard_indices].detach().cpu()
                )
                if projected is not None:
                    projected_hard_background_features.append(
                        projected[batch_idx, hard_indices].detach().cpu()
                    )
            if random_indices.numel() > 0:
                random_background_features.append(
                    normalized[batch_idx, random_indices].detach().cpu()
                )
                random_background_scores.append(
                    probability[batch_idx, random_indices].detach().cpu()
                )
                if projected is not None:
                    projected_random_background_features.append(
                        projected[batch_idx, random_indices].detach().cpu()
                    )

        processed += int(images.shape[0])
        if processed % 20 == 0:
            print(
                f"[Offline-{split_name}] processed={processed}/{maximum_images}",
                flush=True,
            )

    if (
        not foreground_features
        or not hard_background_features
    ):
        raise RuntimeError(
            f"{split_name} did not produce foreground and hard-background support"
        )

    foreground = torch.cat(foreground_features).numpy()
    hard_background = torch.cat(hard_background_features).numpy()
    random_background = (
        torch.cat(random_background_features).numpy()
        if random_background_features
        else hard_background[:0].copy()
    )
    random_background_raw_score = (
        torch.cat(random_background_scores).numpy()
        if random_background_scores
        else np.zeros((0,), dtype=np.float32)
    )
    result = {
        "foreground": foreground,
        "hard_background": hard_background,
        "random_background": random_background,
        "background": np.concatenate([hard_background, random_background], axis=0),
        "foreground_raw_score": torch.cat(foreground_scores).numpy(),
        "hard_background_raw_score": torch.cat(hard_background_scores).numpy(),
        "random_background_raw_score": random_background_raw_score,
        "images": np.asarray([processed], dtype=np.int64),
        "skipped_empty": np.asarray([skipped_empty], dtype=np.int64),
    }
    result["background_raw_score"] = np.concatenate(
        [
            result["hard_background_raw_score"],
            result["random_background_raw_score"],
        ]
    )
    if projector is not None:
        projected_foreground = torch.cat(projected_foreground_features).numpy()
        projected_hard = torch.cat(projected_hard_background_features).numpy()
        projected_random = (
            torch.cat(projected_random_background_features).numpy()
            if projected_random_background_features
            else projected_hard[:0].copy()
        )
        result.update(
            {
                "projected_foreground": projected_foreground,
                "projected_hard_background": projected_hard,
                "projected_random_background": projected_random,
                "projected_background": np.concatenate(
                    [projected_hard, projected_random], axis=0
                ),
            }
        )
    print(
        f"[Offline-{split_name}] images={processed}, skipped_empty={skipped_empty}, "
        f"foreground={len(result['foreground'])}, "
        f"hard_bg={len(result['hard_background'])}, "
        f"random_bg={len(result['random_background'])}",
        flush=True,
    )
    return result


def spherical_kmeans(
    values: torch.Tensor, clusters: int, iterations: int
) -> torch.Tensor:
    values = F.normalize(values, dim=-1, eps=1e-6)
    if values.shape[0] < clusters:
        raise ValueError(f"need at least {clusters} values, got {values.shape[0]}")

    mean = F.normalize(values.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
    first = (2.0 - 2.0 * values.matmul(mean.t()).squeeze(1)).argmin()
    selected = [int(first.item())]
    minimum_distance = torch.full(
        (values.shape[0],), float("inf"), device=values.device
    )
    while len(selected) < clusters:
        center = values[selected[-1]].unsqueeze(0)
        distance = (2.0 - 2.0 * values.matmul(center.t()).squeeze(1)).clamp_min(0)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
        selected.append(int(minimum_distance.argmax().item()))
    centers = values[selected].clone()

    for _ in range(max(iterations, 1)):
        distance = (2.0 - 2.0 * values.matmul(centers.t())).clamp_min(0)
        assignment = distance.argmin(dim=1)
        updated = centers.clone()
        for cluster_idx in range(clusters):
            members = values[assignment == cluster_idx]
            if members.numel() > 0:
                updated[cluster_idx] = members.mean(dim=0)
        updated = F.normalize(updated, dim=-1, eps=1e-6)
        if torch.allclose(updated, centers, atol=1e-5, rtol=1e-5):
            centers = updated
            break
        centers = updated
    return centers


def min_distance(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (2.0 - 2.0 * values.matmul(centers.t())).clamp_min(0).min(dim=1).values


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order].astype(np.int64)
    positives = max(int(labels.sum()), 1)
    negatives = max(int((1 - labels).sum()), 1)
    true_positive = np.concatenate([[0.0], np.cumsum(labels) / positives])
    false_positive = np.concatenate([[0.0], np.cumsum(1 - labels) / negatives])
    return float(np.trapz(true_positive, false_positive))


def classification_metrics(
    labels: np.ndarray, predicted: np.ndarray, scores: np.ndarray
) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    predicted = predicted.astype(np.int64)
    tp = int(((labels == 1) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    tn = int(((labels == 0) & (predicted == 0)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "foreground_recall": recall,
        "background_recall": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
        "auc": binary_auc(labels, scores),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_k(
    fit: Dict[str, np.ndarray],
    evaluation: Dict[str, np.ndarray],
    foreground_clusters: int,
    background_clusters: int,
    iterations: int,
    device: torch.device,
    feature_space: str = "raw",
) -> Dict[str, float]:
    prefix = "" if feature_space == "raw" else "projected_"
    fit_fg = torch.from_numpy(fit[f"{prefix}foreground"]).to(device)
    fit_bg = torch.from_numpy(fit[f"{prefix}background"]).to(device)
    eval_fg = F.normalize(
        torch.from_numpy(evaluation[f"{prefix}foreground"]).to(device), dim=-1
    )
    eval_hard_bg = F.normalize(
        torch.from_numpy(evaluation[f"{prefix}hard_background"]).to(device),
        dim=-1,
    )
    eval_random_bg = F.normalize(
        torch.from_numpy(evaluation[f"{prefix}random_background"]).to(device),
        dim=-1,
    )
    eval_bg = torch.cat([eval_hard_bg, eval_random_bg], dim=0)
    foreground_centers = spherical_kmeans(fit_fg, foreground_clusters, iterations)
    background_centers = spherical_kmeans(fit_bg, background_clusters, iterations)
    values = torch.cat([eval_fg, eval_bg], dim=0)
    foreground_distance = min_distance(values, foreground_centers)
    background_distance = min_distance(values, background_centers)
    margin = background_distance - foreground_distance
    prediction = (margin > 0).long()
    labels = torch.cat(
        [
            torch.ones(eval_fg.shape[0], dtype=torch.long, device=device),
            torch.zeros(eval_bg.shape[0], dtype=torch.long, device=device),
        ]
    )
    metrics = classification_metrics(
        labels.cpu().numpy(), prediction.cpu().numpy(), margin.cpu().numpy()
    )
    metrics.update(
        {
            "k_fg": foreground_clusters,
            "k_bg": background_clusters,
            "feature_space": feature_space,
            "foreground_margin_mean": float(margin[: eval_fg.shape[0]].mean().item()),
            "background_margin_mean": float(margin[eval_fg.shape[0] :].mean().item()),
            "foreground_intra_distance": float(
                min_distance(eval_fg, foreground_centers).mean().item()
            ),
            "background_intra_distance": float(
                min_distance(eval_bg, background_centers).mean().item()
            ),
            "center_cross_similarity_max": float(
                foreground_centers.matmul(background_centers.t()).max().item()
            ),
            "hard_background_recall": float(
                (margin[eval_fg.shape[0] : eval_fg.shape[0] + eval_hard_bg.shape[0]] < 0)
                .float()
                .mean()
                .item()
            ),
            "random_background_recall": (
                float(
                    (margin[eval_fg.shape[0] + eval_hard_bg.shape[0] :] < 0)
                    .float()
                    .mean()
                    .item()
                )
                if eval_random_bg.shape[0] > 0
                else None
            ),
        }
    )
    return metrics


def compare_feature_spaces(
    rows: Sequence[Dict[str, float]], foreground_clusters: int, background_clusters: int
) -> Dict[str, object]:
    selected = {
        str(row["feature_space"]): row
        for row in rows
        if int(row["k_fg"]) == int(foreground_clusters)
        and int(row["k_bg"]) == int(background_clusters)
    }
    comparison: Dict[str, object] = {
        "k_fg": int(foreground_clusters),
        "k_bg": int(background_clusters),
        "available_spaces": sorted(selected),
    }
    raw = selected.get("raw")
    projected = selected.get("projected")
    if raw is None or projected is None:
        comparison["valid"] = False
        comparison["reason"] = "raw and projected rows are both required"
        return comparison

    metric_names = (
        "balanced_accuracy",
        "auc",
        "foreground_recall",
        "hard_background_recall",
        "random_background_recall",
        "center_cross_similarity_max",
    )
    comparison["valid"] = True
    comparison["raw"] = {
        name: (float(raw[name]) if raw.get(name) is not None else None)
        for name in metric_names
    }
    comparison["projected"] = {
        name: (float(projected[name]) if projected.get(name) is not None else None)
        for name in metric_names
    }
    comparison["projected_minus_raw"] = {
        name: (
            float(projected[name]) - float(raw[name])
            if projected.get(name) is not None and raw.get(name) is not None
            else None
        )
        for name in metric_names
    }
    return comparison


def main() -> None:
    args = get_args_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original P2P implementation")
    if args.num_classes != 1:
        raise ValueError("offline prototype validation currently supports one class")
    if args.fit_images < 1 or args.eval_images < 1:
        raise ValueError("fit_images and eval_images must be positive")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0")
    args.proto_enable = bool(args.include_projected_space)
    args.proto_inference_fusion = 0

    mean_std_path = args.mean_std_path or os.path.join(args.dataset, "mean_std.npy")
    if not os.path.isfile(mean_std_path):
        raise FileNotFoundError(f"mean/std not found: {mean_std_path}")
    mean, std = np.load(mean_std_path)

    fit_dataset = deterministic_dataset(args.dataset, "train", mean, std, args.seed)
    eval_dataset = deterministic_dataset(args.dataset, "test", mean, std, args.seed + 1)
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_pad,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_pad,
    )

    model = build_model(args).to(device)
    checkpoint, load_result = load_model_weights(args.checkpoint, model)
    missing_projector_keys = sorted(
        key
        for key in load_result.missing_keys
        if key.startswith("prototype_head.projector.")
    )
    if args.include_projected_space and missing_projector_keys:
        raise RuntimeError(
            "projected-space validation requires a trained prototype projector; "
            f"checkpoint is missing {missing_projector_keys}"
        )
    model.eval()
    matcher = build_matcher(args)
    selector = CandidatePrototypeBank(
        feat_dim=args.hidden_dim,
        embedding_dim=args.hidden_dim,
        num_fg_prototypes=max(args.fg_k_values),
        num_bg_prototypes=max(args.bg_k_values),
        foreground_queue_size=max(args.fg_k_values),
        background_queue_size=max(args.bg_k_values),
        background_radius=args.background_radius,
        max_positive_per_image=args.max_pos_per_image,
        max_background_per_image=args.max_bg_per_image,
        max_hard_background_per_image=args.max_hard_bg_per_image,
        max_random_background_per_image=args.max_random_bg_per_image,
    )
    projector = (
        model.prototype_head.project
        if args.include_projected_space and model.prototype_head is not None
        else None
    )

    fit = extract_support_features(
        model,
        matcher,
        selector,
        fit_loader,
        args.fit_images,
        device,
        "fit-train",
        projector=projector,
    )
    evaluation = extract_support_features(
        model,
        matcher,
        selector,
        eval_loader,
        args.eval_images,
        device,
        "eval-test",
        projector=projector,
    )

    labels = np.concatenate(
        [
            np.ones(len(evaluation["foreground"]), dtype=np.int64),
            np.zeros(len(evaluation["background"]), dtype=np.int64),
        ]
    )
    raw_scores = np.concatenate(
        [evaluation["foreground_raw_score"], evaluation["background_raw_score"]]
    )
    raw_metrics = classification_metrics(labels, raw_scores >= 0.5, raw_scores)

    rows = []
    feature_spaces = ["raw"]
    if projector is not None:
        feature_spaces.append("projected")
    for feature_space in feature_spaces:
        for foreground_clusters in args.fg_k_values:
            for background_clusters in args.bg_k_values:
                metrics = evaluate_k(
                    fit,
                    evaluation,
                    foreground_clusters,
                    background_clusters,
                    args.kmeans_iterations,
                    device,
                    feature_space=feature_space,
                )
                rows.append(metrics)
                print(
                    f"[Offline-{feature_space}-Kfg={foreground_clusters},Kbg={background_clusters}] "
                    f"AUC={metrics['auc']:.6f}, "
                    f"balanced_acc={metrics['balanced_accuracy']:.6f}, "
                    f"fg_recall={metrics['foreground_recall']:.6f}, "
                    f"hard_bg_recall={metrics['hard_background_recall']:.6f}, "
                    f"random_bg_recall="
                    f"{metrics['random_background_recall'] if metrics['random_background_recall'] is not None else 'n/a'}",
                    flush=True,
                )

    selection_space = "projected" if projector is not None else "raw"
    selection_rows = [
        row for row in rows if row["feature_space"] == selection_space
    ]
    ranking = sorted(selection_rows, key=lambda row: (row["balanced_accuracy"], row["auc"]), reverse=True)
    best = ranking[0]
    near_best = [
        row
        for row in rows
        if row["balanced_accuracy"] >= best["balanced_accuracy"] - 0.005
    ]
    recommended = sorted(
        near_best,
        key=lambda row: (
            row["k_fg"] + row["k_bg"],
            max(row["k_fg"], row["k_bg"]),
            -row["balanced_accuracy"],
            -row["auc"],
        ),
    )[0]
    configured_space_comparison = compare_feature_spaces(
        rows, args.proto_num_fg, args.proto_num_bg
    )
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "dataset": os.path.abspath(args.dataset),
        "fit_images": int(fit["images"][0]),
        "eval_images": int(evaluation["images"][0]),
        "fit_foreground": len(fit["foreground"]),
        "fit_background": len(fit["background"]),
        "eval_foreground": len(evaluation["foreground"]),
        "eval_background": len(evaluation["background"]),
        "feature_spaces": feature_spaces,
        "selection_feature_space": selection_space,
        "configured_feature_space_comparison": configured_space_comparison,
        "sample_policy": {
            "positive": "hungarian_matched_score_quantiles",
            "ignored": "unmatched_predicted_or_anchor_distance_below_radius",
            "hard_background": "top_raw_foreground_score_among_far_unmatched",
            "random_background": "seeded_random_sample_from_remaining_far_unmatched",
            "background_radius": args.background_radius,
            "max_positive_per_image": args.max_pos_per_image,
            "max_background_per_image": args.max_bg_per_image,
            "max_hard_background_per_image": args.max_hard_bg_per_image,
            "max_random_background_per_image": args.max_random_bg_per_image,
        },
        "raw_classifier": raw_metrics,
        "prototype_results": rows,
        "best_k_pair_from_source_only": {
            "foreground": int(best["k_fg"]),
            "background": int(best["k_bg"]),
        },
        "recommended_k_pair_from_source_only": {
            "foreground": int(recommended["k_fg"]),
            "background": int(recommended["k_bg"]),
            "selection_rule": "smallest bank within 0.005 balanced accuracy of source best",
        },
        "screening_pass": bool(
            recommended["auc"] >= 0.70
            and recommended["balanced_accuracy"] >= 0.65
            and recommended["foreground_recall"] >= 0.60
            and recommended["background_recall"] >= 0.60
        ),
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
    }

    with open(os.path.join(args.output_dir, "offline_prototype_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    fieldnames = list(rows[0].keys())
    with open(os.path.join(args.output_dir, "offline_prototype_k_sweep.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.save_feature_bank:
        np.savez_compressed(
            os.path.join(args.output_dir, "offline_candidate_feature_bank.npz"),
            fit_foreground=fit["foreground"],
            fit_background=fit["background"],
            eval_foreground=evaluation["foreground"],
            eval_background=evaluation["background"],
            eval_foreground_raw_score=evaluation["foreground_raw_score"],
            eval_background_raw_score=evaluation["background_raw_score"],
            fit_hard_background=fit["hard_background"],
            fit_random_background=fit["random_background"],
            eval_hard_background=evaluation["hard_background"],
            eval_random_background=evaluation["random_background"],
            **(
                {
                    "fit_projected_foreground": fit["projected_foreground"],
                    "fit_projected_background": fit["projected_background"],
                    "eval_projected_foreground": evaluation["projected_foreground"],
                    "eval_projected_background": evaluation["projected_background"],
                }
                if projector is not None
                else {}
            ),
        )

    print(f"[Offline] raw classifier={raw_metrics}", flush=True)
    print(
        f"[Offline] recommended_k_fg={recommended['k_fg']}, "
        f"recommended_k_bg={recommended['k_bg']}, "
        f"screening_pass={summary['screening_pass']}",
        flush=True,
    )
    print(
        "[Offline] configured_feature_space_comparison="
        f"{configured_space_comparison}",
        flush=True,
    )
    print(f"[SUCCESS] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
