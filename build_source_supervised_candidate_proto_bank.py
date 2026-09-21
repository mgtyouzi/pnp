import argparse
import hashlib
import json
import math
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset_zy_src import DataFolder
from matcher import build_matcher
from models.detr import build_model
from models.source_supervised_candidate_proto import (
    SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION,
    SourceSupervisedCandidatePrototype,
)
from transforms import Preprocessing
from utils import load_model_weights


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build source-supervised candidate prototypes from paraffin train data."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mean_std_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default=6, type=int)
    parser.add_argument("--support_fraction", default=0.8, type=float)
    parser.add_argument("--max_images", default=0, type=int)
    parser.add_argument("--num_fg", default=4, type=int)
    parser.add_argument("--num_bg", default=8, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--positive_radius", default=15.0, type=float)
    parser.add_argument("--background_radius", default=30.0, type=float)
    parser.add_argument("--max_hard_positive", default=32, type=int)
    parser.add_argument("--max_random_positive", default=32, type=int)
    parser.add_argument("--max_hard_background", default=32, type=int)
    parser.add_argument("--max_random_background", default=32, type=int)
    parser.add_argument("--kmeans_iterations", default=25, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--gate_auc", default=0.65, type=float)
    parser.add_argument("--gate_min_slide_auc", default=0.55, type=float)
    parser.add_argument("--gate_min_cluster_share", default=0.02, type=float)

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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slide_group_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def split_slide_groups(
    files: Sequence[str], support_fraction: float, seed: int
) -> Tuple[set, set]:
    groups = sorted({slide_group_id(path) for path in files})
    if len(groups) < 2:
        raise ValueError("source-only calibration needs at least two slide groups")
    random.Random(seed).shuffle(groups)
    boundary = min(max(int(round(len(groups) * support_fraction)), 1), len(groups) - 1)
    support_groups = set(groups[:boundary])
    calibration_groups = set(groups[boundary:])
    return support_groups, calibration_groups


class StrictSourceDataset(DataFolder):
    def __getitem__(self, index: int):
        image_path = self.files[index]
        raw_sample = self.read_data(self.data[index], image_path)
        image, points, labels = self.data_transform(raw_sample)
        return image, points, labels, image_path


def strict_collate(batch):
    images, points, labels, paths = zip(*batch)
    lengths = [len(value) for value in labels]
    points = pad_sequence(points, batch_first=True, padding_value=-1).reshape(
        len(batch), -1
    )
    labels = pad_sequence(labels, batch_first=True, padding_value=-1).reshape(
        len(batch), -1
    )
    return torch.stack(images), points.float(), labels.long(), lengths, list(paths)


def build_source_dataset(root: str, mean: np.ndarray, std: np.ndarray):
    dataset = StrictSourceDataset(
        root,
        num_classes=1,
        phase="train",
        data_transform=Preprocessing(mean, std),
    )
    paired = sorted(zip(dataset.data, dataset.files), key=lambda item: item[1])
    dataset.data = [item[0] for item in paired]
    dataset.files = [item[1] for item in paired]
    return dataset


def spherical_kmeans(values: torch.Tensor, clusters: int, iterations: int):
    values = F.normalize(values.float(), dim=-1, eps=1e-6)
    if values.shape[0] < clusters:
        raise RuntimeError(f"need {clusters} samples, found {values.shape[0]}")
    selected = [0]
    nearest = 1.0 - values.matmul(values[:1].t()).squeeze(1)
    for _ in range(1, clusters):
        index = int(nearest.argmax().item())
        selected.append(index)
        distance = 1.0 - values.matmul(values[index:index + 1].t()).squeeze(1)
        nearest = torch.minimum(nearest, distance)
    centers = values[selected].clone()
    assignment = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    for _ in range(max(int(iterations), 1)):
        assignment = values.matmul(centers.t()).argmax(dim=1)
        closest = values.matmul(centers.t()).max(dim=1).values
        updated = []
        for index in range(clusters):
            members = values[assignment == index]
            if members.numel():
                updated.append(F.normalize(members.mean(dim=0), dim=0, eps=1e-6))
            else:
                updated.append(values[closest.argmin()])
        new_centers = torch.stack(updated)
        if torch.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers
    assignment = values.matmul(centers.t()).argmax(dim=1)
    return centers, assignment


def average_rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def class_margin(
    values: torch.Tensor,
    foreground: torch.Tensor,
    background: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    values = F.normalize(values.float(), dim=-1, eps=1e-6)
    foreground_score = temperature * (
        torch.logsumexp(values.matmul(foreground.t()) / temperature, dim=1)
        - math.log(foreground.shape[0])
    )
    background_score = temperature * (
        torch.logsumexp(values.matmul(background.t()) / temperature, dim=1)
        - math.log(background.shape[0])
    )
    return foreground_score - background_score


@torch.inference_mode()
def extract_candidates(model, matcher, selector, loader, groups, maximum_images, device):
    buckets: Dict[str, Dict[str, List[torch.Tensor]]] = {
        split: {
            "foreground": [],
            "background": [],
            "foreground_slides": [],
            "background_slides": [],
        }
        for split in ("support", "calibration")
    }
    counts = {"images": 0, "support_images": 0, "calibration_images": 0}
    for images, points, labels, lengths, paths in loader:
        if maximum_images and counts["images"] >= maximum_images:
            break
        group = slide_group_id(paths[0])
        split = "support" if group in groups[0] else "calibration"
        if group not in groups[0] and group not in groups[1]:
            continue
        images = images.to(device)
        points = points.to(device)
        labels = labels.to(device)
        targets = {
            "gt_nums": lengths,
            "gt_points": [value[value != -1].reshape(-1, 2) for value in points],
            "gt_labels": [value[value != -1] for value in labels],
        }
        anchors = model.get_aps(images)
        pyramid = model.backbone(images)
        reg_features, cls_features, _, _ = model.extract_features(pyramid, anchors)
        predicted_points = model.reg_head(reg_features) + anchors
        raw_logits = model.cls_head(cls_features)
        indices = matcher(
            {"pnt_coords": predicted_points, "cls_logits": raw_logits}, targets
        )
        sampled = selector.sample_candidates(
            predicted_points, raw_logits, targets, indices
        )
        normalized = F.normalize(cls_features.float(), dim=-1, eps=1e-6)
        foreground_indices = sampled["positive_indices"][0]
        background_indices = sampled["background_indices"][0]
        if foreground_indices.numel():
            buckets[split]["foreground"].append(
                normalized[0, foreground_indices].detach().cpu()
            )
            buckets[split]["foreground_slides"].extend(
                [group] * int(foreground_indices.numel())
            )
        if background_indices.numel():
            buckets[split]["background"].append(
                normalized[0, background_indices].detach().cpu()
            )
            buckets[split]["background_slides"].extend(
                [group] * int(background_indices.numel())
            )
        counts["images"] += 1
        counts[f"{split}_images"] += 1
        if counts["images"] % 50 == 0:
            print(f"[SSCP-bank] processed={counts['images']}", flush=True)
    result = {"counts": counts}
    for split in ("support", "calibration"):
        result[split] = {}
        for name in ("foreground", "background"):
            result[split][name] = (
                torch.cat(buckets[split][name], dim=0)
                if buckets[split][name]
                else torch.zeros(0, selector.feat_dim)
            )
        result[split]["foreground_slides"] = buckets[split]["foreground_slides"]
        result[split]["background_slides"] = buckets[split]["background_slides"]
    return result


def main() -> None:
    args = get_args_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original P2P feature extractor")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    mean_std_path = args.mean_std_path or os.path.join(args.dataset, "mean_std.npy")
    mean, std = np.load(mean_std_path)
    dataset = build_source_dataset(args.dataset, mean, std)
    support_groups, calibration_groups = split_slide_groups(
        dataset.files, args.support_fraction, args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=strict_collate,
    )
    model = build_model(args).to(device)
    checkpoint, load_result = load_model_weights(args.checkpoint, model)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={list(load_result.missing_keys)}, "
            f"unexpected={list(load_result.unexpected_keys)}"
        )
    model.eval()
    model.requires_grad_(False)
    matcher = build_matcher(args)
    selector = SourceSupervisedCandidatePrototype(
        feat_dim=args.hidden_dim,
        num_foreground_prototypes=args.num_fg,
        num_background_prototypes=args.num_bg,
        max_hard_positive_per_image=args.max_hard_positive,
        max_random_positive_per_image=args.max_random_positive,
        max_hard_background_per_image=args.max_hard_background,
        max_random_background_per_image=args.max_random_background,
        positive_radius=args.positive_radius,
        background_radius=args.background_radius,
        prototype_temperature=args.temperature,
        sampling_seed=args.seed,
    ).to(device)
    extracted = extract_candidates(
        model,
        matcher,
        selector,
        loader,
        (support_groups, calibration_groups),
        args.max_images,
        device,
    )
    support_fg = extracted["support"]["foreground"].to(device)
    support_bg = extracted["support"]["background"].to(device)
    foreground, foreground_assignment = spherical_kmeans(
        support_fg, args.num_fg, args.kmeans_iterations
    )
    background, background_assignment = spherical_kmeans(
        support_bg, args.num_bg, args.kmeans_iterations
    )
    foreground_share = (
        torch.bincount(foreground_assignment, minlength=args.num_fg).float()
        / foreground_assignment.numel()
    )
    background_share = (
        torch.bincount(background_assignment, minlength=args.num_bg).float()
        / background_assignment.numel()
    )

    calibration_fg = extracted["calibration"]["foreground"].to(device)
    calibration_bg = extracted["calibration"]["background"].to(device)
    values = torch.cat([calibration_fg, calibration_bg], dim=0)
    labels = np.concatenate([
        np.ones(len(calibration_fg), dtype=np.int64),
        np.zeros(len(calibration_bg), dtype=np.int64),
    ])
    scores = class_margin(values, foreground, background, args.temperature)
    scores_np = scores.detach().cpu().numpy()
    auc = average_rank_auc(labels, scores_np)
    slide_auc = {}
    normalized_slide_names = (
        extracted["calibration"]["foreground_slides"]
        + extracted["calibration"]["background_slides"]
    )
    for group in sorted(set(normalized_slide_names)):
        mask = np.asarray([name == group for name in normalized_slide_names])
        slide_auc[group] = average_rank_auc(labels[mask], scores_np[mask])
    finite_slide_auc = [value for value in slide_auc.values() if np.isfinite(value)]
    min_slide_auc = min(finite_slide_auc) if finite_slide_auc else float("nan")
    min_share = min(float(foreground_share.min()), float(background_share.min()))
    gate_pass = bool(
        auc >= args.gate_auc
        and min_slide_auc >= args.gate_min_slide_auc
        and min_share >= args.gate_min_cluster_share
    )
    checkpoint_hash = file_sha256(args.checkpoint)
    bank_identity = {
        "checkpoint_sha256": checkpoint_hash,
        "support_groups": sorted(support_groups),
        "num_fg": args.num_fg,
        "num_bg": args.num_bg,
        "seed": args.seed,
    }
    bank_id = hashlib.sha256(
        json.dumps(bank_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload = {
        "implementation_version": SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION,
        "foreground_prototypes": foreground.detach().cpu(),
        "background_prototypes": background.detach().cpu(),
        "bank_id": bank_id,
        "metadata": {
            "dataset": os.path.abspath(args.dataset),
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "gate_pass": gate_pass,
            "feature_source": "native_p2p_cls_features_256d",
            "support_groups": sorted(support_groups),
            "calibration_groups": sorted(calibration_groups),
        },
    }
    bank_path = os.path.join(
        args.output_dir, "source_supervised_candidate_proto_bank.pth"
    )
    torch.save(payload, bank_path)
    audit = {
        "implementation_version": SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION,
        "gate_pass": gate_pass,
        "bank_id": bank_id,
        "source_only": True,
        "dataset": os.path.abspath(args.dataset),
        "checkpoint": os.path.abspath(args.checkpoint),
        "support_groups": sorted(support_groups),
        "calibration_groups": sorted(calibration_groups),
        "counts": extracted["counts"],
        "support_foreground": int(len(support_fg)),
        "support_background": int(len(support_bg)),
        "calibration_foreground": int(len(calibration_fg)),
        "calibration_background": int(len(calibration_bg)),
        "foreground_assignment_share": foreground_share.cpu().tolist(),
        "background_assignment_share": background_share.cpu().tolist(),
        "prototype_macro_auc": auc,
        "prototype_min_slide_auc": min_slide_auc,
        "slide_auc": slide_auc,
        "thresholds": {
            "prototype_macro_auc": args.gate_auc,
            "prototype_min_slide_auc": args.gate_min_slide_auc,
            "minimum_assignment_share": args.gate_min_cluster_share,
        },
        "next_action": "run_10_epoch_paired_source_gate",
    }
    with open(
        os.path.join(args.output_dir, "prototype_offline_audit.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(
        f"[SSCP-bank] gate_pass={gate_pass}, auc={auc:.6f}, "
        f"min_slide_auc={min_slide_auc:.6f}, min_share={min_share:.6f}",
        flush=True,
    )
    print(f"[SSCP-bank] output={bank_path}", flush=True)


if __name__ == "__main__":
    main()
