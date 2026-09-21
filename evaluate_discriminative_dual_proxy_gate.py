import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_zy_src import build_dataset
from models.detr import build_model
from matcher import build_matcher
from train_p2p import get_args_parser
from utils import collate_fn_pad


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _slide_name(path):
    name = os.path.basename(path)
    return name.split(".kfb_")[0] if ".kfb_" in name else name.split("_")[0]


def _effective_count(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    shares = counts / total
    positive = shares[shares > 0]
    effective = float(np.exp(-(positive * np.log(positive)).sum()))
    return effective, float(shares.min())


def _pairwise_max(prototypes):
    if prototypes.shape[0] < 2:
        return float("nan")
    matrix = prototypes @ prototypes.T
    np.fill_diagonal(matrix, -1.0)
    return float(matrix.max())


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda:0")
    model = build_model(args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    prototype_head = model.prototype_head
    if prototype_head is None or model.prototype_mode != "discriminative_dual_proxy":
        raise RuntimeError("checkpoint is not a discriminative dual-proxy model")
    if not bool(prototype_head.prototype_ready.item()):
        raise RuntimeError("dual-proxy bank is not initialized")
    prototype_head.sampling_step.zero_()

    dataset = build_dataset(args, "test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_pad,
    )
    matcher = build_matcher(args)
    slide_labels = defaultdict(list)
    slide_scores = defaultdict(list)
    foreground_gaps = []
    background_gaps = []
    foreground_counts = np.zeros(prototype_head.num_fg, dtype=np.int64)
    background_counts = np.zeros(prototype_head.num_bg, dtype=np.int64)
    positive_total = background_total = ignored_near_total = rejected_total = 0
    hard_background_total = random_background_total = 0

    fg_prototypes = prototype_head.normalized_foreground_prototypes()
    bg_prototypes = prototype_head.normalized_background_prototypes()
    for image_index, (images, points, labels, lengths) in enumerate(loader):
        images = images.to(device)
        points = points.to(device)
        labels = labels.to(device)
        targets = {
            "gt_nums": lengths,
            "gt_points": [
                row[row != -1].reshape(-1, 2) for row in points
            ],
            "gt_labels": [row[row != -1] for row in labels],
        }
        outputs = model(images, prototype_support_points=targets["gt_points"])
        indices = matcher(outputs, targets)
        sampled = prototype_head.sample_candidates(
            outputs["pnt_coords"], outputs["raw_cls_logits"], targets, indices
        )
        embeddings = outputs["proto_embeddings"][0]
        fg_similarity = embeddings.matmul(fg_prototypes.t())
        bg_similarity = embeddings.matmul(bg_prototypes.t())
        score = fg_similarity.max(dim=-1).values - bg_similarity.max(dim=-1).values
        slide = _slide_name(dataset.files[image_index])

        positive_indices = sampled["reliable_indices"][0]
        background_indices = sampled["background_indices"][0]
        positive_total += int(positive_indices.numel())
        background_total += int(background_indices.numel())
        hard_background_total += int(
            sampled["hard_background_indices"][0].numel()
        )
        random_background_total += int(
            sampled["random_background_indices"][0].numel()
        )
        ignored_near_total += int(sampled["ignored_near_indices"][0].numel())
        rejected_total += int(sampled["rejected_match_indices"][0].numel())
        if positive_indices.numel():
            values = score[positive_indices]
            slide_labels[slide].extend([1] * int(values.numel()))
            slide_scores[slide].extend(values.cpu().tolist())
            foreground_gaps.extend(values.cpu().tolist())
            assignments = fg_similarity[positive_indices].argmax(dim=-1).cpu().numpy()
            foreground_counts += np.bincount(
                assignments, minlength=prototype_head.num_fg
            )
        if background_indices.numel():
            values = score[background_indices]
            slide_labels[slide].extend([0] * int(values.numel()))
            slide_scores[slide].extend(values.cpu().tolist())
            background_gaps.extend((-values).cpu().tolist())
            assignments = bg_similarity[background_indices].argmax(dim=-1).cpu().numpy()
            background_counts += np.bincount(
                assignments, minlength=prototype_head.num_bg
            )
        if (image_index + 1) % 50 == 0:
            print(
                f"[Dual-Proxy-gate-eval] processed={image_index + 1}/{len(dataset)}",
                flush=True,
            )

    slide_auc = {
        slide: _auc(slide_labels[slide], slide_scores[slide])
        for slide in sorted(slide_labels)
    }
    valid_auc = [value for value in slide_auc.values() if np.isfinite(value)]
    fg_effective, fg_min_share = _effective_count(foreground_counts)
    bg_effective, bg_min_share = _effective_count(background_counts)
    fg_numpy = fg_prototypes.cpu().numpy()
    bg_numpy = bg_prototypes.cpu().numpy()
    initialization_path = os.path.join(
        os.path.dirname(args.checkpoint), "run_initialization.json"
    )
    initialization = (
        json.load(open(initialization_path, encoding="utf-8"))
        if os.path.isfile(initialization_path)
        else {}
    )
    metrics = {
        "prototype_macro_auc": float(np.mean(valid_auc)) if valid_auc else float("nan"),
        "prototype_min_slide_auc": float(np.min(valid_auc)) if valid_auc else float("nan"),
        "foreground_similarity_gap": float(np.mean(foreground_gaps)) if foreground_gaps else float("nan"),
        "background_similarity_gap": float(np.mean(background_gaps)) if background_gaps else float("nan"),
        "foreground_effective_prototypes": fg_effective,
        "background_effective_prototypes": bg_effective,
        "foreground_assignment_share_min": fg_min_share,
        "background_assignment_share_min": bg_min_share,
        "foreground_pairwise_similarity_max": _pairwise_max(fg_numpy),
        "background_pairwise_similarity_max": _pairwise_max(bg_numpy),
        "cross_bank_similarity_max": float((fg_numpy @ bg_numpy.T).max()),
        "optimizer_projector_only": initialization.get("optimizer_projector_only") is True,
        "positive_candidates": positive_total,
        "background_candidates": background_total,
        "hard_background_candidates": hard_background_total,
        "random_background_candidates": random_background_total,
        "ignored_near_candidates": ignored_near_total,
        "rejected_matched_candidates": rejected_total,
        "slide_auc": slide_auc,
        "checkpoint": args.checkpoint,
    }
    os.makedirs(os.path.dirname(args.gate_output), exist_ok=True)
    with open(args.gate_output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = get_args_parser()
    parser.add_argument(
        "--mean_std_path",
        default="",
        type=str,
        help="explicit mean_std.npy used by the gate dataset",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate_output", required=True)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
