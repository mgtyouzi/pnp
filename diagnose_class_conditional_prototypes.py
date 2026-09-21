import argparse
import csv
import json
import math
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from build_frozen_teacher_fg_bank import spherical_kmeans


TYPE_POSITIVE = 0
TYPE_HARD_NEGATIVE = 1
TYPE_RANDOM_NEGATIVE = 2


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Leave-one-group-out audit for class-conditional prototypes."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--foreground_prototypes", default=4, type=int)
    parser.add_argument("--hard_background_prototypes", default=4, type=int)
    parser.add_argument("--random_background_prototypes", default=2, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--kmeans_iterations", default=20, type=int)
    parser.add_argument("--positive_per_group", default=4096, type=int)
    parser.add_argument("--hard_negative_per_group", default=2048, type=int)
    parser.add_argument("--random_negative_per_group", default=2048, type=int)
    parser.add_argument("--ridge", default=0.01, type=float)
    parser.add_argument("--seed", default=0, type=int)
    return parser


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tpr = np.concatenate(
        [[0.0], np.cumsum(sorted_labels == 1) / positives, [1.0]]
    )
    fpr = np.concatenate(
        [[0.0], np.cumsum(sorted_labels == 0) / negatives, [1.0]]
    )
    return float(np.trapz(tpr, fpr))


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 501)))
    best = (-1.0, 0.0)
    for threshold in thresholds:
        prediction = scores >= threshold
        tpr = float(prediction[labels == 1].mean())
        tnr = float((~prediction[labels == 0]).mean())
        value = 0.5 * (tpr + tnr)
        if value > best[0]:
            best = (value, float(threshold))
    return best


def deterministic_subset(indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(indices) <= count:
        return indices
    random = np.random.RandomState(seed)
    return indices[random.choice(len(indices), size=count, replace=False)]


def group_balanced_subset(
    candidate_type: np.ndarray,
    group_index: np.ndarray,
    training_groups: np.ndarray,
    requested_type: int,
    per_group: int,
    seed: int,
) -> np.ndarray:
    selected = []
    for group in sorted(int(value) for value in training_groups):
        indices = np.where(
            (candidate_type == requested_type) & (group_index == group)
        )[0]
        if len(indices):
            selected.append(
                deterministic_subset(indices, per_group, seed + 1009 * group)
            )
    if not selected:
        raise RuntimeError(
            f"no candidates type={requested_type} in LOSO training groups"
        )
    return np.concatenate(selected)


def fit_prototypes(
    features: np.ndarray,
    indices: np.ndarray,
    count: int,
    iterations: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    prototypes, _ = spherical_kmeans(
        torch.from_numpy(features[indices]).float().to(device),
        count,
        iterations,
        seed,
    )
    return F.normalize(prototypes, dim=-1, eps=1e-6)


def logmeanexp(logits: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(logits, dim=dim) - math.log(logits.shape[dim])


def score_class_prototypes(
    values: np.ndarray,
    foreground: torch.Tensor,
    hard_background: torch.Tensor,
    random_background: torch.Tensor,
    temperature: float,
) -> Dict[str, np.ndarray]:
    output = {"class_prototype_logit": [], "nearest_class_margin": []}
    device = foreground.device
    for start in range(0, len(values), 4096):
        query = F.normalize(
            torch.from_numpy(values[start : start + 4096]).float().to(device),
            dim=-1,
            eps=1e-6,
        )
        fg_logits = query.matmul(foreground.t()) / temperature
        hard_logits = query.matmul(hard_background.t()) / temperature
        random_logits = query.matmul(random_background.t()) / temperature
        fg_energy = logmeanexp(fg_logits, 1)
        hard_energy = logmeanexp(hard_logits, 1)
        random_energy = logmeanexp(random_logits, 1)
        background_energy = torch.logsumexp(
            torch.stack([hard_energy, random_energy], dim=1), dim=1
        ) - math.log(2.0)
        output["class_prototype_logit"].append(
            (fg_energy - background_energy).detach().cpu().numpy()
        )
        nearest_background = torch.maximum(
            hard_logits.max(1).values, random_logits.max(1).values
        )
        output["nearest_class_margin"].append(
            (fg_logits.max(1).values - nearest_background)
            .mul(temperature)
            .detach()
            .cpu()
            .numpy()
        )
    return {name: np.concatenate(parts) for name, parts in output.items()}


def fit_ridge(
    features: np.ndarray,
    positive_indices: np.ndarray,
    hard_indices: np.ndarray,
    random_indices: np.ndarray,
    ridge: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    negative_count = min(len(hard_indices), len(random_indices))
    hard = deterministic_subset(hard_indices, negative_count, seed + 31)
    random = deterministic_subset(random_indices, negative_count, seed + 37)
    negatives = np.concatenate([hard, random])
    class_count = min(len(positive_indices), len(negatives))
    positives = deterministic_subset(positive_indices, class_count, seed + 41)
    negatives = deterministic_subset(negatives, class_count, seed + 43)
    indices = np.concatenate([positives, negatives])
    targets = np.concatenate(
        [np.ones(class_count, dtype=np.float32), -np.ones(class_count, dtype=np.float32)]
    )
    x = F.normalize(
        torch.from_numpy(features[indices]).float().to(device), dim=-1, eps=1e-6
    )
    x = torch.cat([x, torch.ones((len(x), 1), device=device)], dim=1)
    y = torch.from_numpy(targets).to(device)
    identity = torch.eye(x.shape[1], device=device)
    identity[-1, -1] = 0.0
    return torch.linalg.solve(x.t().matmul(x) + ridge * identity, x.t().matmul(y))


def score_ridge(values: np.ndarray, weights: torch.Tensor) -> np.ndarray:
    parts = []
    device = weights.device
    for start in range(0, len(values), 4096):
        x = F.normalize(
            torch.from_numpy(values[start : start + 4096]).float().to(device),
            dim=-1,
            eps=1e-6,
        )
        x = torch.cat([x, torch.ones((len(x), 1), device=device)], dim=1)
        parts.append(x.matmul(weights).detach().cpu().numpy())
    return np.concatenate(parts)


def evaluate_scores(positive: np.ndarray, negative: np.ndarray) -> Dict[str, float]:
    scores = np.concatenate([positive, negative])
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64)]
    )
    ba, threshold = balanced_accuracy(labels, scores)
    return {
        "auc": roc_auc(labels, scores),
        "balanced_accuracy": ba,
        "threshold": threshold,
        "positive_mean": float(positive.mean()),
        "negative_mean": float(negative.mean()),
    }


def leave_one_group_out(args, data, manifest, device):
    features = data["features"].astype(np.float32, copy=False)
    candidate_type = data["candidate_type"]
    group_index = data["group_index"]
    teacher_score = data["teacher_score"]
    all_groups = np.unique(group_index)
    group_names = {
        int(index): name for name, index in manifest["group_to_index"].items()
    }
    rows = []
    prototype_audits = []

    for fold_index, held_group in enumerate(all_groups):
        training_groups = all_groups[all_groups != held_group]
        positive_train = group_balanced_subset(
            candidate_type,
            group_index,
            training_groups,
            TYPE_POSITIVE,
            args.positive_per_group,
            args.seed + 10000 * fold_index,
        )
        hard_train = group_balanced_subset(
            candidate_type,
            group_index,
            training_groups,
            TYPE_HARD_NEGATIVE,
            args.hard_negative_per_group,
            args.seed + 20000 * fold_index,
        )
        random_train = group_balanced_subset(
            candidate_type,
            group_index,
            training_groups,
            TYPE_RANDOM_NEGATIVE,
            args.random_negative_per_group,
            args.seed + 30000 * fold_index,
        )
        foreground_prototypes = fit_prototypes(
            features,
            positive_train,
            args.foreground_prototypes,
            args.kmeans_iterations,
            args.seed + fold_index,
            device,
        )
        hard_background_prototypes = fit_prototypes(
            features,
            hard_train,
            args.hard_background_prototypes,
            args.kmeans_iterations,
            args.seed + 100 + fold_index,
            device,
        )
        random_background_prototypes = fit_prototypes(
            features,
            random_train,
            args.random_background_prototypes,
            args.kmeans_iterations,
            args.seed + 200 + fold_index,
            device,
        )
        ridge_weights = fit_ridge(
            features,
            positive_train,
            hard_train,
            random_train,
            args.ridge,
            args.seed + fold_index,
            device,
        )
        fold_mask = group_index == held_group
        fold_indices = np.where(fold_mask)[0]
        fold_scores = score_class_prototypes(
            features[fold_indices],
            foreground_prototypes,
            hard_background_prototypes,
            random_background_prototypes,
            args.temperature,
        )
        fold_scores["ridge_linear_probe"] = score_ridge(
            features[fold_indices], ridge_weights
        )
        fold_scores["teacher_score"] = teacher_score[fold_indices]
        fold_types = candidate_type[fold_indices]
        local = {
            candidate: np.where(fold_types == candidate_id)[0]
            for candidate, candidate_id in (
                ("positive", TYPE_POSITIVE),
                ("hard_negative", TYPE_HARD_NEGATIVE),
                ("random_negative", TYPE_RANDOM_NEGATIVE),
            )
        }
        if min(len(indices) for indices in local.values()) == 0:
            raise RuntimeError(f"held group {held_group} lacks a candidate class")
        negative_sets = {
            "hard_negative": local["hard_negative"],
            "random_negative": local["random_negative"],
            "all_negative": np.concatenate(
                [local["hard_negative"], local["random_negative"]]
            ),
        }
        for method, scores in fold_scores.items():
            for negative_name, negative_indices in negative_sets.items():
                metrics = evaluate_scores(
                    scores[local["positive"]], scores[negative_indices]
                )
                rows.append(
                    {
                        "held_group": int(held_group),
                        "held_group_name": group_names.get(
                            int(held_group), str(int(held_group))
                        ),
                        "held_images": sum(
                            1
                            for image in manifest["images"]
                            if int(image["group_index"]) == int(held_group)
                        ),
                        "method": method,
                        "negative_group": negative_name,
                        "positive_count": len(local["positive"]),
                        "negative_count": len(negative_indices),
                        **metrics,
                    }
                )
        prototype_audits.append(
            {
                "held_group": int(held_group),
                "foreground_train": len(positive_train),
                "hard_background_train": len(hard_train),
                "random_background_train": len(random_train),
                "foreground_pairwise_cosine": foreground_prototypes.matmul(
                    foreground_prototypes.t()
                ).detach().cpu().tolist(),
                "hard_background_pairwise_cosine": hard_background_prototypes.matmul(
                    hard_background_prototypes.t()
                ).detach().cpu().tolist(),
            }
        )
        print(
            f"[Class-prototype-LOSO] fold={fold_index + 1}/{len(all_groups)}, "
            f"held_group={group_names.get(int(held_group), held_group)}",
            flush=True,
        )
    return rows, prototype_audits


def main():
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA LOSO requested but CUDA is unavailable")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    with np.load(args.features) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows, prototype_audits = leave_one_group_out(args, data, manifest, device)

    fold_path = os.path.join(args.output_dir, "fold_metrics.csv")
    with open(fold_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    macro_rows = []
    combinations = sorted({(row["method"], row["negative_group"]) for row in rows})
    for method, negative_group in combinations:
        selected = [
            row
            for row in rows
            if row["method"] == method and row["negative_group"] == negative_group
        ]
        aucs = np.asarray([row["auc"] for row in selected], dtype=np.float64)
        bas = np.asarray(
            [row["balanced_accuracy"] for row in selected], dtype=np.float64
        )
        macro_rows.append(
            {
                "method": method,
                "negative_group": negative_group,
                "groups": len(selected),
                "macro_auc": float(np.nanmean(aucs)),
                "min_auc": float(np.nanmin(aucs)),
                "max_auc": float(np.nanmax(aucs)),
                "macro_balanced_accuracy": float(np.nanmean(bas)),
            }
        )
    macro_path = os.path.join(args.output_dir, "macro_summary.csv")
    with open(macro_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(macro_rows[0]))
        writer.writeheader()
        writer.writerows(macro_rows)

    macro = {
        (row["method"], row["negative_group"]): row for row in macro_rows
    }
    class_hard = macro[("class_prototype_logit", "hard_negative")]["macro_auc"]
    teacher_hard = macro[("teacher_score", "hard_negative")]["macro_auc"]
    class_random = macro[("class_prototype_logit", "random_negative")]["macro_auc"]
    macro_hard_auc = class_hard
    hard_auc_gain_over_teacher = class_hard - teacher_hard
    gate_pass = bool(
        macro_hard_auc >= 0.60
        and hard_auc_gain_over_teacher >= 0.02
        and class_random >= 0.80
    )
    decision = {
        "gate_pass": gate_pass,
        "macro_hard_auc": macro_hard_auc,
        "teacher_macro_hard_auc": teacher_hard,
        "hard_auc_gain_over_teacher": hard_auc_gain_over_teacher,
        "macro_random_auc": class_random,
        "gates": {
            "macro_hard_auc_min": 0.60,
            "hard_auc_gain_over_teacher_min": 0.02,
            "macro_random_auc_min": 0.80,
        },
        "configuration": vars(args),
        "prototype_audits": prototype_audits,
    }
    with open(
        os.path.join(args.output_dir, "offline_decision.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print(
        f"[Class-prototype-LOSO] gate_pass={gate_pass}, "
        f"macro_hard_auc={macro_hard_auc:.6f}, "
        f"gain={hard_auc_gain_over_teacher:+.6f}, "
        f"macro_random_auc={class_random:.6f}"
    )
    print(f"[Class-prototype-LOSO] output={args.output_dir}")


if __name__ == "__main__":
    main()
