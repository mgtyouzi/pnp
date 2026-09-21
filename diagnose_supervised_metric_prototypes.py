import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from build_frozen_teacher_fg_bank import spherical_kmeans
from diagnose_class_conditional_prototypes import (
    TYPE_HARD_NEGATIVE,
    TYPE_POSITIVE,
    TYPE_RANDOM_NEGATIVE,
    balanced_accuracy,
    deterministic_subset,
    evaluate_scores,
    fit_ridge,
    roc_auc,
    score_ridge,
)


IMPLEMENTATION_VERSION = "supervised_metric_prototypes_v1_20260816"
CLASS_NAMES = ("positive", "hard_negative", "random_negative")


@dataclass
class FoldSplit:
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    train_images: np.ndarray
    calibration_images: np.ndarray
    test_images: np.ndarray


def get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free LOSO audit for a supervised metric projection followed by "
            "three-class multi-prototype scoring."
        )
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--embedding_dim", default=32, type=int)
    parser.add_argument("--foreground_prototypes", default=4, type=int)
    parser.add_argument("--hard_background_prototypes", default=4, type=int)
    parser.add_argument("--random_background_prototypes", default=2, type=int)
    parser.add_argument("--temperature", default=0.15, type=float)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--calibration_fraction", default=0.20, type=float)
    parser.add_argument("--positive_per_group", default=4096, type=int)
    parser.add_argument("--hard_negative_per_group", default=2048, type=int)
    parser.add_argument("--random_negative_per_group", default=2048, type=int)
    parser.add_argument("--hard_negative_weight", default=2.0, type=float)
    parser.add_argument("--diversity_weight", default=0.02, type=float)
    parser.add_argument("--projection_anchor_weight", default=1e-3, type=float)
    parser.add_argument("--kmeans_iterations", default=20, type=int)
    parser.add_argument("--ridge", default=0.01, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hard_auc_gate", default=0.63, type=float)
    parser.add_argument("--hard_min_auc_gate", default=0.55, type=float)
    parser.add_argument("--random_auc_gate", default=0.95, type=float)
    parser.add_argument("--teacher_gain_gate", default=0.05, type=float)
    return parser


def _choose_calibration_images(
    image_values: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    images = np.unique(image_values)
    if len(images) < 2:
        raise RuntimeError("each LOSO training group needs at least two images")
    count = max(1, int(round(len(images) * fraction)))
    count = min(count, len(images) - 1)
    random = np.random.RandomState(seed)
    return np.sort(random.choice(images, size=count, replace=False))


def build_fold_split(
    candidate_type: np.ndarray,
    group_index: np.ndarray,
    image_index: np.ndarray,
    held_group: int,
    calibration_fraction: float,
    seed: int,
) -> FoldSplit:
    del candidate_type  # Candidate classes are checked after image-level splitting.
    train_mask = np.zeros(len(group_index), dtype=bool)
    calibration_mask = np.zeros(len(group_index), dtype=bool)
    training_groups = sorted(int(v) for v in np.unique(group_index) if v != held_group)
    for group in training_groups:
        group_mask = group_index == group
        calibration_images = _choose_calibration_images(
            image_index[group_mask], calibration_fraction, seed + 1009 * group
        )
        group_calibration = group_mask & np.isin(image_index, calibration_images)
        calibration_mask |= group_calibration
        train_mask |= group_mask & ~group_calibration
    test_mask = group_index == held_group
    split = FoldSplit(
        train_indices=np.where(train_mask)[0],
        calibration_indices=np.where(calibration_mask)[0],
        test_indices=np.where(test_mask)[0],
        train_images=np.unique(image_index[train_mask]),
        calibration_images=np.unique(image_index[calibration_mask]),
        test_images=np.unique(image_index[test_mask]),
    )
    if set(split.train_images).intersection(set(split.calibration_images)):
        raise RuntimeError("train/calibration image leakage")
    if set(group_index[split.test_indices]).intersection(
        set(group_index[split.train_indices]) | set(group_index[split.calibration_indices])
    ):
        raise RuntimeError("held-out group leakage")
    return split


def subset_by_group_and_class(
    indices: np.ndarray,
    candidate_type: np.ndarray,
    group_index: np.ndarray,
    caps: Tuple[int, int, int],
    seed: int,
) -> np.ndarray:
    selected = []
    for group in sorted(int(value) for value in np.unique(group_index[indices])):
        for class_id, cap in enumerate(caps):
            available = indices[
                (group_index[indices] == group)
                & (candidate_type[indices] == class_id)
            ]
            if len(available) == 0:
                raise RuntimeError(
                    f"group={group} has no {CLASS_NAMES[class_id]} candidates"
                )
            selected.append(
                deterministic_subset(
                    available, cap, seed + 10007 * group + 101 * class_id
                )
            )
    return np.concatenate(selected)


def logmeanexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim) - math.log(values.shape[dim])


class SupervisedMetricPrototypes(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        prototype_counts: Tuple[int, int, int],
        temperature: float,
    ):
        super().__init__()
        self.projector = nn.Linear(input_dim, embedding_dim, bias=True)
        self.prototype_counts = tuple(int(value) for value in prototype_counts)
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(
            torch.empty(sum(self.prototype_counts), embedding_dim)
        )
        offsets = np.cumsum((0,) + self.prototype_counts)
        self.class_slices = tuple(
            (int(offsets[index]), int(offsets[index + 1]))
            for index in range(3)
        )
        nn.init.orthogonal_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)
        nn.init.normal_(self.prototypes, std=0.02)

    def embed(self, values: torch.Tensor) -> torch.Tensor:
        values = F.normalize(values.float(), dim=-1, eps=1e-6)
        return F.normalize(self.projector(values), dim=-1, eps=1e-6)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1e-6)

    def class_logits(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(values)
        similarities = embedded.matmul(self.normalized_prototypes().t())
        logits = []
        for start, end in self.class_slices:
            logits.append(
                logmeanexp(similarities[:, start:end] / self.temperature, dim=1)
            )
        return torch.stack(logits, dim=1), similarities

    def foreground_score(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, similarities = self.class_logits(values)
        background = torch.logsumexp(logits[:, 1:], dim=1) - math.log(2.0)
        return logits[:, 0] - background, similarities.argmax(dim=1)


def _solve_multiclass_ridge(
    values: torch.Tensor, labels: torch.Tensor, ridge: float
) -> torch.Tensor:
    x = F.normalize(values.float(), dim=-1, eps=1e-6)
    x = torch.cat([x, torch.ones((len(x), 1), device=x.device)], dim=1)
    target = F.one_hot(labels.long(), num_classes=3).float().mul(2.0).sub(1.0)
    identity = torch.eye(x.shape[1], device=x.device)
    identity[-1, -1] = 0.0
    return torch.linalg.solve(x.t().matmul(x) + ridge * identity, x.t().matmul(target))


@torch.no_grad()
def initialize_discriminative_projection(
    model: SupervisedMetricPrototypes,
    values: torch.Tensor,
    labels: torch.Tensor,
    ridge: float = 0.01,
) -> None:
    solution = _solve_multiclass_ridge(values, labels, ridge)
    weight = torch.zeros_like(model.projector.weight)
    bias = torch.zeros_like(model.projector.bias)
    discriminative_dims = min(3, weight.shape[0])
    weight[:discriminative_dims] = solution[:-1, :discriminative_dims].t()
    bias[:discriminative_dims] = solution[-1, :discriminative_dims]

    remaining = weight.shape[0] - discriminative_dims
    if remaining > 0:
        normalized = F.normalize(values.float(), dim=-1, eps=1e-6)
        centered = normalized - normalized.mean(dim=0, keepdim=True)
        q = min(remaining, centered.shape[1], max(1, centered.shape[0] - 1))
        _, _, components = torch.pca_lowrank(centered, q=q, center=False)
        weight[discriminative_dims : discriminative_dims + q] = components[:, :q].t()
        bias[discriminative_dims : discriminative_dims + q] = -normalized.mean(
            dim=0
        ).matmul(components[:, :q])
    model.projector.weight.copy_(weight)
    model.projector.bias.copy_(bias)


@torch.no_grad()
def initialize_class_prototypes(
    model: SupervisedMetricPrototypes,
    values: torch.Tensor,
    labels: torch.Tensor,
    iterations: int,
    seed: int,
) -> None:
    embedded = model.embed(values)
    centers = []
    for class_id, count in enumerate(model.prototype_counts):
        class_values = embedded[labels == class_id]
        prototypes, _ = spherical_kmeans(
            class_values, count, iterations, seed + 1009 * class_id
        )
        centers.append(prototypes)
    model.prototypes.copy_(torch.cat(centers, dim=0))


def prototype_diversity_loss(model: SupervisedMetricPrototypes) -> torch.Tensor:
    prototypes = model.normalized_prototypes()
    penalties = []
    for start, end in model.class_slices:
        count = end - start
        if count <= 1:
            continue
        similarities = prototypes[start:end].matmul(prototypes[start:end].t())
        off_diagonal = ~torch.eye(count, dtype=torch.bool, device=prototypes.device)
        penalties.append(F.relu(similarities[off_diagonal] - 0.50).mean())
    if not penalties:
        return prototypes.sum() * 0.0
    return torch.stack(penalties).mean()


def _macro_group_auc(
    scores: np.ndarray,
    candidate_type: np.ndarray,
    group_index: np.ndarray,
    positive_type: int,
    negative_type: int,
) -> Tuple[float, float]:
    aucs = []
    for group in np.unique(group_index):
        positive = scores[(group_index == group) & (candidate_type == positive_type)]
        negative = scores[(group_index == group) & (candidate_type == negative_type)]
        if len(positive) and len(negative):
            labels = np.concatenate(
                [np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64)]
            )
            aucs.append(roc_auc(labels, np.concatenate([positive, negative])))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.min(aucs))


@torch.no_grad()
def score_supervised_prototypes(
    model: SupervisedMetricPrototypes, values: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    scores = []
    assignments = []
    device = next(model.parameters()).device
    model.eval()
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[start : start + batch_size]).float().to(device)
        batch_scores, batch_assignments = model.foreground_score(batch)
        scores.append(batch_scores.cpu().numpy())
        assignments.append(batch_assignments.cpu().numpy())
    return np.concatenate(scores), np.concatenate(assignments)


def _clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def fit_supervised_prototypes(
    features: np.ndarray,
    candidate_type: np.ndarray,
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    args,
    device: torch.device,
    group_index: np.ndarray = None,
):
    if group_index is None:
        group_index = np.zeros(len(candidate_type), dtype=np.int16)
    model = SupervisedMetricPrototypes(
        input_dim=features.shape[1],
        embedding_dim=args.embedding_dim,
        prototype_counts=(
            args.foreground_prototypes,
            args.hard_background_prototypes,
            args.random_background_prototypes,
        ),
        temperature=args.temperature,
    ).to(device)
    train_values = torch.from_numpy(features[train_indices]).float().to(device)
    train_labels = torch.from_numpy(candidate_type[train_indices].astype(np.int64)).to(device)
    initialize_discriminative_projection(model, train_values, train_labels)
    initialize_class_prototypes(
        model,
        train_values,
        train_labels,
        getattr(args, "kmeans_iterations", 20),
        args.seed,
    )
    projector_reference = {
        name: value.detach().clone()
        for name, value in model.projector.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    class_weights = torch.tensor(
        [1.0, args.hard_negative_weight, 1.0], device=device
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    history = []
    best_state = _clone_state(model)
    best_epoch = -1
    best_objective = -float("inf")
    stale_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_indices), generator=generator)
        epoch_losses = []
        for start in range(0, len(order), args.batch_size):
            local = order[start : start + args.batch_size].to(device)
            values = train_values[local]
            labels = train_labels[local]
            logits, _ = model.class_logits(values)
            classification = F.cross_entropy(logits, labels, weight=class_weights)
            diversity = prototype_diversity_loss(model)
            anchor = sum(
                (parameter - projector_reference[name]).square().mean()
                for name, parameter in model.projector.named_parameters()
            )
            loss = (
                classification
                + args.diversity_weight * diversity
                + getattr(args, "projection_anchor_weight", 1e-3) * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        calibration_scores, _ = score_supervised_prototypes(
            model, features[calibration_indices], args.batch_size
        )
        cal_types = candidate_type[calibration_indices]
        cal_groups = group_index[calibration_indices]
        hard_auc, hard_min_auc = _macro_group_auc(
            calibration_scores,
            cal_types,
            cal_groups,
            TYPE_POSITIVE,
            TYPE_HARD_NEGATIVE,
        )
        random_auc, _ = _macro_group_auc(
            calibration_scores,
            cal_types,
            cal_groups,
            TYPE_POSITIVE,
            TYPE_RANDOM_NEGATIVE,
        )
        objective = hard_auc + 0.10 * min(random_auc, 0.95)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "calibration_hard_auc": hard_auc,
                "calibration_hard_min_auc": hard_min_auc,
                "calibration_random_auc": random_auc,
                "selection_objective": objective,
            }
        )
        if objective > best_objective + 1e-5:
            best_objective = objective
            best_epoch = epoch
            best_state = _clone_state(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"[Supervised-prototype-train] epoch={epoch + 1}/{args.epochs}, "
            f"loss={history[-1]['train_loss']:.6f}, "
            f"cal_hard_auc={hard_auc:.6f}, "
            f"cal_hard_min={hard_min_auc:.6f}, "
            f"cal_random_auc={random_auc:.6f}, stale={stale_epochs}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    prototypes = model.normalized_prototypes().detach().cpu()
    projector = model.projector.weight.detach().cpu()
    singular_values = torch.linalg.svdvals(projector)
    audit = {
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "best_calibration_objective": best_objective,
        "projector_frobenius_norm": float(projector.norm()),
        "projector_singular_values": singular_values.tolist(),
        "prototype_pairwise_cosine": prototypes.matmul(prototypes.t()).tolist(),
    }
    return model, history, audit


def _assignment_audit(
    assignments: np.ndarray,
    candidate_type: np.ndarray,
    class_slices: Iterable[Tuple[int, int]],
) -> Dict[str, object]:
    slices = tuple(class_slices)
    output = {}
    for class_id, (start, end) in enumerate(slices):
        class_assignments = assignments[candidate_type == class_id]
        counts = [int((class_assignments == index).sum()) for index in range(start, end)]
        total = max(len(class_assignments), 1)
        predicted_class_counts = []
        for predicted_start, predicted_end in slices:
            predicted_class_counts.append(
                int(
                    (
                        (class_assignments >= predicted_start)
                        & (class_assignments < predicted_end)
                    ).sum()
                )
            )
        output[CLASS_NAMES[class_id]] = {
            "counts": counts,
            "shares": [count / total for count in counts],
            "active": int(sum(count > 0 for count in counts)),
            "predicted_class_counts": predicted_class_counts,
            "correct_class_assignment_rate": sum(counts) / total,
        }
    return output


def _fold_score_rows(
    held_group: int,
    held_group_name: str,
    held_images: int,
    candidate_type: np.ndarray,
    score_sets: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    local = {
        name: np.where(candidate_type == class_id)[0]
        for class_id, name in enumerate(CLASS_NAMES)
    }
    negative_sets = {
        "hard_negative": local["hard_negative"],
        "random_negative": local["random_negative"],
        "all_negative": np.concatenate(
            [local["hard_negative"], local["random_negative"]]
        ),
    }
    rows = []
    for method, scores in score_sets.items():
        for negative_name, negative_indices in negative_sets.items():
            rows.append(
                {
                    "held_group": held_group,
                    "held_group_name": held_group_name,
                    "held_images": held_images,
                    "method": method,
                    "negative_group": negative_name,
                    "positive_count": len(local["positive"]),
                    "negative_count": len(negative_indices),
                    **evaluate_scores(scores[local["positive"]], scores[negative_indices]),
                }
            )
    return rows


def make_gate_decision(
    hard_auc: float,
    hard_min_auc: float,
    random_auc: float,
    teacher_hard_auc: float,
    ridge_hard_auc: float,
    hard_auc_gate: float = 0.63,
    hard_min_auc_gate: float = 0.55,
    random_auc_gate: float = 0.95,
    teacher_gain_gate: float = 0.05,
) -> Dict[str, object]:
    values = {
        "macro_hard_auc": (hard_auc, hard_auc_gate),
        "min_hard_auc": (hard_min_auc, hard_min_auc_gate),
        "macro_random_auc": (random_auc, random_auc_gate),
        "hard_auc_gain_over_teacher": (
            hard_auc - teacher_hard_auc,
            teacher_gain_gate,
        ),
    }
    failures = [name for name, (value, threshold) in values.items() if value < threshold]
    return {
        "gate_pass": not failures,
        "failures": failures,
        "macro_hard_auc": hard_auc,
        "min_hard_auc": hard_min_auc,
        "macro_random_auc": random_auc,
        "teacher_macro_hard_auc": teacher_hard_auc,
        "ridge_macro_hard_auc": ridge_hard_auc,
        "hard_auc_gain_over_teacher": hard_auc - teacher_hard_auc,
        "hard_auc_gap_to_ridge": hard_auc - ridge_hard_auc,
        "gates": {name: threshold for name, (_, threshold) in values.items()},
    }


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_loso(args, data, manifest, device):
    features = data["features"].astype(np.float32, copy=False)
    candidate_type = data["candidate_type"].astype(np.int64, copy=False)
    group_index = data["group_index"].astype(np.int64, copy=False)
    image_index = data["image_index"].astype(np.int64, copy=False)
    teacher_score = data["teacher_score"].astype(np.float32, copy=False)
    all_groups = np.unique(group_index)
    group_names = {int(index): name for name, index in manifest["group_to_index"].items()}
    fold_rows = []
    history_rows = []
    fold_audits = []
    caps = (
        args.positive_per_group,
        args.hard_negative_per_group,
        args.random_negative_per_group,
    )

    for fold_index, held_group_value in enumerate(all_groups):
        held_group = int(held_group_value)
        split = build_fold_split(
            candidate_type,
            group_index,
            image_index,
            held_group,
            args.calibration_fraction,
            args.seed + 100000 * fold_index,
        )
        train_indices = subset_by_group_and_class(
            split.train_indices,
            candidate_type,
            group_index,
            caps,
            args.seed + 200000 * fold_index,
        )
        calibration_indices = subset_by_group_and_class(
            split.calibration_indices,
            candidate_type,
            group_index,
            caps,
            args.seed + 300000 * fold_index,
        )
        model, history, audit = fit_supervised_prototypes(
            features,
            candidate_type,
            train_indices,
            calibration_indices,
            args,
            device,
            group_index=group_index,
        )
        test_indices = split.test_indices
        supervised_scores, assignments = score_supervised_prototypes(
            model, features[test_indices], args.batch_size
        )

        training_types = candidate_type[train_indices]
        positive_train = train_indices[training_types == TYPE_POSITIVE]
        hard_train = train_indices[training_types == TYPE_HARD_NEGATIVE]
        random_train = train_indices[training_types == TYPE_RANDOM_NEGATIVE]
        ridge_weights = fit_ridge(
            features,
            positive_train,
            hard_train,
            random_train,
            args.ridge,
            args.seed + fold_index,
            device,
        )
        score_sets = {
            "supervised_metric_prototype": supervised_scores,
            "ridge_linear_probe": score_ridge(features[test_indices], ridge_weights),
            "teacher_score": teacher_score[test_indices],
        }
        held_name = group_names.get(held_group, str(held_group))
        fold_rows.extend(
            _fold_score_rows(
                held_group,
                held_name,
                len(split.test_images),
                candidate_type[test_indices],
                score_sets,
            )
        )
        for row in history:
            history_rows.append({"held_group": held_group, "held_group_name": held_name, **row})
        audit.update(
            {
                "held_group": held_group,
                "held_group_name": held_name,
                "train_groups": sorted(set(int(v) for v in group_index[train_indices])),
                "test_groups": sorted(set(int(v) for v in group_index[test_indices])),
                "train_images": len(split.train_images),
                "calibration_images": len(split.calibration_images),
                "test_images": len(split.test_images),
                "train_candidates": len(train_indices),
                "calibration_candidates": len(calibration_indices),
                "test_candidates": len(test_indices),
                "assignment": _assignment_audit(
                    assignments,
                    candidate_type[test_indices],
                    model.class_slices,
                ),
            }
        )
        fold_audits.append(audit)
        print(
            f"[Supervised-prototype-LOSO] fold={fold_index + 1}/{len(all_groups)}, "
            f"held={held_name}, selected_epoch={audit['selected_epoch']}",
            flush=True,
        )
    return fold_rows, history_rows, fold_audits


def summarize_macro(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output = []
    combinations = sorted({(row["method"], row["negative_group"]) for row in rows})
    for method, negative_group in combinations:
        selected = [
            row
            for row in rows
            if row["method"] == method and row["negative_group"] == negative_group
        ]
        aucs = np.asarray([row["auc"] for row in selected], dtype=np.float64)
        bas = np.asarray([row["balanced_accuracy"] for row in selected], dtype=np.float64)
        output.append(
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
    return output


def main():
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    with np.load(args.features) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    required = {"features", "candidate_type", "teacher_score", "group_index", "image_index"}
    missing = sorted(required.difference(data))
    if missing:
        raise RuntimeError(f"candidate archive missing arrays: {missing}")
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)

    fold_rows, history_rows, fold_audits = run_loso(args, data, manifest, device)
    macro_rows = summarize_macro(fold_rows)
    _write_csv(os.path.join(args.output_dir, "fold_metrics.csv"), fold_rows)
    _write_csv(os.path.join(args.output_dir, "macro_summary.csv"), macro_rows)
    _write_csv(os.path.join(args.output_dir, "training_history.csv"), history_rows)
    with open(
        os.path.join(args.output_dir, "fold_audits.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(fold_audits, handle, ensure_ascii=False, indent=2)

    macro = {(row["method"], row["negative_group"]): row for row in macro_rows}
    supervised_hard = macro[("supervised_metric_prototype", "hard_negative")]
    supervised_random = macro[("supervised_metric_prototype", "random_negative")]
    teacher_hard = macro[("teacher_score", "hard_negative")]
    ridge_hard = macro[("ridge_linear_probe", "hard_negative")]
    decision = make_gate_decision(
        hard_auc=supervised_hard["macro_auc"],
        hard_min_auc=supervised_hard["min_auc"],
        random_auc=supervised_random["macro_auc"],
        teacher_hard_auc=teacher_hard["macro_auc"],
        ridge_hard_auc=ridge_hard["macro_auc"],
        hard_auc_gate=args.hard_auc_gate,
        hard_min_auc_gate=args.hard_min_auc_gate,
        random_auc_gate=args.random_auc_gate,
        teacher_gain_gate=args.teacher_gain_gate,
    )
    decision.update(
        {
            "version": IMPLEMENTATION_VERSION,
            "configuration": vars(args),
            "candidate_archive": os.path.abspath(args.features),
            "manifest": os.path.abspath(args.manifest),
            "next_action": (
                "integrate_supervised_metric_prototypes_into_p2p"
                if decision["gate_pass"]
                else "stop_prototype_training_and_use_discriminative_hard_negative_head"
            ),
        }
    )
    with open(
        os.path.join(args.output_dir, "offline_decision.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print(
        f"[Supervised-prototype-LOSO] gate_pass={decision['gate_pass']}, "
        f"hard_auc={decision['macro_hard_auc']:.6f}, "
        f"hard_min={decision['min_hard_auc']:.6f}, "
        f"random_auc={decision['macro_random_auc']:.6f}, "
        f"ridge_gap={decision['hard_auc_gap_to_ridge']:+.6f}",
        flush=True,
    )
    print(f"[Supervised-prototype-LOSO] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
