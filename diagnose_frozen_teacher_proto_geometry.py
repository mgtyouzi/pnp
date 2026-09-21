import argparse
import csv
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose frozen-teacher feature geometry without training."
    )
    parser.add_argument("--bank", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--negative_sample_size", default=256, type=int)
    parser.add_argument("--negative_energy_blocks", default=4, type=int)
    parser.add_argument("--negative_sampling_seed", default=0, type=int)
    parser.add_argument("--ridge", default=0.01, type=float)
    parser.add_argument("--linear_probe_samples_per_class", default=8192, type=int)
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


def best_balanced_accuracy(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[float, float]:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 501)))
    best_accuracy = -1.0
    best_threshold = 0.0
    for threshold in thresholds:
        prediction = scores >= threshold
        tpr = float(prediction[labels == 1].mean())
        tnr = float((~prediction[labels == 0]).mean())
        balanced = 0.5 * (tpr + tnr)
        if balanced > best_accuracy:
            best_accuracy = balanced
            best_threshold = float(threshold)
    return best_accuracy, best_threshold


def deterministic_rows(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(values) <= count:
        return values
    random = np.random.RandomState(seed)
    return values[random.choice(len(values), size=count, replace=False)]


def score_geometry(
    values: np.ndarray,
    prototypes: torch.Tensor,
    negative_bank: torch.Tensor,
    temperature: float,
    sample_size: int,
    blocks: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    device = prototypes.device
    offsets = torch.remainder(
        int(seed) + torch.arange(blocks, device=device) * sample_size,
        len(negative_bank),
    )
    result = {
        "foreground_similarity": [],
        "nearest_bank_margin": [],
        "protonce_relative_energy": [],
    }
    for start in range(0, len(values), 4096):
        query = F.normalize(
            torch.from_numpy(values[start : start + 4096]).float().to(device),
            dim=-1,
            eps=1e-6,
        )
        foreground_logits = query.matmul(prototypes.t())
        foreground_similarity = foreground_logits.max(dim=1).values
        positive_energy = torch.logsumexp(
            foreground_logits / temperature, dim=1
        )
        margins = []
        relative_energies = []
        for offset in offsets:
            indices = (
                torch.arange(sample_size, device=device) + int(offset.item())
            ) % len(negative_bank)
            negative_logits = query.matmul(negative_bank[indices].t())
            margins.append(
                foreground_similarity - negative_logits.max(dim=1).values
            )
            negative_energy = torch.logsumexp(
                negative_logits / temperature, dim=1
            )
            relative_energies.append(positive_energy - negative_energy)
        result["foreground_similarity"].append(
            foreground_similarity.detach().cpu().numpy()
        )
        result["nearest_bank_margin"].append(
            torch.stack(margins).mean(dim=0).detach().cpu().numpy()
        )
        result["protonce_relative_energy"].append(
            torch.stack(relative_energies).mean(dim=0).detach().cpu().numpy()
        )
    return {name: np.concatenate(parts) for name, parts in result.items()}


def fit_ridge_linear_probe(
    support_positive: np.ndarray,
    negative_bank: np.ndarray,
    device: torch.device,
    samples_per_class: int,
    ridge: float,
) -> torch.Tensor:
    positive = deterministic_rows(support_positive, samples_per_class, 101)
    negative = deterministic_rows(negative_bank, samples_per_class, 103)
    values = np.concatenate([positive, negative], axis=0)
    targets = np.concatenate(
        [np.ones(len(positive), dtype=np.float32), -np.ones(len(negative), dtype=np.float32)]
    )
    x = F.normalize(torch.from_numpy(values).float().to(device), dim=-1, eps=1e-6)
    x = torch.cat([x, torch.ones((len(x), 1), device=device)], dim=1)
    y = torch.from_numpy(targets).to(device)
    identity = torch.eye(x.shape[1], device=device)
    identity[-1, -1] = 0.0
    return torch.linalg.solve(x.t().matmul(x) + ridge * identity, x.t().matmul(y))


def score_linear_probe(
    values: np.ndarray, weights: torch.Tensor
) -> np.ndarray:
    output = []
    device = weights.device
    for start in range(0, len(values), 4096):
        x = F.normalize(
            torch.from_numpy(values[start : start + 4096]).float().to(device),
            dim=-1,
            eps=1e-6,
        )
        x = torch.cat([x, torch.ones((len(x), 1), device=device)], dim=1)
        output.append(x.matmul(weights).detach().cpu().numpy())
    return np.concatenate(output)


def evaluate_method(
    positive_score: np.ndarray, negative_score: np.ndarray
) -> Dict[str, float]:
    scores = np.concatenate([positive_score, negative_score])
    labels = np.concatenate(
        [
            np.ones(len(positive_score), dtype=np.int64),
            np.zeros(len(negative_score), dtype=np.int64),
        ]
    )
    balanced, threshold = best_balanced_accuracy(labels, scores)
    return {
        "auc": roc_auc(labels, scores),
        "balanced_accuracy": balanced,
        "threshold": threshold,
        "positive_mean": float(np.mean(positive_score)),
        "negative_mean": float(np.mean(negative_score)),
    }


def main() -> None:
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnosis requested but CUDA is unavailable")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    payload = torch.load(args.bank, map_location="cpu")
    prototypes = F.normalize(
        payload["foreground_prototypes"].float().to(device), dim=-1, eps=1e-6
    )
    negative_bank_array = payload["negative_bank"].float().cpu().numpy()
    negative_bank = F.normalize(
        payload["negative_bank"].float().to(device), dim=-1, eps=1e-6
    )
    sample_size = min(args.negative_sample_size, len(negative_bank))
    block_count = min(args.negative_energy_blocks, len(negative_bank))
    with np.load(args.features) as features:
        groups = {
            "positive": features["calibration_positive"].astype(np.float32, copy=True),
            "hard_negative": features["calibration_hard_negative"].astype(
                np.float32, copy=True
            ),
            "random_negative": features["calibration_random_negative"].astype(
                np.float32, copy=True
            ),
        }
        support_positive = features["support_positive"].astype(np.float32, copy=True)

    group_scores = {
        name: score_geometry(
            values,
            prototypes,
            negative_bank,
            args.temperature,
            sample_size,
            block_count,
            args.negative_sampling_seed,
        )
        for name, values in groups.items()
    }
    linear_weights = fit_ridge_linear_probe(
        support_positive,
        negative_bank_array,
        device,
        args.linear_probe_samples_per_class,
        args.ridge,
    )
    for name, values in groups.items():
        group_scores[name]["ridge_linear_probe"] = score_linear_probe(
            values, linear_weights
        )

    methods = (
        "foreground_similarity",
        "nearest_bank_margin",
        "protonce_relative_energy",
        "ridge_linear_probe",
    )
    negative_sets = {
        "hard_negative": ("hard_negative",),
        "random_negative": ("random_negative",),
        "all_negative": ("hard_negative", "random_negative"),
    }
    rows = []
    for method in methods:
        for negative_name, members in negative_sets.items():
            negative_score = np.concatenate(
                [group_scores[group][method] for group in members]
            )
            metrics = evaluate_method(group_scores["positive"][method], negative_score)
            rows.append(
                {
                    "method": method,
                    "negative_group": negative_name,
                    **metrics,
                }
            )

    with open(
        os.path.join(args.output_dir, "method_separation.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {
        (row["method"], row["negative_group"]): row for row in rows
    }
    protonce_auc = lookup[("protonce_relative_energy", "all_negative")]["auc"]
    linear_auc = lookup[("ridge_linear_probe", "all_negative")]["auc"]
    if linear_auc < 0.60:
        conclusion = "feature_not_linearly_separable"
        next_action = (
            "Do not train prototypes. Change the frozen feature source or audit "
            "positive/negative labels before rebuilding any prototype objective."
        )
    elif linear_auc >= 0.70 and protonce_auc < 0.60:
        conclusion = "prototype_objective_mismatch"
        next_action = (
            "Frozen features contain usable signal, but one-sided foreground "
            "ProtoNCE is wrong. Replace it with a discriminative two-class head "
            "or class-conditional foreground/background prototypes."
        )
    else:
        conclusion = "weak_or_mixed_separation"
        next_action = (
            "Inspect hard-negative and random-negative rows separately before "
            "choosing a new objective."
        )
    report = {
        "bank": os.path.abspath(args.bank),
        "features": os.path.abspath(args.features),
        "configuration": vars(args),
        "counts": {name: len(values) for name, values in groups.items()},
        "method_results": rows,
        "conclusion": conclusion,
        "next_action": next_action,
    }
    with open(
        os.path.join(args.output_dir, "geometry_diagnosis.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"[Prototype-geometry] ProtoNCE AUC={protonce_auc:.6f}, "
        f"ridge-linear AUC={linear_auc:.6f}"
    )
    print(f"[Prototype-geometry] conclusion={conclusion}")
    print(f"[Prototype-geometry] output={args.output_dir}")


if __name__ == "__main__":
    main()
