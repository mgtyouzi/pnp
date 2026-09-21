import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np


CANDIDATE_NAMES = {0: "positive", 1: "hard_negative", 2: "random_negative"}
AUDIT_VERSION = "frozen_supervised_bank_transfer_v1_20260827"


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool, copy=False).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have identical shapes")
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
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
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def summarize_separation(
    scores: np.ndarray,
    candidate_type: np.ndarray,
    group_index: np.ndarray,
    positive_type: int,
    negative_type: int,
) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    candidate_type = np.asarray(candidate_type)
    group_index = np.asarray(group_index)
    selected = (candidate_type == positive_type) | (candidate_type == negative_type)
    labels = candidate_type[selected] == positive_type
    selected_scores = scores[selected]
    selected_groups = group_index[selected]
    group_auc = []
    for group in np.unique(selected_groups):
        group_mask = selected_groups == group
        value = binary_roc_auc(labels[group_mask], selected_scores[group_mask])
        if np.isfinite(value):
            group_auc.append(float(value))
    return {
        "global_auc": float(binary_roc_auc(labels, selected_scores)),
        "macro_group_auc": (
            float(np.mean(group_auc)) if group_auc else float("nan")
        ),
        "min_group_auc": float(np.min(group_auc)) if group_auc else float("nan"),
        "valid_groups": int(len(group_auc)),
        "positive_count": int((candidate_type == positive_type).sum()),
        "negative_count": int((candidate_type == negative_type).sum()),
    }


def decide_transfer(
    hard_auc: float,
    min_group_auc: float,
    random_auc: float,
    gain_over_teacher: float,
) -> Dict[str, object]:
    pass_checks = {
        "hard_auc_at_least_0p70": hard_auc >= 0.70,
        "min_group_auc_at_least_0p60": min_group_auc >= 0.60,
        "random_auc_at_least_0p90": random_auc >= 0.90,
        "gain_over_teacher_at_least_0p02": gain_over_teacher >= 0.02,
    }
    weak_checks = {
        "hard_auc_at_least_0p62": hard_auc >= 0.62,
        "min_group_auc_at_least_0p52": min_group_auc >= 0.52,
        "random_auc_at_least_0p85": random_auc >= 0.85,
    }
    if all(pass_checks.values()):
        status = "pass"
        action = "keep_bank_and_fix_training_integration"
    elif all(weak_checks.values()):
        status = "weak"
        action = "rebuild_bank_for_cross_style_invariance_before_training"
    else:
        status = "fail"
        action = "stop_current_source_only_frozen_prototype_scheme"
    return {
        "status": status,
        "action": action,
        "pass_checks": pass_checks,
        "weak_checks": weak_checks,
    }


def _write_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _score_bank(
    features: np.ndarray,
    payload: Dict,
    device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    weight = torch.as_tensor(payload["projector_weight"]).float().to(device)
    bias = torch.as_tensor(payload["projector_bias"]).float().to(device)
    prototypes = F.normalize(
        torch.as_tensor(payload["prototypes"]).float().to(device), dim=-1, eps=1e-6
    )
    prototype_counts = tuple(int(value) for value in payload["prototype_counts"])
    if len(prototype_counts) != 3 or sum(prototype_counts) != prototypes.shape[0]:
        raise ValueError("bank prototype_counts are incompatible with prototypes")
    if features.ndim != 2 or features.shape[1] != weight.shape[1]:
        raise ValueError(
            f"candidate feature shape {features.shape} is incompatible with "
            f"projector weight {tuple(weight.shape)}"
        )
    temperature = float(payload["temperature"])
    offsets = np.cumsum((0,) + prototype_counts)
    score_parts: List[np.ndarray] = []
    assignment_parts: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            values = torch.from_numpy(
                np.asarray(features[start : start + batch_size], dtype=np.float32)
            ).to(device)
            values = F.normalize(values, dim=-1, eps=1e-6)
            embedded = F.normalize(F.linear(values, weight, bias), dim=-1, eps=1e-6)
            similarities = embedded.matmul(prototypes.t())
            logits = []
            for class_id, count in enumerate(prototype_counts):
                left, right = int(offsets[class_id]), int(offsets[class_id + 1])
                logits.append(
                    torch.logsumexp(
                        similarities[:, left:right] / temperature, dim=-1
                    )
                    - math.log(count)
                )
            logits = torch.stack(logits, dim=-1)
            score_parts.append(logits.softmax(dim=-1)[:, 0].cpu().numpy())
            assignment_parts.append(similarities.argmax(dim=-1).cpu().numpy())
    return np.concatenate(score_parts), np.concatenate(assignment_parts)


def _score_quantiles(
    method: str, scores: np.ndarray, candidate_type: np.ndarray
) -> List[Dict[str, object]]:
    rows = []
    for class_id, name in CANDIDATE_NAMES.items():
        values = np.asarray(scores[candidate_type == class_id], dtype=np.float64)
        if not len(values):
            continue
        quantiles = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
        rows.append(
            {
                "method": method,
                "candidate_type": name,
                "count": len(values),
                "mean": float(values.mean()),
                "p10": float(quantiles[0]),
                "p25": float(quantiles[1]),
                "p50": float(quantiles[2]),
                "p75": float(quantiles[3]),
                "p90": float(quantiles[4]),
            }
        )
    return rows


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit transfer of a frozen source prototype bank to target candidates."
    )
    parser.add_argument("--bank", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--batch_size", default=8192, type=int)
    return parser


def main() -> None:
    import torch

    args = get_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("requested GPU scoring but CUDA is unavailable")
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    archive = np.load(args.features)
    required = {"features", "candidate_type", "teacher_score", "group_index"}
    missing = sorted(required.difference(archive.files))
    if missing:
        raise KeyError(f"candidate archive missing arrays: {missing}")
    features = archive["features"].astype(np.float32, copy=False)
    candidate_type = archive["candidate_type"].astype(np.int64, copy=False)
    teacher_score = archive["teacher_score"].astype(np.float64, copy=False)
    group_index = archive["group_index"].astype(np.int64, copy=False)
    if not (
        len(features) == len(candidate_type) == len(teacher_score) == len(group_index)
    ):
        raise ValueError("candidate archive arrays are misaligned")

    payload = torch.load(args.bank, map_location="cpu")
    prototype_score, assignment = _score_bank(
        features, payload, device, args.batch_size
    )
    methods = {"raw_p2p": teacher_score, "frozen_prototype": prototype_score}
    separation_rows = []
    summaries = {}
    for method, scores in methods.items():
        summaries[method] = {}
        for negative_type, negative_name in ((1, "hard_negative"), (2, "random_negative")):
            result = summarize_separation(
                scores, candidate_type, group_index, 0, negative_type
            )
            summaries[method][negative_name] = result
            separation_rows.append(
                {"method": method, "negative_type": negative_name, **result}
            )

    prototype_hard = summaries["frozen_prototype"]["hard_negative"]
    prototype_random = summaries["frozen_prototype"]["random_negative"]
    teacher_hard = summaries["raw_p2p"]["hard_negative"]
    hard_auc = float(prototype_hard["macro_group_auc"])
    min_group_auc = float(prototype_hard["min_group_auc"])
    random_auc = float(prototype_random["macro_group_auc"])
    gain = hard_auc - float(teacher_hard["macro_group_auc"])
    decision = decide_transfer(hard_auc, min_group_auc, random_auc, gain)

    occupancy_rows = []
    prototype_count = int(sum(int(value) for value in payload["prototype_counts"]))
    for class_id, name in CANDIDATE_NAMES.items():
        selected = assignment[candidate_type == class_id]
        counts = np.bincount(selected, minlength=prototype_count)
        total = max(int(counts.sum()), 1)
        for prototype_id, count in enumerate(counts):
            occupancy_rows.append(
                {
                    "candidate_type": name,
                    "prototype_id": prototype_id,
                    "count": int(count),
                    "share": float(count / total),
                }
            )

    quantile_rows = []
    for method, scores in methods.items():
        quantile_rows.extend(_score_quantiles(method, scores, candidate_type))
    _write_csv(os.path.join(args.output_dir, "separation_metrics.csv"), separation_rows)
    _write_csv(os.path.join(args.output_dir, "score_quantiles.csv"), quantile_rows)
    _write_csv(os.path.join(args.output_dir, "prototype_occupancy.csv"), occupancy_rows)

    report = {
        "version": AUDIT_VERSION,
        "bank": os.path.abspath(args.bank),
        "features": os.path.abspath(args.features),
        "candidates": int(len(features)),
        "prototype_counts": [int(value) for value in payload["prototype_counts"]],
        "bank_id": str(payload.get("bank_id", "")),
        "source_reference": payload.get("metadata", {}).get("decision", {}),
        "target_separation": summaries,
        "hard_auc_gain_over_raw_p2p": gain,
        "decision": decision,
    }
    with open(
        os.path.join(args.output_dir, "target_transfer_audit.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.output_dir, "decision.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)

    print(
        "[Frozen-bank-transfer] "
        f"status={decision['status']}, hard_auc={hard_auc:.6f}, "
        f"hard_min={min_group_auc:.6f}, random_auc={random_auc:.6f}, "
        f"gain_over_raw={gain:+.6f}",
        flush=True,
    )
    print(f"[Frozen-bank-transfer] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
