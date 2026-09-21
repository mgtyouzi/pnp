import argparse
import csv
import json
import os

import numpy as np

from paired_detection_threshold_sweep import (
    _load_records,
    deduplicate_cell_candidates,
    greedy_match_assignments,
    safe_div,
)
from prototype_consensus_union_audit import parse_values


IMPLEMENTATION_VERSION = "prototype_image_adaptive_union_v1_20260909"


def _normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if not positive_count or not negative_count:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and ordered_scores[end] == ordered_scores[start]:
            end += 1
        ordered_ranks[start:end] = 0.5 * ((start + 1) + end)
        start = end
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ordered_ranks
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def choose_next_action(gain, minimum_gain, ready_fraction, margin_auc):
    if gain >= minimum_gain and ready_fraction >= 0.9:
        return "integrate_frozen_baseline_image_adaptive_prototype_rescue"
    if ready_fraction < 0.9:
        return "insufficient_per_image_foreground_or_background_support"
    if not np.isfinite(margin_auc) or margin_auc <= 0.55:
        return "prototype_margin_does_not_separate_rescue_candidates"
    return "margin_separates_candidates_but_rescue_precision_is_insufficient"


def spherical_kmeans(values, num_prototypes, iterations=10):
    values = _normalize(values)
    if not len(values):
        return values
    count = min(int(num_prototypes), len(values))
    mean_direction = _normalize(values.mean(axis=0, keepdims=True))[0]
    selected = [int(np.argmax(values @ mean_direction))]
    while len(selected) < count:
        similarity = values @ values[selected].T
        nearest_similarity = similarity.max(axis=1)
        nearest_similarity[selected] = np.inf
        selected.append(int(np.argmin(nearest_similarity)))
    centers = values[selected].copy()
    for _ in range(int(iterations)):
        assignment = np.argmax(values @ centers.T, axis=1)
        updated = centers.copy()
        for prototype_index in range(count):
            members = values[assignment == prototype_index]
            if len(members):
                updated[prototype_index] = _normalize(
                    members.mean(axis=0, keepdims=True)
                )[0]
        if np.allclose(updated, centers, atol=1e-6):
            centers = updated
            break
        centers = updated
    return centers


def _deduplicated_indices(points, scores, interval, minimum_score):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    candidates = np.where(scores >= float(minimum_score))[0]
    fused = np.zeros(len(candidates), dtype=bool)
    selected = []
    for position in range(len(candidates)):
        if fused[position]:
            continue
        later = np.arange(position, len(candidates))
        distance = np.linalg.norm(
            points[candidates[later]] - points[candidates[position]], axis=1
        )
        group = later[distance < interval]
        fused[group] = True
        selected.append(int(candidates[group[np.argmax(scores[candidates[group]])]]))
    return np.asarray(selected, dtype=np.int64)


def _foreground_support_indices(
    control_points,
    control_scores,
    prototype_points,
    foreground_support_threshold,
    support_match_radius,
    maximum_supports,
):
    reliable = np.where(control_scores >= foreground_support_threshold)[0]
    reliable = reliable[np.argsort(-control_scores[reliable])]
    selected = []
    used = set()
    for control_index in reliable:
        if not len(prototype_points):
            break
        distance = np.linalg.norm(
            prototype_points - control_points[control_index], axis=1
        )
        prototype_index = int(np.argmin(distance))
        if distance[prototype_index] <= support_match_radius and prototype_index not in used:
            selected.append(prototype_index)
            used.add(prototype_index)
        if len(selected) >= maximum_supports:
            break
    return np.asarray(selected, dtype=np.int64)


def _bounded_background_indices(
    points,
    scores,
    threshold,
    maximum_supports,
    exclusion_points=None,
    exclusion_radius=30.0,
):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    exclusion_points = np.asarray(
        exclusion_points if exclusion_points is not None else [],
        dtype=np.float64,
    ).reshape(-1, 2)
    eligible = np.where(scores <= threshold)[0]
    if len(eligible) and len(exclusion_points):
        distance = np.linalg.norm(
            points[eligible, None, :] - exclusion_points[None, :, :], axis=2
        )
        eligible = eligible[distance.min(axis=1) > exclusion_radius]
    if len(eligible) <= maximum_supports:
        return eligible
    ordered = eligible[np.argsort(scores[eligible])]
    positions = np.linspace(0, len(ordered) - 1, maximum_supports).astype(np.int64)
    return ordered[positions]


def _validated_gt(control, prototype):
    control_gt = np.asarray(control["gt_points"], dtype=np.float64).reshape(-1, 2)
    prototype_gt = np.asarray(prototype["gt_points"], dtype=np.float64).reshape(-1, 2)
    if control_gt.shape != prototype_gt.shape or not np.allclose(
        control_gt, prototype_gt, atol=1e-5
    ):
        raise ValueError("paired records contain different ground-truth points")
    return control_gt


def evaluate_image_adaptive_prototype_union(
    control_records,
    prototype_records,
    control_threshold,
    prototype_thresholds,
    margin_thresholds,
    foreground_support_threshold=0.8,
    background_support_threshold=0.05,
    support_match_radius=8.0,
    num_foreground_prototypes=4,
    num_background_prototypes=4,
    max_foreground_supports=64,
    max_background_supports=256,
    kmeans_iterations=10,
    match_distance=15.0,
    dedup_interval=15.0,
    near_radius=30.0,
):
    if len(control_records) != len(prototype_records):
        raise ValueError("paired records must have the same image count")
    prototype_thresholds = sorted({float(value) for value in prototype_thresholds})
    margin_thresholds = sorted({float(value) for value in margin_thresholds})
    configurations = [
        (prototype_threshold, margin_threshold)
        for prototype_threshold in prototype_thresholds
        for margin_threshold in margin_thresholds
    ]
    if not configurations:
        raise ValueError("prototype and margin threshold grids must be non-empty")
    additions = {
        configuration: {
            "added_pred": 0,
            "added_tp": 0,
            "added_background_far_fp": 0,
            "added_margin_sum": 0.0,
        }
        for configuration in configurations
    }
    baseline = {
        "images": 0,
        "gt": 0,
        "baseline_pred": 0,
        "baseline_tp": 0,
        "baseline_background_far_fp": 0,
    }
    support_totals = {
        "images_with_ready_local_prototypes": 0,
        "images_with_insufficient_foreground_support": 0,
        "images_with_insufficient_background_support": 0,
        "foreground_support_count": 0,
        "background_support_count": 0,
        "foreground_prototype_count": 0,
        "background_prototype_count": 0,
    }
    candidate_margin_labels = []
    candidate_margin_values = []
    minimum_prototype_threshold = min(prototype_thresholds)

    for control, prototype in zip(control_records, prototype_records):
        gt_points = _validated_gt(control, prototype)
        if not len(gt_points):
            continue
        baseline["images"] += 1
        baseline["gt"] += len(gt_points)
        control_points_all = np.asarray(control["points"], dtype=np.float64).reshape(-1, 2)
        control_scores_all = np.asarray(control["scores"], dtype=np.float64).reshape(-1)
        control_selected = control_scores_all >= float(control_threshold)
        control_points, control_scores = deduplicate_cell_candidates(
            control_points_all[control_selected],
            control_scores_all[control_selected],
            dedup_interval,
        )
        control_matched_predictions, control_matched_gt = greedy_match_assignments(
            control_points, control_scores, gt_points, match_distance
        )
        baseline["baseline_pred"] += len(control_points)
        baseline["baseline_tp"] += int(control_matched_gt.sum())
        for prediction_index, point in enumerate(control_points):
            if prediction_index in control_matched_predictions:
                continue
            if np.linalg.norm(gt_points - point, axis=1).min() > near_radius:
                baseline["baseline_background_far_fp"] += 1

        prototype_points = np.asarray(prototype["points"], dtype=np.float64).reshape(-1, 2)
        prototype_scores = np.asarray(prototype["scores"], dtype=np.float64).reshape(-1)
        embeddings = _normalize(prototype["embeddings"])
        if len(prototype_points) != len(embeddings):
            raise ValueError("prototype points and embeddings must have the same length")

        foreground_indices = _foreground_support_indices(
            control_points,
            control_scores,
            prototype_points,
            foreground_support_threshold,
            support_match_radius,
            max_foreground_supports,
        )
        background_indices = _bounded_background_indices(
            prototype_points,
            prototype_scores,
            background_support_threshold,
            max_background_supports,
            exclusion_points=control_points,
            exclusion_radius=near_radius,
        )
        support_totals["foreground_support_count"] += len(foreground_indices)
        support_totals["background_support_count"] += len(background_indices)
        foreground_ready = len(foreground_indices) >= num_foreground_prototypes
        background_ready = len(background_indices) >= num_background_prototypes
        if not foreground_ready:
            support_totals["images_with_insufficient_foreground_support"] += 1
        if not background_ready:
            support_totals["images_with_insufficient_background_support"] += 1
        if not foreground_ready or not background_ready:
            continue
        foreground_prototypes = spherical_kmeans(
            embeddings[foreground_indices],
            num_foreground_prototypes,
            kmeans_iterations,
        )
        background_prototypes = spherical_kmeans(
            embeddings[background_indices],
            num_background_prototypes,
            kmeans_iterations,
        )
        support_totals["images_with_ready_local_prototypes"] += 1
        support_totals["foreground_prototype_count"] += len(foreground_prototypes)
        support_totals["background_prototype_count"] += len(background_prototypes)

        candidate_indices = _deduplicated_indices(
            prototype_points,
            prototype_scores,
            dedup_interval,
            minimum_prototype_threshold,
        )
        if len(control_points) and len(candidate_indices):
            distance_to_control = np.linalg.norm(
                prototype_points[candidate_indices, None, :]
                - control_points[None, :, :],
                axis=2,
            )
            candidate_indices = candidate_indices[
                distance_to_control.min(axis=1) >= dedup_interval
            ]
        candidate_embeddings = embeddings[candidate_indices]
        if len(candidate_indices):
            foreground_similarity = (
                candidate_embeddings @ foreground_prototypes.T
            ).max(axis=1)
            background_similarity = (
                candidate_embeddings @ background_prototypes.T
            ).max(axis=1)
            margins = foreground_similarity - background_similarity
        else:
            margins = np.empty(0, dtype=np.float32)
        remaining_gt = gt_points[~control_matched_gt]
        if len(candidate_indices):
            if len(remaining_gt):
                candidate_distance = np.linalg.norm(
                    prototype_points[candidate_indices, None, :]
                    - remaining_gt[None, :, :],
                    axis=2,
                )
                candidate_labels = candidate_distance.min(axis=1) <= match_distance
            else:
                candidate_labels = np.zeros(len(candidate_indices), dtype=bool)
            candidate_margin_labels.extend(candidate_labels.tolist())
            candidate_margin_values.extend(margins.tolist())

        for prototype_threshold, margin_threshold in configurations:
            eligible = (
                (prototype_scores[candidate_indices] >= prototype_threshold)
                & (margins >= margin_threshold)
            )
            selected_indices = candidate_indices[eligible]
            selected_points = prototype_points[selected_indices]
            selected_scores = prototype_scores[selected_indices]
            selected_margins = margins[eligible]
            matched_predictions, matched_gt = greedy_match_assignments(
                selected_points, selected_scores, remaining_gt, match_distance
            )
            aggregate = additions[(prototype_threshold, margin_threshold)]
            aggregate["added_pred"] += len(selected_points)
            aggregate["added_tp"] += int(matched_gt.sum())
            aggregate["added_margin_sum"] += float(selected_margins.sum())
            for prediction_index, point in enumerate(selected_points):
                if prediction_index in matched_predictions:
                    continue
                if np.linalg.norm(gt_points - point, axis=1).min() > near_radius:
                    aggregate["added_background_far_fp"] += 1

    margin_labels = np.asarray(candidate_margin_labels, dtype=bool)
    margin_values = np.asarray(candidate_margin_values, dtype=np.float64)
    positive_margins = margin_values[margin_labels]
    background_margins = margin_values[~margin_labels]
    positive_margin_mean = (
        float(positive_margins.mean()) if len(positive_margins) else float("nan")
    )
    background_margin_mean = (
        float(background_margins.mean()) if len(background_margins) else float("nan")
    )
    margin_diagnostics = {
        "candidate_margin_auc": binary_auc(margin_labels, margin_values),
        "candidate_positive_margin_mean": positive_margin_mean,
        "candidate_background_margin_mean": background_margin_mean,
        "candidate_margin_gap": positive_margin_mean - background_margin_mean,
        "candidate_margin_count": len(margin_values),
        "candidate_positive_count": int(margin_labels.sum()),
    }

    baseline_precision = safe_div(baseline["baseline_tp"], baseline["baseline_pred"])
    baseline_recall = safe_div(baseline["baseline_tp"], baseline["gt"])
    baseline_f1 = safe_div(
        2.0 * baseline_precision * baseline_recall,
        baseline_precision + baseline_recall,
    )
    rows = []
    for prototype_threshold, margin_threshold in configurations:
        aggregate = additions[(prototype_threshold, margin_threshold)]
        added_pred = aggregate["added_pred"]
        added_tp = aggregate["added_tp"]
        final_pred = baseline["baseline_pred"] + added_pred
        final_tp = baseline["baseline_tp"] + added_tp
        final_precision = safe_div(final_tp, final_pred)
        final_recall = safe_div(final_tp, baseline["gt"])
        final_f1 = safe_div(
            2.0 * final_precision * final_recall,
            final_precision + final_recall,
        )
        rows.append(
            {
                "prototype_threshold": prototype_threshold,
                "margin_threshold": margin_threshold,
                "control_threshold": float(control_threshold),
                **baseline,
                **support_totals,
                **margin_diagnostics,
                "baseline_fp": baseline["baseline_pred"] - baseline["baseline_tp"],
                "baseline_fn": baseline["gt"] - baseline["baseline_tp"],
                "baseline_precision": baseline_precision,
                "baseline_recall": baseline_recall,
                "baseline_f1": baseline_f1,
                "added_pred": added_pred,
                "added_tp": added_tp,
                "added_fp": added_pred - added_tp,
                "added_precision": safe_div(added_tp, added_pred),
                "added_background_far_fp": aggregate["added_background_far_fp"],
                "mean_added_margin": safe_div(
                    aggregate["added_margin_sum"], added_pred
                ),
                "rescued_fn": added_tp,
                "newly_lost_tp": 0,
                "final_pred": final_pred,
                "final_tp": final_tp,
                "final_fp": final_pred - final_tp,
                "final_fn": baseline["gt"] - final_tp,
                "final_precision": final_precision,
                "final_recall": final_recall,
                "final_f1": final_f1,
                "f1_gain": final_f1 - baseline_f1,
                "required_marginal_precision": baseline_f1 / 2.0,
            }
        )
    return rows


def get_parser():
    parser = argparse.ArgumentParser(description="Image-adaptive prototype union audit")
    parser.add_argument("--pair_root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mean_std_path", required=True)
    parser.add_argument("--prototype_subdir", default="prototype_local_soft_positive")
    parser.add_argument("--prior_decision", default="")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--prototype_thresholds", default="0.56:0.76:0.02")
    parser.add_argument("--margin_thresholds", default="-0.20:0.40:0.025")
    parser.add_argument("--foreground_support_threshold", default=0.8, type=float)
    parser.add_argument("--background_support_threshold", default=0.05, type=float)
    parser.add_argument("--support_match_radius", default=8.0, type=float)
    parser.add_argument("--num_foreground_prototypes", default=4, type=int)
    parser.add_argument("--num_background_prototypes", default=4, type=int)
    parser.add_argument("--max_foreground_supports", default=64, type=int)
    parser.add_argument("--max_background_supports", default=256, type=int)
    parser.add_argument("--kmeans_iterations", default=10, type=int)
    parser.add_argument("--match_distance", default=15.0, type=float)
    parser.add_argument("--dedup_interval", default=15.0, type=float)
    parser.add_argument("--near_radius", default=30.0, type=float)
    parser.add_argument("--minimum_gain", default=0.002, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--output_dir", required=True)
    return parser


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = get_parser().parse_args()
    import torch

    pair_root = os.path.abspath(args.pair_root)
    output_dir = os.path.abspath(args.output_dir)
    prior_decision_path = os.path.abspath(
        args.prior_decision
        or os.path.join(
            pair_root, "raw_p2p_threshold_sweep", "paired_threshold_decision.json"
        )
    )
    control_checkpoint = os.path.join(pair_root, "control_p2p", "recent_model.pth")
    prototype_checkpoint = os.path.join(
        pair_root, args.prototype_subdir, "recent_model.pth"
    )
    for path in (
        control_checkpoint,
        prototype_checkpoint,
        args.mean_std_path,
        prior_decision_path,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    os.makedirs(output_dir, exist_ok=True)
    with open(prior_decision_path, encoding="utf-8") as handle:
        prior_decision = json.load(handle)
    control_threshold = float(prior_decision["control_best_threshold"])
    device = torch.device(f"cuda:{args.gpu}")

    control_records, _, _ = _load_records(
        control_checkpoint,
        args.dataset,
        args.mean_std_path,
        device,
        args.num_workers,
    )
    torch.cuda.empty_cache()
    prototype_records, _, _ = _load_records(
        prototype_checkpoint,
        args.dataset,
        args.mean_std_path,
        device,
        args.num_workers,
        collect_proto_embeddings=True,
    )
    torch.cuda.empty_cache()
    rows = evaluate_image_adaptive_prototype_union(
        control_records,
        prototype_records,
        control_threshold,
        parse_values(args.prototype_thresholds),
        parse_values(args.margin_thresholds),
        args.foreground_support_threshold,
        args.background_support_threshold,
        args.support_match_radius,
        args.num_foreground_prototypes,
        args.num_background_prototypes,
        args.max_foreground_supports,
        args.max_background_supports,
        args.kmeans_iterations,
        args.match_distance,
        args.dedup_interval,
        args.near_radius,
    )
    _write_csv(os.path.join(output_dir, "image_adaptive_union.csv"), rows)
    best = max(
        rows,
        key=lambda row: (
            float(row["final_f1"]),
            float(row["final_precision"]),
            float(row["added_precision"]),
        ),
    )
    gain = float(best["f1_gain"])
    prior_f1 = float(prior_decision["control_best"]["f1"])
    ready_fraction = safe_div(
        best["images_with_ready_local_prototypes"], best["images"]
    )
    decision = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "diagnostic_only": True,
        "gate_pass": gain >= args.minimum_gain and ready_fraction >= 0.9,
        "minimum_f1_gain": float(args.minimum_gain),
        "minimum_ready_image_fraction": 0.9,
        "ready_image_fraction": ready_fraction,
        "best": best,
        "sweep_size": len(rows),
        "baseline_integrity": {
            "prior_best_f1": prior_f1,
            "measured_f1": float(best["baseline_f1"]),
            "absolute_drift": abs(prior_f1 - float(best["baseline_f1"])),
        },
        "representation_diagnostic": {
            "candidate_margin_auc": float(best["candidate_margin_auc"]),
            "candidate_positive_margin_mean": float(
                best["candidate_positive_margin_mean"]
            ),
            "candidate_background_margin_mean": float(
                best["candidate_background_margin_mean"]
            ),
            "candidate_margin_gap": float(best["candidate_margin_gap"]),
            "candidate_margin_count": int(best["candidate_margin_count"]),
            "candidate_positive_count": int(best["candidate_positive_count"]),
        },
        "next_action": choose_next_action(
            gain,
            args.minimum_gain,
            ready_fraction,
            float(best["candidate_margin_auc"]),
        ),
        "warning": (
            "Source-test sweep is diagnostic only; final thresholds require "
            "source-train slide-group LOSO calibration."
        ),
    }
    with open(
        os.path.join(output_dir, "image_adaptive_union_decision.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print(f"[Image-adaptive-union] output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
