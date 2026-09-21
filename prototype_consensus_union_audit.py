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


IMPLEMENTATION_VERSION = "prototype_consensus_union_audit_v1_20260908"


def parse_values(specification):
    text = str(specification).strip()
    if ":" not in text:
        return [float(item) for item in text.split(",") if item.strip()]
    start, stop, step = (float(item) for item in text.split(":"))
    if step <= 0 or stop < start:
        raise ValueError("range must be start:stop:positive_step")
    count = int(round((stop - start) / step))
    return [round(start + index * step, 10) for index in range(count + 1)]


def local_control_support(query_points, control_points, control_scores, radii):
    query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
    control_points = np.asarray(control_points, dtype=np.float32).reshape(-1, 2)
    control_scores = np.asarray(control_scores, dtype=np.float32).reshape(-1)
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    if len(control_points) != len(control_scores):
        raise ValueError("control points and scores must have the same length")
    support = np.full((len(query_points), len(radii)), -1.0, dtype=np.float32)
    if not len(query_points) or not len(control_points):
        return support

    for start in range(0, len(query_points), 64):
        stop = min(start + 64, len(query_points))
        difference = (
            query_points[start:stop, None, :] - control_points[None, :, :]
        )
        squared_distance = np.sum(difference * difference, axis=2)
        for radius_index, radius in enumerate(radii):
            inside = squared_distance <= float(radius * radius)
            masked = np.where(inside, control_scores[None, :], -1.0)
            support[start:stop, radius_index] = masked.max(axis=1)
    return support


def _validated_gt(control, prototype):
    control_gt = np.asarray(control["gt_points"], dtype=np.float64).reshape(-1, 2)
    prototype_gt = np.asarray(prototype["gt_points"], dtype=np.float64).reshape(-1, 2)
    if control_gt.shape != prototype_gt.shape or not np.allclose(
        control_gt, prototype_gt, atol=1e-5
    ):
        raise ValueError("paired records contain different ground-truth points")
    return control_gt


def evaluate_consensus_gated_union(
    control_records,
    prototype_records,
    control_threshold,
    prototype_thresholds,
    support_floors,
    agreement_radii,
    match_distance=15.0,
    dedup_interval=15.0,
    near_radius=30.0,
):
    if len(control_records) != len(prototype_records):
        raise ValueError("paired records must have the same image count")
    prototype_thresholds = sorted({float(value) for value in prototype_thresholds})
    support_floors = sorted({float(value) for value in support_floors})
    agreement_radii = sorted({float(value) for value in agreement_radii})
    if not prototype_thresholds or not support_floors or not agreement_radii:
        raise ValueError("all consensus sweep dimensions must be non-empty")

    configurations = [
        (prototype_threshold, support_floor, agreement_radius)
        for prototype_threshold in prototype_thresholds
        for support_floor in support_floors
        for agreement_radius in agreement_radii
    ]
    additions = {
        configuration: {
            "added_pred": 0,
            "added_tp": 0,
            "added_background_far_fp": 0,
            "support_score_sum": 0.0,
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
    minimum_prototype_threshold = min(prototype_thresholds)

    for control, prototype in zip(control_records, prototype_records):
        gt_points = _validated_gt(control, prototype)
        if not len(gt_points):
            continue
        baseline["images"] += 1
        baseline["gt"] += len(gt_points)

        control_points_all = np.asarray(
            control["points"], dtype=np.float64
        ).reshape(-1, 2)
        control_scores_all = np.asarray(
            control["scores"], dtype=np.float64
        ).reshape(-1)
        control_selected = control_scores_all >= float(control_threshold)
        control_points, control_scores = deduplicate_cell_candidates(
            control_points_all[control_selected],
            control_scores_all[control_selected],
            dedup_interval,
        )
        control_matched_predictions, control_matched_gt = (
            greedy_match_assignments(
                control_points, control_scores, gt_points, match_distance
            )
        )
        baseline["baseline_pred"] += len(control_points)
        baseline["baseline_tp"] += int(control_matched_gt.sum())
        for prediction_index, point in enumerate(control_points):
            if prediction_index in control_matched_predictions:
                continue
            if np.linalg.norm(gt_points - point, axis=1).min() > near_radius:
                baseline["baseline_background_far_fp"] += 1

        prototype_points_all = np.asarray(
            prototype["points"], dtype=np.float64
        ).reshape(-1, 2)
        prototype_scores_all = np.asarray(
            prototype["scores"], dtype=np.float64
        ).reshape(-1)
        above_minimum = prototype_scores_all >= minimum_prototype_threshold
        candidate_points, candidate_scores = deduplicate_cell_candidates(
            prototype_points_all[above_minimum],
            prototype_scores_all[above_minimum],
            dedup_interval,
        )
        if len(control_points) and len(candidate_points):
            distance_to_detection = np.linalg.norm(
                candidate_points[:, None, :] - control_points[None, :, :], axis=2
            )
            novel = distance_to_detection.min(axis=1) >= dedup_interval
            candidate_points = candidate_points[novel]
            candidate_scores = candidate_scores[novel]

        support = local_control_support(
            candidate_points,
            control_points_all,
            control_scores_all,
            agreement_radii,
        )
        remaining_gt = gt_points[~control_matched_gt]
        for prototype_threshold, support_floor, agreement_radius in configurations:
            radius_index = agreement_radii.index(agreement_radius)
            support_scores = support[:, radius_index]
            eligible = (
                (candidate_scores >= prototype_threshold)
                & (support_scores >= support_floor)
                & (support_scores < float(control_threshold))
            )
            selected_points = candidate_points[eligible]
            selected_scores = candidate_scores[eligible]
            selected_support = support_scores[eligible]
            matched_predictions, matched_gt = greedy_match_assignments(
                selected_points, selected_scores, remaining_gt, match_distance
            )
            aggregate = additions[
                (prototype_threshold, support_floor, agreement_radius)
            ]
            aggregate["added_pred"] += len(selected_points)
            aggregate["added_tp"] += int(matched_gt.sum())
            aggregate["support_score_sum"] += float(selected_support.sum())
            for prediction_index, point in enumerate(selected_points):
                if prediction_index in matched_predictions:
                    continue
                if np.linalg.norm(gt_points - point, axis=1).min() > near_radius:
                    aggregate["added_background_far_fp"] += 1

    baseline_precision = safe_div(
        baseline["baseline_tp"], baseline["baseline_pred"]
    )
    baseline_recall = safe_div(baseline["baseline_tp"], baseline["gt"])
    baseline_f1 = safe_div(
        2.0 * baseline_precision * baseline_recall,
        baseline_precision + baseline_recall,
    )
    rows = []
    for prototype_threshold, support_floor, agreement_radius in configurations:
        aggregate = additions[(prototype_threshold, support_floor, agreement_radius)]
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
                "support_floor": support_floor,
                "agreement_radius": agreement_radius,
                "control_threshold": float(control_threshold),
                **baseline,
                "baseline_fp": baseline["baseline_pred"] - baseline["baseline_tp"],
                "baseline_fn": baseline["gt"] - baseline["baseline_tp"],
                "baseline_precision": baseline_precision,
                "baseline_recall": baseline_recall,
                "baseline_f1": baseline_f1,
                "added_pred": added_pred,
                "added_tp": added_tp,
                "added_fp": added_pred - added_tp,
                "added_precision": safe_div(added_tp, added_pred),
                "added_background_far_fp": aggregate[
                    "added_background_far_fp"
                ],
                "mean_control_support": safe_div(
                    aggregate["support_score_sum"], added_pred
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


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Baseline-preserving prototype consensus union audit"
    )
    parser.add_argument("--pair_root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mean_std_path", required=True)
    parser.add_argument(
        "--prototype_subdir", default="prototype_local_soft_positive"
    )
    parser.add_argument("--prior_decision", default="")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--prototype_thresholds", default="0.56:0.72:0.02")
    parser.add_argument(
        "--support_floors", default="0.05,0.10,0.20,0.30,0.40,0.50"
    )
    parser.add_argument("--agreement_radii", default="3,5,8,12")
    parser.add_argument("--match_distance", default=15.0, type=float)
    parser.add_argument("--dedup_interval", default=15.0, type=float)
    parser.add_argument("--near_radius", default=30.0, type=float)
    parser.add_argument("--minimum_gain", default=0.002, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--output_dir", required=True)
    return parser


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
    )
    torch.cuda.empty_cache()

    rows = evaluate_consensus_gated_union(
        control_records,
        prototype_records,
        control_threshold,
        parse_values(args.prototype_thresholds),
        parse_values(args.support_floors),
        parse_values(args.agreement_radii),
        args.match_distance,
        args.dedup_interval,
        args.near_radius,
    )
    _write_csv(os.path.join(output_dir, "consensus_gated_union.csv"), rows)
    best = max(
        rows,
        key=lambda row: (
            float(row["final_f1"]),
            float(row["final_precision"]),
            float(row["added_precision"]),
        ),
    )
    gain = float(best["f1_gain"])
    decision = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "diagnostic_only": True,
        "gate_pass": gain >= args.minimum_gain,
        "minimum_f1_gain": float(args.minimum_gain),
        "best": best,
        "sweep_size": len(rows),
        "baseline_integrity": {
            "prior_best_f1": float(prior_decision["control_best"]["f1"]),
            "measured_f1": float(best["baseline_f1"]),
            "absolute_drift": abs(
                float(prior_decision["control_best"]["f1"])
                - float(best["baseline_f1"])
            ),
        },
        "next_action": (
            "build_frozen_baseline_parallel_prototype_rescue_branch"
            if gain >= args.minimum_gain
            else "prototype_consensus_still_lacks_precision"
        ),
        "warning": (
            "Source-test sweep is diagnostic only; final thresholds require "
            "source-train slide-group LOSO calibration."
        ),
    }
    with open(
        os.path.join(output_dir, "consensus_union_decision.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print(f"[Consensus-union] output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
