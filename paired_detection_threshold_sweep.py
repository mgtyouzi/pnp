import argparse
import csv
import json
import os
from argparse import Namespace

import numpy as np


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def parse_thresholds(specification):
    text = str(specification).strip()
    if ":" not in text:
        values = [float(item) for item in text.split(",") if item.strip()]
    else:
        start, stop, step = (float(item) for item in text.split(":"))
        if step <= 0 or stop < start:
            raise ValueError("threshold range must be start:stop:positive_step")
        count = int(round((stop - start) / step))
        values = [start + index * step for index in range(count + 1)]
    values.append(0.5)
    return sorted({round(value, 10) for value in values if 0.0 <= value <= 1.0})


def deduplicate_cell_candidates(points, scores, interval):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(points) != len(scores):
        raise ValueError("points and scores must have the same length")
    if not len(points):
        return points, scores

    fused = np.zeros(len(points), dtype=bool)
    selected = []
    for index in range(len(points)):
        if fused[index]:
            continue
        later = np.arange(index, len(points))
        distances = np.linalg.norm(points[later] - points[index], axis=1)
        group = later[distances < interval]
        fused[group] = True
        selected.append(int(group[np.argmax(scores[group])]))
    selected = np.asarray(selected, dtype=np.int64)
    return points[selected], scores[selected]


def greedy_match_count(pred_points, pred_scores, gt_points, threshold):
    matched_predictions, matched_gt = greedy_match_assignments(
        pred_points, pred_scores, gt_points, threshold
    )
    return int(matched_gt.sum()), matched_predictions


def greedy_match_assignments(pred_points, pred_scores, gt_points, threshold):
    pred_points = np.asarray(pred_points, dtype=np.float64).reshape(-1, 2)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 2)
    if not len(pred_points) or not len(gt_points):
        return set(), np.zeros(len(gt_points), dtype=bool)

    order = np.argsort(-pred_scores)
    difference = pred_points[order, None, :] - gt_points[None, :, :]
    distances = np.sqrt(np.sum(difference * difference, axis=2))
    unmatched_gt = np.ones(len(gt_points), dtype=bool)
    matched_gt = np.zeros(len(gt_points), dtype=bool)
    matched_predictions = set()
    for sorted_index, original_index in enumerate(order):
        available = np.where(unmatched_gt)[0]
        if not len(available):
            break
        nearest = int(available[np.argmin(distances[sorted_index, available])])
        if distances[sorted_index, nearest] <= threshold:
            unmatched_gt[nearest] = False
            matched_gt[nearest] = True
            matched_predictions.add(int(original_index))
    return matched_predictions, matched_gt


def _detected_gt_mask(record, threshold, match_distance, dedup_interval):
    gt_points = np.asarray(record["gt_points"], dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(record["scores"], dtype=np.float64).reshape(-1)
    selected = scores >= float(threshold)
    pred_points, pred_scores = deduplicate_cell_candidates(
        np.asarray(record["points"], dtype=np.float64).reshape(-1, 2)[selected],
        scores[selected],
        dedup_interval,
    )
    _, matched_gt = greedy_match_assignments(
        pred_points, pred_scores, gt_points, match_distance
    )
    return matched_gt


def compare_detection_transitions(
    control_records,
    prototype_records,
    control_threshold,
    prototype_threshold,
    match_distance=15.0,
    dedup_interval=15.0,
):
    if len(control_records) != len(prototype_records):
        raise ValueError("paired records must have the same image count")
    totals = {
        "gt": 0,
        "stable_tp": 0,
        "stable_fn": 0,
        "rescued_fn": 0,
        "newly_lost_tp": 0,
    }
    for control, prototype in zip(control_records, prototype_records):
        control_gt = np.asarray(control["gt_points"], dtype=np.float64).reshape(-1, 2)
        prototype_gt = np.asarray(prototype["gt_points"], dtype=np.float64).reshape(-1, 2)
        if control_gt.shape != prototype_gt.shape or not np.allclose(
            control_gt, prototype_gt, atol=1e-5
        ):
            raise ValueError("paired records contain different ground-truth points")
        control_detected = _detected_gt_mask(
            control, control_threshold, match_distance, dedup_interval
        )
        prototype_detected = _detected_gt_mask(
            prototype, prototype_threshold, match_distance, dedup_interval
        )
        totals["gt"] += int(len(control_gt))
        totals["stable_tp"] += int((control_detected & prototype_detected).sum())
        totals["stable_fn"] += int((~control_detected & ~prototype_detected).sum())
        totals["rescued_fn"] += int((~control_detected & prototype_detected).sum())
        totals["newly_lost_tp"] += int((control_detected & ~prototype_detected).sum())
    totals["net_rescue"] = totals["rescued_fn"] - totals["newly_lost_tp"]
    totals["control_threshold"] = float(control_threshold)
    totals["prototype_threshold"] = float(prototype_threshold)
    return totals


def evaluate_baseline_preserving_union(
    control_records,
    prototype_records,
    control_threshold,
    addition_thresholds,
    match_distance=15.0,
    dedup_interval=15.0,
    near_radius=30.0,
):
    if len(control_records) != len(prototype_records):
        raise ValueError("paired records must have the same image count")
    rows = []
    for addition_threshold in addition_thresholds:
        totals = {
            "images": 0,
            "gt": 0,
            "baseline_pred": 0,
            "baseline_tp": 0,
            "baseline_background_far_fp": 0,
            "added_pred": 0,
            "added_tp": 0,
            "added_background_far_fp": 0,
        }
        for control, prototype in zip(control_records, prototype_records):
            gt_points = np.asarray(
                control["gt_points"], dtype=np.float64
            ).reshape(-1, 2)
            prototype_gt = np.asarray(
                prototype["gt_points"], dtype=np.float64
            ).reshape(-1, 2)
            if gt_points.shape != prototype_gt.shape or not np.allclose(
                gt_points, prototype_gt, atol=1e-5
            ):
                raise ValueError(
                    "paired records contain different ground-truth points"
                )
            if not len(gt_points):
                continue
            totals["images"] += 1
            totals["gt"] += len(gt_points)

            control_scores = np.asarray(
                control["scores"], dtype=np.float64
            ).reshape(-1)
            control_selected = control_scores >= float(control_threshold)
            control_points, retained_control_scores = deduplicate_cell_candidates(
                np.asarray(control["points"], dtype=np.float64).reshape(-1, 2)[
                    control_selected
                ],
                control_scores[control_selected],
                dedup_interval,
            )
            control_matched_predictions, control_matched_gt = (
                greedy_match_assignments(
                    control_points,
                    retained_control_scores,
                    gt_points,
                    match_distance,
                )
            )
            totals["baseline_pred"] += len(control_points)
            totals["baseline_tp"] += int(control_matched_gt.sum())
            for prediction_index, point in enumerate(control_points):
                if prediction_index in control_matched_predictions:
                    continue
                if np.linalg.norm(gt_points - point, axis=1).min() > near_radius:
                    totals["baseline_background_far_fp"] += 1

            prototype_scores = np.asarray(
                prototype["scores"], dtype=np.float64
            ).reshape(-1)
            prototype_selected = prototype_scores >= float(addition_threshold)
            addition_points, addition_scores = deduplicate_cell_candidates(
                np.asarray(prototype["points"], dtype=np.float64).reshape(-1, 2)[
                    prototype_selected
                ],
                prototype_scores[prototype_selected],
                dedup_interval,
            )
            if len(control_points) and len(addition_points):
                distances_to_control = np.linalg.norm(
                    addition_points[:, None, :] - control_points[None, :, :],
                    axis=2,
                )
                novel = distances_to_control.min(axis=1) >= dedup_interval
                addition_points = addition_points[novel]
                addition_scores = addition_scores[novel]

            remaining_gt = gt_points[~control_matched_gt]
            addition_matched_predictions, addition_matched_gt = (
                greedy_match_assignments(
                    addition_points,
                    addition_scores,
                    remaining_gt,
                    match_distance,
                )
            )
            totals["added_pred"] += len(addition_points)
            totals["added_tp"] += int(addition_matched_gt.sum())
            for prediction_index, point in enumerate(addition_points):
                if prediction_index in addition_matched_predictions:
                    continue
                if np.linalg.norm(gt_points - point, axis=1).min() > near_radius:
                    totals["added_background_far_fp"] += 1

        baseline_fp = totals["baseline_pred"] - totals["baseline_tp"]
        baseline_precision = safe_div(
            totals["baseline_tp"], totals["baseline_pred"]
        )
        baseline_recall = safe_div(totals["baseline_tp"], totals["gt"])
        baseline_f1 = safe_div(
            2.0 * baseline_precision * baseline_recall,
            baseline_precision + baseline_recall,
        )
        added_fp = totals["added_pred"] - totals["added_tp"]
        final_pred = totals["baseline_pred"] + totals["added_pred"]
        final_tp = totals["baseline_tp"] + totals["added_tp"]
        final_precision = safe_div(final_tp, final_pred)
        final_recall = safe_div(final_tp, totals["gt"])
        final_f1 = safe_div(
            2.0 * final_precision * final_recall,
            final_precision + final_recall,
        )
        rows.append(
            {
                "threshold": float(addition_threshold),
                "control_threshold": float(control_threshold),
                **totals,
                "baseline_fp": baseline_fp,
                "baseline_fn": totals["gt"] - totals["baseline_tp"],
                "baseline_precision": baseline_precision,
                "baseline_recall": baseline_recall,
                "baseline_f1": baseline_f1,
                "added_fp": added_fp,
                "added_precision": safe_div(totals["added_tp"], totals["added_pred"]),
                "rescued_fn": totals["added_tp"],
                "newly_lost_tp": 0,
                "final_pred": final_pred,
                "final_tp": final_tp,
                "final_fp": final_pred - final_tp,
                "final_fn": totals["gt"] - final_tp,
                "final_precision": final_precision,
                "final_recall": final_recall,
                "final_f1": final_f1,
                "f1_gain": final_f1 - baseline_f1,
            }
        )
    return rows


def evaluate_thresholds(
    records,
    thresholds,
    match_distance=15.0,
    dedup_interval=15.0,
    near_radius=30.0,
):
    rows = []
    for threshold in thresholds:
        totals = {"images": 0, "gt": 0, "pred": 0, "tp": 0, "fp_background_far": 0}
        for record in records:
            gt_points = np.asarray(record["gt_points"], dtype=np.float64).reshape(-1, 2)
            if not len(gt_points):
                continue
            totals["images"] += 1
            totals["gt"] += len(gt_points)
            scores = np.asarray(record["scores"], dtype=np.float64)
            selected = scores >= float(threshold)
            pred_points, pred_scores = deduplicate_cell_candidates(
                np.asarray(record["points"])[selected],
                scores[selected],
                dedup_interval,
            )
            matched_count, matched_predictions = greedy_match_count(
                pred_points, pred_scores, gt_points, match_distance
            )
            totals["pred"] += len(pred_points)
            totals["tp"] += matched_count
            for pred_index, point in enumerate(pred_points):
                if pred_index in matched_predictions:
                    continue
                distance = np.sqrt(np.sum((gt_points - point) ** 2, axis=1)).min()
                if distance > near_radius:
                    totals["fp_background_far"] += 1

        precision = safe_div(totals["tp"], totals["pred"])
        recall = safe_div(totals["tp"], totals["gt"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        rows.append(
            {
                "threshold": float(threshold),
                **totals,
                "fp": totals["pred"] - totals["tp"],
                "fn": totals["gt"] - totals["tp"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pred_gt_ratio": safe_div(totals["pred"], totals["gt"]),
            }
        )
    return rows


def _row_at(rows, threshold):
    return min(rows, key=lambda row: abs(float(row["threshold"]) - threshold))


def _best_row(rows):
    return max(
        rows,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            -abs(float(row["threshold"]) - 0.5),
        ),
    )


def compare_sweeps(control_rows, prototype_rows, minimum_gain=0.002):
    control_default = _row_at(control_rows, 0.5)
    prototype_default = _row_at(prototype_rows, 0.5)
    control_best = _best_row(control_rows)
    prototype_best = _best_row(prototype_rows)
    best_gain = float(prototype_best["f1"] - control_best["f1"])
    default_gain = float(prototype_default["f1"] - control_default["f1"])

    if best_gain >= minimum_gain:
        status = (
            "ranking_gain_at_default_threshold"
            if default_gain >= minimum_gain
            else "calibration_shift_with_recoverable_gain"
        )
        next_action = "calibrate_threshold_on_source_train_loso_before_long_training"
    else:
        status = "no_recoverable_ranking_gain"
        next_action = "change_training_gradient_routing_before_long_training"

    return {
        "diagnostic_only": True,
        "status": status,
        "minimum_gain": float(minimum_gain),
        "control_default": control_default,
        "prototype_default": prototype_default,
        "control_best": control_best,
        "prototype_best": prototype_best,
        "control_best_threshold": float(control_best["threshold"]),
        "prototype_best_threshold": float(prototype_best["threshold"]),
        "default_f1_gain": default_gain,
        "best_f1_gain": best_gain,
        "next_action": next_action,
        "warning": (
            "Threshold selection on source test is diagnostic only; choose a deployable "
            "threshold from source-train LOSO calibration."
        ),
    }


def _checkpoint_state_dict(checkpoint):
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "net", "network"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def _model_args(checkpoint, dataset, mean_std_path, num_workers):
    import train_p2p as p2p

    defaults = vars(p2p.get_args_parser().parse_args([]))
    saved = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved, Namespace):
        saved = vars(saved)
    defaults.update(saved if isinstance(saved, dict) else {})
    defaults.update(
        {
            "dataset": dataset,
            "mean_std_path": mean_std_path,
            "test_mean_std_path": mean_std_path,
            "eval_split": "test",
            "num_workers": int(num_workers),
            "batch_size": 1,
            "distributed": False,
            "world_size": 1,
            "rank": 0,
            "gpu": 0,
            "local_rank": None,
            "output_dir": "",
            "proto_inference_fusion": 0,
        }
    )
    return Namespace(**defaults)


def _load_records(
    checkpoint_path,
    dataset_path,
    mean_std_path,
    device,
    num_workers,
    collect_proto_embeddings=False,
):
    import torch
    from torch.utils.data import DataLoader

    import train_p2p as p2p
    from dataset_zy_src import build_dataset
    from models.detr import build_model
    from utils import collate_fn_pad

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = _model_args(checkpoint, dataset_path, mean_std_path, num_workers)
    p2p.args = args
    dataset = build_dataset(args, "test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_pad,
    )
    model = build_model(args).to(device)
    state = _checkpoint_state_dict(checkpoint)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint load mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    model.eval()
    if collect_proto_embeddings:
        if model.prototype_head is None:
            raise RuntimeError("checkpoint has no prototype head for embedding audit")
        model.prototype_diagnostic_eval = True

    records = []
    with torch.no_grad():
        for index, (images, points, labels, lengths) in enumerate(loader):
            del labels
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            height, width = images.shape[-2:]
            predicted_points = outputs["pnt_coords"][0].detach().cpu().numpy()
            cell_scores = (
                torch.softmax(outputs["raw_cls_logits"][0], dim=-1)[:, 0]
                .detach()
                .cpu()
                .numpy()
            )
            inside = (
                (predicted_points[:, 0] >= 0)
                & (predicted_points[:, 0] < width)
                & (predicted_points[:, 1] >= 0)
                & (predicted_points[:, 1] < height)
            )
            gt_count = int(lengths[0])
            gt_points = points[0].reshape(-1, 2)[:gt_count].cpu().numpy()
            record = {
                "points": predicted_points[inside].astype(np.float32),
                "scores": cell_scores[inside].astype(np.float32),
                "gt_points": gt_points.astype(np.float32),
            }
            if collect_proto_embeddings:
                if "proto_embeddings" not in outputs:
                    raise RuntimeError(
                        "diagnostic prototype evaluation returned no embeddings"
                    )
                record["embeddings"] = (
                    outputs["proto_embeddings"][0]
                    .detach()
                    .cpu()
                    .numpy()[inside]
                    .astype(np.float32)
                )
            records.append(record)
            if (index + 1) % 50 == 0 or index + 1 == len(loader):
                print(
                    f"[Threshold-cache] checkpoint={os.path.basename(checkpoint_path)}, "
                    f"images={index + 1}/{len(loader)}",
                    flush=True,
                )
    stored = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    stored_values = stored.get("检测指标", stored.get("分类指标", []))
    stored_f1 = float(stored_values[2]) if len(stored_values) >= 3 else None
    return records, stored_f1, args


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Paired raw-P2P score threshold diagnosis without retraining"
    )
    parser.add_argument("--pair_root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mean_std_path", required=True)
    parser.add_argument("--prototype_subdir", default="positive_only_train_proto")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--thresholds", default="0.45:0.75:0.01")
    parser.add_argument("--match_distance", default=15.0, type=float)
    parser.add_argument("--dedup_interval", default=15.0, type=float)
    parser.add_argument("--near_radius", default=30.0, type=float)
    parser.add_argument("--minimum_gain", default=0.002, type=float)
    parser.add_argument("--max_default_f1_drift", default=0.002, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--output_dir", default="")
    return parser


def main():
    args = get_parser().parse_args()
    import torch

    pair_root = os.path.abspath(args.pair_root)
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(pair_root, "raw_p2p_threshold_sweep")
    )
    os.makedirs(output_dir, exist_ok=True)
    control_checkpoint = os.path.join(pair_root, "control_p2p", "recent_model.pth")
    prototype_checkpoint = os.path.join(
        pair_root, args.prototype_subdir, "recent_model.pth"
    )
    for path in (control_checkpoint, prototype_checkpoint, args.mean_std_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    device = torch.device(f"cuda:{args.gpu}")
    thresholds = parse_thresholds(args.thresholds)

    control_records, control_stored_f1, _ = _load_records(
        control_checkpoint, args.dataset, args.mean_std_path, device, args.num_workers
    )
    torch.cuda.empty_cache()
    prototype_records, prototype_stored_f1, _ = _load_records(
        prototype_checkpoint, args.dataset, args.mean_std_path, device, args.num_workers
    )
    torch.cuda.empty_cache()
    if len(control_records) != len(prototype_records):
        raise RuntimeError("paired checkpoints were evaluated on different image counts")

    control_rows = evaluate_thresholds(
        control_records,
        thresholds,
        args.match_distance,
        args.dedup_interval,
        args.near_radius,
    )
    prototype_rows = evaluate_thresholds(
        prototype_records,
        thresholds,
        args.match_distance,
        args.dedup_interval,
        args.near_radius,
    )
    _write_csv(os.path.join(output_dir, "control_threshold_sweep.csv"), control_rows)
    _write_csv(os.path.join(output_dir, "prototype_threshold_sweep.csv"), prototype_rows)

    decision = compare_sweeps(control_rows, prototype_rows, args.minimum_gain)
    decision["target_transitions"] = compare_detection_transitions(
        control_records,
        prototype_records,
        decision["control_best_threshold"],
        decision["prototype_best_threshold"],
        args.match_distance,
        args.dedup_interval,
    )
    union_rows = evaluate_baseline_preserving_union(
        control_records,
        prototype_records,
        decision["control_best_threshold"],
        sorted(set(thresholds + [1.0])),
        args.match_distance,
        args.dedup_interval,
        args.near_radius,
    )
    _write_csv(
        os.path.join(output_dir, "baseline_preserving_union.csv"), union_rows
    )
    best_union = max(
        union_rows,
        key=lambda row: (
            float(row["final_f1"]),
            float(row["final_precision"]),
            float(row["threshold"]),
        ),
    )
    union_gain = float(best_union["f1_gain"])
    decision["baseline_preserving_union"] = {
        "diagnostic_only": True,
        "gate_pass": union_gain >= args.minimum_gain,
        "minimum_f1_gain": float(args.minimum_gain),
        "best_addition_threshold": float(best_union["threshold"]),
        "best_f1_gain": union_gain,
        "baseline_f1": float(best_union["baseline_f1"]),
        "final_f1": float(best_union["final_f1"]),
        "added_pred": int(best_union["added_pred"]),
        "added_tp": int(best_union["added_tp"]),
        "added_fp": int(best_union["added_fp"]),
        "added_precision": float(best_union["added_precision"]),
        "added_background_far_fp": int(
            best_union["added_background_far_fp"]
        ),
        "rescued_fn": int(best_union["rescued_fn"]),
        "newly_lost_tp": 0,
        "next_action": (
            "build_frozen_baseline_parallel_prototype_rescue_branch"
            if union_gain >= args.minimum_gain
            else "prototype_additions_are_not_precise_enough_for_parallel_rescue"
        ),
    }
    integrity = {}
    for name, rows, stored_f1 in (
        ("control", control_rows, control_stored_f1),
        ("prototype", prototype_rows, prototype_stored_f1),
    ):
        measured_f1 = float(_row_at(rows, 0.5)["f1"])
        drift = abs(measured_f1 - stored_f1) if stored_f1 is not None else None
        integrity[name] = {
            "stored_f1": stored_f1,
            "measured_f1_at_0p5": measured_f1,
            "absolute_drift": drift,
            "pass": drift is None or drift <= args.max_default_f1_drift,
        }
    decision["integrity"] = integrity
    decision["integrity_pass"] = all(item["pass"] for item in integrity.values())
    if not decision["integrity_pass"]:
        decision["status"] = "invalid_metric_reproduction"
        decision["next_action"] = "inspect_metric_protocol_before_interpreting_sweep"

    decision_path = os.path.join(output_dir, "paired_threshold_decision.json")
    with open(decision_path, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"[Threshold-sweep] output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
