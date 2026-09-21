import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import train_p2p as p2p
from dataset_zy_src import build_dataset
from models.detr import build_model
from utils import collate_fn_pad, cleanup, deduplicate_scores, get_rank, init_distributed_mode


def _as_float(value):
    return float(value) if value is not None else 0.0


def _safe_div(num, den):
    return float(num) / float(den) if float(den) > 0 else 0.0


def _strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint):
    if hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, dict):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict", "model_state_dict", "net", "network"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _resize_eval_only_prototype_buffers(model, state_dict):
    """Match training-only prototype reservoirs to checkpoint shapes for eval."""
    prototype_head = getattr(model, "prototype_head", None)
    if prototype_head is None:
        return

    buffer_size_attributes = {
        "foreground_queue": "foreground_queue_size",
        "background_queue": "background_queue_size",
        "foreground_queue_priorities": None,
        "background_queue_priorities": None,
    }
    for buffer_name, size_attribute in buffer_size_attributes.items():
        state_key = f"prototype_head.{buffer_name}"
        checkpoint_buffer = state_dict.get(state_key)
        current_buffer = getattr(prototype_head, buffer_name, None)
        if checkpoint_buffer is None or current_buffer is None:
            continue
        if tuple(current_buffer.shape) == tuple(checkpoint_buffer.shape):
            continue
        setattr(
            prototype_head,
            buffer_name,
            current_buffer.new_empty(tuple(checkpoint_buffer.shape)),
        )
        if size_attribute is not None:
            setattr(prototype_head, size_attribute, int(checkpoint_buffer.shape[0]))
        print(
            f"[Checkpoint-eval-buffer] {state_key}: "
            f"{tuple(current_buffer.shape)} -> {tuple(checkpoint_buffer.shape)}"
        )


def load_checkpoint_for_eval(model, checkpoint_path, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    _resize_eval_only_prototype_buffers(model, state_dict)
    load_result = model.load_state_dict(state_dict, strict=strict)
    return checkpoint, load_result


def distance_matrix(points_a, points_b):
    if len(points_a) == 0 or len(points_b) == 0:
        return np.zeros((len(points_a), len(points_b)), dtype=np.float64)
    diff = points_a[:, None, :].astype(np.float64) - points_b[None, :, :].astype(np.float64)
    return np.sqrt(np.sum(diff * diff, axis=2))


def greedy_match(pred_points, pred_scores, gt_points, thr):
    if len(pred_points) == 0 or len(gt_points) == 0:
        return [], set(), set()

    sorted_indices = np.argsort(-pred_scores)
    sorted_pred_points = pred_points[sorted_indices]
    dis = distance_matrix(sorted_pred_points, gt_points)
    unmatched_gt = np.ones(len(gt_points), dtype=bool)
    matches = []

    for sorted_i, pred_i in enumerate(sorted_indices):
        if not np.any(unmatched_gt):
            break
        candidate_gt_indices = np.where(unmatched_gt)[0]
        nearest_local = np.argmin(dis[sorted_i, candidate_gt_indices])
        gt_i = int(candidate_gt_indices[nearest_local])
        dist = float(dis[sorted_i, gt_i])
        if dist <= thr:
            matches.append((int(pred_i), gt_i, dist))
            unmatched_gt[gt_i] = False

    matched_pred = {item[0] for item in matches}
    matched_gt = {item[1] for item in matches}
    return matches, matched_pred, matched_gt


@torch.no_grad()
def predict_detailed(model, images, num_classes, dedup_interval):
    h, w = images.shape[-2:]
    outputs = model(images)
    raw_points = outputs["pnt_coords"][0].detach().cpu().numpy()
    raw_anchors = outputs.get("anchor_points", outputs["pnt_coords"])[0].detach().cpu().numpy()
    raw_scores = torch.softmax(outputs["cls_logits"][0], dim=-1).detach().cpu().numpy()

    cross_border = (
        (raw_points[:, 0] < 0) | (raw_points[:, 0] >= w) |
        (raw_points[:, 1] < 0) | (raw_points[:, 1] >= h)
    )
    points_in = raw_points[~cross_border]
    anchors_in = raw_anchors[~cross_border]
    scores_in = raw_scores[~cross_border]
    classes_in = np.argmax(scores_in, axis=-1) if len(scores_in) else np.array([], dtype=np.int64)
    reserved = classes_in < num_classes

    if len(scores_in):
        cell_scores_in = scores_in[:, :num_classes].max(axis=1)
        bg_scores_in = scores_in[:, -1]
    else:
        cell_scores_in = np.array([], dtype=np.float64)
        bg_scores_in = np.array([], dtype=np.float64)

    if np.any(reserved):
        pred_points, pred_classes, pred_scores = deduplicate_scores(
            points_in[reserved], scores_in[reserved], dedup_interval)
    else:
        pred_points = np.zeros((0, 2), dtype=np.float64)
        pred_classes = np.array([], dtype=np.float64)
        pred_scores = np.zeros((0, num_classes + 1), dtype=np.float64)

    if len(pred_scores):
        pred_cell_scores = pred_scores[:, :num_classes].max(axis=1)
    else:
        pred_cell_scores = np.array([], dtype=np.float64)

    attn = {}
    for key in ("reg_attn", "cls_attn"):
        if key in outputs:
            value = outputs[key][0].detach().cpu().numpy()
            value = np.squeeze(value)
            if value.ndim == 2:
                attn[key] = value.mean(axis=0)

    debug = {
        "raw_points": len(raw_points),
        "cross_border": int(cross_border.sum()),
        "in_border": len(points_in),
        "reserved_before_dedup": int(reserved.sum()),
        "reserved_after_dedup": len(pred_points),
        "points_in": points_in,
        "anchors_in": anchors_in,
        "scores_in": scores_in,
        "classes_in": classes_in,
        "reserved": reserved,
        "cell_scores_in": cell_scores_in,
        "bg_scores_in": bg_scores_in,
        "attn": attn,
    }
    if "proto_logits" in outputs:
        proto_logits = outputs["proto_logits"][0].detach().cpu().numpy()[~cross_border]
        debug["proto_margin_in"] = proto_logits[:, 0] - proto_logits[:, 1]
        if "proto_distances" in outputs:
            debug["proto_distances_in"] = (
                outputs["proto_distances"][0].detach().cpu().numpy()[~cross_border]
            )
        else:
            debug["proto_distances_in"] = None
    else:
        debug["proto_margin_in"] = None
        debug["proto_distances_in"] = None
    return pred_points, pred_classes.astype(np.int64), pred_scores, pred_cell_scores, debug


def classify_fn(gt_i, gt_points, pred_points, raw_debug, matched_gt, match_dis, near_radius, num_classes):
    gt = gt_points[gt_i]
    nearest_reserved_dist = None
    if len(pred_points):
        d_reserved = distance_matrix(pred_points, gt[None, :])[:, 0]
        nearest_reserved_dist = float(d_reserved.min())

    raw_points = raw_debug["points_in"]
    raw_classes = raw_debug["classes_in"]
    cell_scores = raw_debug["cell_scores_in"]
    bg_scores = raw_debug["bg_scores_in"]

    nearest_raw_dist = None
    nearest_raw_cell_score = 0.0
    nearest_raw_bg_score = 0.0
    nearest_raw_is_bg = 1
    if len(raw_points):
        d_raw = distance_matrix(raw_points, gt[None, :])[:, 0]
        raw_i = int(np.argmin(d_raw))
        nearest_raw_dist = float(d_raw[raw_i])
        nearest_raw_cell_score = float(cell_scores[raw_i])
        nearest_raw_bg_score = float(bg_scores[raw_i])
        nearest_raw_is_bg = int(raw_classes[raw_i] >= num_classes)

    if nearest_reserved_dist is not None and match_dis < nearest_reserved_dist <= near_radius:
        return "localization_shift", nearest_reserved_dist, nearest_raw_dist, nearest_raw_cell_score, nearest_raw_bg_score
    if nearest_raw_dist is not None and nearest_raw_dist <= near_radius:
        if nearest_raw_is_bg:
            return "low_score_or_background", nearest_reserved_dist, nearest_raw_dist, nearest_raw_cell_score, nearest_raw_bg_score
        return "near_candidate_unmatched", nearest_reserved_dist, nearest_raw_dist, nearest_raw_cell_score, nearest_raw_bg_score
    return "no_candidate", nearest_reserved_dist, nearest_raw_dist, nearest_raw_cell_score, nearest_raw_bg_score


def classify_fp(pred_i, pred_points, gt_points, matched_gt, match_dis, near_radius):
    if len(gt_points) == 0:
        return "empty_image_fp", None, -1

    d = distance_matrix(pred_points[pred_i][None, :], gt_points)[0]
    nearest_gt = int(np.argmin(d))
    nearest_dist = float(d[nearest_gt])
    if nearest_dist <= match_dis and nearest_gt in matched_gt:
        return "duplicate", nearest_dist, nearest_gt
    if nearest_dist <= near_radius:
        return "near_miss", nearest_dist, nearest_gt
    return "background_far", nearest_dist, nearest_gt


def write_header(path, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def append_rows(path, rows):
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def run_diagnosis(args):
    rank = args.gpu if getattr(args, "distributed", False) else 0
    os.makedirs(args.output_dir, exist_ok=True)
    p2p.args = args
    include_empty_gt = getattr(args, "include_empty_gt_in_eval", False)

    dataset = build_dataset(args, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn_pad)
    evaluator = p2p.Evaluator(loader)

    model = build_model(args).cuda(rank)
    checkpoint, load_result = load_checkpoint_for_eval(model, args.checkpoint, strict=args.strict_load)
    model.eval()

    process_rank = getattr(args, "rank", rank)
    world_size = getattr(args, "world_size", 1)
    num_levels = getattr(model, "num_levels", 0)

    sample_path = os.path.join(args.output_dir, f"sample_stats_rank{process_rank}.csv")
    score_path = os.path.join(args.output_dir, f"score_stats_rank{process_rank}.csv")
    fn_path = os.path.join(args.output_dir, f"fn_stats_rank{process_rank}.csv")
    fp_path = os.path.join(args.output_dir, f"fp_stats_rank{process_rank}.csv")

    write_header(sample_path, [
        "case", "rank", "sample_index", "image_path", "gt", "pred", "tp", "fp", "fn",
        "precision", "recall", "f1", "empty_gt", "metric_image", "raw_points", "cross_border",
        "reserved_before_dedup", "reserved_after_dedup", "tp_distance_mean",
        "tp_l1_mean", "fn_no_candidate", "fn_low_score_or_background",
        "fn_localization_shift", "fn_near_candidate_unmatched", "fp_empty_image",
        "fp_duplicate", "fp_near_miss", "fp_background_far"
    ])
    write_header(score_path, [
        "case", "rank", "sample_index", "image_path", "kind", "score", "distance",
        "gt_index", "pred_index"
    ])
    write_header(fn_path, [
        "case", "rank", "sample_index", "image_path", "fn_type", "gt_index",
        "nearest_reserved_dist", "nearest_raw_dist", "nearest_raw_cell_score",
        "nearest_raw_bg_score"
    ])
    write_header(fp_path, [
        "case", "rank", "sample_index", "image_path", "fp_type", "pred_index",
        "score", "nearest_gt_dist", "nearest_gt_index"
    ])

    total = {
        "images": 0.0, "total_eval_images": 0.0, "skipped_empty_images": 0.0,
        "empty_images": 0.0, "gt": 0.0, "pred": 0.0, "tp": 0.0,
        "raw_points": 0.0, "cross_border": 0.0, "reserved_before": 0.0,
        "reserved_after": 0.0, "mse_sum": 0.0, "mae_sum": 0.0,
        "tp_distance_sum": 0.0, "tp_distance_sq_sum": 0.0,
        "fn_no_candidate": 0.0, "fn_low_score_or_background": 0.0,
        "fn_localization_shift": 0.0, "fn_near_candidate_unmatched": 0.0,
        "fp_empty_image": 0.0, "fp_duplicate": 0.0, "fp_near_miss": 0.0,
        "fp_background_far": 0.0,
        "proto_near_count": 0.0, "proto_far_count": 0.0,
        "proto_near_margin_sum": 0.0, "proto_far_margin_sum": 0.0,
        "proto_near_fg_win": 0.0, "proto_far_fg_win": 0.0,
        "proto_near_fg_distance_sum": 0.0, "proto_near_bg_distance_sum": 0.0,
        "proto_far_fg_distance_sum": 0.0, "proto_far_bg_distance_sum": 0.0,
    }
    reg_attn_sum = np.zeros(num_levels, dtype=np.float64)
    cls_attn_sum = np.zeros(num_levels, dtype=np.float64)
    attn_count = 0.0

    for i, (images, points, labels, lengths) in enumerate(loader):
        if getattr(args, "distributed", False) and i % world_size != process_rank:
            continue

        images = images.cuda(rank, non_blocking=True)
        pred_points, pred_classes, pred_scores, pred_cell_scores, raw_debug = predict_detailed(
            model, images, args.num_classes, args.dedup_interval)
        gt_points = np.zeros((0, 2), dtype=np.float64)
        for c in range(args.num_classes):
            gt_points = np.concatenate([gt_points, evaluator.gds[i][c]], axis=0)

        global_scores = pred_scores[:, :args.num_classes].sum(axis=1) if len(pred_scores) else np.array([])
        matches, matched_pred, matched_gt = greedy_match(pred_points, global_scores, gt_points, args.match_dis)
        unmatched_pred = [idx for idx in range(len(pred_points)) if idx not in matched_pred]
        unmatched_gt = [idx for idx in range(len(gt_points)) if idx not in matched_gt]

        tp = len(matches)
        pred_count = len(pred_points)
        gt_count = len(gt_points)
        fp = pred_count - tp
        fn = gt_count - tp

        total["total_eval_images"] += 1
        total["empty_images"] += 1 if gt_count == 0 else 0

        image_path = dataset.files[i] if hasattr(dataset, "files") else ""
        if gt_count == 0 and not include_empty_gt:
            total["skipped_empty_images"] += 1
            append_rows(sample_path, [[
                args.case_name, process_rank, i, image_path, gt_count, pred_count, 0, 0, 0,
                0.0, 0.0, 0.0, 1, 0, raw_debug["raw_points"], raw_debug["cross_border"],
                raw_debug["reserved_before_dedup"], raw_debug["reserved_after_dedup"],
                0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]])
            continue

        total["images"] += 1
        total["gt"] += gt_count
        total["pred"] += pred_count
        total["tp"] += tp
        total["raw_points"] += raw_debug["raw_points"]
        total["cross_border"] += raw_debug["cross_border"]
        total["reserved_before"] += raw_debug["reserved_before_dedup"]
        total["reserved_after"] += raw_debug["reserved_after_dedup"]

        if (
            raw_debug["proto_margin_in"] is not None
            and len(raw_debug["points_in"])
            and len(gt_points)
        ):
            predicted_distance = distance_matrix(raw_debug["points_in"], gt_points).min(axis=1)
            anchor_distance = distance_matrix(raw_debug["anchors_in"], gt_points).min(axis=1)
            feature_distance = np.minimum(predicted_distance, anchor_distance)
            proto_margin = raw_debug["proto_margin_in"]
            proto_distances = raw_debug["proto_distances_in"]
            near = feature_distance <= args.match_dis
            far = feature_distance > args.near_radius
            total["proto_near_count"] += float(near.sum())
            total["proto_far_count"] += float(far.sum())
            total["proto_near_margin_sum"] += float(proto_margin[near].sum())
            total["proto_far_margin_sum"] += float(proto_margin[far].sum())
            total["proto_near_fg_win"] += float((proto_margin[near] > 0).sum())
            total["proto_far_fg_win"] += float((proto_margin[far] > 0).sum())
            if raw_debug["proto_distances_in"] is not None:
                proto_distances = raw_debug["proto_distances_in"]
                total["proto_near_fg_distance_sum"] += float(proto_distances[near, 0].sum())
                total["proto_near_bg_distance_sum"] += float(proto_distances[near, 1].sum())
                total["proto_far_fg_distance_sum"] += float(proto_distances[far, 0].sum())
                total["proto_far_bg_distance_sum"] += float(proto_distances[far, 1].sum())

        sample_fn_counts = {
            "no_candidate": 0, "low_score_or_background": 0,
            "localization_shift": 0, "near_candidate_unmatched": 0,
        }
        sample_fp_counts = {
            "empty_image_fp": 0, "duplicate": 0, "near_miss": 0, "background_far": 0,
        }
        score_rows = []
        fn_rows = []
        fp_rows = []
        tp_l1_sum = 0.0
        tp_dist_sum = 0.0

        for pred_i, gt_i, match_dist in matches:
            pred = pred_points[pred_i].astype(np.float64)
            gt = gt_points[gt_i].astype(np.float64)
            diff = pred - gt
            sq = float(np.sum(diff * diff))
            l1 = float(np.sum(np.abs(diff)))
            total["mse_sum"] += sq
            total["mae_sum"] += l1
            total["tp_distance_sum"] += match_dist
            total["tp_distance_sq_sum"] += match_dist * match_dist
            tp_l1_sum += l1
            tp_dist_sum += match_dist
            score_rows.append([
                args.case_name, process_rank, i, image_path, "TP",
                float(global_scores[pred_i]) if len(global_scores) else 0.0,
                match_dist, gt_i, pred_i
            ])

        for gt_i in unmatched_gt:
            fn_type, nearest_reserved_dist, nearest_raw_dist, raw_cell_score, raw_bg_score = classify_fn(
                gt_i, gt_points, pred_points, raw_debug, matched_gt,
                args.match_dis, args.near_radius, args.num_classes)
            sample_fn_counts[fn_type] += 1
            total[f"fn_{fn_type}"] += 1
            fn_rows.append([
                args.case_name, process_rank, i, image_path, fn_type, gt_i,
                nearest_reserved_dist if nearest_reserved_dist is not None else "",
                nearest_raw_dist if nearest_raw_dist is not None else "",
                raw_cell_score, raw_bg_score
            ])

        for pred_i in unmatched_pred:
            fp_type, nearest_gt_dist, nearest_gt_index = classify_fp(
                pred_i, pred_points, gt_points, matched_gt,
                args.match_dis, args.near_radius)
            sample_fp_counts[fp_type] += 1
            fp_total_key = {
                "empty_image_fp": "fp_empty_image",
                "duplicate": "fp_duplicate",
                "near_miss": "fp_near_miss",
                "background_far": "fp_background_far",
            }[fp_type]
            total[fp_total_key] += 1
            score = float(global_scores[pred_i]) if len(global_scores) else 0.0
            score_rows.append([
                args.case_name, process_rank, i, image_path, "FP",
                score, nearest_gt_dist if nearest_gt_dist is not None else "",
                nearest_gt_index, pred_i
            ])
            fp_rows.append([
                args.case_name, process_rank, i, image_path, fp_type, pred_i,
                score, nearest_gt_dist if nearest_gt_dist is not None else "",
                nearest_gt_index
            ])

        if "reg_attn" in raw_debug["attn"]:
            reg_attn_sum += raw_debug["attn"]["reg_attn"]
        if "cls_attn" in raw_debug["attn"]:
            cls_attn_sum += raw_debug["attn"]["cls_attn"]
        if "reg_attn" in raw_debug["attn"] or "cls_attn" in raw_debug["attn"]:
            attn_count += 1

        p = _safe_div(tp, pred_count)
        r = _safe_div(tp, gt_count)
        f1 = _safe_div(2 * p * r, p + r)
        sample_rows = [[
            args.case_name, process_rank, i, image_path, gt_count, pred_count, tp, fp, fn,
            p, r, f1, int(gt_count == 0), 1, raw_debug["raw_points"], raw_debug["cross_border"],
            raw_debug["reserved_before_dedup"], raw_debug["reserved_after_dedup"],
            _safe_div(tp_dist_sum, tp), _safe_div(tp_l1_sum, tp),
            sample_fn_counts["no_candidate"], sample_fn_counts["low_score_or_background"],
            sample_fn_counts["localization_shift"], sample_fn_counts["near_candidate_unmatched"],
            sample_fp_counts["empty_image_fp"], sample_fp_counts["duplicate"],
            sample_fp_counts["near_miss"], sample_fp_counts["background_far"],
        ]]
        append_rows(sample_path, sample_rows)
        append_rows(score_path, score_rows)
        append_rows(fn_path, fn_rows)
        append_rows(fp_path, fp_rows)

    count_names = list(total.keys())
    count_tensor = torch.tensor([total[name] for name in count_names], dtype=torch.float64).cuda(rank)
    attn_tensor = torch.tensor(
        list(reg_attn_sum) + list(cls_attn_sum) + [attn_count],
        dtype=torch.float64
    ).cuda(rank)
    if getattr(args, "distributed", False):
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(attn_tensor, op=dist.ReduceOp.SUM)

    reduced = {name: float(count_tensor[j].item()) for j, name in enumerate(count_names)}
    reg_attn = attn_tensor[:num_levels].detach().cpu().numpy()
    cls_attn = attn_tensor[num_levels:2 * num_levels].detach().cpu().numpy()
    reduced_attn_count = float(attn_tensor[-1].item())
    if reduced_attn_count > 0:
        reg_attn = reg_attn / reduced_attn_count
        cls_attn = cls_attn / reduced_attn_count

    if process_rank == 0:
        tp = reduced["tp"]
        pred = reduced["pred"]
        gt = reduced["gt"]
        fp = pred - tp
        fn = gt - tp
        precision = _safe_div(tp, pred)
        recall = _safe_div(tp, gt)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        mae = _safe_div(reduced["mae_sum"], tp)
        mse = _safe_div(reduced["mse_sum"], tp)
        mean_tp_distance = _safe_div(reduced["tp_distance_sum"], tp)
        proto_near_margin = _safe_div(reduced["proto_near_margin_sum"], reduced["proto_near_count"])
        proto_far_margin = _safe_div(reduced["proto_far_margin_sum"], reduced["proto_far_count"])
        prototype_head = getattr(model, "prototype_head", None)

        summary = {
            "case": args.case_name,
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch", "") if isinstance(checkpoint, dict) else "",
            "images": reduced["images"],
            "total_eval_images": reduced["total_eval_images"],
            "skipped_empty_images": reduced["skipped_empty_images"],
            "empty_images": reduced["empty_images"],
            "include_empty_gt_in_eval": int(include_empty_gt),
            "gt": gt,
            "pred": pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mae": mae,
            "mse": mse,
            "pred_per_image": _safe_div(pred, reduced["images"]),
            "gt_per_image": _safe_div(gt, reduced["images"]),
            "pred_gt_ratio": _safe_div(pred, gt),
            "empty_image_pred": reduced["fp_empty_image"],
            "raw_points_per_image": _safe_div(reduced["raw_points"], reduced["images"]),
            "reserved_before_dedup_per_image": _safe_div(reduced["reserved_before"], reduced["images"]),
            "reserved_after_dedup_per_image": _safe_div(reduced["reserved_after"], reduced["images"]),
            "mean_tp_distance": mean_tp_distance,
            "fn_no_candidate": reduced["fn_no_candidate"],
            "fn_low_score_or_background": reduced["fn_low_score_or_background"],
            "fn_localization_shift": reduced["fn_localization_shift"],
            "fn_near_candidate_unmatched": reduced["fn_near_candidate_unmatched"],
            "fp_empty_image": reduced["fp_empty_image"],
            "fp_duplicate": reduced["fp_duplicate"],
            "fp_near_miss": reduced["fp_near_miss"],
            "fp_background_far": reduced["fp_background_far"],
            "prototype_ready": int(prototype_head.prototype_ready.item()) if prototype_head is not None else 0,
            "prototype_fg_queue_count": int(
                getattr(
                    prototype_head,
                    "foreground_queue_count",
                    getattr(prototype_head, "fg_queue_count", torch.tensor(0)),
                ).item()
            ) if prototype_head is not None else 0,
            "prototype_bg_queue_count": int(
                getattr(
                    prototype_head,
                    "background_queue_count",
                    getattr(prototype_head, "bg_queue_count", torch.tensor(0)),
                ).item()
            ) if prototype_head is not None else 0,
            "prototype_near_count": reduced["proto_near_count"],
            "prototype_far_count": reduced["proto_far_count"],
            "prototype_near_margin": proto_near_margin,
            "prototype_far_margin": proto_far_margin,
            "prototype_margin_separation": proto_near_margin - proto_far_margin,
            "prototype_near_fg_win_rate": _safe_div(reduced["proto_near_fg_win"], reduced["proto_near_count"]),
            "prototype_far_fg_win_rate": _safe_div(reduced["proto_far_fg_win"], reduced["proto_far_count"]),
            "prototype_near_fg_distance": _safe_div(reduced["proto_near_fg_distance_sum"], reduced["proto_near_count"]),
            "prototype_near_bg_distance": _safe_div(reduced["proto_near_bg_distance_sum"], reduced["proto_near_count"]),
            "prototype_far_fg_distance": _safe_div(reduced["proto_far_fg_distance_sum"], reduced["proto_far_count"]),
            "prototype_far_bg_distance": _safe_div(reduced["proto_far_bg_distance_sum"], reduced["proto_far_count"]),
            "reg_attn_mean": json.dumps(reg_attn.tolist()),
            "cls_attn_mean": json.dumps(cls_attn.tolist()),
            "load_missing_keys": len(getattr(load_result, "missing_keys", [])),
            "load_unexpected_keys": len(getattr(load_result, "unexpected_keys", [])),
        }

        summary_path = os.path.join(args.output_dir, "summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(os.path.join(args.output_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write(f"Run started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"case={args.case_name}\n")
            f.write(f"dataset={args.dataset}\n")
            f.write(f"checkpoint={args.checkpoint}\n")
            f.write(f"checkpoint_epoch={summary['checkpoint_epoch']}\n")
            f.write(f"include_empty_gt_in_eval={int(include_empty_gt)}\n")
            f.write(f"metric_images={reduced['images']:.0f}, total_eval_images={reduced['total_eval_images']:.0f}, "
                    f"skipped_empty_images={reduced['skipped_empty_images']:.0f}, empty_images={reduced['empty_images']:.0f}\n")
            f.write(f"missing_keys={getattr(load_result, 'missing_keys', [])}\n")
            f.write(f"unexpected_keys={getattr(load_result, 'unexpected_keys', [])}\n")
            f.write(f"precision={precision:.6f}, recall={recall:.6f}, f1={f1:.6f}\n")
            f.write(f"tp={tp:.0f}, fp={fp:.0f}, fn={fn:.0f}, pred={pred:.0f}, gt={gt:.0f}\n")
            f.write(f"fn_breakdown=no_candidate:{reduced['fn_no_candidate']:.0f}, "
                    f"low_score_or_background:{reduced['fn_low_score_or_background']:.0f}, "
                    f"localization_shift:{reduced['fn_localization_shift']:.0f}, "
                    f"near_candidate_unmatched:{reduced['fn_near_candidate_unmatched']:.0f}\n")
            f.write(f"fp_breakdown=empty_image:{reduced['fp_empty_image']:.0f}, "
                    f"duplicate:{reduced['fp_duplicate']:.0f}, near_miss:{reduced['fp_near_miss']:.0f}, "
                    f"background_far:{reduced['fp_background_far']:.0f}\n")

        print(json.dumps(summary, ensure_ascii=False, indent=2))


def get_parser():
    parser = p2p.get_args_parser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint path to diagnose")
    parser.add_argument("--case_name", required=True, help="case name written to CSV files")
    parser.add_argument("--dedup_interval", default=15, type=float, help="deduplication radius")
    parser.add_argument("--near_radius", default=30, type=float, help="loose radius for FN/FP decomposition")
    parser.add_argument("--strict_load", action="store_true", help="strict checkpoint loading")
    parser.add_argument("--include_empty_gt_in_eval", dest="include_empty_gt_in_eval",
                        action="store_true", default=False,
                        help="count empty-GT test images in metrics; use for sensitivity analysis only")
    parser.add_argument("--skip_empty_gt_in_eval", dest="include_empty_gt_in_eval",
                        action="store_false",
                        help="default protocol: skip empty-GT images from metrics")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    p2p.args = args
    init_distributed_mode(args)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if not args.output_dir:
        args.output_dir = os.path.join("experiments", f"diagnosis_{args.case_name}")

    run_diagnosis(args)

    if getattr(args, "distributed", False):
        cleanup()


if __name__ == "__main__":
    main()
