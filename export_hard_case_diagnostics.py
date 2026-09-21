import argparse
import csv
import json
import math
import os
import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw

import train_p2p as p2p
from dataset_zy_src import build_dataset
from diagnose_cross_domain import distance_matrix, greedy_match, load_checkpoint_for_eval
from models.detr import build_model


DEFAULT_IMAGE_NAMES = [
    "2022-06-28_11_20_39.kfb_3171.jpg",
    "2022-06-28_11_20_39.kfb_2991.jpg",
    "2022-06-28_11_21_44.kfb_2229.jpg",
    "2022-06-28_11_20_39.kfb_974.jpg",
    "2022-06-28_11_21_44.kfb_1804.jpg",
    "2022-06-28_11_11_09.kfb_2543.jpg",
    "2022-06-28_11_21_44.kfb_2123.jpg",
    "2022-06-28_11_21_44.kfb_1053.jpg",
    "2022-06-28_11_18_14.kfb_2274.jpg",
    "2022-06-28_11_18_14.kfb_3471.jpg",
    "2022-06-28_11_11_09.kfb_446.jpg",
    "2022-06-28_11_18_14.kfb_593.jpg",
    "2022-06-28_11_11_09.kfb_2336.jpg",
    "2022-06-28_11_21_44.kfb_2170.jpg",
    "2022-06-28_11_11_09.kfb_1092.jpg",
    "2022-06-28_11_20_39.kfb_1335.jpg",
    "2022-06-28_11_20_39.kfb_373.jpg",
    "2022-06-28_11_12_06.kfb_1226.jpg",
    "2022-06-28_11_20_39.kfb_430.jpg",
    "2022-06-28_11_21_44.kfb_1227.jpg",
    "2022-06-28_11_12_06.kfb_956.jpg",
    "2022-06-28_11_21_44.kfb_1199.jpg",
    "2022-06-28_11_12_06.kfb_1853.jpg",
    "2022-06-28_11_21_44.kfb_2174.jpg",
]


class FeatureCapture:
    def __init__(self):
        self.cls_features = None
        self.reg_features = None

    def clear(self):
        self.cls_features = None
        self.reg_features = None


def _safe_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_sample_rows(case_dir):
    rows = []
    if not case_dir or not os.path.isdir(case_dir):
        return rows
    for name in os.listdir(case_dir):
        if name.startswith("sample_stats_rank") and name.endswith(".csv"):
            with open(os.path.join(case_dir, name), newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
    return rows


def select_images_from_diagnosis(diagnosis_dir, max_images):
    para_rows = read_sample_rows(os.path.join(diagnosis_dir, "0318_paraffin_to_frozen"))
    frozen_rows = read_sample_rows(os.path.join(diagnosis_dir, "0326bd_frozen_to_frozen"))
    if not para_rows or not frozen_rows:
        return []

    para = {
        os.path.basename(row["image_path"]): row
        for row in para_rows
        if row.get("metric_image", "1") == "1"
    }
    frozen = {
        os.path.basename(row["image_path"]): row
        for row in frozen_rows
        if row.get("metric_image", "1") == "1"
    }

    buckets = {
        "extra_fp": [],
        "f1_gap": [],
        "paraffin_low_recall": [],
        "paraffin_localization": [],
        "paraffin_background_fp": [],
    }
    for image_name, row in para.items():
        other = frozen.get(image_name)
        if other is None:
            continue
        buckets["extra_fp"].append((_safe_float(row, "fp") - _safe_float(other, "fp"), image_name))
        buckets["f1_gap"].append((_safe_float(other, "f1") - _safe_float(row, "f1"), image_name))
        buckets["paraffin_low_recall"].append((1.0 - _safe_float(row, "recall"), image_name))
        buckets["paraffin_localization"].append((_safe_float(row, "fn_localization_shift"), image_name))
        buckets["paraffin_background_fp"].append((_safe_float(row, "fp_background_far"), image_name))

    selected = OrderedDict()
    per_bucket = max(4, max_images // len(buckets))
    for key in ("extra_fp", "f1_gap", "paraffin_low_recall", "paraffin_localization",
                "paraffin_background_fp"):
        for _, image_name in sorted(buckets[key], reverse=True)[:per_bucket]:
            selected[image_name] = True
            if len(selected) >= max_images:
                return list(selected.keys())
    return list(selected.keys())


def normalize_image(image):
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo + 1e-6)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def crop_box(center, image_shape, crop_size):
    h, w = image_shape[:2]
    x, y = float(center[0]), float(center[1])
    half = crop_size / 2.0
    left = int(max(0, round(x - half)))
    top = int(max(0, round(y - half)))
    right = int(min(w, left + crop_size))
    bottom = int(min(h, top + crop_size))
    left = max(0, right - crop_size)
    top = max(0, bottom - crop_size)
    return left, top, right, bottom


def draw_cross(draw, xy, color, radius=5, width=2):
    x, y = float(xy[0]), float(xy[1])
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=width)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=width)


def draw_circle(draw, xy, color, radius=4, width=2):
    x, y = float(xy[0]), float(xy[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def save_crop(raw_image, center, out_path, title, gt_points=None, pred_points=None,
              selected_pred=None, selected_gt=None, crop_size=256):
    arr = normalize_image(raw_image)
    left, top, right, bottom = crop_box(center, arr.shape, crop_size)
    crop = Image.fromarray(arr[top:bottom, left:right]).convert("RGB")
    draw = ImageDraw.Draw(crop)
    draw.rectangle((0, 0, crop.width, 28), fill=(0, 0, 0))
    draw.text((6, 6), title[:160], fill=(255, 255, 255))

    if gt_points is not None:
        for j, gt in enumerate(gt_points):
            x, y = float(gt[0]) - left, float(gt[1]) - top
            if 0 <= x < crop.width and 0 <= y < crop.height:
                if selected_gt is not None and j == selected_gt:
                    draw_cross(draw, (x, y), (255, 220, 0), radius=6, width=3)
                else:
                    draw_circle(draw, (x, y), (80, 255, 80), radius=3, width=2)

    if pred_points is not None:
        for j, pred in enumerate(pred_points):
            x, y = float(pred[0]) - left, float(pred[1]) - top
            if 0 <= x < crop.width and 0 <= y < crop.height:
                if selected_pred is not None and j == selected_pred:
                    draw_cross(draw, (x, y), (255, 40, 40), radius=7, width=3)
                else:
                    draw_circle(draw, (x, y), (0, 220, 255), radius=3, width=1)

    crop.save(out_path, quality=95)


def entropy(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return float(-np.sum(scores * np.log(scores + 1e-12)))


def deduplicate_with_indices(points, scores, raw_indices, interval):
    n = len(points)
    if n == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.array([], dtype=np.int64),
            np.zeros((0, scores.shape[1] if scores.ndim == 2 else 0), dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    fused = np.full(n, False)
    kept_points = []
    kept_classes = []
    kept_scores = []
    kept_raw_indices = []
    cluster_sizes = []
    for i in range(n):
        if fused[i]:
            continue
        fused_index = np.where(np.linalg.norm(points[[i]] - points[i:], 2, axis=1) < interval)[0] + i
        fused[fused_index] = True
        r_, c_ = np.where(scores[fused_index] == np.max(scores[fused_index]))
        row = int(fused_index[int(r_[0])])
        kept_points.append(points[row])
        kept_classes.append(int(c_[0]))
        kept_scores.append(scores[row])
        kept_raw_indices.append(int(raw_indices[row]))
        cluster_sizes.append(int(len(fused_index)))
    return (
        np.asarray(kept_points, dtype=np.float64).reshape(-1, 2),
        np.asarray(kept_classes, dtype=np.int64),
        np.asarray(kept_scores, dtype=np.float64),
        np.asarray(kept_raw_indices, dtype=np.int64),
        np.asarray(cluster_sizes, dtype=np.int64),
    )


def attach_feature_hooks(model, capture):
    def cls_hook(module, module_input, module_output):
        capture.cls_features = module_input[0].detach().cpu()

    def reg_hook(module, module_input, module_output):
        capture.reg_features = module_input[0].detach().cpu()

    handles = [
        model.cls_head.register_forward_hook(cls_hook),
        model.reg_head.register_forward_hook(reg_hook),
    ]
    return handles


@torch.no_grad()
def predict_with_features(model, image_tensor, args, device, capture):
    capture.clear()
    images = image_tensor.unsqueeze(0).cuda(device, non_blocking=True)
    h, w = images.shape[-2:]
    outputs = model(images)
    raw_points = outputs["pnt_coords"][0].detach().cpu().numpy()
    logits = outputs["cls_logits"][0].detach().cpu()
    scores = torch.softmax(logits, dim=-1).numpy()
    logits_np = logits.numpy()
    cls_features = capture.cls_features[0].numpy() if capture.cls_features is not None else None
    reg_features = capture.reg_features[0].numpy() if capture.reg_features is not None else None

    raw_indices = np.arange(len(raw_points), dtype=np.int64)
    cross_border = (
        (raw_points[:, 0] < 0) | (raw_points[:, 0] >= w) |
        (raw_points[:, 1] < 0) | (raw_points[:, 1] >= h)
    )
    points_in = raw_points[~cross_border]
    scores_in = scores[~cross_border]
    logits_in = logits_np[~cross_border]
    raw_indices_in = raw_indices[~cross_border]
    classes_in = np.argmax(scores_in, axis=-1) if len(scores_in) else np.array([], dtype=np.int64)
    reserved = classes_in < args.num_classes

    points_reserved = points_in[reserved]
    scores_reserved = scores_in[reserved]
    raw_reserved = raw_indices_in[reserved]
    pred_points, pred_classes, pred_scores, pred_raw_indices, cluster_sizes = deduplicate_with_indices(
        points_reserved, scores_reserved, raw_reserved, args.dedup_interval)

    if len(pred_scores):
        cell_scores = pred_scores[:, :args.num_classes].max(axis=1)
        bg_scores = pred_scores[:, -1]
        margins = cell_scores - bg_scores
        entropies = np.array([entropy(row) for row in pred_scores], dtype=np.float64)
    else:
        cell_scores = np.array([], dtype=np.float64)
        bg_scores = np.array([], dtype=np.float64)
        margins = np.array([], dtype=np.float64)
        entropies = np.array([], dtype=np.float64)

    raw_debug = {
        "raw_points": raw_points,
        "points_in": points_in,
        "scores_in": scores_in,
        "logits_in": logits_in,
        "raw_indices_in": raw_indices_in,
        "classes_in": classes_in,
        "cell_scores_in": scores_in[:, :args.num_classes].max(axis=1) if len(scores_in) else np.array([]),
        "bg_scores_in": scores_in[:, -1] if len(scores_in) else np.array([]),
        "cross_border_count": int(cross_border.sum()),
        "reserved_before_dedup": int(reserved.sum()),
        "reserved_after_dedup": int(len(pred_points)),
        "cls_features": cls_features,
        "reg_features": reg_features,
    }
    return {
        "pred_points": pred_points,
        "pred_classes": pred_classes,
        "pred_scores": pred_scores,
        "pred_raw_indices": pred_raw_indices,
        "cluster_sizes": cluster_sizes,
        "cell_scores": cell_scores,
        "bg_scores": bg_scores,
        "margins": margins,
        "entropies": entropies,
        "raw_debug": raw_debug,
    }


def classify_fp(pred_i, pred_points, gt_points, matched_gt, match_dis, near_radius):
    if len(gt_points) == 0:
        return "empty_image_fp", "", -1
    d = distance_matrix(pred_points[pred_i][None, :], gt_points)[0]
    nearest_gt = int(np.argmin(d))
    nearest_dist = float(d[nearest_gt])
    if nearest_dist <= match_dis and nearest_gt in matched_gt:
        return "duplicate", nearest_dist, nearest_gt
    if nearest_dist <= near_radius:
        return "near_miss", nearest_dist, nearest_gt
    return "background_far", nearest_dist, nearest_gt


def classify_fn(gt_i, gt_points, pred_points, pred_scores, raw_debug, match_dis, near_radius):
    gt = gt_points[gt_i]
    nearest_pred_dist = ""
    if len(pred_points):
        d_pred = distance_matrix(pred_points, gt[None, :])[:, 0]
        nearest_pred_dist = float(d_pred.min())

    raw_points = raw_debug["points_in"]
    raw_cell_scores = raw_debug["cell_scores_in"]
    raw_bg_scores = raw_debug["bg_scores_in"]
    raw_classes = raw_debug["classes_in"]
    if len(raw_points):
        d_raw = distance_matrix(raw_points, gt[None, :])[:, 0]
        raw_i = int(np.argmin(d_raw))
        nearest_raw_dist = float(d_raw[raw_i])
        raw_cell_score = float(raw_cell_scores[raw_i])
        raw_bg_score = float(raw_bg_scores[raw_i])
        raw_margin = raw_cell_score - raw_bg_score
        raw_is_bg = int(raw_classes[raw_i] >= 1)
    else:
        nearest_raw_dist = ""
        raw_cell_score = 0.0
        raw_bg_score = 0.0
        raw_margin = 0.0
        raw_is_bg = 1

    if nearest_pred_dist != "" and match_dis < nearest_pred_dist <= near_radius:
        fn_type = "localization_shift"
    elif nearest_raw_dist != "" and nearest_raw_dist <= near_radius and raw_is_bg:
        fn_type = "low_score_or_background"
    elif nearest_raw_dist != "" and nearest_raw_dist <= near_radius:
        fn_type = "near_candidate_unmatched"
    else:
        fn_type = "no_candidate"
    return fn_type, nearest_pred_dist, nearest_raw_dist, raw_cell_score, raw_bg_score, raw_margin


def summarize_prediction(pred, gt_points, args):
    scores = pred["pred_scores"][:, :args.num_classes].sum(axis=1) if len(pred["pred_scores"]) else np.array([])
    matches, matched_pred, matched_gt = greedy_match(
        pred["pred_points"], scores, gt_points, args.match_dis)
    unmatched_pred = [i for i in range(len(pred["pred_points"])) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_points)) if i not in matched_gt]
    tp = len(matches)
    pred_count = len(pred["pred_points"])
    gt_count = len(gt_points)
    fp = pred_count - tp
    fn = gt_count - tp
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "matched_pred": matched_pred,
        "matched_gt": matched_gt,
        "unmatched_pred": unmatched_pred,
        "unmatched_gt": unmatched_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred": pred_count,
        "gt": gt_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def point_common_fields(model_name, image_name, pred, pred_i):
    raw_i = int(pred["pred_raw_indices"][pred_i]) if len(pred["pred_raw_indices"]) else -1
    return {
        "model": model_name,
        "image_name": image_name,
        "pred_index": int(pred_i),
        "raw_index": raw_i,
        "x": float(pred["pred_points"][pred_i][0]),
        "y": float(pred["pred_points"][pred_i][1]),
        "cell_score": float(pred["cell_scores"][pred_i]),
        "bg_score": float(pred["bg_scores"][pred_i]),
        "margin_cell_minus_bg": float(pred["margins"][pred_i]),
        "entropy": float(pred["entropies"][pred_i]),
        "cluster_size": int(pred["cluster_sizes"][pred_i]),
    }


def load_model(args, checkpoint, device):
    model = build_model(args).cuda(device)
    checkpoint_obj, load_result = load_checkpoint_for_eval(model, checkpoint, strict=False)
    model.eval()
    return model, checkpoint_obj, load_result


def feature_for_pred(pred, pred_i):
    raw_i = int(pred["pred_raw_indices"][pred_i])
    features = pred["raw_debug"]["cls_features"]
    if features is None or raw_i < 0 or raw_i >= len(features):
        return None
    return features[raw_i].astype(np.float32, copy=False)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_feature_pca(feature_rows, features, out_dir):
    if len(features) < 3:
        return
    x = np.asarray(features, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    pca_rows = []
    for row, coord in zip(feature_rows, coords):
        new_row = dict(row)
        new_row["pc1"] = float(coord[0])
        new_row["pc2"] = float(coord[1])
        pca_rows.append(new_row)
    write_csv(os.path.join(out_dir, "feature_pca_points.csv"), pca_rows)

    width, height = 1200, 900
    pad = 60
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x0, x1 = float(coords[:, 0].min()), float(coords[:, 0].max())
    y0, y1 = float(coords[:, 1].min()), float(coords[:, 1].max())
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1

    colors = {
        ("paraffin", "TP"): (0, 130, 255),
        ("paraffin", "FP"): (255, 30, 30),
        ("frozen", "TP"): (0, 170, 80),
        ("frozen", "FP"): (255, 140, 0),
    }
    for row, coord in zip(feature_rows, coords):
        px = pad + (coord[0] - x0) / (x1 - x0) * (width - 2 * pad)
        py = height - pad - (coord[1] - y0) / (y1 - y0) * (height - 2 * pad)
        color = colors.get((row["model"], row["kind"]), (0, 0, 0))
        r = 3 if row["kind"] == "TP" else 4
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color)
    legend = [
        ("paraffin TP", colors[("paraffin", "TP")]),
        ("paraffin FP", colors[("paraffin", "FP")]),
        ("frozen TP", colors[("frozen", "TP")]),
        ("frozen FP", colors[("frozen", "FP")]),
    ]
    y = 16
    for text, color in legend:
        draw.rectangle((16, y, 34, y + 18), fill=color)
        draw.text((42, y), text, fill=(0, 0, 0))
        y += 24
    canvas.save(os.path.join(out_dir, "feature_pca.png"), quality=95)


def main():
    parser = p2p.get_args_parser()
    parser.add_argument("--paraffin_checkpoint", required=True)
    parser.add_argument("--frozen_checkpoint", required=True)
    parser.add_argument("--diagnosis_dir", default="")
    parser.add_argument("--image_names", nargs="*", default=None)
    parser.add_argument("--max_images", type=int, default=24)
    parser.add_argument("--max_points_per_kind", type=int, default=40)
    parser.add_argument("--max_crops_per_kind", type=int, default=12)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--dedup_interval", type=float, default=15)
    parser.add_argument("--near_radius", type=float, default=30)
    args = parser.parse_args()
    p2p.args = args

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "crops"), exist_ok=True)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    device = 0
    dataset = build_dataset(args, "test")
    file_by_name = {os.path.basename(path): i for i, path in enumerate(dataset.files)}
    image_names = args.image_names or select_images_from_diagnosis(args.diagnosis_dir, args.max_images)
    if not image_names:
        image_names = DEFAULT_IMAGE_NAMES[:args.max_images]
    image_names = [name for name in image_names if name in file_by_name][:args.max_images]

    para_model, _, para_load = load_model(args, args.paraffin_checkpoint, device)
    frozen_model, _, frozen_load = load_model(args, args.frozen_checkpoint, device)
    para_capture, frozen_capture = FeatureCapture(), FeatureCapture()
    handles = attach_feature_hooks(para_model, para_capture) + attach_feature_hooks(frozen_model, frozen_capture)

    point_rows = []
    image_rows = []
    feature_rows = []
    feature_vectors = []
    crop_counter = defaultdict(int)

    for image_name in image_names:
        index = file_by_name[image_name]
        raw_sample = dataset.read_data(dataset.data[index], dataset.files[index])
        raw_image = raw_sample["image"]
        gt_points = raw_sample["keypoints"].astype(np.float64)
        image_tensor, _, _ = dataset[index]

        predictions = {
            "paraffin": predict_with_features(para_model, image_tensor, args, device, para_capture),
            "frozen": predict_with_features(frozen_model, image_tensor, args, device, frozen_capture),
        }

        summaries = {
            model_name: summarize_prediction(pred, gt_points, args)
            for model_name, pred in predictions.items()
        }

        for model_name, pred in predictions.items():
            summary = summaries[model_name]
            fp_types = defaultdict(int)
            fn_types = defaultdict(int)

            image_rows.append({
                "image_name": image_name,
                "model": model_name,
                "gt": summary["gt"],
                "pred": summary["pred"],
                "tp": summary["tp"],
                "fp": summary["fp"],
                "fn": summary["fn"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "reserved_before_dedup": pred["raw_debug"]["reserved_before_dedup"],
                "reserved_after_dedup": pred["raw_debug"]["reserved_after_dedup"],
                "cross_border": pred["raw_debug"]["cross_border_count"],
            })

            # TP rows: keep matched points with high confidence first for feature comparison.
            tp_candidates = []
            for pred_i, gt_i, match_dist in summary["matches"]:
                row = point_common_fields(model_name, image_name, pred, pred_i)
                row.update({
                    "kind": "TP",
                    "subtype": "matched",
                    "gt_index": int(gt_i),
                    "nearest_gt_dist": float(match_dist),
                    "nearest_pred_dist": "",
                    "nearest_raw_dist": "",
                    "nearest_raw_cell_score": "",
                    "nearest_raw_bg_score": "",
                    "nearest_raw_margin": "",
                })
                tp_candidates.append(row)

            # FP rows: sort by score and then background-far status.
            fp_candidates = []
            for pred_i in summary["unmatched_pred"]:
                fp_type, nearest_gt_dist, nearest_gt_index = classify_fp(
                    pred_i, pred["pred_points"], gt_points, summary["matched_gt"],
                    args.match_dis, args.near_radius)
                fp_types[fp_type] += 1
                row = point_common_fields(model_name, image_name, pred, pred_i)
                row.update({
                    "kind": "FP",
                    "subtype": fp_type,
                    "gt_index": nearest_gt_index,
                    "nearest_gt_dist": nearest_gt_dist,
                    "nearest_pred_dist": "",
                    "nearest_raw_dist": "",
                    "nearest_raw_cell_score": "",
                    "nearest_raw_bg_score": "",
                    "nearest_raw_margin": "",
                })
                fp_candidates.append(row)

            # FN rows use raw nearest candidate diagnostics.
            fn_candidates = []
            for gt_i in summary["unmatched_gt"]:
                fn_type, nearest_pred_dist, nearest_raw_dist, raw_cell_score, raw_bg_score, raw_margin = classify_fn(
                    gt_i, gt_points, pred["pred_points"], pred["pred_scores"],
                    pred["raw_debug"], args.match_dis, args.near_radius)
                fn_types[fn_type] += 1
                row = {
                    "model": model_name,
                    "image_name": image_name,
                    "kind": "FN",
                    "subtype": fn_type,
                    "pred_index": -1,
                    "raw_index": -1,
                    "gt_index": int(gt_i),
                    "x": float(gt_points[gt_i][0]),
                    "y": float(gt_points[gt_i][1]),
                    "cell_score": "",
                    "bg_score": "",
                    "margin_cell_minus_bg": "",
                    "entropy": "",
                    "cluster_size": "",
                    "nearest_gt_dist": "",
                    "nearest_pred_dist": nearest_pred_dist,
                    "nearest_raw_dist": nearest_raw_dist,
                    "nearest_raw_cell_score": raw_cell_score,
                    "nearest_raw_bg_score": raw_bg_score,
                    "nearest_raw_margin": raw_margin,
                }
                fn_candidates.append(row)

            tp_candidates = sorted(tp_candidates, key=lambda r: _safe_float(r, "cell_score"), reverse=True)
            fp_candidates = sorted(fp_candidates, key=lambda r: (
                r["subtype"] == "background_far",
                _safe_float(r, "cell_score")
            ), reverse=True)
            fn_candidates = sorted(fn_candidates, key=lambda r: (
                r["subtype"] in ("low_score_or_background", "localization_shift"),
                _safe_float(r, "nearest_raw_cell_score")
            ), reverse=True)

            kept_rows = (
                tp_candidates[:args.max_points_per_kind] +
                fp_candidates[:args.max_points_per_kind] +
                fn_candidates[:args.max_points_per_kind]
            )
            point_rows.extend(kept_rows)

            for row in tp_candidates[:args.max_points_per_kind] + fp_candidates[:args.max_points_per_kind]:
                pred_i = _safe_int(row["pred_index"])
                feat = feature_for_pred(pred, pred_i) if pred_i >= 0 else None
                if feat is not None:
                    feature_rows.append({
                        "model": model_name,
                        "image_name": image_name,
                        "kind": row["kind"],
                        "subtype": row["subtype"],
                        "cell_score": row["cell_score"],
                        "bg_score": row["bg_score"],
                        "margin_cell_minus_bg": row["margin_cell_minus_bg"],
                    })
                    feature_vectors.append(feat)

            for row in fp_candidates[:args.max_crops_per_kind]:
                key = (model_name, "FP", row["subtype"])
                if crop_counter[key] >= args.max_crops_per_kind:
                    continue
                pred_i = _safe_int(row["pred_index"])
                if pred_i < 0:
                    continue
                title = (
                    f"{model_name} FP {row['subtype']} score={_safe_float(row, 'cell_score'):.3f} "
                    f"bg={_safe_float(row, 'bg_score'):.3f} margin={_safe_float(row, 'margin_cell_minus_bg'):.3f}"
                )
                out_name = f"{model_name}_{row['kind']}_{row['subtype']}_{crop_counter[key]:03d}_{image_name}.jpg"
                save_crop(
                    raw_image, (row["x"], row["y"]),
                    os.path.join(args.output_dir, "crops", out_name),
                    title, gt_points=gt_points, pred_points=pred["pred_points"],
                    selected_pred=pred_i, crop_size=args.crop_size)
                crop_counter[key] += 1

            for row in fn_candidates[:args.max_crops_per_kind]:
                key = (model_name, "FN", row["subtype"])
                if crop_counter[key] >= args.max_crops_per_kind:
                    continue
                gt_i = _safe_int(row["gt_index"])
                title = (
                    f"{model_name} FN {row['subtype']} raw_cell={_safe_float(row, 'nearest_raw_cell_score'):.3f} "
                    f"raw_bg={_safe_float(row, 'nearest_raw_bg_score'):.3f} raw_margin={_safe_float(row, 'nearest_raw_margin'):.3f}"
                )
                out_name = f"{model_name}_{row['kind']}_{row['subtype']}_{crop_counter[key]:03d}_{image_name}.jpg"
                save_crop(
                    raw_image, (row["x"], row["y"]),
                    os.path.join(args.output_dir, "crops", out_name),
                    title, gt_points=gt_points, pred_points=pred["pred_points"],
                    selected_gt=gt_i, crop_size=args.crop_size)
                crop_counter[key] += 1

            image_rows[-1].update({
                "fp_background_far": fp_types["background_far"],
                "fp_near_miss": fp_types["near_miss"],
                "fp_duplicate": fp_types["duplicate"],
                "fn_low_score_or_background": fn_types["low_score_or_background"],
                "fn_localization_shift": fn_types["localization_shift"],
                "fn_near_candidate_unmatched": fn_types["near_candidate_unmatched"],
                "fn_no_candidate": fn_types["no_candidate"],
            })

    for handle in handles:
        handle.remove()

    write_csv(os.path.join(args.output_dir, "image_hardcase_summary.csv"), image_rows)
    write_csv(os.path.join(args.output_dir, "point_logit_diagnostics.csv"), point_rows)
    save_feature_pca(feature_rows, feature_vectors, args.output_dir)
    if feature_vectors:
        np.savez_compressed(
            os.path.join(args.output_dir, "point_cls_features.npz"),
            features=np.asarray(feature_vectors, dtype=np.float32),
            meta=np.asarray([json.dumps(row, ensure_ascii=False) for row in feature_rows], dtype=object),
        )

    with open(os.path.join(args.output_dir, "diagnostic_readme.txt"), "w", encoding="utf-8") as f:
        f.write("point_logit_diagnostics.csv: TP/FP/FN rows with score, bg_score, margin, entropy and nearest-candidate diagnostics.\n")
        f.write("crops/: local crops for top FP and FN points. Red cross marks selected FP; yellow cross marks selected FN GT.\n")
        f.write("feature_pca.png/csv: PCA of cls_head input features for sampled TP/FP points.\n")
        f.write("Use this to decide whether errors are feature mixing, classifier calibration, or localization/response failures.\n")

    print(json.dumps({
        "output_dir": args.output_dir,
        "num_images": len(image_names),
        "point_rows": len(point_rows),
        "feature_rows": len(feature_rows),
        "paraffin_missing_keys": len(getattr(para_load, "missing_keys", [])),
        "frozen_missing_keys": len(getattr(frozen_load, "missing_keys", [])),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
