import argparse
import csv
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw

import train_p2p as p2p
from dataset_zy_src import build_dataset
from diagnose_cross_domain import (
    greedy_match,
    load_checkpoint_for_eval,
    predict_detailed,
)
from models.detr import build_model


DEFAULT_IMAGE_NAMES = [
    "2022-06-28_11_20_39.kfb_3171.jpg",
    "2022-06-28_11_20_39.kfb_2991.jpg",
    "2022-06-28_11_21_44.kfb_2229.jpg",
    "2022-06-28_11_20_39.kfb_974.jpg",
    "2022-06-28_11_21_44.kfb_1804.jpg",
    "2022-06-28_11_11_09.kfb_2543.jpg",
    "2022-06-28_11_18_14.kfb_2274.jpg",
    "2022-06-28_11_11_09.kfb_446.jpg",
    "2022-06-28_11_21_44.kfb_2170.jpg",
    "2022-06-28_11_12_06.kfb_1853.jpg",
    "2022-06-28_11_12_06.kfb_1226.jpg",
    "2022-06-28_11_20_39.kfb_373.jpg",
]


def _safe_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _read_sample_rows(case_dir):
    rows = []
    if not case_dir or not os.path.isdir(case_dir):
        return rows
    for name in os.listdir(case_dir):
        if name.startswith("sample_stats_rank") and name.endswith(".csv"):
            path = os.path.join(case_dir, name)
            with open(path, newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
    return rows


def select_images_from_diagnosis(diagnosis_dir, max_images):
    para_rows = _read_sample_rows(os.path.join(diagnosis_dir, "0318_paraffin_to_frozen"))
    frozen_rows = _read_sample_rows(os.path.join(diagnosis_dir, "0326bd_frozen_to_frozen"))
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

    extra_fp = []
    f1_gap = []
    loc_shift = []
    low_score_fn = []
    for image_name, row in para.items():
        other = frozen.get(image_name)
        if other is None:
            continue
        extra_fp.append((_safe_float(row, "fp") - _safe_float(other, "fp"), image_name))
        f1_gap.append((_safe_float(other, "f1") - _safe_float(row, "f1"), image_name))
        loc_shift.append((_safe_float(row, "fn_localization_shift"), image_name))
        low_score_fn.append((_safe_float(row, "fn_low_score_or_background"), image_name))

    selected = OrderedDict()
    for bucket in (
        sorted(extra_fp, reverse=True)[:max(4, max_images // 3)],
        sorted(f1_gap, reverse=True)[:max(4, max_images // 3)],
        sorted(loc_shift, reverse=True)[:max(3, max_images // 4)],
        sorted(low_score_fn, reverse=True)[:max(3, max_images // 4)],
    ):
        for _, image_name in bucket:
            selected[image_name] = True
            if len(selected) >= max_images:
                return list(selected.keys())
    return list(selected.keys())


def image_to_uint8(image):
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


def draw_cross(draw, xy, color, radius=4, width=2):
    x, y = float(xy[0]), float(xy[1])
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=width)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=width)


def draw_circle(draw, xy, color, radius=3, width=2, fill=None):
    x, y = float(xy[0]), float(xy[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 outline=color, width=width, fill=fill)


def overlay_panel(raw_image, gt_points, pred_points=None, pred_scores=None,
                  matches=None, title="", max_draw=2000):
    panel = Image.fromarray(image_to_uint8(raw_image)).convert("RGB")
    draw = ImageDraw.Draw(panel)
    matches = matches or []
    matched_pred = {pred_i for pred_i, _, _ in matches}
    matched_gt = {gt_i for _, gt_i, _ in matches}

    draw.rectangle((0, 0, panel.width, 34), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))

    for j, gt in enumerate(gt_points[:max_draw]):
        if j not in matched_gt:
            draw_cross(draw, gt, color=(255, 220, 0), radius=5, width=2)
        else:
            draw_circle(draw, gt, color=(80, 255, 80), radius=3, width=2)

    if pred_points is not None:
        for j, pred in enumerate(pred_points[:max_draw]):
            if j in matched_pred:
                draw_circle(draw, pred, color=(0, 230, 255), radius=4, width=2)
            else:
                draw_cross(draw, pred, color=(255, 40, 40), radius=5, width=2)

    return panel


def stack_panels(panels, max_panel_width):
    resized = []
    for panel in panels:
        if panel.width > max_panel_width:
            ratio = max_panel_width / float(panel.width)
            new_size = (max_panel_width, int(panel.height * ratio))
            panel = panel.resize(new_size, Image.BILINEAR)
        resized.append(panel)
    w = sum(p.width for p in resized)
    h = max(p.height for p in resized)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for panel in resized:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def load_model(args, checkpoint, device):
    model = build_model(args).cuda(device)
    checkpoint_obj, load_result = load_checkpoint_for_eval(model, checkpoint, strict=False)
    model.eval()
    return model, checkpoint_obj, load_result


def predict_for_model(model, image_tensor, args, device):
    images = image_tensor.unsqueeze(0).cuda(device, non_blocking=True)
    pred_points, pred_classes, pred_scores, pred_cell_scores, raw_debug = predict_detailed(
        model, images, args.num_classes, args.dedup_interval)
    global_scores = pred_scores[:, :args.num_classes].sum(axis=1) if len(pred_scores) else np.array([])
    return pred_points, global_scores, raw_debug


def metrics_for_prediction(pred_points, scores, gt_points, match_dis):
    matches, matched_pred, matched_gt = greedy_match(pred_points, scores, gt_points, match_dis)
    tp = len(matches)
    pred = len(pred_points)
    gt = len(gt_points)
    fp = pred - tp
    fn = gt - tp
    precision = tp / pred if pred else 0.0
    recall = tp / gt if gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "matches": matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred": pred,
        "gt": gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = p2p.get_args_parser()
    parser.add_argument("--paraffin_checkpoint", required=True)
    parser.add_argument("--frozen_checkpoint", required=True)
    parser.add_argument("--diagnosis_dir", default="")
    parser.add_argument("--image_names", nargs="*", default=None)
    parser.add_argument("--max_images", type=int, default=16)
    parser.add_argument("--dedup_interval", type=float, default=15)
    parser.add_argument("--max_panel_width", type=int, default=960)
    args = parser.parse_args()
    p2p.args = args

    os.makedirs(args.output_dir, exist_ok=True)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    device = 0
    dataset = build_dataset(args, "test")
    file_by_name = {os.path.basename(path): i for i, path in enumerate(dataset.files)}

    image_names = args.image_names
    if not image_names:
        image_names = select_images_from_diagnosis(args.diagnosis_dir, args.max_images)
    if not image_names:
        image_names = DEFAULT_IMAGE_NAMES[:args.max_images]
    image_names = [name for name in image_names if name in file_by_name][:args.max_images]

    para_model, para_ckpt, para_load = load_model(args, args.paraffin_checkpoint, device)
    frozen_model, frozen_ckpt, frozen_load = load_model(args, args.frozen_checkpoint, device)

    rows = []
    for image_name in image_names:
        index = file_by_name[image_name]
        raw_sample = dataset.read_data(dataset.data[index], dataset.files[index])
        raw_image = raw_sample["image"]
        gt_points = raw_sample["keypoints"].astype(np.float64)
        image_tensor, _, _ = dataset[index]

        para_points, para_scores, para_debug = predict_for_model(para_model, image_tensor, args, device)
        frozen_points, frozen_scores, frozen_debug = predict_for_model(frozen_model, image_tensor, args, device)
        para_metrics = metrics_for_prediction(para_points, para_scores, gt_points, args.match_dis)
        frozen_metrics = metrics_for_prediction(frozen_points, frozen_scores, gt_points, args.match_dis)

        base_title = f"GT only | gt={len(gt_points)} | {image_name}"
        para_title = (
            f"0318 paraffin model | P={para_metrics['precision']:.3f} "
            f"R={para_metrics['recall']:.3f} F1={para_metrics['f1']:.3f} "
            f"TP={para_metrics['tp']} FP={para_metrics['fp']} FN={para_metrics['fn']}"
        )
        frozen_title = (
            f"0326 frozen model | P={frozen_metrics['precision']:.3f} "
            f"R={frozen_metrics['recall']:.3f} F1={frozen_metrics['f1']:.3f} "
            f"TP={frozen_metrics['tp']} FP={frozen_metrics['fp']} FN={frozen_metrics['fn']}"
        )

        panels = [
            overlay_panel(raw_image, gt_points, title=base_title),
            overlay_panel(raw_image, gt_points, para_points, para_scores,
                          para_metrics["matches"], para_title),
            overlay_panel(raw_image, gt_points, frozen_points, frozen_scores,
                          frozen_metrics["matches"], frozen_title),
        ]
        canvas = stack_panels(panels, args.max_panel_width)
        out_name = os.path.splitext(image_name)[0] + "_compare.png"
        canvas.save(os.path.join(args.output_dir, out_name), quality=95)

        rows.append({
            "image_name": image_name,
            "gt": len(gt_points),
            "paraffin_pred": para_metrics["pred"],
            "paraffin_tp": para_metrics["tp"],
            "paraffin_fp": para_metrics["fp"],
            "paraffin_fn": para_metrics["fn"],
            "paraffin_precision": para_metrics["precision"],
            "paraffin_recall": para_metrics["recall"],
            "paraffin_f1": para_metrics["f1"],
            "frozen_pred": frozen_metrics["pred"],
            "frozen_tp": frozen_metrics["tp"],
            "frozen_fp": frozen_metrics["fp"],
            "frozen_fn": frozen_metrics["fn"],
            "frozen_precision": frozen_metrics["precision"],
            "frozen_recall": frozen_metrics["recall"],
            "frozen_f1": frozen_metrics["f1"],
            "extra_fp_paraffin_minus_frozen": para_metrics["fp"] - frozen_metrics["fp"],
            "paraffin_reserved_after_dedup": para_debug["reserved_after_dedup"],
            "frozen_reserved_after_dedup": frozen_debug["reserved_after_dedup"],
        })

    summary_path = os.path.join(args.output_dir, "visualized_cases.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_name"])
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(args.output_dir, "legend.txt"), "w", encoding="utf-8") as f:
        f.write("green circle = GT matched by this model\n")
        f.write("cyan circle = TP prediction\n")
        f.write("red cross = FP prediction\n")
        f.write("yellow cross = FN ground truth\n")
        f.write("left panel = GT only; middle = paraffin-trained model; right = frozen-trained model\n")

    print(json.dumps({
        "output_dir": args.output_dir,
        "num_images": len(rows),
        "summary": summary_path,
        "paraffin_checkpoint": args.paraffin_checkpoint,
        "frozen_checkpoint": args.frozen_checkpoint,
        "paraffin_missing_keys": len(getattr(para_load, "missing_keys", [])),
        "frozen_missing_keys": len(getattr(frozen_load, "missing_keys", [])),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
