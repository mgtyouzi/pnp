import csv
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import train_p2p as base
from train_p2p_no_empty_v2 import (
    KEY_CLS_METRICS,
    KEY_DET,
    build_dataset_no_empty,
    calculate_metrics_skip_empty,
    get_args_parser,
)


WSI_RE = re.compile(r"(.+?\.kfb)_\d+\.[^.]+$")


def extract_wsi_name(image_name):
    match = WSI_RE.match(image_name)
    if match:
        return match.group(1)
    return image_name.rsplit("_", 1)[0]


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def f1_score(precision, recall):
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def flatten_metrics(metrics):
    det = metrics.get(KEY_DET, [0.0, 0.0, 0.0])
    cls = metrics.get(KEY_CLS_METRICS, [0.0, 0.0, 0.0])
    counts = metrics.get("eval_counts", {})
    return {
        "det_precision": det[0],
        "det_recall": det[1],
        "det_f1": det[2],
        "cls_precision": cls[0],
        "cls_recall": cls[1],
        "cls_f1": cls[2],
        "mse": metrics.get("MSE", 0.0),
        "mae": metrics.get("MAE", 0.0),
        "eval_protocol": metrics.get("eval_protocol", "unknown"),
        "metric_images": counts.get("metric_images", ""),
        "total_eval_images": counts.get("total_eval_images", ""),
        "skipped_empty_images": counts.get("skipped_empty_images", ""),
        "det_tp": counts.get("det_tp", ""),
        "det_pred": counts.get("det_pred", ""),
        "det_gt": counts.get("det_gt", ""),
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_eval_loader(args):
    dataset_eval = base.build_dataset(args, "test")
    return DataLoader(
        dataset_eval,
        shuffle=False,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=base.collate_fn_pad,
    )


@torch.no_grad()
def evaluate_by_image_and_wsi(evaluator, model, args, rank):
    if getattr(args, "distributed", False):
        return [], []

    model.eval()
    image_rows = []
    wsi_stats = {}

    for i, (images, points, labels, lengths) in enumerate(evaluator.data_loader):
        image_path = evaluator.data_loader.dataset.files[i]
        image_name = os.path.basename(image_path)
        wsi = extract_wsi_name(image_name)
        images = images.cuda(rank, non_blocking=True)
        pd_points, pd_classes, pred_scores = evaluator.predict_scores(
            model, images, apply_deduplication=True)

        gd_points = np.zeros((0, 2), dtype=int)
        class_items = []
        for c in range(evaluator.num_classes):
            category_pd_points = pd_points[pd_classes == c]
            category_gd_points = evaluator.gds[i][c]
            category_pred_scores = pred_scores[pd_classes == c][:, 0]
            gd_points = np.concatenate([gd_points, category_gd_points], axis=0)
            class_items.append((category_pd_points, category_gd_points, category_pred_scores))

        if len(gd_points) == 0 and getattr(args, "skip_empty_eval", True):
            continue

        pred_total = 0
        gt_total = 0
        tp_total = 0
        mse_sum = 0.0
        mae_sum = 0.0
        matched_total = 0

        for category_pd_points, category_gd_points, category_pred_scores in class_items:
            pred_total += len(category_pd_points)
            gt_total += len(category_gd_points)
            if len(category_pd_points) and len(category_gd_points):
                tp, matched_pred, matched_gd = evaluator.get_tp(
                    category_pd_points,
                    category_pred_scores,
                    category_gd_points,
                    thr=args.match_dis,
                )
                tp_total += tp
                if tp > 0:
                    mse_sum += float(np.sum((matched_pred - matched_gd) ** 2))
                    mae_sum += float(np.sum(np.abs(matched_pred - matched_gd)))
                    matched_total += int(tp)

        precision = safe_div(tp_total, pred_total)
        recall = safe_div(tp_total, gt_total)
        row = {
            "image_name": image_name,
            "wsi": wsi,
            "gt": gt_total,
            "pred": pred_total,
            "tp": tp_total,
            "fp": pred_total - tp_total,
            "fn": gt_total - tp_total,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "mse": safe_div(mse_sum, matched_total),
            "mae": safe_div(mae_sum, matched_total),
            "matched_points": matched_total,
            "image_path": image_path,
        }
        image_rows.append(row)

        stat = wsi_stats.setdefault(wsi, {
            "wsi": wsi,
            "images": 0,
            "gt": 0,
            "pred": 0,
            "tp": 0,
            "mse_sum": 0.0,
            "mae_sum": 0.0,
            "matched_points": 0,
        })
        stat["images"] += 1
        stat["gt"] += gt_total
        stat["pred"] += pred_total
        stat["tp"] += tp_total
        stat["mse_sum"] += mse_sum
        stat["mae_sum"] += mae_sum
        stat["matched_points"] += matched_total

    wsi_rows = []
    for stat in sorted(wsi_stats.values(), key=lambda item: item["wsi"]):
        precision = safe_div(stat["tp"], stat["pred"])
        recall = safe_div(stat["tp"], stat["gt"])
        wsi_rows.append({
            "wsi": stat["wsi"],
            "images": stat["images"],
            "gt": stat["gt"],
            "pred": stat["pred"],
            "tp": stat["tp"],
            "fp": stat["pred"] - stat["tp"],
            "fn": stat["gt"] - stat["tp"],
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "mse": safe_div(stat["mse_sum"], stat["matched_points"]),
            "mae": safe_div(stat["mae_sum"], stat["matched_points"]),
            "matched_points": stat["matched_points"],
        })
    return image_rows, wsi_rows


def load_checkpoint(model, checkpoint_path, strict_load=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    result = model.load_state_dict(state_dict, strict=strict_load)
    epoch = checkpoint.get("epoch", "") if isinstance(checkpoint, dict) else ""
    metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    return result, epoch, metrics


def main():
    parser = get_args_parser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint path, usually best_model.pth")
    parser.add_argument("--eval_output_dir", default="", help="directory for final test metrics")
    parser.add_argument("--strict_load", action="store_true", help="load checkpoint with strict=True")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not args.eval_output_dir:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        args.eval_output_dir = os.path.join(checkpoint_dir, f"eval_{args.eval_split}_{timestamp}")
    if not args.output_dir:
        args.output_dir = args.eval_output_dir

    base.args = args
    base._original_build_dataset = base.build_dataset
    base.build_dataset = build_dataset_no_empty
    if args.skip_empty_eval:
        base.Evaluator.calculate_metrics = calculate_metrics_skip_empty

    base.init_distributed_mode(args)
    seed = args.seed + base.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    rank = args.gpu if getattr(args, "distributed", False) else 0
    if rank == 0:
        os.makedirs(args.eval_output_dir, exist_ok=True)
        print("[NoEmptyWSIEval] entry=eval_p2p_no_empty_wsi.py", flush=True)
        print(
            f"[NoEmptyWSIEval] dataset={args.dataset}, eval_split={args.eval_split}, "
            f"checkpoint={args.checkpoint}, output={args.eval_output_dir}",
            flush=True,
        )

    model = base.build_model(args).cuda(rank)
    load_result, checkpoint_epoch, checkpoint_metrics = load_checkpoint(
        model, args.checkpoint, strict_load=args.strict_load)
    if rank == 0:
        print(f"[NoEmptyWSIEval] checkpoint_epoch={checkpoint_epoch}", flush=True)
        print(f"[NoEmptyWSIEval] checkpoint_metrics={checkpoint_metrics}", flush=True)
        print(
            f"[NoEmptyWSIEval] missing_keys={len(load_result.missing_keys)}, "
            f"unexpected_keys={len(load_result.unexpected_keys)}",
            flush=True,
        )

    if getattr(args, "distributed", False):
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    data_loader = build_eval_loader(args)
    evaluator = base.Evaluator(data_loader)
    metrics = evaluator.calculate_metrics(model, effective_matching_dis=args.match_dis, rank=rank)
    image_rows, wsi_rows = evaluate_by_image_and_wsi(evaluator, model, args, rank)

    if rank == 0:
        summary_path = os.path.join(args.eval_output_dir, "eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        write_csv(
            os.path.join(args.eval_output_dir, "eval_summary.csv"),
            [flatten_metrics(metrics)],
            list(flatten_metrics(metrics).keys()),
        )
        if image_rows:
            write_csv(
                os.path.join(args.eval_output_dir, "eval_by_image.csv"),
                image_rows,
                list(image_rows[0].keys()),
            )
        if wsi_rows:
            write_csv(
                os.path.join(args.eval_output_dir, "eval_by_wsi.csv"),
                wsi_rows,
                list(wsi_rows[0].keys()),
            )
        print(f"[NoEmptyWSIEval] metrics={metrics}", flush=True)
        print(f"[NoEmptyWSIEval] summary_path={summary_path}", flush=True)

    if getattr(args, "distributed", False):
        base.cleanup()


if __name__ == "__main__":
    main()
