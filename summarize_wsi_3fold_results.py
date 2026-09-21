import argparse
import csv
import json
import os
import statistics
from pathlib import Path


KEY_DET = "\u68c0\u6d4b\u6307\u6807"
KEY_CLS_METRICS = "\u5206\u7c7b\u6307\u6807"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_summary_string(rows, phase):
    phase_rows = [row for row in rows if row.get("phase") == phase]
    wsis = [row.get("wsi", "") for row in phase_rows]
    patches = sum(int(float(row.get("non_empty_patches", 0))) for row in phase_rows)
    points = sum(int(float(row.get("total_points", 0))) for row in phase_rows)
    density = points / patches if patches else 0.0
    return ";".join(wsis), patches, points, density


def load_metric_row(fold_dir, fold_name, checkpoint_name, eval_dir_name):
    eval_dir = fold_dir / eval_dir_name
    summary_path = eval_dir / "eval_summary.json"
    if not summary_path.is_file():
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    det = metrics.get(KEY_DET, [0.0, 0.0, 0.0])
    cls = metrics.get(KEY_CLS_METRICS, [0.0, 0.0, 0.0])
    counts = metrics.get("eval_counts", {})
    split_rows = read_csv_rows(fold_dir / "split_summary.csv")
    image_rows = read_csv_rows(eval_dir / "eval_by_image.csv")
    wsi_rows = read_csv_rows(eval_dir / "eval_by_wsi.csv")

    train_wsi, train_patches, train_points, train_density = split_summary_string(split_rows, "train")
    val_wsi, val_patches, val_points, val_density = split_summary_string(split_rows, "val")
    test_wsi, test_patches, test_points, test_density = split_summary_string(split_rows, "test")
    eval_wsi = ";".join(row.get("wsi", "") for row in wsi_rows)

    return {
        "fold": fold_name,
        "checkpoint": checkpoint_name,
        "det_precision": det[0],
        "det_recall": det[1],
        "det_f1": det[2],
        "cls_precision": cls[0],
        "cls_recall": cls[1],
        "cls_f1": cls[2],
        "mse": metrics.get("MSE", 0.0),
        "mae": metrics.get("MAE", 0.0),
        "metric_images": counts.get("metric_images", ""),
        "det_tp": counts.get("det_tp", ""),
        "det_pred": counts.get("det_pred", ""),
        "det_gt": counts.get("det_gt", ""),
        "matched_points": counts.get("matched_points", ""),
        "train_wsi": train_wsi,
        "train_patches": train_patches,
        "train_points": train_points,
        "train_density": train_density,
        "val_wsi": val_wsi,
        "val_patches": val_patches,
        "val_points": val_points,
        "val_density": val_density,
        "test_wsi": test_wsi,
        "eval_wsi": eval_wsi,
        "test_patches": test_patches,
        "test_points": test_points,
        "test_density": test_density,
        "eval_dir": str(eval_dir),
        "num_eval_images_csv": len(image_rows),
    }


def aggregate_rows(metric_rows):
    aggregate = []
    for checkpoint in sorted({row["checkpoint"] for row in metric_rows}):
        rows = [row for row in metric_rows if row["checkpoint"] == checkpoint]
        result = {"checkpoint": checkpoint, "folds": len(rows)}
        for key in ("det_precision", "det_recall", "det_f1", "cls_precision", "cls_recall", "cls_f1", "mse", "mae"):
            values = [safe_float(row[key]) for row in rows]
            result[f"{key}_mean"] = statistics.mean(values) if values else 0.0
            result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(result)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Summarize paraffin WSI 3-fold train/eval results.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--folds", nargs="+", default=["fold1", "fold2", "fold3"])
    args = parser.parse_args()

    run_root = Path(args.run_root)
    metric_rows = []
    for fold_name in args.folds:
        fold_dir = run_root / fold_name
        for checkpoint_name, eval_dir_name in (("best", "eval_best_test"), ("recent", "eval_recent_test")):
            row = load_metric_row(fold_dir, fold_name, checkpoint_name, eval_dir_name)
            if row is not None:
                metric_rows.append(row)

    if not metric_rows:
        raise RuntimeError(f"No eval_summary.json files found under {run_root}")

    metric_fields = list(metric_rows[0].keys())
    metrics_csv = run_root / "wsi_3fold_metrics.csv"
    write_csv(metrics_csv, metric_rows, metric_fields)

    aggregate = aggregate_rows(metric_rows)
    aggregate_csv = run_root / "wsi_3fold_aggregate.csv"
    write_csv(aggregate_csv, aggregate, list(aggregate[0].keys()))

    print(f"[WSI-3Fold-Summary] metrics_csv={metrics_csv}")
    print(f"[WSI-3Fold-Summary] aggregate_csv={aggregate_csv}")
    for row in metric_rows:
        print(
            "[WSI-3Fold-Metric] "
            f"fold={row['fold']} checkpoint={row['checkpoint']} "
            f"P={safe_float(row['det_precision']):.6f} "
            f"R={safe_float(row['det_recall']):.6f} "
            f"F1={safe_float(row['det_f1']):.6f} "
            f"test_wsi={row['test_wsi']}",
            flush=True,
        )
    for row in aggregate:
        print(
            "[WSI-3Fold-Aggregate] "
            f"checkpoint={row['checkpoint']} folds={row['folds']} "
            f"F1_mean={safe_float(row['det_f1_mean']):.6f} "
            f"F1_std={safe_float(row['det_f1_std']):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
