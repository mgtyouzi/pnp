import argparse
import csv
import json
import os
import re
import time
from pathlib import Path


DEFAULT_DATASETS = ["\u77f3\u8721-2025", "\u51b0\u51bb-2025"]
TARGET_LABELS = ["\u5370\u6212\u7ec6\u80de"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
WSI_RE = re.compile(r"(.+?\.kfb)_\d+\.[^.]+$")


def dataset_root(project_root, data_root, dataset):
    dataset_path = Path(dataset)
    if dataset_path.is_absolute():
        return dataset_path
    return Path(project_root) / data_root / dataset


def extract_wsi_name(image_name):
    match = WSI_RE.match(image_name)
    if match:
        return match.group(1)
    return image_name.rsplit("_", 1)[0]


def count_points(json_path, target_labels):
    if not json_path.is_file():
        return 0, False
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = 0
    for ann in data.get("annotation", []):
        labels = ann.get("label", [])
        label = labels[0] if labels else ""
        if not target_labels or any(item in label for item in target_labels):
            points += 1
    return points, True


def iter_split_samples(root, split):
    image_dir = root / f"{split}_image"
    point_dir = root / f"{split}_point"
    if not image_dir.is_dir():
        return
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        json_path = point_dir / f"{image_path.name}.json"
        yield split, image_path, json_path


def analyze_dataset(root, dataset_name, target_labels):
    rows_by_wsi = {}
    sample_rows = []
    for split in ("train", "test"):
        for source_split, image_path, json_path in iter_split_samples(root, split):
            wsi = extract_wsi_name(image_path.name)
            points, json_exists = count_points(json_path, target_labels)
            empty = points == 0
            key = (dataset_name, wsi)
            if key not in rows_by_wsi:
                rows_by_wsi[key] = {
                    "dataset": dataset_name,
                    "dataset_root": str(root),
                    "wsi": wsi,
                    "total_patches": 0,
                    "non_empty_patches": 0,
                    "empty_patches": 0,
                    "total_points": 0,
                    "train_source_patches": 0,
                    "test_source_patches": 0,
                    "missing_json": 0,
                }
            row = rows_by_wsi[key]
            row["total_patches"] += 1
            row["total_points"] += points
            row[f"{source_split}_source_patches"] += 1
            if empty:
                row["empty_patches"] += 1
            else:
                row["non_empty_patches"] += 1
            if not json_exists:
                row["missing_json"] += 1

            sample_rows.append({
                "dataset": dataset_name,
                "wsi": wsi,
                "source_split": source_split,
                "image_path": str(image_path),
                "json_path": str(json_path),
                "points": points,
                "empty": int(empty),
                "json_exists": int(json_exists),
            })

    rows = []
    for row in rows_by_wsi.values():
        denom = max(row["non_empty_patches"], 1)
        row["mean_points_per_non_empty_patch"] = row["total_points"] / denom
        rows.append(row)
    rows.sort(key=lambda item: (item["dataset"], item["wsi"]))
    sample_rows.sort(key=lambda item: (item["dataset"], item["wsi"], item["source_split"], item["image_path"]))
    return rows, sample_rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze patch distribution by WSI before WSI-level splitting.")
    parser.add_argument("--project_root", default="/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--target_label", action="append", default=None,
                        help="target label substring; can be repeated")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.project_root) / "result" / f"wsi_distribution_{timestamp}"
    target_labels = args.target_label if args.target_label is not None else TARGET_LABELS

    all_rows = []
    all_sample_rows = []
    for dataset in args.datasets:
        root = dataset_root(args.project_root, args.data_root, dataset)
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {root}")
        rows, sample_rows = analyze_dataset(root, dataset, target_labels)
        all_rows.extend(rows)
        all_sample_rows.extend(sample_rows)

    summary_fields = [
        "dataset", "dataset_root", "wsi", "total_patches", "non_empty_patches", "empty_patches",
        "total_points", "mean_points_per_non_empty_patch", "train_source_patches",
        "test_source_patches", "missing_json",
    ]
    sample_fields = [
        "dataset", "wsi", "source_split", "image_path", "json_path", "points", "empty", "json_exists",
    ]
    write_csv(output_dir / "wsi_distribution_summary.csv", all_rows, summary_fields)
    write_csv(output_dir / "wsi_sample_index.csv", all_sample_rows, sample_fields)

    print(f"[WSI-Distribution] output_dir={output_dir}")
    for row in all_rows:
        print(
            "[WSI] "
            f"dataset={row['dataset']} wsi={row['wsi']} total={row['total_patches']} "
            f"non_empty={row['non_empty_patches']} empty={row['empty_patches']} "
            f"points={row['total_points']} train_src={row['train_source_patches']} "
            f"test_src={row['test_source_patches']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
