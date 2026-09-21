import argparse
import csv
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm


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
        return 0
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = 0
    for ann in data.get("annotation", []):
        labels = ann.get("label", [])
        label = labels[0] if labels else ""
        if not target_labels or any(item in label for item in target_labels):
            points += 1
    return points


def collect_samples(root, target_labels):
    samples = []
    for split in ("train", "test"):
        image_dir = root / f"{split}_image"
        point_dir = root / f"{split}_point"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            json_path = point_dir / f"{image_path.name}.json"
            points = count_points(json_path, target_labels)
            samples.append({
                "source_split": split,
                "image_path": image_path,
                "json_path": json_path,
                "image_name": image_path.name,
                "json_name": f"{image_path.name}.json",
                "wsi": extract_wsi_name(image_path.name),
                "points": points,
                "empty": points == 0,
            })
    return samples


def summarize_by_wsi(samples):
    summary = {}
    for sample in samples:
        row = summary.setdefault(sample["wsi"], {
            "wsi": sample["wsi"],
            "total_patches": 0,
            "non_empty_patches": 0,
            "empty_patches": 0,
            "total_points": 0,
        })
        row["total_patches"] += 1
        row["total_points"] += sample["points"]
        if sample["empty"]:
            row["empty_patches"] += 1
        else:
            row["non_empty_patches"] += 1
    return summary


def auto_split_wsi(summary, train_count=0, val_count=1, test_count=1):
    rows = [item for item in summary.values() if item["non_empty_patches"] > 0]
    if len(rows) < val_count + test_count + 1:
        raise RuntimeError("not enough non-empty WSI for train/val/test split")
    rows = sorted(rows, key=lambda item: (item["non_empty_patches"], item["total_points"], item["wsi"]))
    target_points = np.median([item["total_points"] for item in rows])
    ranked = sorted(
        rows,
        key=lambda item: (abs(item["total_points"] - target_points), -item["non_empty_patches"], item["wsi"]),
    )
    test = [item["wsi"] for item in ranked[:test_count]]
    remaining = [item for item in ranked[test_count:] if item["wsi"] not in test]
    val = [item["wsi"] for item in remaining[:val_count]]
    used = set(test + val)
    train = [item["wsi"] for item in rows if item["wsi"] not in used]
    if train_count > 0:
        train = train[:train_count]
    return {"train": train, "val": val, "test": test}


def parse_split_arg(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_split(args, summary):
    if args.split_json:
        with open(args.split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        return {key: list(value) for key, value in split.items()}
    if args.train_wsi or args.val_wsi or args.test_wsi:
        split = {
            "train": parse_split_arg(args.train_wsi),
            "val": parse_split_arg(args.val_wsi),
            "test": parse_split_arg(args.test_wsi),
        }
        if not split["train"] or not split["val"] or not split["test"]:
            raise RuntimeError("--train_wsi, --val_wsi and --test_wsi must all be set together")
        return split
    return auto_split_wsi(summary, val_count=args.val_count, test_count=args.test_count)


def check_no_overlap(split):
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(set(split[left]) & set(split[right]))
        if overlap:
            raise RuntimeError(f"WSI overlap between {left} and {right}: {overlap}")


def safe_link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")


def image_to_float_rgb(path):
    img = io.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    img = img.astype(np.float64, copy=False)
    if img.max(initial=0) > 1.0:
        img = img / 255.0
    return img


def compute_mean_std(image_paths):
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sumsq = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in tqdm(image_paths, desc="mean/std train"):
        img = image_to_float_rgb(path)
        flat = img.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sumsq += np.square(flat).sum(axis=0)
        pixel_count += flat.shape[0]
    mean = channel_sum / pixel_count
    var = channel_sumsq / pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32), int(pixel_count)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_dataset(args, dataset):
    src_root = dataset_root(args.project_root, args.src_data_root, dataset)
    dst_name = args.output_name or f"{dataset}-wsi"
    dst_root = dataset_root(args.project_root, args.dst_data_root, dst_name)
    target_labels = args.target_label if args.target_label is not None else TARGET_LABELS
    samples = collect_samples(src_root, target_labels)
    summary = summarize_by_wsi(samples)
    split = resolve_split(args, summary)
    check_no_overlap(split)

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"destination exists: {dst_root}; use --overwrite")
        shutil.rmtree(dst_root)
    for phase in ("train", "val", "test"):
        (dst_root / f"{phase}_image").mkdir(parents=True, exist_ok=True)
        (dst_root / f"{phase}_point").mkdir(parents=True, exist_ok=True)

    split_lookup = {}
    for phase, wsis in split.items():
        for wsi in wsis:
            split_lookup[wsi] = phase

    copied_rows = []
    skipped_rows = []
    train_images = []
    for sample in samples:
        phase = split_lookup.get(sample["wsi"])
        if phase is None:
            skipped_rows.append({**sample, "skip_reason": "wsi_not_in_split"})
            continue
        if sample["empty"] and args.exclude_empty:
            skipped_rows.append({**sample, "skip_reason": "empty_gt"})
            continue
        if not sample["json_path"].is_file():
            skipped_rows.append({**sample, "skip_reason": "missing_json"})
            continue
        dst_image = dst_root / f"{phase}_image" / sample["image_name"]
        dst_json = dst_root / f"{phase}_point" / sample["json_name"]
        safe_link_or_copy(sample["image_path"], dst_image, args.mode)
        safe_link_or_copy(sample["json_path"], dst_json, args.mode)
        if phase == "train":
            train_images.append(dst_image)
        copied_rows.append({
            "phase": phase,
            "dataset": dataset,
            "wsi": sample["wsi"],
            "points": sample["points"],
            "image_path": str(dst_image),
            "json_path": str(dst_json),
            "source_image_path": str(sample["image_path"]),
            "source_json_path": str(sample["json_path"]),
            "source_split": sample["source_split"],
        })

    if not train_images:
        raise RuntimeError("no train images copied")
    mean, std, pixel_count = compute_mean_std(train_images)
    np.save(dst_root / "mean_std.npy", np.stack([mean, std], axis=0))
    np.save(dst_root / "mean_std_train.npy", np.stack([mean, std], axis=0))

    split_rows = []
    for phase, wsis in split.items():
        for wsi in wsis:
            row = summary[wsi].copy()
            row.update({"phase": phase, "dataset": dataset})
            split_rows.append(row)
    copied_fields = [
        "phase", "dataset", "wsi", "points", "image_path", "json_path",
        "source_image_path", "source_json_path", "source_split",
    ]
    skipped_fields = [
        "source_split", "image_path", "json_path", "image_name", "json_name",
        "wsi", "points", "empty", "skip_reason",
    ]
    split_fields = [
        "dataset", "phase", "wsi", "total_patches", "non_empty_patches",
        "empty_patches", "total_points",
    ]
    write_csv(dst_root / "sample_index.csv", copied_rows, copied_fields)
    write_csv(dst_root / "skipped_samples.csv", skipped_rows, skipped_fields)
    write_csv(dst_root / "split_summary.csv", split_rows, split_fields)
    with open(dst_root / "split_wsi.json", "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    with open(dst_root / "mean_std_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "mean": mean.tolist(),
            "std": std.tolist(),
            "pixel_count": pixel_count,
            "computed_from": "train_image",
            "exclude_empty": bool(args.exclude_empty),
        }, f, ensure_ascii=False, indent=2)

    print(f"[Dataset-New] dataset={dataset} dst={dst_root}")
    print(f"[Dataset-New] split={split}")
    print(f"[Dataset-New] copied={len(copied_rows)} skipped={len(skipped_rows)}")
    print(f"[Dataset-New] mean={mean.tolist()} std={std.tolist()}")
    return dst_root


def main():
    parser = argparse.ArgumentParser(description="Create dataset-new folders with strict WSI-level train/val/test split.")
    parser.add_argument("--project_root", default="/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025")
    parser.add_argument("--src_data_root", default="datasets")
    parser.add_argument("--dst_data_root", default="dataset-new")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_name", default="")
    parser.add_argument("--split_json", default="")
    parser.add_argument("--train_wsi", default="")
    parser.add_argument("--val_wsi", default="")
    parser.add_argument("--test_wsi", default="")
    parser.add_argument("--val_count", default=1, type=int)
    parser.add_argument("--test_count", default=1, type=int)
    parser.add_argument("--mode", default="symlink", choices=("symlink", "hardlink", "copy"))
    parser.add_argument("--target_label", action="append", default=None)
    parser.add_argument("--include_empty", dest="exclude_empty", action="store_false")
    parser.add_argument("--exclude_empty", dest="exclude_empty", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    create_dataset(args, args.dataset)


if __name__ == "__main__":
    main()
