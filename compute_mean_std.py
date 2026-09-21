import argparse
import json
import os
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


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


def iter_images(dataset_root, split):
    splits = ["train", "test"] if split == "all" else [split]
    for item in splits:
        image_dir = dataset_root / f"{item}_image"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image directory not found: {image_dir}")
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                yield path


def compute_stats(dataset_root, split, max_images=0):
    paths = list(iter_images(dataset_root, split))
    if max_images and max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError(f"no images found for split={split} under {dataset_root}")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sumsq = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in tqdm(paths, desc=f"mean/std {dataset_root.name} {split}"):
        img = image_to_float_rgb(path)
        flat = img.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sumsq += np.square(flat).sum(axis=0)
        pixel_count += flat.shape[0]

    mean = channel_sum / pixel_count
    var = channel_sumsq / pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32), len(paths), int(pixel_count)


def load_mean_std(path):
    arr = np.load(path)
    mean, std = arr
    return mean.astype(np.float64), std.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="Compute P2P mean_std.npy without overwriting historical files by default.")
    parser.add_argument("--dataset", required=True, help="dataset name under data_root, or absolute dataset root")
    parser.add_argument("--data_root", default="./datasets", help="directory containing dataset folders")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"], help="image split used for statistics")
    parser.add_argument("--output", default="", help="output .npy path; default is dataset_root/mean_std_<split>.npy")
    parser.add_argument("--compare", default="", help="optional existing mean_std.npy to compare against")
    parser.add_argument("--max_images", default=0, type=int, help="debug only: limit image count")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting output path")
    args = parser.parse_args()

    dataset_root = Path(args.dataset) if os.path.isabs(args.dataset) else Path(args.data_root) / args.dataset
    output = Path(args.output) if args.output else dataset_root / f"mean_std_{args.split}.npy"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite or choose another --output")

    mean, std, image_count, pixel_count = compute_stats(dataset_root, args.split, args.max_images)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.stack([mean, std], axis=0))

    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "image_count": image_count,
        "pixel_count": pixel_count,
        "output": str(output),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    if args.compare:
        cmp_mean, cmp_std = load_mean_std(args.compare)
        summary["compare_path"] = args.compare
        summary["mean_delta_new_minus_compare"] = (mean.astype(np.float64) - cmp_mean).tolist()
        summary["std_delta_new_minus_compare"] = (std.astype(np.float64) - cmp_std).tolist()
        summary["mean_abs_delta"] = np.abs(mean.astype(np.float64) - cmp_mean).tolist()
        summary["std_abs_delta"] = np.abs(std.astype(np.float64) - cmp_std).tolist()

    json_path = output.with_suffix(output.suffix + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
