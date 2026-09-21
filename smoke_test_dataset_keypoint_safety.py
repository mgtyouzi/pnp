import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from dataset_zy_src import build_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Exercise real training augmentation and audit keypoint safety."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--max_source_points_dropped",
        type=int,
        default=0,
        help="maximum explicitly logged invalid source annotations allowed",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = build_dataset(
        SimpleNamespace(
            dataset=args.dataset,
            num_classes=1,
            mean_std_path=os.path.join(args.dataset, "mean_std.npy"),
            train_mean_std_path="",
            test_mean_std_path="",
            eval_split="test",
        ),
        "train",
    )
    samples = len(dataset) if args.samples <= 0 else min(args.samples, len(dataset))
    for index in range(samples):
        dataset[index]
        if (index + 1) % 200 == 0 or index + 1 == samples:
            print(f"[Dataset-safety] processed={index + 1}/{samples}", flush=True)

    state = dataset.get_debug_state(reset=False)
    counts = state["counts"]
    failures = []
    for key in ("load_or_transform_errors", "replacement_samples", "crop_size_errors"):
        if int(counts.get(key, 0)) != 0:
            failures.append(f"{key}={counts[key]}")
    dropped = int(counts.get("source_points_dropped", 0))
    if dropped > args.max_source_points_dropped:
        failures.append(f"source_points_dropped={counts['source_points_dropped']}")
    warnings = []
    if dropped > 0 and dropped <= args.max_source_points_dropped:
        warnings.append(
            f"explicitly dropped {dropped} invalid source point(s); see records"
        )

    payload = {
        "dataset": os.path.abspath(args.dataset),
        "samples_requested": samples,
        "counts": counts,
        "records": state["records"],
        "source_points_dropped_rate": dropped
        / max(int(counts.get("source_points_seen", 0)), 1),
        "status": (
            "FAIL" if failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")
        ),
        "failures": failures,
        "warnings": warnings,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise RuntimeError("dataset keypoint safety failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
