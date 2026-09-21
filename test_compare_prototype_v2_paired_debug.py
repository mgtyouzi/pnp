import json
import os
import shutil
import subprocess
import sys


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def write_jsonl(path, payloads):
    with open(path, "w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload) + "\n")


def make_run(path, sha256):
    os.makedirs(path)
    write_json(
        os.path.join(path, "run_initialization.json"),
        {
            "init_checkpoint": "/same/baseline.pth",
            "parameter_groups": {
                "backbone": {
                    "elements": 10,
                    "sum": 1.0,
                    "l2_norm": 2.0,
                    "sha256": sha256,
                }
            },
        },
    )
    write_jsonl(
        os.path.join(path, "run_debug.jsonl"),
        [
            {
                "epoch": 0,
                "rng_start": {"torch": "a"},
                "rng_end": {"torch": "b"},
                "losses": {"classification": 2.0, "regression": 3.0},
                "optimizer": {
                    "optimizer_total_grad_norm_before_clip": 1.0,
                    "optimizer_clip_rate": 0.0,
                },
                "dataset_ranks": [
                    {
                        "rank": 0,
                        "sample_trace": ["a.jpg", "b.jpg"],
                        "counts": {
                            "load_or_transform_errors": 0,
                            "replacement_samples": 0,
                            "source_points_clipped": 1,
                            "source_points_dropped": 0,
                        },
                    }
                ],
            }
        ],
    )
    tensor = {
        "shape": [1],
        "dtype": "torch.float32",
        "sha256": "tensor-hash",
        "sum": 1.0,
        "l2_norm": 1.0,
        "min": 1.0,
        "max": 1.0,
    }
    write_jsonl(
        os.path.join(path, "first_batch_debug.jsonl"),
        [
            {
                "epoch": 0,
                "images": tensor,
                "points": tensor,
                "labels": tensor,
                "pnt_coords": tensor,
                "raw_cls_logits": tensor,
                "base_losses": [1.0, 2.0],
                "matcher_source_indices": [tensor],
                "matcher_target_indices": [tensor],
            }
        ],
    )


def run_compare(root, control, prototype, name):
    output = os.path.join(root, name)
    subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "compare_prototype_v2_paired_debug.py"),
            "--control",
            control,
            "--prototype",
            prototype,
            "--output",
            output,
        ],
        check=True,
    )
    with open(
        os.path.join(output, "paired_debug_summary.json"), encoding="utf-8"
    ) as handle:
        return json.load(handle)


def main():
    root = os.path.join(os.path.dirname(__file__), ".test_compare_prototype_v2")
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    try:
        control = os.path.join(root, "control")
        prototype = os.path.join(root, "prototype")
        make_run(control, "same")
        make_run(prototype, "same")
        valid = run_compare(root, control, prototype, "valid")
        assert valid["attribution_valid"] is True
        assert valid["data_clean"] is True
        assert valid["initial_forward_equal"] is True

        init_path = os.path.join(prototype, "run_initialization.json")
        with open(init_path, encoding="utf-8") as handle:
            changed = json.load(handle)
        changed["parameter_groups"]["backbone"]["sha256"] = "different"
        write_json(init_path, changed)
        invalid = run_compare(root, control, prototype, "invalid")
        assert invalid["attribution_valid"] is False
        assert invalid["base_initialization_exact_match"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("Prototype-v2 paired debug comparison tests passed")


if __name__ == "__main__":
    main()
