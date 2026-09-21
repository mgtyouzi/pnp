import argparse
import csv
import json
import os


FIELDS = [
    "mode",
    "label",
    "checkpoint_epoch",
    "precision",
    "recall",
    "f1",
    "mae",
    "pred_gt_ratio",
    "fp_background_far",
    "fn_low_score_or_background",
    "prototype_margin_separation",
    "prototype_near_fg_win_rate",
    "prototype_far_fg_win_rate",
    "checkpoint",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--top_labels", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for current, _, files in os.walk(args.root):
        if "summary.json" not in files:
            continue
        path = os.path.join(current, "summary.json")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        relative = os.path.relpath(current, args.root).replace("\\", "/")
        parts = relative.split("/", 1)
        mode = parts[0]
        label = parts[1] if len(parts) > 1 else ""
        rows.append({
            "mode": mode,
            "label": label,
            "checkpoint_epoch": payload.get("checkpoint_epoch", ""),
            "precision": payload.get("precision", ""),
            "recall": payload.get("recall", ""),
            "f1": payload.get("f1", ""),
            "mae": payload.get("mae", ""),
            "pred_gt_ratio": payload.get("pred_gt_ratio", ""),
            "fp_background_far": payload.get("fp_background_far", ""),
            "fn_low_score_or_background": payload.get("fn_low_score_or_background", ""),
            "prototype_margin_separation": payload.get("prototype_margin_separation", ""),
            "prototype_near_fg_win_rate": payload.get("prototype_near_fg_win_rate", ""),
            "prototype_far_fg_win_rate": payload.get("prototype_far_fg_win_rate", ""),
            "checkpoint": payload.get("checkpoint", ""),
        })

    rows.sort(key=lambda row: float(row["f1"] or -1), reverse=True)
    output = os.path.join(args.root, "prototype_eval_ranked.csv")
    with open(output, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"summaries={len(rows)}")
    print(f"output={output}")
    if args.top_labels > 0:
        seen = set()
        labels = []
        for row in rows:
            if row["mode"] != "raw" or row["label"] in seen:
                continue
            seen.add(row["label"])
            labels.append(row["label"])
            if len(labels) >= args.top_labels:
                break
        label_path = os.path.join(args.root, "top_raw_checkpoint_labels.txt")
        with open(label_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(labels))
            if labels:
                handle.write("\n")
        print(f"top_labels={','.join(labels)}")
        print(f"top_labels_file={label_path}")
    for row in rows[:10]:
        print(
            f"{row['mode']:>12} {row['label']:<10} epoch={row['checkpoint_epoch']} "
            f"P={float(row['precision']):.6f} R={float(row['recall']):.6f} "
            f"F1={float(row['f1']):.6f} bg_far={row['fp_background_far']} "
            f"cls_FN={row['fn_low_score_or_background']}"
        )


if __name__ == "__main__":
    main()
