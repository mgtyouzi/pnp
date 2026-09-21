import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch

from candidate_conditioned_proto_data import (
    CATEGORY_BACKGROUND_FAR_FP,
    CATEGORY_ID_TO_NAME,
    CATEGORY_NEAR_MISS_FP,
    CATEGORY_TP,
    build_candidate_feature_matrix,
)
from frozen_proto_rescoring import candidate_separation_report
from models.candidate_conditioned_prototype import CandidateConditionedPrototype


SCORE_VERSION = "candidate_conditioned_proto_fixed_score_v1_20260828"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a fixed candidate-conditioned bank without refitting it."
    )
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--mode", required=True, choices=("source_test_fixed", "target_fixed")
    )
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--conditional_score_bins", default=4, type=int)
    return parser


def _write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def _score(model, features: np.ndarray, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    values = []
    model.eval()
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).float().to(device)
        values.append(model.margin(batch).cpu().numpy())
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _joint_score(
    raw: np.ndarray, prototype: np.ndarray, calibration: Dict[str, float]
) -> np.ndarray:
    raw_z = (raw - float(calibration["raw_mean"])) / float(
        calibration["raw_std"]
    )
    proto_z = (prototype - float(calibration["prototype_mean"])) / float(
        calibration["prototype_std"]
    )
    beta = float(calibration["beta"])
    return (1.0 - beta) * raw_z + beta * proto_z


def _find(report, comparison: str, score: str, metric: str) -> float:
    return float(
        next(
            row[metric]
            for row in report
            if row["comparison"] == comparison and row["score"] == score
        )
    )


def _conditional_auc(rows, bins: int) -> float:
    from frozen_proto_rescoring import _binary_roc_auc

    values = []
    selected = [
        row
        for row in rows
        if row["category"] in ("tp", "background_far_fp")
    ]
    for group in sorted(set(row["slide_group"] for row in selected)):
        local = [row for row in selected if row["slide_group"] == group]
        raw = np.asarray([row["raw_margin"] for row in local], dtype=np.float64)
        proto = np.asarray(
            [row["prototype_margin"] for row in local], dtype=np.float64
        )
        labels = np.asarray([row["category"] == "tp" for row in local])
        edges = np.unique(np.quantile(raw, np.linspace(0.0, 1.0, bins + 1)))
        assigned = np.digitize(raw, edges[1:-1] if len(edges) > 2 else [])
        for bin_index in np.unique(assigned):
            keep = assigned == bin_index
            auc = _binary_roc_auc(labels[keep], proto[keep])
            if np.isfinite(auc):
                values.append(float(auc))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    with np.load(args.archive) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    payload = torch.load(args.bank, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    if manifest.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
        raise RuntimeError("archive and prototype bank detector hashes do not match")
    feature_mode = metadata.get("feature_mode")
    features, feature_source = build_candidate_feature_matrix(data, feature_mode)
    if metadata.get("feature_source") != feature_source:
        raise RuntimeError("prototype bank feature contract does not match archive")
    available_sources = dict(manifest.get("available_feature_sources", {}))
    if available_sources.get(feature_mode) != feature_source:
        raise RuntimeError("candidate archive feature contract is incompatible")
    counts = tuple(int(value) for value in payload["prototype_counts"])
    model = CandidateConditionedPrototype(
        feat_dim=int(payload["projector_weight"].shape[1]),
        embedding_dim=int(payload["projector_weight"].shape[0]),
        prototype_counts=counts,
        temperature=float(payload["temperature"]),
    ).to(device)
    model.load_bank_file(args.bank)
    calibration = dict(metadata.get("source_train_calibration", {}))
    required_calibration = {
        "beta",
        "raw_mean",
        "raw_std",
        "prototype_mean",
        "prototype_std",
    }
    if required_calibration.difference(calibration):
        raise RuntimeError("prototype bank lacks fixed source-train calibration")

    categories = data["category"].astype(np.int64, copy=False)
    raw_margin = data["raw_margin"].astype(np.float64, copy=False)
    prototype_margin = _score(model, features, args.batch_size)
    joint = _joint_score(raw_margin, prototype_margin, calibration)
    group_names = {
        int(value): key for key, value in manifest["group_to_index"].items()
    }
    rows = []
    for index in range(len(features)):
        rows.append(
            {
                "archive_index": index,
                "slide_group": group_names[int(data["group_index"][index])],
                "image_index": int(data["image_index"][index]),
                "category": CATEGORY_ID_TO_NAME.get(
                    int(categories[index]), "unknown"
                ),
                "raw_margin": float(raw_margin[index]),
                "prototype_margin": float(prototype_margin[index]),
                "joint_score": float(joint[index]),
            }
        )
    report = candidate_separation_report(rows)
    raw_macro = _find(
        report, "tp_vs_background_far_fp", "raw_margin", "macro_slide_auc"
    )
    proto_macro = _find(
        report,
        "tp_vs_background_far_fp",
        "prototype_margin",
        "macro_slide_auc",
    )
    proto_min = _find(
        report,
        "tp_vs_background_far_fp",
        "prototype_margin",
        "min_slide_auc",
    )
    joint_macro = _find(
        report, "tp_vs_background_far_fp", "joint_score", "macro_slide_auc"
    )
    raw_near = _find(
        report, "tp_vs_near_miss_fp", "raw_margin", "macro_slide_auc"
    )
    joint_near = _find(
        report, "tp_vs_near_miss_fp", "joint_score", "macro_slide_auc"
    )
    conditional = _conditional_auc(rows, args.conditional_score_bins)
    if args.mode == "source_test_fixed":
        failures = []
        if proto_macro < 0.65:
            failures.append("prototype_macro_auc")
        if proto_min < 0.55:
            failures.append("prototype_min_slide_auc")
        if joint_macro - raw_macro < 0.01:
            failures.append("joint_gain")
        if conditional < 0.60:
            failures.append("conditional_macro_auc")
        if raw_near - joint_near > 0.01:
            failures.append("near_miss_drop")
        status = "source_test_pass" if not failures else "source_test_fail"
    else:
        failures = []
        status = "target_fixed_diagnostic_only"
    decision = {
        "version": SCORE_VERSION,
        "mode": args.mode,
        "status": status,
        "failures": failures,
        "target_labels_used_for_selection": False,
        "bank_id": payload["bank_id"],
        "feature_mode": feature_mode,
        "feature_source": feature_source,
        "fixed_source_train_beta": calibration["beta"],
        "metrics": {
            "raw_macro_auc_tp_bgfar": raw_macro,
            "prototype_macro_auc_tp_bgfar": proto_macro,
            "prototype_min_slide_auc_tp_bgfar": proto_min,
            "joint_macro_auc_tp_bgfar": joint_macro,
            "joint_gain_tp_bgfar": joint_macro - raw_macro,
            "conditional_prototype_auc": conditional,
            "raw_macro_auc_tp_near_miss": raw_near,
            "joint_macro_auc_tp_near_miss": joint_near,
            "near_miss_drop": raw_near - joint_near,
        },
    }
    _write_csv(os.path.join(args.output_dir, "candidate_scores.csv"), rows)
    _write_csv(os.path.join(args.output_dir, "candidate_separation.csv"), report)
    _write_json(os.path.join(args.output_dir, "decision.json"), decision)
    print(
        f"[Candidate-conditioned-score] status={status}, "
        f"proto_macro={proto_macro:.6f}, joint_gain={joint_macro - raw_macro:+.6f}",
        flush=True,
    )
    print(f"[Candidate-conditioned-score] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
