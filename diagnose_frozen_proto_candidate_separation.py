import argparse
import json
import os
from typing import Dict, Iterable, List

import numpy as np
import torch

from build_frozen_teacher_fg_bank import get_args_parser, slide_group_id
from diagnose_frozen_proto_suppression import (
    _extract_records,
    _load_model,
    _write_csv,
    _write_json,
)
from frozen_proto_rescoring import (
    CANDIDATE_AUDIT_VERSION,
    apply_candidate_joint_calibration,
    candidate_separation_report,
    evaluate_image,
    fit_candidate_joint_calibration,
    label_candidate_diagnostics,
    summarize_distribution,
    validate_candidate_audit_calibration,
)


CANDIDATE_FIELDS = [
    "dataset_index",
    "slide_group",
    "image",
    "category",
    "candidate_index",
    "gt_index",
    "raw_class",
    "raw_cell_probability",
    "raw_margin",
    "prototype_margin",
    "joint_score",
]


def get_parser() -> argparse.ArgumentParser:
    parser = get_args_parser()
    parser.description = (
        "Audit TP-vs-false-positive separation inside P2P cell candidates."
    )
    parser.add_argument("--bank", required=True)
    parser.add_argument(
        "--mode", required=True, choices=("source_calibrate", "target_fixed")
    )
    parser.add_argument("--calibration_file", default="")
    parser.add_argument("--beta_grid", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1")
    parser.add_argument("--dedup_interval", default=15.0, type=float)
    parser.add_argument("--match_dis", default=15.0, type=float)
    parser.add_argument("--near_radius", default=30.0, type=float)
    parser.set_defaults(phase="test")
    return parser


def _parse_beta_grid(value: str) -> List[float]:
    values = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not values or values[0] < 0 or values[-1] > 1:
        raise ValueError("--beta_grid needs comma-separated values in [0, 1]")
    return values


def _load_fixed_calibration(
    args,
    bank_payload: Dict[str, object],
    checkpoint_sha256: str,
    bank_state_sha256: str,
):
    if args.mode != "target_fixed":
        return None
    if not args.calibration_file:
        raise ValueError("target_fixed mode requires --calibration_file")
    with open(args.calibration_file, encoding="utf-8") as handle:
        calibration = json.load(handle)
    validate_candidate_audit_calibration(
        calibration,
        checkpoint_sha256=checkpoint_sha256,
        bank_id=str(bank_payload.get("bank_id", "")),
        bank_state_sha256=bank_state_sha256,
    )
    return calibration


def _build_candidate_rows(records, args) -> List[Dict[str, object]]:
    rows = []
    fp_categories = {
        "duplicate_fp",
        "near_miss_fp",
        "background_far_fp",
        "empty_image_fp",
    }
    for record in records:
        local_rows = label_candidate_diagnostics(
            record["points"],
            record["raw_logits"],
            record["prototype_logits"],
            record["gt_points"],
            dedup_interval=args.dedup_interval,
            match_dis=args.match_dis,
            near_radius=args.near_radius,
        )
        expected = evaluate_image(
            record["points"],
            record["raw_logits"],
            record["gt_points"],
            dedup_interval=args.dedup_interval,
            match_dis=args.match_dis,
            near_radius=args.near_radius,
        )
        observed_tp = sum(row["category"] == "tp" for row in local_rows)
        observed_fp = sum(row["category"] in fp_categories for row in local_rows)
        observed_low_fn = sum(row["category"] == "low_score_fn" for row in local_rows)
        if (
            observed_tp != int(expected["tp"])
            or observed_fp != int(expected["fp"])
            or observed_low_fn != int(expected["fn_low_score_or_background"])
        ):
            raise RuntimeError(
                "candidate labels diverged from P2P evaluation for "
                f"{record['image']}: observed={observed_tp}/{observed_fp}/"
                f"{observed_low_fn}, expected={int(expected['tp'])}/"
                f"{int(expected['fp'])}/"
                f"{int(expected['fn_low_score_or_background'])}"
            )
        for row in local_rows:
            rows.append(
                {
                    "dataset_index": record["dataset_index"],
                    "slide_group": slide_group_id(record["image"]),
                    "image": record["image"],
                    **row,
                }
            )
    return rows


def _distribution_report(
    rows: List[Dict[str, object]], score_names: Iterable[str]
) -> Dict[str, object]:
    report = {}
    for category in sorted(set(row["category"] for row in rows)):
        selected = [row for row in rows if row["category"] == category]
        report[category] = {
            score_name: summarize_distribution(
                np.asarray([row[score_name] for row in selected], dtype=np.float64)
            )
            for score_name in score_names
        }
    return report


def _fixed_threshold_metrics(
    rows: List[Dict[str, object]], calibration: Dict[str, object]
) -> Dict[str, float]:
    selected = [
        row for row in rows
        if row["category"] in ("tp", "background_far_fp")
    ]
    labels = np.asarray([row["category"] == "tp" for row in selected])
    scores = np.asarray([row["joint_score"] for row in selected], dtype=np.float64)
    predicted = scores >= float(calibration["joint_threshold"])
    sensitivity = float(predicted[labels].mean()) if labels.any() else float("nan")
    specificity = (
        float((~predicted[~labels]).mean()) if (~labels).any() else float("nan")
    )
    return {
        "source_fixed_threshold": float(calibration["joint_threshold"]),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
    }


def _find_auc(
    report: List[Dict[str, object]], comparison: str, score: str
) -> float:
    return float(
        next(
            row["global_auc"]
            for row in report
            if row["comparison"] == comparison and row["score"] == score
        )
    )


def _run_source_calibration(
    args,
    rows: List[Dict[str, object]],
    checkpoint,
    bank_payload: Dict[str, object],
    checkpoint_sha256: str,
    bank_state_sha256: str,
) -> None:
    calibration = fit_candidate_joint_calibration(rows, _parse_beta_grid(args.beta_grid))
    calibration.update(
        mode="source_calibrate",
        diagnostic_only=True,
        checkpoint=os.path.abspath(args.checkpoint),
        checkpoint_epoch=(
            checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        ),
        checkpoint_sha256=checkpoint_sha256,
        bank=os.path.abspath(args.bank),
        bank_id=str(bank_payload.get("bank_id", "")),
        bank_state_sha256=bank_state_sha256,
        dataset=os.path.abspath(args.dataset),
        phase=args.phase,
    )
    joint_scores = apply_candidate_joint_calibration(rows, calibration)
    for row, score in zip(rows, joint_scores):
        row["joint_score"] = float(score)
    report = candidate_separation_report(rows)
    _write_csv(
        os.path.join(args.output_dir, "candidate_rows.csv"),
        rows,
        fieldnames=CANDIDATE_FIELDS,
    )
    _write_csv(os.path.join(args.output_dir, "candidate_separation.csv"), report)
    _write_json(os.path.join(args.output_dir, "candidate_calibration.json"), calibration)
    _write_json(
        os.path.join(args.output_dir, "candidate_distributions.json"),
        _distribution_report(rows, ("raw_margin", "prototype_margin", "joint_score")),
    )
    print(
        "[Candidate-separation-source] "
        f"beta={calibration['beta']}, source_auc={calibration['source_auc']:.6f}",
        flush=True,
    )


def _run_target_fixed(
    args,
    rows: List[Dict[str, object]],
    checkpoint,
    bank_payload: Dict[str, object],
    checkpoint_sha256: str,
    bank_state_sha256: str,
    calibration: Dict[str, object],
) -> None:
    del checkpoint, bank_payload, checkpoint_sha256, bank_state_sha256
    joint_scores = apply_candidate_joint_calibration(rows, calibration)
    for row, score in zip(rows, joint_scores):
        row["joint_score"] = float(score)
    report = candidate_separation_report(rows)
    prototype_auc = _find_auc(
        report, "tp_vs_background_far_fp", "prototype_margin"
    )
    raw_auc = _find_auc(report, "tp_vs_background_far_fp", "raw_margin")
    joint_auc = _find_auc(report, "tp_vs_background_far_fp", "joint_score")
    if prototype_auc >= 0.65:
        status = "keep_and_build_gate"
    elif prototype_auc >= 0.55:
        status = "weak_diagnostic_only"
    else:
        status = "stop_current_prototype_bank"
    decision = {
        "status": status,
        "action": status,
        "diagnostic_only": True,
        "prototype_auc_tp_vs_background_far": prototype_auc,
        "raw_auc_tp_vs_background_far": raw_auc,
        "joint_auc_tp_vs_background_far": joint_auc,
        "joint_auc_gain_over_raw": joint_auc - raw_auc,
        "target_labels_used_for_selection": False,
        "fixed_threshold_metrics": _fixed_threshold_metrics(rows, calibration),
    }
    _write_csv(
        os.path.join(args.output_dir, "candidate_rows.csv"),
        rows,
        fieldnames=CANDIDATE_FIELDS,
    )
    _write_csv(os.path.join(args.output_dir, "candidate_separation.csv"), report)
    _write_json(os.path.join(args.output_dir, "decision.json"), decision)
    _write_json(
        os.path.join(args.output_dir, "candidate_distributions.json"),
        _distribution_report(rows, ("raw_margin", "prototype_margin", "joint_score")),
    )
    print(
        "[Candidate-separation-target] "
        f"status={status}, prototype_auc={prototype_auc:.6f}, "
        f"joint_gain={joint_auc - raw_auc:+.6f}",
        flush=True,
    )


def main() -> None:
    args = get_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original P2P implementation")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    (
        model,
        checkpoint,
        load_result,
        bank_payload,
        checkpoint_sha256,
        bank_state_sha256,
    ) = _load_model(args, device)
    calibration = _load_fixed_calibration(
        args, bank_payload, checkpoint_sha256, bank_state_sha256
    )
    records, selected_indices = _extract_records(args, model, device)
    rows = _build_candidate_rows(records, args)
    metadata = {
        "version": CANDIDATE_AUDIT_VERSION,
        "mode": args.mode,
        "selected_images": len(selected_indices),
        "candidate_rows": len(rows),
        "sampling": "slide_group_round_robin",
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
    }
    _write_json(os.path.join(args.output_dir, "run_metadata.json"), metadata)
    if args.mode == "source_calibrate":
        _run_source_calibration(
            args,
            rows,
            checkpoint,
            bank_payload,
            checkpoint_sha256,
            bank_state_sha256,
        )
    else:
        _run_target_fixed(
            args,
            rows,
            checkpoint,
            bank_payload,
            checkpoint_sha256,
            bank_state_sha256,
            calibration,
        )
    print(f"[Candidate-separation] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
