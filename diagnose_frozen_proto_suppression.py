import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from build_frozen_teacher_fg_bank import (
    deterministic_dataset,
    file_sha256,
    get_args_parser,
    slide_group_id,
)
from extract_frozen_teacher_candidate_audit import strict_teacher_collate
from frozen_proto_rescoring import (
    SUPPRESSION_VERSION,
    decide_suppression_target,
    empty_counts,
    evaluate_image,
    load_mean_std,
    merge_counts,
    parse_alphas,
    parse_quantiles,
    resolve_csv_fieldnames,
    select_source_suppression,
    summarize_distribution,
    summarize_counts,
    suppress_raw_cell_logits,
    validate_checkpoint_load_report,
    validate_suppression_calibration,
)
from models.detr import build_model
from prototype_audit_sampling import stratified_round_robin_indices
from utils import load_model_weights


SUPPRESSION_SAMPLE_FIELDS = [
    "label",
    "alpha",
    "threshold",
    "dataset_index",
    "slide_group",
    "image",
    "gt",
    "pred",
    "tp",
    "fp",
    "fn",
    "fp_background_far",
    "fn_low_score_or_background",
]


def get_parser():
    parser = get_args_parser()
    parser.description = (
        "Source-calibrate or target-test one-way frozen prototype suppression."
    )
    parser.add_argument("--bank", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("source_calibrate", "target_fixed"),
    )
    parser.add_argument("--calibration_file", default="")
    parser.add_argument("--fusion_alphas", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument(
        "--threshold_quantiles", default="0.05,0.10,0.20,0.30,0.40,0.50"
    )
    parser.add_argument("--fusion_clip", default=2.0, type=float)
    parser.add_argument("--dedup_interval", default=15.0, type=float)
    parser.add_argument("--match_dis", default=15.0, type=float)
    parser.add_argument("--near_radius", default=30.0, type=float)
    parser.add_argument("--max_source_f1_drop", default=0.001, type=float)
    parser.add_argument("--max_source_recall_drop", default=0.005, type=float)
    parser.add_argument("--target_min_f1_gain", default=0.003, type=float)
    parser.add_argument(
        "--target_min_bg_far_reduction_fraction", default=0.05, type=float
    )
    parser.add_argument("--target_max_recall_drop", default=0.005, type=float)
    parser.add_argument(
        "--target_max_cls_fn_increase_fraction", default=0.01, type=float
    )
    parser.add_argument(
        "--include_empty_gt_in_eval", action="store_true", default=False
    )
    parser.set_defaults(phase="test")
    return parser


def _write_csv(
    path: str,
    rows: List[Dict[str, object]],
    fieldnames: List[str] = None,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=resolve_csv_fieldnames(rows, fieldnames),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _configure_frozen_bank(args, payload: Dict[str, object]) -> None:
    counts = tuple(int(value) for value in payload["prototype_counts"])
    if len(counts) != 3:
        raise ValueError("frozen supervised bank needs three prototype groups")
    args.proto_enable = True
    args.proto_mode = "frozen_supervised_metric"
    args.proto_bank_path = args.bank
    args.proto_embedding_dim = int(payload["projector_weight"].shape[0])
    args.proto_num_fg, args.proto_num_hard_bg, args.proto_num_random_bg = counts
    args.proto_temperature = float(payload["temperature"])
    args.proto_initial_positive_radius = float(args.positive_radius)
    args.proto_background_radius = float(args.background_radius)
    args.proto_max_pos_per_image = int(args.max_positive_per_image)
    args.proto_max_hard_bg_per_image = int(args.max_hard_negative_per_image)
    args.proto_max_random_bg_per_image = int(args.max_random_negative_per_image)
    args.proto_hard_bg_term_weight = 2.0
    args.proto_sampling_seed = int(args.seed)
    args.proto_inference_fusion = 0
    args.proto_debug_interval = 0


def _load_model(args, device: torch.device):
    bank_payload = torch.load(args.bank, map_location="cpu")
    metadata = dict(bank_payload.get("metadata", {}))
    checkpoint_sha256 = file_sha256(args.checkpoint)
    if checkpoint_sha256 != str(metadata.get("checkpoint_sha256", "")):
        raise RuntimeError("frozen bank and detector checkpoint SHA256 do not match")
    _configure_frozen_bank(args, bank_payload)

    model = build_model(args).to(device)
    bank_state_sha256 = model.prototype_head.fixed_state_sha256()
    report_path = os.path.join(args.output_dir, "checkpoint_load_report.json")
    checkpoint, load_result = load_model_weights(
        args.checkpoint, model, report_path=report_path
    )
    with open(report_path, encoding="utf-8") as handle:
        validate_checkpoint_load_report(json.load(handle))
    if model.prototype_head.fixed_state_sha256() != bank_state_sha256:
        raise RuntimeError("loading the detector checkpoint changed the frozen bank")
    model.eval()
    return (
        model,
        checkpoint,
        load_result,
        bank_payload,
        checkpoint_sha256,
        bank_state_sha256,
    )


def _load_target_calibration(
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
    validate_suppression_calibration(
        calibration,
        checkpoint_sha256=checkpoint_sha256,
        bank_id=str(bank_payload.get("bank_id", "")),
        bank_state_sha256=bank_state_sha256,
    )
    return calibration


@torch.inference_mode()
def _extract_records(args, model, device: torch.device):
    mean_std_path = args.mean_std_path or os.path.join(
        args.dataset, "mean_std.npy"
    )
    mean, std = load_mean_std(mean_std_path)
    dataset = deterministic_dataset(args.dataset, mean, std, phase=args.phase)
    selected_indices = stratified_round_robin_indices(
        dataset.files,
        int(args.max_images),
        key=lambda path: slide_group_id(os.path.basename(path)),
    )
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=strict_teacher_collate,
    )
    records = []
    for local_index, (images, points, labels, lengths, image_paths) in enumerate(loader):
        del labels, lengths
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        predicted_points = outputs["pnt_coords"][0]
        raw_logits = outputs["raw_cls_logits"][0]
        prototype_logits = outputs["proto_logits"][0]
        height, width = images.shape[-2:]
        in_border = (
            (predicted_points[:, 0] >= 0)
            & (predicted_points[:, 0] < width)
            & (predicted_points[:, 1] >= 0)
            & (predicted_points[:, 1] < height)
        )
        gt_points = points[0][points[0] != -1].reshape(-1, 2)
        records.append(
            {
                "dataset_index": int(selected_indices[local_index]),
                "image": os.path.basename(image_paths[0]),
                "points": predicted_points[in_border].detach().cpu().numpy().astype(
                    np.float32, copy=False
                ),
                "raw_logits": raw_logits[in_border].detach().cpu().numpy().astype(
                    np.float32, copy=False
                ),
                "prototype_logits": prototype_logits[in_border]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
                "gt_points": gt_points.detach().cpu().numpy().astype(
                    np.float32, copy=False
                ),
            }
        )
        if (local_index + 1) % 25 == 0:
            print(
                f"[Frozen-suppression] extracted={local_index + 1}/{len(loader)}",
                flush=True,
            )
    if model.prototype_head.fixed_state_sha256() == "":
        raise RuntimeError("frozen bank state hash is unexpectedly empty")
    return records, selected_indices


def _evaluate_configuration(
    records,
    args,
    label: str,
    alpha: float,
    threshold: float,
    fusion_clip: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    total = {
        **empty_counts(),
        "total_eval_images": 0.0,
        "empty_images": 0.0,
        "skipped_empty_images": 0.0,
        "cell_to_background": 0.0,
        "background_to_cell": 0.0,
    }
    sample_rows = []
    identity_max_abs = 0.0
    for record in records:
        raw_logits = record["raw_logits"]
        fused_logits = suppress_raw_cell_logits(
            raw_logits,
            record["prototype_logits"],
            alpha=alpha,
            threshold=threshold,
            clip=fusion_clip,
        )
        if alpha == 0:
            identity_max_abs = max(
                identity_max_abs,
                float(np.max(np.abs(fused_logits - raw_logits)))
                if len(raw_logits)
                else 0.0,
            )
        raw_classes = np.argmax(raw_logits, axis=-1)
        fused_classes = np.argmax(fused_logits, axis=-1)
        total["cell_to_background"] += float(
            ((raw_classes == 0) & (fused_classes == 1)).sum()
        )
        total["background_to_cell"] += float(
            ((raw_classes == 1) & (fused_classes == 0)).sum()
        )
        total["total_eval_images"] += 1
        gt_points = record["gt_points"]
        total["empty_images"] += float(len(gt_points) == 0)
        if len(gt_points) == 0 and not args.include_empty_gt_in_eval:
            total["skipped_empty_images"] += 1
            continue
        result = evaluate_image(
            record["points"],
            fused_logits,
            gt_points,
            dedup_interval=args.dedup_interval,
            match_dis=args.match_dis,
            near_radius=args.near_radius,
        )
        merge_counts(total, result)
        sample_rows.append(
            {
                "label": label,
                "alpha": alpha,
                "threshold": threshold,
                "dataset_index": record["dataset_index"],
                "slide_group": slide_group_id(record["image"]),
                "image": record["image"],
                "gt": int(result["gt"]),
                "pred": int(result["pred"]),
                "tp": int(result["tp"]),
                "fp": int(result["fp"]),
                "fn": int(result["fn"]),
                "fp_background_far": int(result["fp_background_far"]),
                "fn_low_score_or_background": int(
                    result["fn_low_score_or_background"]
                ),
            }
        )
    if identity_max_abs != 0:
        raise RuntimeError(
            f"alpha=0 identity control changed logits: {identity_max_abs}"
        )
    if total["background_to_cell"] != 0:
        raise RuntimeError("one-way suppression promoted background to cell")
    summary = summarize_counts(total)
    summary.update(
        label=label,
        alpha=float(alpha),
        threshold=float(threshold),
        total_eval_images=total["total_eval_images"],
        empty_images=total["empty_images"],
        skipped_empty_images=total["skipped_empty_images"],
        cell_to_background=total["cell_to_background"],
        background_to_cell=total["background_to_cell"],
        identity_max_abs=identity_max_abs,
    )
    return {"label": summary.pop("label"), **summary}, sample_rows


def _source_thresholds(records, quantiles: List[float]):
    margin_parts = []
    for record in records:
        raw_logits = record["raw_logits"]
        raw_cell = np.argmax(raw_logits, axis=-1) == 0
        prototype_logits = record["prototype_logits"]
        margins = prototype_logits[raw_cell, 0] - prototype_logits[raw_cell, 1]
        if len(margins):
            margin_parts.append(margins.astype(np.float64, copy=False))
    if not margin_parts:
        raise RuntimeError("source sample contains no raw-cell candidates")
    margins = np.concatenate(margin_parts)
    values = np.quantile(margins, quantiles)
    pairs = []
    seen = set()
    for quantile, threshold in zip(quantiles, values):
        key = float(threshold)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((float(quantile), key))
    return pairs, summarize_distribution(margins)


def _prototype_margin_audit(records):
    parts = {"raw_cell": [], "raw_background": []}
    for record in records:
        raw_logits = record["raw_logits"]
        prototype_logits = record["prototype_logits"]
        margins = prototype_logits[:, 0] - prototype_logits[:, 1]
        raw_cell = np.argmax(raw_logits, axis=-1) == 0
        if raw_cell.any():
            parts["raw_cell"].append(margins[raw_cell])
        if (~raw_cell).any():
            parts["raw_background"].append(margins[~raw_cell])
    return {
        key: summarize_distribution(np.concatenate(values) if values else [])
        for key, values in parts.items()
    }


def _run_source_calibration(
    args,
    records,
    checkpoint,
    bank_payload,
    checkpoint_sha256: str,
    bank_state_sha256: str,
) -> None:
    alphas = parse_alphas(args.fusion_alphas)
    if len(alphas) < 2:
        raise ValueError("source calibration needs alpha=0 and non-zero alpha")
    quantiles = parse_quantiles(args.threshold_quantiles)
    threshold_pairs, margin_summary = _source_thresholds(records, quantiles)
    baseline, baseline_samples = _evaluate_configuration(
        records, args, "identity", 0.0, threshold_pairs[0][1], args.fusion_clip
    )
    baseline["threshold"] = None
    grid_rows = [baseline]
    candidate_samples = {}
    for alpha in alphas:
        if alpha == 0:
            continue
        for quantile, threshold in threshold_pairs:
            label = f"a{alpha:g}_q{quantile:g}"
            row, samples = _evaluate_configuration(
                records, args, label, alpha, threshold, args.fusion_clip
            )
            row["threshold_quantile"] = quantile
            grid_rows.append(row)
            candidate_samples[(alpha, threshold)] = samples
    selection = select_source_suppression(
        grid_rows,
        max_f1_drop=args.max_source_f1_drop,
        max_recall_drop=args.max_source_recall_drop,
    )
    selected_samples = baseline_samples
    selected_row = baseline
    if selection["status"] == "selected":
        selected_samples = candidate_samples[
            (selection["alpha"], selection["threshold"])
        ]
        selected_row = next(
            row
            for row in grid_rows
            if float(row["alpha"]) == selection["alpha"]
            and float(row["threshold"]) == selection["threshold"]
        )
    calibration = {
        "version": SUPPRESSION_VERSION,
        "mode": "source_calibrate",
        "diagnostic_only": True,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": (
            checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "bank": os.path.abspath(args.bank),
        "bank_id": str(bank_payload.get("bank_id", "")),
        "bank_state_sha256": bank_state_sha256,
        "dataset": os.path.abspath(args.dataset),
        "phase": args.phase,
        "fusion_clip": float(args.fusion_clip),
        "selection": selection,
        "constraints": {
            "max_source_f1_drop": float(args.max_source_f1_drop),
            "max_source_recall_drop": float(args.max_source_recall_drop),
        },
        "raw_cell_prototype_margin": margin_summary,
        "prototype_margin_by_raw_class": _prototype_margin_audit(records),
        "baseline": baseline,
        "selected": selected_row,
        "warning": "Parameters are selected from source labels only.",
    }
    _write_csv(os.path.join(args.output_dir, "suppression_grid.csv"), grid_rows)
    _write_csv(
        os.path.join(args.output_dir, "suppression_selected_sample_stats.csv"),
        selected_samples,
        fieldnames=SUPPRESSION_SAMPLE_FIELDS,
    )
    _write_json(
        os.path.join(args.output_dir, "suppression_calibration.json"), calibration
    )
    print(
        "[Frozen-suppression-source] "
        f"status={selection['status']}, alpha={selection['alpha']}, "
        f"threshold={selection['threshold']}",
        flush=True,
    )


def _run_target_fixed(
    args,
    records,
    checkpoint,
    bank_payload,
    checkpoint_sha256: str,
    bank_state_sha256: str,
    calibration: Dict[str, object],
) -> None:
    selection = calibration["selection"]
    fusion_clip = float(calibration["fusion_clip"])
    identity, identity_samples = _evaluate_configuration(
        records, args, "identity", 0.0, float(selection["threshold"]), fusion_clip
    )
    fixed, fixed_samples = _evaluate_configuration(
        records,
        args,
        "source_fixed",
        float(selection["alpha"]),
        float(selection["threshold"]),
        fusion_clip,
    )
    summary_rows = [identity, fixed]
    decision = decide_suppression_target(
        summary_rows,
        minimum_f1_gain=args.target_min_f1_gain,
        minimum_background_far_reduction_fraction=(
            args.target_min_bg_far_reduction_fraction
        ),
        maximum_recall_drop=args.target_max_recall_drop,
        maximum_cls_fn_increase_fraction=(
            args.target_max_cls_fn_increase_fraction
        ),
    )
    _write_csv(
        os.path.join(args.output_dir, "suppression_target_summary.csv"),
        summary_rows,
    )
    _write_csv(
        os.path.join(args.output_dir, "suppression_target_sample_stats.csv"),
        identity_samples + fixed_samples,
        fieldnames=SUPPRESSION_SAMPLE_FIELDS,
    )
    _write_json(os.path.join(args.output_dir, "decision.json"), decision)
    report = {
        "version": SUPPRESSION_VERSION,
        "mode": "target_fixed",
        "diagnostic_only": True,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": (
            checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "bank": os.path.abspath(args.bank),
        "bank_id": str(bank_payload.get("bank_id", "")),
        "bank_state_sha256": bank_state_sha256,
        "dataset": os.path.abspath(args.dataset),
        "phase": args.phase,
        "calibration_file": os.path.abspath(args.calibration_file),
        "fixed_selection": selection,
        "prototype_margin_by_raw_class": _prototype_margin_audit(records),
        "summaries": summary_rows,
        "decision": decision,
        "warning": "Target labels evaluate but never select suppression parameters.",
    }
    _write_json(os.path.join(args.output_dir, "suppression_target_report.json"), report)
    print(
        "[Frozen-suppression-target] "
        f"status={decision['status']}, f1_gain={decision['f1_gain']:+.6f}, "
        "background_to_cell=0",
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
    calibration = _load_target_calibration(
        args,
        bank_payload,
        checkpoint_sha256,
        bank_state_sha256,
    )
    records, selected_indices = _extract_records(args, model, device)
    if model.prototype_head.fixed_state_sha256() != bank_state_sha256:
        raise RuntimeError("frozen bank changed during diagnostic forward passes")
    metadata = {
        "version": SUPPRESSION_VERSION,
        "mode": args.mode,
        "selected_images": len(selected_indices),
        "sampling": "slide_group_round_robin",
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
    }
    _write_json(os.path.join(args.output_dir, "run_metadata.json"), metadata)
    if args.mode == "source_calibrate":
        _run_source_calibration(
            args,
            records,
            checkpoint,
            bank_payload,
            checkpoint_sha256,
            bank_state_sha256,
        )
    else:
        _run_target_fixed(
            args,
            records,
            checkpoint,
            bank_payload,
            checkpoint_sha256,
            bank_state_sha256,
            calibration,
        )
    print(f"[Frozen-suppression] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
