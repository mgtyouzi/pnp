import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from build_source_supervised_candidate_proto_bank import (
    StrictSourceDataset,
    file_sha256,
    get_args_parser as get_model_parser,
    slide_group_id,
    strict_collate,
)
from diagnose_cross_domain import load_checkpoint_for_eval
from frozen_proto_rescoring import (
    empty_counts,
    evaluate_image,
    merge_counts,
    resolve_csv_fieldnames,
    summarize_counts,
)
from models.detr import build_model
from sscp_selective_rescue import (
    SSCP_SELECTIVE_RESCUE_VERSION,
    audit_dataset_split,
    calculate_promotion_budget,
    decide_fixed_rescue,
    fit_precision_constrained_calibration,
    round_robin_group_indices,
    select_precision_constrained_configuration,
    selective_positive_rescue,
    validate_experiment_contract,
)
from transforms import Preprocessing


def get_parser() -> argparse.ArgumentParser:
    parser = get_model_parser()
    parser.description = (
        "Calibrate and evaluate one-way SSCP positive rescue on paraffin only."
    )
    parser.add_argument("--bank", required=True)
    parser.add_argument("--uncertainty_grid", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--strength_grid", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--evidence_clip", default=3.0, type=float)
    parser.add_argument("--dedup_interval", default=15.0, type=float)
    parser.add_argument("--match_dis", default=15.0, type=float)
    parser.add_argument("--near_radius", default=30.0, type=float)
    parser.add_argument("--minimum_f1_gain", default=0.002, type=float)
    parser.add_argument("--minimum_promotion_precision", default=0.60, type=float)
    parser.add_argument("--minimum_group_precision", default=0.50, type=float)
    parser.add_argument("--minimum_predictions_per_group", default=10, type=int)
    parser.add_argument(
        "--minimum_slides_improved_fraction", default=0.60, type=float
    )
    parser.add_argument("--promotion_budget_ratio", default=0.02, type=float)
    parser.add_argument("--promotion_budget_min", default=1, type=int)
    parser.add_argument("--promotion_budget_max", default=8, type=int)
    parser.add_argument(
        "--maximum_calibration_candidates_per_class", default=100000, type=int
    )
    parser.add_argument("--max_support_images", default=0, type=int)
    parser.add_argument("--max_selection_images", default=0, type=int)
    parser.add_argument("--max_test_images", default=0, type=int)
    return parser


def _parse_positive_grid(value: str, name: str) -> List[float]:
    values = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive values")
    return values


def _build_dataset(root: str, mean: np.ndarray, std: np.ndarray, phase: str):
    dataset = StrictSourceDataset(
        root,
        num_classes=1,
        phase=phase,
        data_transform=Preprocessing(mean, std),
    )
    paired = sorted(zip(dataset.data, dataset.files), key=lambda item: item[1])
    dataset.data = [item[0] for item in paired]
    dataset.files = [item[1] for item in paired]
    return dataset


def _group_indices(
    files: Sequence[str], groups: Iterable[str], maximum: int
) -> List[int]:
    groups = set(str(value) for value in groups)
    candidates = [
        index for index, path in enumerate(files)
        if slide_group_id(path) in groups
    ]
    local = round_robin_group_indices(
        [slide_group_id(files[index]) for index in candidates],
        int(maximum),
    )
    return [candidates[index] for index in local]


def _loader(dataset, indices: Sequence[int], num_workers: int):
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=strict_collate,
    )


def _load_training_contract(checkpoint_path: str, bank_payload: Dict):
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    args_path = os.path.join(checkpoint_dir, "args.json")
    initialization_path = os.path.join(checkpoint_dir, "run_initialization.json")
    missing = [
        path for path in (args_path, initialization_path) if not os.path.isfile(path)
    ]
    if missing:
        raise RuntimeError(
            "cannot prove checkpoint/prototype-bank provenance; missing sidecars: "
            + ", ".join(missing)
        )
    with open(args_path, encoding="utf-8") as handle:
        training_args = json.load(handle)
    with open(initialization_path, encoding="utf-8") as handle:
        initialization = json.load(handle)
    contract = validate_experiment_contract(
        training_args, initialization, bank_payload
    )
    contract.update({
        "args_path": args_path,
        "initialization_path": initialization_path,
    })
    return training_args, initialization, contract


def _configure_model(args, bank_payload: Dict, contract: Dict[str, object]) -> None:
    args.proto_enable = True
    args.proto_mode = "source_supervised_candidate_proto"
    args.proto_bank_path = args.bank
    args.proto_num_fg = int(contract["num_foreground_prototypes"])
    args.proto_num_bg = int(contract["num_background_prototypes"])
    args.proto_fg_queue_size = int(contract["foreground_queue_size"])
    args.proto_bg_queue_size = int(contract["background_queue_size"])
    args.proto_temperature = float(contract["prototype_temperature"])
    args.proto_inference_fusion = 0
    args.proto_debug_interval = 0


def _validate_embedded_checkpoint_args(
    checkpoint: object, training_args: Dict[str, object]
) -> None:
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("args"), dict):
        raise RuntimeError("checkpoint does not contain its training args")
    embedded = checkpoint["args"]
    keys = (
        "proto_mode",
        "proto_num_fg",
        "proto_num_bg",
        "proto_fg_queue_size",
        "proto_bg_queue_size",
        "proto_temperature",
    )
    differences = {
        key: {"checkpoint": embedded.get(key), "sidecar": training_args.get(key)}
        for key in keys
        if embedded.get(key) != training_args.get(key)
    }
    if differences:
        raise RuntimeError(
            "checkpoint args disagree with args.json: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )


def _extract_image(model, images, points, device):
    images = images.to(device, non_blocking=True)
    outputs = model(images)
    predicted_points = outputs["pnt_coords"][0].detach().cpu().numpy()
    raw_logits = outputs["raw_cls_logits"][0].detach().cpu().numpy()
    prototype_logits = outputs["proto_logits"][0].detach().cpu().numpy()
    height, width = images.shape[-2:]
    valid = (
        (predicted_points[:, 0] >= 0)
        & (predicted_points[:, 0] < width)
        & (predicted_points[:, 1] >= 0)
        & (predicted_points[:, 1] < height)
    )
    gt_points = points[0][points[0] != -1].reshape(-1, 2).detach().cpu().numpy()
    return predicted_points[valid], raw_logits[valid], prototype_logits[valid], gt_points


def _nearest_gt_distance(points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.zeros((0,), dtype=np.float64)
    if not len(gt_points):
        return np.full((len(points),), np.inf, dtype=np.float64)
    difference = points[:, None, :].astype(np.float64) - gt_points[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=2)).min(axis=1)


@torch.inference_mode()
def _fit_calibrations(
    model,
    loader,
    uncertainties: Sequence[float],
    args,
    device,
) -> Dict[float, Dict[str, float]]:
    samples = {
        value: {
            "positive": [], "negative": [],
            "positive_groups": [], "negative_groups": [],
            "positive_count": 0, "negative_count": 0,
        }
        for value in uncertainties
    }
    limit = int(args.maximum_calibration_candidates_per_class)
    for image_index, (images, points, labels, lengths, paths) in enumerate(loader):
        del labels, lengths
        group = slide_group_id(paths[0])
        predicted, raw, prototype, gt = _extract_image(model, images, points, device)
        raw_margin = raw[:, 0] - raw[:, 1]
        prototype_margin = prototype[:, 0] - prototype[:, 1]
        distances = _nearest_gt_distance(predicted, gt)
        for uncertainty in uncertainties:
            uncertain_background = (raw_margin < 0.0) & (raw_margin >= -uncertainty)
            for name, mask in (
                ("positive", uncertain_background & (distances <= args.match_dis)),
                ("negative", uncertain_background & (distances > args.match_dis)),
            ):
                remaining = limit - int(samples[uncertainty][f"{name}_count"])
                if remaining <= 0:
                    continue
                values = prototype_margin[mask][:remaining]
                if len(values):
                    samples[uncertainty][name].append(values.astype(np.float64))
                    samples[uncertainty][f"{name}_groups"].append(
                        np.full(len(values), group, dtype=object)
                    )
                    samples[uncertainty][f"{name}_count"] += int(len(values))
        if (image_index + 1) % 50 == 0:
            print(f"[SSCP-rescue-calibration] processed={image_index + 1}/{len(loader)}", flush=True)

    calibrations = {}
    for uncertainty in uncertainties:
        positive_parts = samples[uncertainty]["positive"]
        negative_parts = samples[uncertainty]["negative"]
        if not positive_parts or not negative_parts:
            print(
                f"[SSCP-rescue-calibration] unsupported uncertainty={uncertainty}: "
                f"positive={samples[uncertainty]['positive_count']}, "
                f"negative={samples[uncertainty]['negative_count']}",
                flush=True,
            )
            continue
        positive = np.concatenate(positive_parts)
        negative = np.concatenate(negative_parts)
        labels = np.concatenate([
            np.ones(len(positive), dtype=np.int64),
            np.zeros(len(negative), dtype=np.int64),
        ])
        margins = np.concatenate([positive, negative])
        groups = np.concatenate([
            np.concatenate(samples[uncertainty]["positive_groups"]),
            np.concatenate(samples[uncertainty]["negative_groups"]),
        ])
        try:
            calibration = fit_precision_constrained_calibration(
                labels,
                margins,
                groups,
                minimum_precision=args.minimum_promotion_precision,
                minimum_group_precision=args.minimum_group_precision,
                minimum_predictions_per_group=args.minimum_predictions_per_group,
            )
        except ValueError as error:
            print(
                f"[SSCP-rescue-calibration] unsupported uncertainty={uncertainty}: "
                f"{error}",
                flush=True,
            )
            continue
        calibration["uncertainty"] = float(uncertainty)
        calibrations[uncertainty] = calibration
    return calibrations


def _new_total() -> Dict[str, float]:
    return {
        **empty_counts(),
        "total_eval_images": 0.0,
        "empty_images": 0.0,
        "skipped_empty_images": 0.0,
        "eligible": 0.0,
        "eligible_before_budget": 0.0,
        "promotion_budget": 0.0,
        "background_to_cell": 0.0,
        "cell_to_background": 0.0,
        "promoted_positive_candidate": 0.0,
        "promoted_ignored_near": 0.0,
        "promoted_far_background": 0.0,
    }


def _promotion_budget(raw_classes: np.ndarray, args) -> int:
    raw_positive_count = int((raw_classes == 0).sum())
    return calculate_promotion_budget(
        raw_positive_count,
        ratio=args.promotion_budget_ratio,
        minimum=args.promotion_budget_min,
        maximum=args.promotion_budget_max,
    )


def _accumulate_image(
    total: Dict[str, float],
    diagnostics: Dict[str, int],
    promoted_positive: int,
    promoted_near: int,
    promoted_far: int,
    gt_is_empty: bool,
    detection_counts: Dict[str, float],
) -> None:
    total["total_eval_images"] += 1
    total["empty_images"] += float(gt_is_empty)
    for name in (
        "eligible_before_budget", "eligible", "background_to_cell",
        "cell_to_background",
    ):
        total[name] += float(diagnostics.get(name, 0))
    total["promotion_budget"] += float(diagnostics.get("budget", 0))
    total["promoted_positive_candidate"] += float(promoted_positive)
    total["promoted_ignored_near"] += float(promoted_near)
    total["promoted_far_background"] += float(promoted_far)
    if gt_is_empty:
        total["skipped_empty_images"] += 1
    else:
        merge_counts(total, detection_counts)


@torch.inference_mode()
def _evaluate_configurations(
    model,
    loader,
    configurations: Sequence[Tuple[float, float]],
    calibrations: Dict[float, Dict[str, float]],
    args,
    device,
    stage: str,
) -> List[Dict[str, float]]:
    totals = {configuration: _new_total() for configuration in configurations}
    group_totals = {configuration: {} for configuration in configurations}
    for image_index, (images, points, labels, lengths, paths) in enumerate(loader):
        del labels, lengths
        group = slide_group_id(paths[0])
        predicted, raw, prototype, gt = _extract_image(model, images, points, device)
        raw_classes = np.argmax(raw, axis=-1)
        distances = _nearest_gt_distance(predicted, gt)
        for uncertainty, strength in configurations:
            total = totals[(uncertainty, strength)]
            group_total = group_totals[(uncertainty, strength)].setdefault(
                group, _new_total()
            )
            if uncertainty == 0.0 and strength == 0.0:
                fused = raw.copy()
                diagnostics = {
                    "eligible_before_budget": 0, "eligible": 0,
                    "background_to_cell": 0, "cell_to_background": 0,
                    "budget": 0,
                }
            else:
                calibration = calibrations[uncertainty]
                fused, diagnostics = selective_positive_rescue(
                    raw,
                    prototype,
                    threshold=calibration["threshold"],
                    scale=calibration["scale"],
                    uncertainty=uncertainty,
                    strength=strength,
                    evidence_clip=args.evidence_clip,
                    max_promotions=_promotion_budget(raw_classes, args),
                )
            promoted = (raw_classes == 1) & (np.argmax(fused, axis=-1) == 0)
            promoted_positive = int((promoted & (distances <= args.match_dis)).sum())
            promoted_near = int(
                (promoted & (distances > args.match_dis)
                 & (distances <= args.near_radius)).sum()
            )
            promoted_far = int((promoted & (distances > args.near_radius)).sum())
            detection_counts = (
                evaluate_image(
                    predicted,
                    fused,
                    gt,
                    dedup_interval=args.dedup_interval,
                    match_dis=args.match_dis,
                    near_radius=args.near_radius,
                ) if len(gt) else {}
            )
            for target in (total, group_total):
                _accumulate_image(
                    target,
                    diagnostics,
                    promoted_positive,
                    promoted_near,
                    promoted_far,
                    not len(gt),
                    detection_counts,
                )
        if (image_index + 1) % 25 == 0:
            print(f"[SSCP-rescue-{stage}] processed={image_index + 1}/{len(loader)}", flush=True)

    group_summaries = {
        configuration: {
            group: summarize_counts(total)
            for group, total in values.items()
        }
        for configuration, values in group_totals.items()
    }
    baseline_by_group = group_summaries[(0.0, 0.0)]
    rows = []
    for uncertainty, strength in configurations:
        total = totals[(uncertainty, strength)]
        summary = summarize_counts(total)
        per_group = group_summaries[(uncertainty, strength)]
        f1_values = [float(value["f1"]) for value in per_group.values()]
        f1_deltas = [
            float(value["f1"]) - float(baseline_by_group[group]["f1"])
            for group, value in per_group.items()
        ]
        promotions = float(total["background_to_cell"])
        summary.update({
            "uncertainty": float(uncertainty),
            "strength": float(strength),
            "threshold": (
                float(calibrations[uncertainty]["threshold"])
                if uncertainty else ""
            ),
            "scale": (
                float(calibrations[uncertainty]["scale"])
                if uncertainty else ""
            ),
            "total_eval_images": total["total_eval_images"],
            "empty_images": total["empty_images"],
            "skipped_empty_images": total["skipped_empty_images"],
            "eligible": total["eligible"],
            "eligible_before_budget": total["eligible_before_budget"],
            "promotion_budget": total["promotion_budget"],
            "background_to_cell": total["background_to_cell"],
            "cell_to_background": total["cell_to_background"],
            "promoted_positive_candidate": total["promoted_positive_candidate"],
            "promoted_ignored_near": total["promoted_ignored_near"],
            "promoted_far_background": total["promoted_far_background"],
            "promotion_precision": (
                float(total["promoted_positive_candidate"]) / promotions
                if promotions else 1.0
            ),
            "slide_count": len(f1_values),
            "macro_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
            "minimum_slide_f1_delta": (
                float(min(f1_deltas)) if f1_deltas else float("nan")
            ),
            "slides_improved_fraction": (
                float(np.mean(np.asarray(f1_deltas) > 0.0))
                if f1_deltas else 0.0
            ),
        })
        rows.append(summary)
    return rows


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolve_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def _metric_view(row: Dict[str, object]) -> Dict[str, float]:
    keys = (
        "precision", "recall", "f1", "pred_gt_ratio", "fp_background_far",
        "fn_low_score_or_background", "background_to_cell",
        "promoted_positive_candidate", "promoted_ignored_near",
        "promoted_far_background",
        "promotion_precision", "macro_f1", "minimum_slide_f1_delta",
        "slides_improved_fraction",
    )
    return {key: float(row[key]) for key in keys}


def main() -> None:
    args = get_parser().parse_args()
    uncertainties = _parse_positive_grid(args.uncertainty_grid, "uncertainty_grid")
    strengths = _parse_positive_grid(args.strength_grid, "strength_grid")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original P2P implementation")
    os.makedirs(args.output_dir, exist_ok=True)
    if not 0 < args.promotion_budget_ratio <= 1:
        raise ValueError("promotion_budget_ratio must be in (0, 1]")
    if not 0 <= args.promotion_budget_min <= args.promotion_budget_max:
        raise ValueError("promotion budget bounds are invalid")
    device = torch.device(f"cuda:{args.gpu}")

    bank_payload = torch.load(args.bank, map_location="cpu")
    metadata = dict(bank_payload.get("metadata", {}))
    support_groups = set(metadata.get("support_groups", []))
    calibration_groups = set(metadata.get("calibration_groups", []))
    if not support_groups or not calibration_groups or support_groups & calibration_groups:
        raise RuntimeError("bank does not contain a disjoint source group split")
    training_args, initialization, experiment_contract = _load_training_contract(
        args.checkpoint, bank_payload
    )
    _configure_model(args, bank_payload, experiment_contract)
    model = build_model(args).to(device)
    checkpoint, load_result = load_checkpoint_for_eval(
        model, args.checkpoint, strict=False
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={list(load_result.missing_keys)}, "
            f"unexpected={list(load_result.unexpected_keys)}"
        )
    _validate_embedded_checkpoint_args(checkpoint, training_args)
    model.eval()
    model.requires_grad_(False)

    mean_std_path = args.mean_std_path or os.path.join(args.dataset, "mean_std.npy")
    mean, std = np.load(mean_std_path)
    train_dataset = _build_dataset(args.dataset, mean, std, "train")
    test_dataset = _build_dataset(args.dataset, mean, std, "test")
    split_audit = audit_dataset_split(
        train_dataset.files, test_dataset.files, group_fn=slide_group_id
    )
    support_indices = _group_indices(
        train_dataset.files, support_groups, args.max_support_images
    )
    selection_indices = _group_indices(
        train_dataset.files, calibration_groups, args.max_selection_images
    )
    test_indices = round_robin_group_indices(
        [slide_group_id(path) for path in test_dataset.files],
        int(args.max_test_images),
    )
    print(
        f"[SSCP-rescue] support/selection/test="
        f"{len(support_indices)}/{len(selection_indices)}/{len(test_indices)}",
        flush=True,
    )

    calibrations = _fit_calibrations(
        model,
        _loader(train_dataset, support_indices, args.num_workers),
        uncertainties,
        args,
        device,
    )
    if not calibrations:
        decision = {
            "gate_pass": False,
            "metric_gate_pass": False,
            "status": "no_precision_supported_uncertainty",
            "minimum_f1_gain": args.minimum_f1_gain,
            "minimum_promotion_precision": args.minimum_promotion_precision,
            "source_test_group_disjoint": bool(
                split_audit["source_test_group_disjoint"]
            ),
            "next_action": "stop_current_sscp_prototype",
        }
        report = {
            "version": SSCP_SELECTIVE_RESCUE_VERSION,
            "diagnostic_only": True,
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "bank": os.path.abspath(args.bank),
            "bank_id": str(bank_payload.get("bank_id", "")),
            "experiment_contract": experiment_contract,
            "dataset": os.path.abspath(args.dataset),
            "dataset_split_audit": split_audit,
            "support_groups": sorted(support_groups),
            "calibration_groups": sorted(calibration_groups),
            "unsupported_uncertainties": uncertainties,
            "decision": decision,
        }
        for name, payload in (("report.json", report), ("decision.json", decision)):
            with open(
                os.path.join(args.output_dir, name), "w", encoding="utf-8"
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"[SSCP-rescue] output={args.output_dir}", flush=True)
        return
    supported_uncertainties = sorted(calibrations)
    unsupported_uncertainties = sorted(set(uncertainties) - set(calibrations))
    configurations = [(0.0, 0.0)] + [
        (uncertainty, strength)
        for uncertainty in supported_uncertainties
        for strength in strengths
    ]
    calibration_rows = _evaluate_configurations(
        model,
        _loader(train_dataset, selection_indices, args.num_workers),
        configurations,
        calibrations,
        args,
        device,
        "selection",
    )
    selected = select_precision_constrained_configuration(
        calibration_rows,
        minimum_promotion_precision=args.minimum_promotion_precision,
        minimum_slides_improved_fraction=args.minimum_slides_improved_fraction,
    )
    fixed_configuration = (
        float(selected["uncertainty"]), float(selected["strength"])
    )
    test_configurations = [(0.0, 0.0)]
    if fixed_configuration != (0.0, 0.0):
        test_configurations.append(fixed_configuration)
    test_rows = _evaluate_configurations(
        model,
        _loader(test_dataset, test_indices, args.num_workers),
        test_configurations,
        calibrations,
        args,
        device,
        "test",
    )
    baseline = test_rows[0]
    fixed = test_rows[-1]
    decision = decide_fixed_rescue(
        baseline,
        fixed,
        minimum_f1_gain=args.minimum_f1_gain,
        minimum_promotion_precision=args.minimum_promotion_precision,
    )
    metric_gate_pass = bool(decision["gate_pass"])
    decision["metric_gate_pass"] = metric_gate_pass
    decision["source_test_group_disjoint"] = bool(
        split_audit["source_test_group_disjoint"]
    )
    decision["gate_pass"] = bool(
        metric_gate_pass and split_audit["source_test_group_disjoint"]
    )
    if metric_gate_pass and not decision["gate_pass"]:
        decision["next_action"] = "repeat_on_slide_disjoint_source_test"

    _write_csv(os.path.join(args.output_dir, "calibration_grid.csv"), calibration_rows)
    _write_csv(os.path.join(args.output_dir, "test_fixed_summary.csv"), test_rows)
    report = {
        "version": SSCP_SELECTIVE_RESCUE_VERSION,
        "diagnostic_only": True,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "bank": os.path.abspath(args.bank),
        "bank_id": str(bank_payload.get("bank_id", "")),
        "experiment_contract": experiment_contract,
        "dataset": os.path.abspath(args.dataset),
        "data_contract": "test labels never select rescue parameters",
        "dataset_split_audit": split_audit,
        "support_groups": sorted(support_groups),
        "calibration_groups": sorted(calibration_groups),
        "image_counts": {
            "support": len(support_indices),
            "selection": len(selection_indices),
            "test": len(test_indices),
        },
        "calibrations": {str(key): value for key, value in calibrations.items()},
        "unsupported_uncertainties": unsupported_uncertainties,
        "selected_configuration": selected,
        "rescue_constraints": {
            "minimum_promotion_precision": args.minimum_promotion_precision,
            "minimum_group_precision": args.minimum_group_precision,
            "minimum_predictions_per_group": args.minimum_predictions_per_group,
            "minimum_slides_improved_fraction": (
                args.minimum_slides_improved_fraction
            ),
            "promotion_budget_ratio": args.promotion_budget_ratio,
            "promotion_budget_min": args.promotion_budget_min,
            "promotion_budget_max": args.promotion_budget_max,
        },
        "test_identity": _metric_view(baseline),
        "test_fixed": _metric_view(fixed),
        "decision": decision,
    }
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[SSCP-rescue] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
