import csv
import json
import os
from typing import Dict, List

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
    decide_rescoring,
    empty_counts,
    evaluate_image,
    fuse_binary_logits,
    load_mean_std,
    merge_counts,
    parse_alphas,
    resolve_csv_fieldnames,
    summarize_counts,
    validate_checkpoint_load_report,
)
from models.detr import build_model
from prototype_audit_sampling import stratified_round_robin_indices
from utils import load_model_weights


VERSION = "frozen_proto_detection_rescoring_v1_20260827"


def get_parser():
    parser = get_args_parser()
    parser.description = (
        "Evaluate frozen prototype logit rescoring with one target forward pass."
    )
    parser.add_argument("--bank", required=True)
    parser.add_argument(
        "--fusion_alphas", default="0,0.05,0.1,0.2,0.3,0.5"
    )
    parser.add_argument("--fusion_clip", default=2.0, type=float)
    parser.add_argument("--dedup_interval", default=15.0, type=float)
    parser.add_argument("--match_dis", default=15.0, type=float)
    parser.add_argument("--near_radius", default=30.0, type=float)
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


def _configure_frozen_bank(args, payload: Dict) -> None:
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


def main() -> None:
    args = get_parser().parse_args()
    alphas = parse_alphas(args.fusion_alphas)
    if len(alphas) < 2:
        raise ValueError("rescoring diagnostic needs alpha=0 and a non-zero alpha")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original P2P implementation")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")

    bank_payload = torch.load(args.bank, map_location="cpu")
    metadata = dict(bank_payload.get("metadata", {}))
    checkpoint_hash = file_sha256(args.checkpoint)
    expected_hash = str(metadata.get("checkpoint_sha256", ""))
    if not expected_hash or checkpoint_hash != expected_hash:
        raise RuntimeError(
            "frozen bank and P2P checkpoint SHA256 do not match; rescoring is invalid"
        )
    _configure_frozen_bank(args, bank_payload)

    model = build_model(args).to(device)
    bank_hash_before = model.prototype_head.fixed_state_sha256()
    checkpoint_report_path = os.path.join(
        args.output_dir, "checkpoint_load_report.json"
    )
    checkpoint, load_result = load_model_weights(
        args.checkpoint,
        model,
        report_path=checkpoint_report_path,
    )
    with open(checkpoint_report_path, encoding="utf-8") as handle:
        validate_checkpoint_load_report(json.load(handle))
    bank_hash_after = model.prototype_head.fixed_state_sha256()
    if bank_hash_before != bank_hash_after:
        raise RuntimeError("loading the detector checkpoint changed the frozen bank")
    model.eval()

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

    totals = {}
    for alpha in alphas:
        totals[alpha] = {
            **empty_counts(),
            "total_eval_images": 0.0,
            "empty_images": 0.0,
            "skipped_empty_images": 0.0,
            "cell_to_background": 0.0,
            "background_to_cell": 0.0,
        }
    sample_rows = []
    identity_max_abs = 0.0

    with torch.inference_mode():
        for local_index, (images, points, labels, lengths, image_paths) in enumerate(loader):
            del labels
            dataset_index = int(selected_indices[local_index])
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predicted_points = outputs["pnt_coords"][0].detach().cpu().numpy()
            raw_logits = outputs["raw_cls_logits"][0].detach().cpu().numpy()
            prototype_logits = outputs["proto_logits"][0].detach().cpu().numpy()
            height, width = images.shape[-2:]
            cross_border = (
                (predicted_points[:, 0] < 0)
                | (predicted_points[:, 0] >= width)
                | (predicted_points[:, 1] < 0)
                | (predicted_points[:, 1] >= height)
            )
            predicted_points = predicted_points[~cross_border]
            raw_logits = raw_logits[~cross_border]
            prototype_logits = prototype_logits[~cross_border]
            gt_points = (
                points[0][points[0] != -1].reshape(-1, 2).detach().cpu().numpy()
            )
            raw_classes = np.argmax(raw_logits, axis=-1)

            for alpha in alphas:
                fused_logits = fuse_binary_logits(
                    raw_logits, prototype_logits, alpha, args.fusion_clip
                )
                if alpha == 0.0:
                    identity_max_abs = max(
                        identity_max_abs,
                        float(np.max(np.abs(fused_logits - raw_logits)))
                        if len(raw_logits)
                        else 0.0,
                    )
                fused_classes = np.argmax(fused_logits, axis=-1)
                total = totals[alpha]
                total["total_eval_images"] += 1
                total["empty_images"] += float(len(gt_points) == 0)
                total["cell_to_background"] += float(
                    ((raw_classes == 0) & (fused_classes == 1)).sum()
                )
                total["background_to_cell"] += float(
                    ((raw_classes == 1) & (fused_classes == 0)).sum()
                )
                if len(gt_points) == 0 and not args.include_empty_gt_in_eval:
                    total["skipped_empty_images"] += 1
                    continue
                result = evaluate_image(
                    predicted_points,
                    fused_logits,
                    gt_points,
                    dedup_interval=args.dedup_interval,
                    match_dis=args.match_dis,
                    near_radius=args.near_radius,
                )
                merge_counts(total, result)
                sample_rows.append(
                    {
                        "alpha": alpha,
                        "dataset_index": dataset_index,
                        "image": os.path.basename(image_paths[0]),
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
            if (local_index + 1) % 25 == 0:
                print(
                    f"[Frozen-rescoring] processed={local_index + 1}/{len(loader)}",
                    flush=True,
                )

    if identity_max_abs != 0.0:
        raise RuntimeError(f"alpha=0 identity control changed logits: {identity_max_abs}")

    summary_rows = []
    for alpha in alphas:
        row = summarize_counts(totals[alpha])
        row.update(
            alpha=alpha,
            total_eval_images=totals[alpha]["total_eval_images"],
            empty_images=totals[alpha]["empty_images"],
            skipped_empty_images=totals[alpha]["skipped_empty_images"],
            cell_to_background=totals[alpha]["cell_to_background"],
            background_to_cell=totals[alpha]["background_to_cell"],
        )
        summary_rows.append({"alpha": row.pop("alpha"), **row})
    decision = decide_rescoring(summary_rows)

    _write_csv(os.path.join(args.output_dir, "rescoring_summary.csv"), summary_rows)
    _write_csv(
        os.path.join(args.output_dir, "rescoring_sample_stats.csv"),
        sample_rows,
        fieldnames=[
            "alpha",
            "dataset_index",
            "image",
            "gt",
            "pred",
            "tp",
            "fp",
            "fn",
            "fp_background_far",
            "fn_low_score_or_background",
        ],
    )
    report = {
        "version": VERSION,
        "diagnostic_only": True,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_sha256": checkpoint_hash,
        "bank": os.path.abspath(args.bank),
        "bank_id": str(bank_payload.get("bank_id", "")),
        "bank_state_unchanged": bank_hash_before == bank_hash_after,
        "dataset": os.path.abspath(args.dataset),
        "phase": args.phase,
        "sampling": {
            "strategy": "slide_group_round_robin",
            "selected_images": len(selected_indices),
        },
        "fusion_clip": args.fusion_clip,
        "alphas": alphas,
        "identity_max_abs": identity_max_abs,
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
        "summaries": summary_rows,
        "decision": decision,
        "warning": (
            "Alpha ranking uses labeled target diagnostics and must not be reported "
            "as an unbiased final test result."
        ),
    }
    with open(
        os.path.join(args.output_dir, "rescoring_report.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.output_dir, "decision.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)

    print(
        "[Frozen-rescoring] "
        f"status={decision['status']}, best_alpha={decision['best_alpha']:.3f}, "
        f"f1={decision['baseline_f1']:.6f}->{decision['best_f1']:.6f}, "
        f"gain={decision['f1_gain']:+.6f}",
        flush=True,
    )
    print(f"[Frozen-rescoring] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
