import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from build_frozen_teacher_fg_bank import evaluate_separation


RECALIBRATION_VERSION = "protonce_gate_v1_20260816"


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate a saved frozen-teacher bank with the same relative "
            "ProtoNCE energy used during training."
        )
    )
    parser.add_argument("--bank", required=True)
    parser.add_argument(
        "--features",
        required=True,
        help="Existing prototype_calibration_features.npz from bank extraction.",
    )
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--output_audit", required=True)
    parser.add_argument(
        "--gpu",
        default=-1,
        type=int,
        help="Logical CUDA device for matrix scoring; use -1 for CPU.",
    )
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--negative_sample_size", default=256, type=int)
    parser.add_argument("--negative_energy_blocks", default=4, type=int)
    parser.add_argument("--negative_sampling_seed", default=0, type=int)
    parser.add_argument("--gate_auc", default=0.70, type=float)
    parser.add_argument("--gate_balanced_accuracy", default=0.65, type=float)
    parser.add_argument("--gate_min_cluster_share", default=0.05, type=float)
    return parser


def atomic_torch_save(payload: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary = output_path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output_path)


def atomic_json_save(payload: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, output_path)


def main() -> None:
    args = get_args_parser().parse_args()
    for path in (args.bank, args.features, args.audit):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if os.path.abspath(args.output_bank) == os.path.abspath(args.bank):
        raise ValueError("output_bank must differ from the original bank")
    if os.path.abspath(args.output_audit) == os.path.abspath(args.audit):
        raise ValueError("output_audit must differ from the original audit")

    payload = torch.load(args.bank, map_location="cpu")
    required = {"foreground_prototypes", "negative_bank", "bank_id", "metadata"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"bank is missing required fields: {missing}")
    with np.load(args.features) as features:
        feature_keys = {
            "calibration_positive",
            "calibration_hard_negative",
            "calibration_random_negative",
        }
        missing_features = sorted(feature_keys - set(features.files))
        if missing_features:
            raise KeyError(
                f"prototype_calibration_features.npz is missing: {missing_features}"
            )
        calibration_positive = features["calibration_positive"].astype(
            np.float32, copy=True
        )
        calibration_hard_negative = features[
            "calibration_hard_negative"
        ].astype(np.float32, copy=True)
        calibration_random_negative = features[
            "calibration_random_negative"
        ].astype(np.float32, copy=True)

    with open(args.audit, encoding="utf-8") as handle:
        audit = json.load(handle)
    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA scoring requested but CUDA is unavailable")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    negative_bank = payload["negative_bank"].float().cpu().numpy()
    metrics = evaluate_separation(
        payload["foreground_prototypes"].float().to(device),
        negative_bank,
        calibration_positive,
        calibration_hard_negative,
        calibration_random_negative,
        temperature=args.temperature,
        negative_sample_size=args.negative_sample_size,
        negative_energy_blocks=args.negative_energy_blocks,
        negative_sampling_seed=args.negative_sampling_seed,
    )
    cluster_shares = payload["metadata"].get(
        "cluster_shares", audit.get("cluster_shares", [])
    )
    if not cluster_shares:
        raise RuntimeError("cluster shares are missing from bank and audit")
    min_share = float(min(cluster_shares))
    gate_pass = bool(
        metrics["auc"] >= args.gate_auc
        and metrics["balanced_accuracy"] >= args.gate_balanced_accuracy
        and min_share >= args.gate_min_cluster_share
    )
    gate_config = {
        "auc_min": args.gate_auc,
        "balanced_accuracy_min": args.gate_balanced_accuracy,
        "cluster_share_min": args.gate_min_cluster_share,
        "temperature": args.temperature,
        "negative_sample_size": args.negative_sample_size,
        "negative_energy_blocks": args.negative_energy_blocks,
        "negative_sampling_seed": args.negative_sampling_seed,
    }

    metadata = dict(payload["metadata"])
    metadata.update(
        {
            "gate_pass": gate_pass,
            "gate_score": "protonce_relative_energy",
            "metrics": metrics,
            "gate_recalibration_version": RECALIBRATION_VERSION,
            "gate_recalibrated_from": os.path.abspath(args.bank),
            "gate_config": gate_config,
        }
    )
    corrected_payload = dict(payload)
    corrected_payload["metadata"] = metadata

    corrected_audit = dict(audit)
    corrected_audit.update(
        {
            "gate_pass": gate_pass,
            "gate_score": "protonce_relative_energy",
            "metrics": metrics,
            "gates": gate_config,
            "gate_recalibration_version": RECALIBRATION_VERSION,
            "gate_recalibrated_from": os.path.abspath(args.audit),
        }
    )
    atomic_torch_save(corrected_payload, args.output_bank)
    atomic_json_save(corrected_audit, args.output_audit)

    print(
        f"[Teacher-bank-recalibration] gate_pass={gate_pass}, "
        f"AUC={metrics['auc']:.6f}, "
        f"BA={metrics['balanced_accuracy']:.6f}, min_share={min_share:.6f}",
        flush=True,
    )
    print(f"[Teacher-bank-recalibration] output_bank={args.output_bank}")
    print(f"[Teacher-bank-recalibration] output_audit={args.output_audit}")
    if not gate_pass:
        raise RuntimeError(
            "ProtoNCE-aligned offline foreground prototype gate failed; "
            "do not start prototype training"
        )


if __name__ == "__main__":
    main()
