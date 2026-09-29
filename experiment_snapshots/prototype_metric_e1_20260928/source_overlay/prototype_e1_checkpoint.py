import json
import os

import torch


def load_baseline_checkpoint(model, checkpoint_path, allowed_missing_prefixes=("proto_metric_adapter.",)):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError("checkpoint must contain a model state dictionary")

    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}

    model_state = model.state_dict()
    unexpected = sorted(set(state) - set(model_state))
    shape_mismatch = sorted(
        key for key in set(state).intersection(model_state)
        if tuple(state[key].shape) != tuple(model_state[key].shape)
    )
    if unexpected or shape_mismatch:
        raise RuntimeError(
            "baseline checkpoint is incompatible: "
            f"unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}"
        )

    incompatible = model.load_state_dict(state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected_after_load = sorted(incompatible.unexpected_keys)
    disallowed_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing or unexpected_after_load:
        raise RuntimeError(
            "baseline checkpoint did not initialize the full detector: "
            f"missing={disallowed_missing[:8]}, "
            f"unexpected={unexpected_after_load[:8]}"
        )
    if not state:
        raise RuntimeError("baseline checkpoint has an empty state dictionary")

    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "loaded_tensors": len(state),
        "missing_new_parameters": missing,
        "unexpected_keys": unexpected_after_load,
        "shape_mismatch_keys": shape_mismatch,
    }


def save_initialization_report(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "checkpoint_initialization_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return path
