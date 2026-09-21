import argparse
import json
import math
import os


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relative_delta(left, right):
    return (right - left) / max(abs(left), 1e-12)


def mean(values):
    return sum(values) / len(values) if values else None


def read_optional_jsonl(path):
    return read_jsonl(path) if os.path.isfile(path) else []


def fingerprints_equal(left, right):
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and bool(left.get("sha256"))
        and left.get("shape") == right.get("shape")
        and left.get("dtype") == right.get("dtype")
        and left.get("sha256") == right.get("sha256")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classification_relative_limit", default=0.0, type=float)
    parser.add_argument("--regression_relative_limit", default=0.005, type=float)
    args = parser.parse_args()

    control_init = read_json(os.path.join(args.control, "run_initialization.json"))
    prototype_init = read_json(os.path.join(args.prototype, "run_initialization.json"))
    control_rows = read_jsonl(os.path.join(args.control, "run_debug.jsonl"))
    prototype_rows = read_jsonl(os.path.join(args.prototype, "run_debug.jsonl"))
    control_first_rows = read_optional_jsonl(
        os.path.join(args.control, "first_batch_debug.jsonl")
    )
    prototype_first_rows = read_optional_jsonl(
        os.path.join(args.prototype, "first_batch_debug.jsonl")
    )
    control_by_epoch = {int(row["epoch"]): row for row in control_rows}
    prototype_by_epoch = {int(row["epoch"]): row for row in prototype_rows}
    control_first_by_epoch = {
        int(row["epoch"]): row for row in control_first_rows
    }
    prototype_first_by_epoch = {
        int(row["epoch"]): row for row in prototype_first_rows
    }
    epochs = sorted(set(control_by_epoch) & set(prototype_by_epoch))

    base_groups = sorted(
        (
            set(control_init["parameter_groups"])
            & set(prototype_init["parameter_groups"])
        )
        - {"prototype_head"}
    )
    initialization_deltas = {}
    for group in base_groups:
        control_group = control_init["parameter_groups"][group]
        prototype_group = prototype_init["parameter_groups"][group]
        initialization_deltas[group] = {
            "sum_delta": prototype_group["sum"] - control_group["sum"],
            "l2_norm_delta": prototype_group["l2_norm"] - control_group["l2_norm"],
            "elements_equal": prototype_group["elements"] == control_group["elements"],
            "sha256_equal": (
                bool(control_group.get("sha256"))
                and control_group.get("sha256") == prototype_group.get("sha256")
            ),
        }

    epoch_rows = []
    trace_equal_all = True
    rng_start_equal_all = True
    rng_end_equal_all = True
    for epoch in epochs:
        control = control_by_epoch[epoch]
        prototype = prototype_by_epoch[epoch]
        control_ranks = {
            int(item["rank"]): item for item in control.get("dataset_ranks", [])
        }
        prototype_ranks = {
            int(item["rank"]): item for item in prototype.get("dataset_ranks", [])
        }
        trace_equal = all(
            rank in prototype_ranks
            and item.get("sample_trace", [])
            == prototype_ranks[rank].get("sample_trace", [])
            for rank, item in control_ranks.items()
        ) and set(control_ranks) == set(prototype_ranks)
        rng_start_equal = control.get("rng_start") == prototype.get("rng_start")
        rng_end_equal = control.get("rng_end") == prototype.get("rng_end")
        trace_equal_all &= trace_equal
        rng_start_equal_all &= rng_start_equal
        rng_end_equal_all &= rng_end_equal

        control_cls = float(control["losses"]["classification"])
        prototype_cls = float(prototype["losses"]["classification"])
        control_reg = float(control["losses"]["regression"])
        prototype_reg = float(prototype["losses"]["regression"])
        control_grad = control.get("optimizer", {}).get(
            "optimizer_total_grad_norm_before_clip"
        )
        prototype_grad = prototype.get("optimizer", {}).get(
            "optimizer_total_grad_norm_before_clip"
        )
        control_clip = control.get("optimizer", {}).get("optimizer_clip_rate")
        prototype_clip = prototype.get("optimizer", {}).get("optimizer_clip_rate")
        epoch_rows.append(
            {
                "epoch": epoch,
                "sample_trace_equal": trace_equal,
                "rng_start_equal": rng_start_equal,
                "rng_end_equal": rng_end_equal,
                "control_classification_loss": control_cls,
                "prototype_classification_loss": prototype_cls,
                "classification_loss_delta": prototype_cls - control_cls,
                "classification_loss_relative_delta": relative_delta(
                    control_cls, prototype_cls
                ),
                "control_regression_loss": control_reg,
                "prototype_regression_loss": prototype_reg,
                "regression_loss_delta": prototype_reg - control_reg,
                "control_grad_norm": control_grad,
                "prototype_grad_norm": prototype_grad,
                "grad_norm_delta": (
                    float(prototype_grad) - float(control_grad)
                    if control_grad is not None and prototype_grad is not None
                    else None
                ),
                "control_clip_rate": control_clip,
                "prototype_clip_rate": prototype_clip,
            }
        )

    init_checkpoint_equal = (
        control_init.get("init_checkpoint")
        == prototype_init.get("init_checkpoint")
    )
    init_exact = bool(base_groups) and init_checkpoint_equal and all(
        item["elements_equal"]
        and item["sha256_equal"]
        and math.isclose(item["sum_delta"], 0.0, abs_tol=1e-8)
        and math.isclose(item["l2_norm_delta"], 0.0, abs_tol=1e-8)
        for item in initialization_deltas.values()
    )
    data_errors = {}
    for name, rows in (("control", control_rows), ("prototype", prototype_rows)):
        totals = {
            "load_or_transform_errors": 0,
            "replacement_samples": 0,
            "source_points_clipped": 0,
            "source_points_dropped": 0,
        }
        for row in rows:
            for rank_state in row.get("dataset_ranks", []):
                counts = rank_state.get("counts", {})
                for key in totals:
                    totals[key] += int(counts.get(key, 0))
        data_errors[name] = totals
    data_runtime_clean = all(
        totals["load_or_transform_errors"] == 0
        and totals["replacement_samples"] == 0
        for totals in data_errors.values()
    )
    data_sanitization_equal = (
        data_errors.get("control", {}).get("source_points_clipped")
        == data_errors.get("prototype", {}).get("source_points_clipped")
        and data_errors.get("control", {}).get("source_points_dropped")
        == data_errors.get("prototype", {}).get("source_points_dropped")
    )
    data_clean = data_runtime_clean and data_sanitization_equal
    epochs_present = bool(epochs)
    first_batch_epochs = sorted(
        set(control_first_by_epoch) & set(prototype_first_by_epoch)
    )
    first_batch_inputs_equal = bool(first_batch_epochs) and all(
        all(
            fingerprints_equal(
                control_first_by_epoch[epoch].get(key),
                prototype_first_by_epoch[epoch].get(key),
            )
            for key in ("images", "points", "labels")
        )
        for epoch in first_batch_epochs
    )
    initial_forward_equal = False
    initial_matcher_equal = False
    if first_batch_epochs:
        initial_epoch = first_batch_epochs[0]
        control_first = control_first_by_epoch[initial_epoch]
        prototype_first = prototype_first_by_epoch[initial_epoch]
        initial_forward_equal = all(
            fingerprints_equal(control_first.get(key), prototype_first.get(key))
            for key in ("pnt_coords", "raw_cls_logits")
        ) and control_first.get("base_losses") == prototype_first.get("base_losses")
        initial_matcher_equal = (
            control_first.get("matcher_source_indices")
            == prototype_first.get("matcher_source_indices")
            and control_first.get("matcher_target_indices")
            == prototype_first.get("matcher_target_indices")
        )

    attribution_valid = bool(
        epochs_present
        and init_exact
        and data_clean
        and first_batch_inputs_equal
        and initial_forward_equal
        and initial_matcher_equal
        and trace_equal_all
        and rng_start_equal_all
        and rng_end_equal_all
    )
    mean_classification_relative_delta = mean(
        [row["classification_loss_relative_delta"] for row in epoch_rows]
    )
    mean_regression_relative_delta = mean(
        [row["regression_loss_delta"] / max(abs(row["control_regression_loss"]), 1e-12) for row in epoch_rows]
    )
    optimization_gates = {
        "classification_relative_delta_within_limit": (
            mean_classification_relative_delta is not None
            and mean_classification_relative_delta <= args.classification_relative_limit
        ),
        "regression_relative_delta_within_limit": (
            mean_regression_relative_delta is not None
            and mean_regression_relative_delta <= args.regression_relative_limit
        ),
    }
    summary = {
        "control": os.path.abspath(args.control),
        "prototype": os.path.abspath(args.prototype),
        "epochs": epochs,
        "paired_epochs_present": epochs_present,
        "init_checkpoint_equal": init_checkpoint_equal,
        "base_initialization_exact_match": init_exact,
        "initialization_group_deltas": initialization_deltas,
        "sample_trace_equal_all_epochs": trace_equal_all,
        "rng_start_equal_all_epochs": rng_start_equal_all,
        "rng_end_equal_all_epochs": rng_end_equal_all,
        "data_errors": data_errors,
        "data_runtime_clean": data_runtime_clean,
        "data_sanitization_equal": data_sanitization_equal,
        "data_clean": data_clean,
        "first_batch_epochs": first_batch_epochs,
        "first_batch_inputs_equal_all_epochs": first_batch_inputs_equal,
        "initial_forward_equal": initial_forward_equal,
        "initial_matcher_equal": initial_matcher_equal,
        "mean_classification_loss_delta": mean(
            [row["classification_loss_delta"] for row in epoch_rows]
        ),
        "mean_regression_loss_delta": mean(
            [row["regression_loss_delta"] for row in epoch_rows]
        ),
        "mean_classification_loss_relative_delta": mean_classification_relative_delta,
        "mean_regression_loss_relative_delta": mean_regression_relative_delta,
        "classification_relative_limit": args.classification_relative_limit,
        "regression_relative_limit": args.regression_relative_limit,
        "optimization_gates": optimization_gates,
        "epoch_rows": epoch_rows,
        "attribution_valid": attribution_valid,
        "paired_optimization_ready": bool(
            attribution_valid and all(optimization_gates.values())
        ),
    }
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "paired_debug_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    markdown_path = os.path.join(args.output, "paired_debug_report.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("# Prototype-v2 paired mechanism check\n\n")
        handle.write(f"- Base initialization exact: `{init_exact}`\n")
        handle.write(f"- Init checkpoint exact: `{init_checkpoint_equal}`\n")
        handle.write(f"- Paired epochs present: `{epochs_present}`\n")
        handle.write(f"- Dataset clean: `{data_clean}`\n")
        handle.write(f"- Runtime data errors absent: `{data_runtime_clean}`\n")
        handle.write(f"- Data sanitization exact across runs: `{data_sanitization_equal}`\n")
        handle.write(f"- First-batch inputs exact: `{first_batch_inputs_equal}`\n")
        handle.write(f"- Initial raw P2P forward exact: `{initial_forward_equal}`\n")
        handle.write(f"- Initial matcher exact: `{initial_matcher_equal}`\n")
        handle.write(f"- Sample order exact: `{trace_equal_all}`\n")
        handle.write(f"- RNG start exact: `{rng_start_equal_all}`\n")
        handle.write(f"- RNG end exact: `{rng_end_equal_all}`\n")
        handle.write(f"- Attribution valid: `{summary['attribution_valid']}`\n\n")
        handle.write(
            f"- Mean classification relative delta: "
            f"`{mean_classification_relative_delta}` "
            f"(limit `{args.classification_relative_limit}`)\n"
        )
        handle.write(
            f"- Mean regression relative delta: `{mean_regression_relative_delta}` "
            f"(limit `{args.regression_relative_limit}`)\n"
        )
        handle.write(
            f"- Paired optimization ready: `{summary['paired_optimization_ready']}`\n\n"
        )
        handle.write("| epoch | same samples | same RNG end | cls delta | reg delta | grad delta | control/proto clip |\n")
        handle.write("|---:|---|---|---:|---:|---:|---|\n")
        for row in epoch_rows:
            grad_delta = row["grad_norm_delta"]
            handle.write(
                f"| {row['epoch']} | {row['sample_trace_equal']} | {row['rng_end_equal']} | "
                f"{row['classification_loss_delta']:.6f} | {row['regression_loss_delta']:.6f} | "
                f"{grad_delta if grad_delta is not None else 'n/a'} | "
                f"{row['control_clip_rate']}/{row['prototype_clip_rate']} |\n"
            )

    print(f"base_initialization_exact_match={init_exact}")
    print(f"init_checkpoint_equal={init_checkpoint_equal}")
    print(f"paired_epochs_present={epochs_present}")
    print(f"data_clean={data_clean}")
    print(f"data_runtime_clean={data_runtime_clean}")
    print(f"data_sanitization_equal={data_sanitization_equal}")
    print(f"first_batch_inputs_equal_all_epochs={first_batch_inputs_equal}")
    print(f"initial_forward_equal={initial_forward_equal}")
    print(f"initial_matcher_equal={initial_matcher_equal}")
    print(f"sample_trace_equal_all_epochs={trace_equal_all}")
    print(f"rng_start_equal_all_epochs={rng_start_equal_all}")
    print(f"rng_end_equal_all_epochs={rng_end_equal_all}")
    print(f"attribution_valid={summary['attribution_valid']}")
    print(f"paired_optimization_ready={summary['paired_optimization_ready']}")
    print(f"json={json_path}")
    print(f"report={markdown_path}")


if __name__ == "__main__":
    main()
