# Frozen Prototype One-Way Suppression Design

## Goal

Use the audited frozen supervised prototype bank only to suppress likely background false positives from the original P2P detector. The prototype must never promote an original background candidate to cell.

## Scope

- Keep the 0318 P2P checkpoint, coordinates, frozen bank, and detector architecture unchanged.
- Do not train P2P, HBS, or the prototype module.
- Select `threshold` and `alpha` using paraffin source validation only.
- Apply the selected pair unchanged to frozen target diagnostics.

## Fusion Contract

For raw binary logits `z` and prototype margin `m = proto_cell - proto_bg`:

1. Raw-background rows are returned bit-for-bit unchanged.
2. For raw-cell rows, compute `penalty = alpha * clip(max(threshold - m, 0), 0, fusion_clip)`.
3. Subtract half the penalty from the cell logit and add half to the background logit.
4. `alpha=0` is an exact identity control.

## Source Calibration

- Evaluate `alpha = 0, 0.05, 0.10, 0.20, 0.30`.
- Derive threshold candidates from source raw-cell prototype-margin quantiles.
- A candidate is eligible only when source F1 drops by at most `0.001` and source recall drops by at most `0.005` absolute versus alpha zero.
- Among eligible candidates, choose the largest far-background FP reduction, then higher F1, then smaller alpha and smaller threshold.
- Save checkpoint SHA256, bank identity, parameters, source metrics, and constraints in the calibration artifact.

## Target Validation

- Load exactly one source-selected parameter pair plus alpha-zero control.
- Validate checkpoint and bank identities before evaluation.
- Require zero background-to-cell flips by construction.
- Mark the method promising only if target F1 gains at least `0.003`, far-background FP falls at least 5%, recall falls by at most `0.005`, and classification FN rises by at most 1% of GT.
- Target labels may evaluate the fixed method but may not select or alter parameters.

