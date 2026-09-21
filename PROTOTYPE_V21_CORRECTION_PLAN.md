# Prototype-v2.1 correction plan

## Why v2 must not enter a long run

The paired diagnostic is attribution-valid, but v2 learned easy positives and
random background while hard-background correctness fell from 0.6731 to
0.2256. Effective prototypes also fell to 2.39/4 foreground and 4.26/8
background, with foreground/background mean cosine similarity 0.7696.

## Controlled changes

Prototype-v2.1 changes only the mechanisms implicated by that diagnostic:

1. Separate positive, hard-background, and random-background CE terms.
2. Use term weights 1.0, 2.0, and 0.25 respectively.
3. Sample 24 hard and 8 random background candidates per image.
4. Recluster current-epoch support, align fresh centers to persistent slots,
   then update with EMA momentum 0.9.
5. Refresh a center after two epochs below 2% assignment share.
6. Keep total prototype loss weight 0.01 and shared-feature gradient scale 0.1.
7. Keep inference fusion disabled.

The old pooled-background loss and assignment-EMA update remain available as
`pooled` and `assignment_ema` for controlled comparisons.

## Diagnostic before training

Run `run_p2p_prototype_v21_paired_diagnostic_gpu3.sh`. It executes a control
and Prototype-v2.1 from the exact same checkpoint, data order, augmentation,
and RNG state. Each run uses 8 epochs and 100 batches per epoch; this is a
mechanism test, not model training.

Proceed to a full run only when both conditions hold:

- `comparison/paired_debug_summary.json`: `attribution_valid=true`
- `prototype_v21/prototype_v2_audit.json`: `long_run_ready=true`

The mechanism gates require positive correctness >= 0.75, hard-background
correctness >= 0.50 without a drop larger than 0.10, effective foreground and
background prototypes >= 70% of configured counts, mean cross-class center
similarity <= 0.75, and a bounded non-conflicting prototype gradient.
