# Discriminative Dual-Proxy P2P Design

## Goal

Replace the failed foreground-only GT proxy with a training-time discriminative
embedding objective that explicitly separates reliable cell candidates from
far-background false positives. The first milestone is a mechanism gate, not a
long detector run.

## Evidence Driving the Redesign

The foreground-only proxy changed shared gradients but did not improve the
detector: frozen-domain F1 changed by -0.00043, far-background FP increased by
2323, and source-domain F1 was unchanged. Its background-to-foreground-proxy
similarity (0.9926) exceeded reliable-query similarity (0.9803), while the four
foreground proxies were highly redundant (maximum pairwise similarity 0.9518).
The failure is therefore representational, not a missing training connection.

## Architecture

Add a new isolated prototype mode named `discriminative_dual_proxy`. Existing
prototype modes and checkpoints remain loadable and unchanged.

The new head consumes the shared P2P candidate classification features and an
independent GT-point support stream. A dedicated two-layer MLP projector maps
both streams into a normalized metric space. It owns two independent proxy
banks:

- four foreground proxies, initialized and updated from exact GT-point support;
- four background proxies, initialized and updated from verified far-background
  candidates.

The head is training-only by default. Standard inference returns the original
P2P logits and points. Optional prototype-logit fusion exists only as a
diagnostic switch and defaults to zero.

## Sampling Contract

Positive support consists of exact GT-point features. Reliable positive queries
are matcher-assigned candidates whose localization error is at most 15 pixels.

Background support consists only of unmatched candidates farther than 30 pixels
from every GT point. At most 16 candidates per image are selected by raw P2P
cell score, so the bank focuses on difficult far-background false positives.

The following samples are ignored by the prototype objective:

- unmatched candidates within 30 pixels of a GT point;
- matched candidates with localization error above 15 pixels;
- padded support positions.

No near-GT candidate may be inserted into the background bank.

## Bank Lifecycle

Each class has a bounded, deterministic per-epoch reservoir to prevent dense
slides and late batches from dominating. Foreground input is capped at 64
samples per image and background input at 16 samples per image. All DDP ranks
contribute before proxy refresh.

The projector is trained by gradient descent. Proxy targets are refreshed from
detached embeddings with spherical k-means after warm-up, then updated with EMA.
Foreground and background banks are refreshed independently. Empty or
single-class refreshes are rejected instead of silently producing invalid
centers.

## Objective

The raw auxiliary loss contains four terms:

1. proxy cross-entropy over all foreground and background proxies;
2. supervised contrastive loss using foreground/background class labels;
3. foreground/background proxy separation penalty;
4. within-class assignment-balance penalty.

The detector receives the weighted auxiliary loss only after the representation
gate passes. Initial detector weight is 0.02. The main P2P matcher, regression
loss, classification loss, and inference path are unchanged.

## Two-Stage Gate

### Stage A: Frozen-Detector Representation Gate

Load the 0318 baseline and freeze Backbone, FPN, and P2P head. Train only the
projector and proxy banks for five epochs. Do not start detector training unless
all conditions pass on held-out source slides:

- TP versus far-background macro AUC >= 0.75;
- foreground-query minus foreground-proxy background similarity >= 0.05;
- background-query minus background-proxy foreground similarity >= 0.05;
- foreground and background effective proxy count >= 3.0;
- maximum same-bank pairwise proxy similarity < 0.90;
- every proxy assignment share >= 0.05.

The gate report records per-slide metrics and must not average away a failed
slide.

### Stage B: Paired Detector Gate

Run the control and dual-proxy detector from the identical 0318 checkpoint,
RNG state, data order, and ten-epoch schedule. The candidate method advances
only if:

- frozen-domain F1 is not below the paired control;
- frozen far-background FP does not increase;
- source-domain F1 decreases by no more than 0.005;
- Stage A geometry remains valid after detector training;
- prototype-to-classification gradient cosine is reported, not missing.

No 100- or 200-epoch run is launched automatically.

## Diagnostics and Failure Handling

Log class counts, rejected near-GT counts, reservoir coverage, all four loss
components, foreground/background similarity gaps, per-bank effective proxy
counts, assignment shares, pairwise similarities, gradient norm ratio, and
gradient cosine. Non-finite losses, contaminated background samples, missing
DDP contributions, or invalid banks fail fast and save a compact snapshot.

## Compatibility

The feature is opt-in. Existing command lines behave exactly as before. New
state keys are confined to the new prototype head. Loading an old P2P
checkpoint uses non-strict initialization for the new head during training;
evaluation without the new mode does not instantiate it.

## Verification

Tests must cover sampling ownership, near-GT exclusion, independent bank
refresh, deterministic reservoirs, loss ordering on a separable toy problem,
collapse detection, DDP aggregation, default inference invariance, parser and
model wiring, and run-script stop conditions. Production code is written only
after the corresponding test fails for the intended missing behavior.
