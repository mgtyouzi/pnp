# P2P + Candidate Prototype Design

## Scope

This project is isolated from SET, SET_2, SET_3, and SET-api. The first experiment is
plain P2P plus a candidate-level prototype branch. It does not include HBS, API,
HoVer-Net masks, or any change to candidate generation and point regression.

## Data flow

1. ResNet-50 and the six-level FPN produce six 256-channel feature maps.
2. P2P samples every anchor from all six levels and computes the existing learned
   classification aggregation. The resulting `cls_features` tensor is `[B, N, 256]`.
3. The original classification head maps `cls_features` to raw cell/background logits.
4. The prototype projector maps the same candidate features through
   `LayerNorm -> Linear(256,128) -> ReLU -> Linear(128,128) -> L2 normalize`.
5. The original Hungarian assignment is computed once from raw P2P predictions and is
   reused by regression loss, classification loss, and prototype sampling.

## Prototype samples

- Foreground support: a Hungarian-matched candidate is accepted only when its
  regressed point is within 10 px of the assigned GT during initialization, or within
  15 px after the bank is ready. A rejected match is ignored and can never be relabelled
  as background. If an image has more than 64 accepted candidates, deterministic
  score-quantile sampling keeps hard, medium, and easy positives.
- Ignored candidates: unmatched candidates whose predicted point or original anchor is
  less than 30 px from any GT point. This avoids treating a cell-near classification
  feature as background when its early regression output is inaccurate.
- Background support: unmatched candidates whose regressed point and original anchor
  are both at least 30 px from every GT. Each image contributes at most 16 hard
  backgrounds ranked by raw P2P foreground probability and 16 random far backgrounds.
- The source-domain point annotations are the only supervision. No target-domain labels
  or test images are used during training.

## Prototype update and loss

- The v2 default uses four foreground and eight background prototypes. This retains
  multiple cell appearances while giving the heterogeneous background more capacity.
- Accepted embeddings are appended to persistent bounded FIFO queues. Rank 0 runs
  deterministic spherical Lloyd K-means exactly once when the bank first becomes ready.
- Later epochs assign current support to the nearest existing center and update centers
  with normalized EMA (`momentum=0.99`). A center unused for three epochs is reinitialized
  from a poorly represented current sample. Rank 0 broadcasts all state to every rank.
- Candidate-to-class distance is the minimum squared Euclidean distance to that class's
  normalized prototypes.
- Prototype loss is class-balanced CE over matched positives and hard far backgrounds.
- Total loss is `P2P regression + P2P classification + 0.01 * prototype CE`.
- Prototype loss updates the projector normally, but its gradient returning to the P2P
  candidate feature is scaled by 0.1. This keeps prototype supervision auxiliary rather
  than allowing it to overwrite the detector representation.
- Epochs 0-19 train plain P2P only. Epoch 20 collects support and initializes prototypes;
  prototype CE becomes effective from epoch 21.
  The loss at epoch `t` uses prototypes finalized at epoch `t-1`, so a sample never
  supervises itself through a center built from the same epoch.

## Inference and fair comparison

- Raw mode is the primary result: inference uses the unchanged P2P logits. This tests
  whether prototype supervision improved the shared candidate features.
- Fusion mode is a post-training ablation on the same checkpoint. Prototype log odds
  are clipped and added symmetrically to cell/background logits.
- The primary fusion alpha is fixed at 0.10 before frozen-domain testing. Extra alpha
  values are sensitivity analysis only and must not be selected by frozen-test F1.
- All saved checkpoints may be evaluated in raw mode for a training-trajectory audit.
  The reportable models remain the source-domain-selected `best` checkpoint and the
  pre-registered epoch-50 checkpoint; the frozen test set never selects a checkpoint.
- Candidate coordinates, candidate count, deduplication, and 15 px evaluation matching
  remain unchanged in every mode.

## Default parameters

| Parameter | Value |
|---|---:|
| embedding dimension | 128 |
| foreground prototypes | 4 |
| background prototypes | 8 |
| foreground bank | 4096 |
| background bank | 8192 |
| temperature | 0.20 |
| prototype loss weight | 0.01 |
| P2P-only warm-up | 20 epochs |
| initialization/update positive radius | 10 / 15 px |
| near-GT ignore radius | 30 px |
| positives per image | 64 |
| hard/random backgrounds per image | 16 / 16 |
| one-time K-means iterations | 20 |
| prototype EMA momentum | 0.99 |
| dead prototype patience | 3 epochs |
| prototype-to-P2P gradient scale | 0.10 |
| training-time fusion | disabled |

## Decision rule

Compare against the established baseline and old HBS results using frozen-domain P, R,
F1, `pred_gt_ratio`, `fp_background_far`, and `fn_low_score_or_background`.
The same evaluation pass also records prototype margin separation and foreground-win
rates for GT-near candidates versus far-background candidates, so a failed experiment
does not require a second inference run for mechanism diagnosis.

- Raw F1 improves: prototype supervision transfers into the ordinary P2P branch.
- Only fused F1 improves: prototype distances are useful, but feature transfer is weak.
- FN falls while far-background FP rises sharply: prototype correction is too positive;
  reduce fusion alpha or increase hard-background coverage.
- Before a full run, a 22-epoch limited-batch smoke experiment must show exactly one
  K-means initialization, both hard and random background samples, a non-zero prototype
  loss after initialization, finite gradients, and EMA rather than repeated K-means.
- Neither raw nor fused improves after v2 passes its mechanism audit: stop prototype
  tuning and do not add HBS yet.
