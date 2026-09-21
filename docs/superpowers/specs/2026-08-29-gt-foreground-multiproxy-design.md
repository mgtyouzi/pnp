# GT-Guided Foreground Multi-Proxy P2P Design

## Goal

Add a training-only prototype regularizer to P2P that improves the source-trained
cell representation without changing inference. The first experiment isolates
the prototype mechanism; HBS and prototype score fusion remain disabled.

## Non-negotiable contracts

- Both paired runs initialize from the same 0318 baseline checkpoint and reset
  Python, NumPy, CPU Torch, and CUDA RNG to the same seed.
- The original P2P classification and regression heads remain unchanged.
- GT support and detector queries use the same six-level P2P classification
  feature extraction path.
- A matched query is a positive only when its predicted point is within 15 px
  of the assigned GT point, regardless of its classification score.
- Hard background is unmatched, more than 30 px from every GT point, and chosen
  from the highest original cell scores. Near-miss and duplicate candidates are
  ignored by prototype supervision.
- Only foreground prototypes are modeled. Background samples are repelled from
  all foreground prototypes instead of being clustered into background centers.
- Prototype logits never alter `cls_logits`; inference is original P2P only.

## Architecture

The P2P forward pass extracts the normal candidate `cls_features`. During
prototype training it additionally samples the same multi-scale classification
representation at padded GT coordinates. A student projector embeds candidate
features and gradient-carrying GT support features; a momentum projector embeds
the same GT supports as stable stop-gradient teachers. Four
non-parametric foreground prototypes are initialized after five active epochs
from gathered GT support embeddings and then updated once per epoch by EMA.
Each image contributes at most 64 spatially distributed GT supports, and a
deterministic epoch reservoir prevents the last batches or dense slides from
dominating the 8192-entry support bank without consuming the training RNG.

The training loss contains student-support attraction, reliable matched-query
attraction to the support-selected prototype, hard-far-background repulsion,
support-query warm-up alignment, and a low-weight assignment-balance term.
Dead prototypes are detected from global assignment share and reinitialized
from the least represented support samples.

## Initial settings

- embedding dimension: 64
- foreground prototypes: 4
- active warm-up before prototype initialization: 5 epochs
- global support queue: 8192 embeddings
- maximum support contribution per image: 64 embeddings
- prototype EMA momentum: 0.99
- momentum projector coefficient: 0.999
- query distance gate: 15 px
- far-background distance gate: 30 px
- hard backgrounds per image: 16
- background cosine margin: 0.20
- minimum prototype assignment share: 0.05
- dead prototype patience: 2 epochs
- outer prototype loss weight: 0.02

## Diagnostics and gates

Every debug interval records sample counts, reliable/rejected matched queries,
hard-background count, support-query cosine, support/query/background losses,
proxy usage, effective K, pairwise proxy similarity, and prototype/classification
gradient ratio and cosine. Every epoch records global support counts, prototype
initialization or update events, per-prototype usage, dead ages, and reinitialization.

The paired 10-epoch experiment is the first gate. Do not start 50 or 100 epochs
unless effective K is at least 3, minimum share is at least 0.05 after warm-up,
the prototype branch is finite, source classification loss is not materially
worse than control, and frozen-domain F1 or the FP/FN trade-off improves.
