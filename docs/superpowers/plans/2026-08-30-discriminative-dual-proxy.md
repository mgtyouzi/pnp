# Discriminative Dual-Proxy P2P Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and gate a foreground/background dual-proxy metric head that must prove source-slide candidate separation before it can affect P2P detector training.

**Architecture:** Add an isolated `discriminative_dual_proxy` mode with a dedicated normalized MLP embedding, independent foreground/background reservoirs, spherical proxy refresh, and discriminative losses. A five-epoch frozen-detector gate runs first; only a passing bank can start a ten-epoch paired detector experiment, while standard inference remains raw P2P.

**Tech Stack:** Python 3.8, PyTorch 1.10, existing P2P/DETR model, torch.distributed, NumPy/CSV/JSON, shell launchers.

## Global Constraints

- Preserve every existing prototype mode and checkpoint contract.
- Use exact GT points and matcher-localized queries as foreground evidence.
- Background samples must be unmatched and farther than 30 pixels from every GT point.
- Ignore unmatched near-GT candidates and matched candidates farther than 15 pixels from their GT.
- Use four foreground and four background proxies with 64-dimensional embeddings.
- Standard inference must return unmodified P2P logits; diagnostic fusion defaults to zero.
- Do not launch 100- or 200-epoch training from this pipeline.
- The local directory is not a Git repository, so each task ends with an explicit test checkpoint instead of a commit.

---

### Task 1: Dual-Class Sampling and Reservoirs

**Files:**
- Create: `models/discriminative_dual_proxy.py`
- Create: `test_discriminative_dual_proxy.py`

**Interfaces:**
- Produces: `DiscriminativeDualProxy(feat_dim: int, embedding_dim: int = 64, num_fg: int = 4, num_bg: int = 4, ...)`.
- Produces: `sample_candidates(points, raw_logits, targets, indices) -> Dict[str, List[Tensor]]`.
- Produces: `begin_epoch()`, `_cache_foreground(Tensor)`, `_cache_background(Tensor)`.

- [ ] **Step 1: Write failing sampling tests**

Create toy candidates containing reliable matches, far matched candidates, unmatched near-GT candidates, and unmatched far-background candidates. Assert that only reliable matches are positive, only unmatched far candidates are background, and no candidate appears in both sets.

```python
sampled = head.sample_candidates(points, raw_logits, targets, indices)
assert sampled["reliable_indices"][0].tolist() == [0]
assert sampled["background_indices"][0].tolist() == [3, 4]
assert sampled["ignored_near_indices"][0].tolist() == [2]
assert sampled["rejected_match_indices"][0].tolist() == [1]
```

- [ ] **Step 2: Verify RED**

Run: `python test_discriminative_dual_proxy.py`

Expected: import failure for `models.discriminative_dual_proxy`.

- [ ] **Step 3: Implement strict candidate ownership**

Implement distance-based masks before score-based top-k. Maintain `matched_mask`; compute nearest GT distance for all candidates; select far-background only from `(~matched_mask) & (nearest > background_radius)`.

```python
hard_eligible = (~matched_mask) & (nearest > self.background_radius)
hard_indices = torch.where(hard_eligible)[0]
selected = hard_indices[
    torch.topk(cell_scores[hard_indices], min(limit, hard_indices.numel())).indices
]
```

- [ ] **Step 4: Add independent deterministic reservoirs**

Register `foreground_queue`, `foreground_priorities`, `background_queue`, and `background_priorities`. Use the existing hash-priority reservoir pattern with different class seeds. Cap per-image foreground support at 64 and background support at 16.

- [ ] **Step 5: Verify GREEN**

Run: `python test_discriminative_dual_proxy.py`

Expected: sampling and bounded-reservoir tests pass.

---

### Task 2: Dual Proxy Refresh, Losses, and Collapse Metrics

**Files:**
- Modify: `models/discriminative_dual_proxy.py`
- Modify: `test_discriminative_dual_proxy.py`

**Interfaces:**
- Produces: `forward(query_features, raw_logits, support_features=None) -> Dict[str, Tensor]`.
- Produces: `compute_loss_and_cache(...) -> Tuple[Tensor, Dict[str, float]]`.
- Produces: `finalize_epoch() -> Dict[str, float]`.

- [ ] **Step 1: Write failing geometry and loss tests**

Use a separable two-dimensional toy embedding. Assert that the correct class proxy loss is lower than a label-swapped loss, background embeddings select background proxies, and collapsed banks fail geometry checks.

```python
loss_good, diag_good = head.compute_metric_loss(fg, bg)
loss_bad, _ = head.compute_metric_loss(bg, fg)
assert loss_good < loss_bad
assert diag_good["dualproxy_fg_bg_similarity_gap"] > 0
```

- [ ] **Step 2: Verify RED**

Run: `python test_discriminative_dual_proxy.py`

Expected: missing refresh/loss methods or diagnostic keys.

- [ ] **Step 3: Implement normalized projector and proxy refresh**

Use `LayerNorm -> Linear -> ReLU -> Linear -> L2 normalize`. Initialize each bank independently with deterministic spherical k-means. Reject refresh when either bank has fewer samples than its proxy count.

- [ ] **Step 4: Implement the four-term objective**

Concatenate foreground and background proxies. Targets for foreground samples are the best foreground proxy index; background targets are offset by `num_fg`. Compute proxy cross-entropy, supervised contrastive loss, cross-bank separation, and within-bank balance.

```python
all_proxies = torch.cat([fg_proxies, bg_proxies], dim=0)
logits = embeddings @ all_proxies.t() / self.temperature
proxy_ce = F.cross_entropy(logits, proxy_targets)
separation = F.relu(fg_proxies @ bg_proxies.t() - self.separation_margin).square().mean()
loss = proxy_ce + self.supcon_weight * supcon + self.separation_weight * separation + self.balance_weight * balance
```

- [ ] **Step 5: Emit complete geometry diagnostics**

Report class counts, losses, foreground-query and background-query margins, per-bank effective proxy count, minimum assignment share, maximum same-bank pairwise similarity, and cross-bank maximum similarity.

- [ ] **Step 6: Verify GREEN**

Run: `python test_discriminative_dual_proxy.py`

Expected: all tensor-level tests pass with finite gradients in projector parameters.

---

### Task 3: Model and Parser Wiring Without Inference Changes

**Files:**
- Modify: `models/detr.py`
- Modify: `train_p2p.py`
- Create: `test_discriminative_dual_proxy_wiring.py`

**Interfaces:**
- Adds parser mode: `--proto_mode=discriminative_dual_proxy`.
- Adds arguments: `--proto_dual_num_fg`, `--proto_dual_num_bg`, `--proto_dual_fg_queue_size`, `--proto_dual_bg_queue_size`, `--proto_dual_supcon_weight`, `--proto_dual_separation_weight`, `--proto_dual_separation_margin`, and `--proto_dual_balance_weight`.

- [ ] **Step 1: Write failing AST wiring tests**

Assert the new mode is accepted, imports `DiscriminativeDualProxy`, uses GT support through `extract_features`, dispatches a dedicated loss branch, and keeps `fused_logits` equal to raw logits by default.

- [ ] **Step 2: Verify RED**

Run: `python test_discriminative_dual_proxy_wiring.py`

Expected: assertions fail because the new mode is absent.

- [ ] **Step 3: Wire the new head in DETR**

Instantiate the head only for the new mode. Reuse `_pack_support_points()` and the shared six-level feature extractor. Add dual-proxy output tensors without changing matcher inputs.

- [ ] **Step 4: Wire training dispatch and diagnostics**

Supply GT support points for both GT-based modes. Add an explicit `elif prototype_mode == 'discriminative_dual_proxy'` dispatch. Add `dualproxy_*` CSV fields and a dedicated health log line.

- [ ] **Step 5: Verify inference invariance**

Test exact tensor equality between `raw_cls_logits` and `cls_logits` in eval mode when diagnostic fusion is zero.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
python test_discriminative_dual_proxy_wiring.py
python test_gt_foreground_proxy.py
python test_gt_foreground_proxy_wiring.py
python -m py_compile models/discriminative_dual_proxy.py models/detr.py train_p2p.py
```

Expected: all commands pass.

---

### Task 4: Frozen-Detector Representation Gate

**Files:**
- Create: `train_discriminative_dual_proxy_gate.py`
- Create: `decide_discriminative_dual_proxy_gate.py`
- Create: `test_discriminative_dual_proxy_gate.py`

**Interfaces:**
- `train_discriminative_dual_proxy_gate.py` loads the 0318 detector, freezes all parameters except the dual-proxy projector, runs five source epochs, and writes `representation_gate_metrics.json` plus `dual_proxy_bank.pth` only on pass.
- `decide_discriminative_dual_proxy_gate.py --metrics <json> --output <json>` returns exit code 0 on pass and 2 on fail.

- [ ] **Step 1: Write failing gate-decision tests**

Test all thresholds independently: macro AUC 0.75, both class gaps 0.05, both effective counts 3.0, pairwise maximum 0.90, and minimum assignment share 0.05.

- [ ] **Step 2: Verify RED**

Run: `python test_discriminative_dual_proxy_gate.py`

Expected: missing decision module.

- [ ] **Step 3: Implement deterministic held-out-slide aggregation**

Aggregate metrics per slide first, then macro-average. Record the minimum slide AUC alongside macro AUC. The bank export block executes only after `gate_pass` is true.

- [ ] **Step 4: Implement frozen training safety**

Set `requires_grad=False` for Backbone, FPN, P2P regression/classification heads, and aggregation modules. Assert that the optimizer parameter IDs exactly equal projector parameter IDs. Save this assertion result in the gate report.

- [ ] **Step 5: Verify GREEN**

Run: `python test_discriminative_dual_proxy_gate.py`

Expected: threshold, no-export-on-failure, and optimizer-ownership tests pass.

---

### Task 5: Representation-Gate Launcher

**Files:**
- Create: `run_discriminative_dual_proxy_gate_gpu.sh`
- Create: `test_discriminative_dual_proxy_run_script.py`

**Interfaces:**
- Inputs: `GPU`, `DATASET`, `INIT_CHECKPOINT`, `OUTPUT_DIR`, `BATCH_SIZE`.
- Output: `representation_gate_metrics.json`, optional `dual_proxy_bank.pth`, and `decision.json`.

- [ ] **Step 1: Write failing run-script structure tests**

Assert `set -euo pipefail`, explicit checkpoint and mean/std preflight, five epochs, exit-on-failed-gate, and absence of any detector-training command before the gate check.

- [ ] **Step 2: Verify RED**

Run: `python test_discriminative_dual_proxy_run_script.py`

Expected: launcher missing.

- [ ] **Step 3: Implement launcher**

Run tensor/wiring/gate tests first, then the frozen representation gate. Print `[STOP] representation gate failed; no detector training` and exit 2 when failed.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python test_discriminative_dual_proxy_run_script.py
bash -n run_discriminative_dual_proxy_gate_gpu.sh
```

Expected: both pass.

---

### Task 6: Ten-Epoch Paired Detector Gate

**Files:**
- Create: `run_discriminative_dual_proxy_paired.sh`
- Create: `decide_discriminative_dual_proxy_paired.py`
- Create: `test_discriminative_dual_proxy_paired.py`

**Interfaces:**
- Requires a passed `dual_proxy_bank.pth` and gate decision.
- Trains `control_p2p` and `discriminative_dual_proxy` for ten epochs from the identical initialization and seed.
- Writes `paired_decision.json`.

- [ ] **Step 1: Write failing paired-decision tests**

Test rejection when frozen F1 decreases, far-background FP increases, source F1 drops by more than 0.005, geometry becomes invalid, or gradient cosine is missing.

- [ ] **Step 2: Verify RED**

Run: `python test_discriminative_dual_proxy_paired.py`

Expected: missing decision module.

- [ ] **Step 3: Implement paired launcher and decision**

Reset RNG immediately after loading the same 0318 checkpoint in both arms. Save first-batch fingerprints and compare them before prototype activation. Use outer prototype weight 0.02 and inference fusion zero.

- [ ] **Step 4: Add evaluation-only resume**

Support `EVAL_ONLY=1` with explicit `PAIR_ROOT`; verify both checkpoints before skipping training. Never derive the evaluation root from a mutable latest-pointer file.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
python test_discriminative_dual_proxy_paired.py
python test_discriminative_dual_proxy_run_script.py
bash -n run_discriminative_dual_proxy_paired.sh
```

Expected: all pass.

---

### Task 7: Full Local Verification and Remote Handoff

**Files:**
- Verify all files from Tasks 1-6.

- [ ] **Step 1: Run regression tests**

```bash
python test_discriminative_dual_proxy.py
python test_discriminative_dual_proxy_wiring.py
python test_discriminative_dual_proxy_gate.py
python test_discriminative_dual_proxy_paired.py
python test_discriminative_dual_proxy_run_script.py
python test_gt_foreground_proxy.py
python test_gt_foreground_proxy_structure.py
python test_gt_foreground_proxy_wiring.py
python test_gt_foreground_proxy_run_script.py
```

Expected: every test prints its success message.

- [ ] **Step 2: Compile and parse all changed files**

```bash
python -m py_compile models/discriminative_dual_proxy.py models/detr.py train_p2p.py train_discriminative_dual_proxy_gate.py decide_discriminative_dual_proxy_gate.py decide_discriminative_dual_proxy_paired.py
bash -n run_discriminative_dual_proxy_gate_gpu.sh
bash -n run_discriminative_dual_proxy_paired.sh
```

Expected: no output and exit code zero.

- [ ] **Step 3: Produce the remote synchronization list**

List only created and modified runtime/test files. On the server, rerun Task 7 before occupying a GPU. Then launch Stage A on one selected GPU; Stage B starts only after a passing `decision.json`.
