# Reviewer muQn Rebuttal Code Design

## Goal

Add reproducible evidence and experiment infrastructure for Reviewer muQn's
three concerns:

1. condition-compatible retrieval evidence on the five main datasets;
2. fair comparison with retrieval and subsequence baselines;
3. memory-purification sensitivity, contamination, and coverage audits.

The implementation must preserve the default SCAR training and inference
behavior. Existing Stage-A checkpoints remain reusable, and existing result
directories must never be overwritten.

## Architecture

The work is divided into three layers.

### Core provenance and audit layer

`ScaleMemory` records the raw start position of every retained patch. Memory
construction records raw, purified, and final patch counts and can optionally
export per-patch audit arrays. Query details expose both Level-1 source-window
starts and Level-2 source-patch starts.

The model-training seed and memory-construction seed are separated. The
existing `seed` continues to control Stage-A. `memory_seed` defaults to
`seed`, preserving historical behavior, but can be varied independently for
Stage-B repetitions.

### Rebuttal experiment layer

All reviewer-specific orchestration lives under `scripts/rebuttal/muqn/`.
These scripts load existing checkpoints, export retrieval logs, compute
label-free proxy metrics, run perturbation and temporal-consistency studies,
and execute purification/contamination sweeps.

Proxy metrics are computed from raw signals, not from SCAR embeddings:

- state proxy: per-channel mean, standard deviation, linear slope, first
  difference magnitude, and low-frequency energy ratio;
- context proxy: channel-wise normalized L1 distance and one minus Pearson
  correlation.

### Baseline adapter layer

Official baseline repositories under `third_party/baselines/` remain
read-only. Project-side adapters load SCAR datasets, run the upstream model,
align subsequence scores to test points, and evaluate point-wise AUROC/AP with
one shared evaluator.

PaAno receives an executable Python adapter. DAMP receives input preparation,
a MATLAB function derived from the official multidimensional implementation
that returns scores and accepts an explicit train/test split, and a result
collector. GDFlex receives a machine-readable compatibility audit because its
official implementation does not define a multivariate dense-score protocol.

## Core Data Contracts

### ScaleMemory

Each scale stores:

```python
ScaleMemory(
    z: Tensor[N, Dz],
    c: Tensor[N, Dz],
    window_ids: Tensor[N],
    raw_starts: Tensor[N],
)
```

`raw_starts[i]` is the start index of retained patch `i` in the normalized
training sequence. Old memory artifacts load with `raw_starts=-1`; raw-space
mechanism scripts reject such artifacts with an actionable error.

### Query details

`MemoryBank.query_preencoded(..., return_details=True)` additionally returns:

- `coarse_window_starts`;
- `neighbor_raw_starts`;
- the existing neighbor IDs, distances, and validity masks.

### Memory audit

Summary mode records per-scale:

- patch size;
- raw count;
- purification threshold;
- count after purification;
- coreset mode;
- final count.

Full mode also writes one compressed NPZ per scale:

- `raw_start`;
- `window_id`;
- `completion_score`;
- `kept_after_purification`;
- `kept_after_coreset`.

## Contamination Protocol

Contamination is introduced only into Stage-B. Stage-A remains clean.
Anomaly events are assigned stable event IDs and split into disjoint injection
and evaluation folds. Explicit anomaly windows are appended to a custom memory
loader, making the requested contamination fraction exact in window/patch
space. Injected events are excluded from evaluation.

The audit identifies injected entries through their window-ID range, enabling
measurement of:

- anomalous-patch removal rate;
- normal-patch removal rate;
- residual contamination after coreset selection;
- survival rate of the lowest reconstruction-error anomaly quartile.

## Compatibility and Reproducibility

- Default CLI values reproduce historical behavior.
- Existing memory artifacts remain loadable.
- Old artifacts cannot be used for raw patch provenance analysis.
- Stage-B metadata includes `memory_seed` and the coreset cap.
- Reviewer experiments write only below a dedicated output root.
- No test labels are used for SCAR model selection or threshold tuning.
- All baseline scores are cut to the test range before AUROC/AP evaluation.

## Tests

Tests cover:

- raw-start preservation through purification, coreset selection, sorting,
  save, and load;
- backward loading of memory artifacts without raw starts;
- independent Stage-A and memory seeds;
- proxy metrics on deterministic synthetic examples;
- perturbation overlap and temporal lag calculations;
- contamination event disjointness and exact requested fraction;
- baseline score alignment and label-independent AUROC/AP.

## Documentation

Every implementation batch updates `PROJECT_PLAN.md` and
`CHANGELOG_MAINTENANCE.md`. Experiment results are recorded only after commands
have actually run; planned outcomes are never entered as results.
