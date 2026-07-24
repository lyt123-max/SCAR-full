# Reviewer muQn Rebuttal Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build reproducible code for main-dataset retrieval evidence, closest
baseline comparison, and memory-purification audits.

**Architecture:** Add provenance and audit metadata to the existing memory
pipeline without changing default scores, then build reviewer-specific scripts
on those public interfaces. Keep all upstream baseline repositories read-only.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, scikit-learn, matplotlib, FAISS,
MATLAB R2021a+ for DAMP/GDFlex.

## Global Constraints

- Preserve default SCAR numerical behavior.
- Reuse existing Stage-A checkpoints.
- Never overwrite existing main-result directories.
- Keep `third_party/baselines/` read-only.
- Use point-wise AUROC and AP without test-label threshold tuning.
- Update both long-term maintenance documents.

---

### Task 1: Memory provenance and independent memory seed

**Files:**
- Modify: `coremad/config.py`
- Modify: `coremad/memory.py`
- Modify: `coremad/trainer.py`
- Modify: `run.py`
- Create: `tests/test_memory_provenance.py`

**Interfaces:**
- Produces `CoReMADConfig.memory_seed`.
- Produces `ScaleMemory.raw_starts`.
- Produces `coarse_window_starts` and `neighbor_raw_starts` query details.

- [ ] Write tests for effective memory seed, raw-start selection, persistence,
  backward compatibility, and query-detail shapes.
- [ ] Run the focused tests and confirm they fail for missing interfaces.
- [ ] Add config and CLI fields with `memory_seed=None` resolving to `seed`.
- [ ] Propagate raw starts through disk and in-memory memory builders.
- [ ] Save/load raw starts with backward compatibility.
- [ ] Return raw provenance from detailed queries.
- [ ] Run focused tests and syntax checks.

### Task 2: Memory purification audit

**Files:**
- Modify: `coremad/config.py`
- Modify: `coremad/memory.py`
- Modify: `coremad/trainer.py`
- Modify: `run.py`
- Create: `tests/test_memory_audit.py`

**Interfaces:**
- Produces `memory_audit_mode` with `none`, `summary`, and `full`.
- Produces `MemoryBank.build_stats`.
- Produces `memory_audit_scale<P>.npz`.

- [ ] Write failing tests for summary counts and full audit masks.
- [ ] Implement build statistics in both memory construction paths.
- [ ] Export full audit arrays before temporary memmaps are removed.
- [ ] Add build statistics to `memory_meta.json`.
- [ ] Verify that audit mode does not alter retained tensors.

### Task 3: Retrieval mechanism evidence

**Files:**
- Create: `scripts/rebuttal/muqn/common.py`
- Create: `scripts/rebuttal/muqn/proxy_metrics.py`
- Create: `scripts/rebuttal/muqn/export_retrieval_logs.py`
- Create: `scripts/rebuttal/muqn/run_retrieval_stability.py`
- Create: `scripts/rebuttal/muqn/run_temporal_consistency.py`
- Create: `scripts/rebuttal/muqn/plot_retrieval_evidence.py`
- Create: `tests/test_muqn_proxy_metrics.py`

**Interfaces:**
- Produces CSV/JSON/NPZ retrieval evidence under an explicit output directory.
- Consumes only raw data, checkpoints, and detailed retrieval output.

- [ ] Write deterministic tests for state features, context distances,
  Jaccard overlap, and lagged consistency.
- [ ] Implement pure metric helpers.
- [ ] Implement sampled multi-strategy retrieval-log export.
- [ ] Implement perturbation and lagged temporal studies.
- [ ] Implement aggregate tables and figures.
- [ ] Run pure tests and CLI help smoke tests.

### Task 4: Purification and contamination experiments

**Files:**
- Modify: `scripts/sensitivity/run_dataset_sensitivity.py`
- Create: `scripts/rebuttal/muqn/run_purification_sweep.py`
- Create: `scripts/rebuttal/muqn/contamination.py`
- Create: `scripts/rebuttal/muqn/run_contamination_sweep.py`
- Create: `scripts/rebuttal/muqn/analyze_purification.py`
- Create: `tests/test_muqn_contamination.py`

**Interfaces:**
- Produces clean-ratio sweep manifests and commands.
- Produces event-disjoint injection/evaluation manifests.
- Produces retention, survival, coverage, AUROC, and AP tables.

- [ ] Register the clean-ratio sweep.
- [ ] Write tests for anomaly-event extraction and fold disjointness.
- [ ] Implement explicit memory-window contamination datasets.
- [ ] Implement Stage-B/test orchestration with Stage-A bootstrap.
- [ ] Implement audit analysis and result collection.
- [ ] Run tests and dry-run command generation.

### Task 5: External baseline adapters

**Files:**
- Create: `coremad/evaluation.py`
- Create: `scripts/rebuttal/muqn/baselines/common_protocol.py`
- Create: `scripts/rebuttal/muqn/baselines/run_paano.py`
- Create: `scripts/rebuttal/muqn/baselines/prepare_damp_inputs.py`
- Create: `scripts/rebuttal/muqn/baselines/damp_multidim_scores.m`
- Create: `scripts/rebuttal/muqn/baselines/collect_damp_results.py`
- Create: `scripts/rebuttal/muqn/baselines/audit_gdflex.py`
- Create: `tests/test_muqn_baseline_protocol.py`

**Interfaces:**
- Produces aligned point scores and threshold-free metrics.
- Does not modify upstream repositories.

- [ ] Write score-alignment and metric tests.
- [ ] Add shared point-wise AUROC/AP evaluation.
- [ ] Add PaAno data/model wrapper.
- [ ] Add DAMP data preparation, score-returning MATLAB adapter, and collector.
- [ ] Add GDFlex compatibility audit.
- [ ] Run Python tests and CLI smoke checks; record MATLAB as unexecuted when
  unavailable.

### Task 6: Verification and maintenance

**Files:**
- Modify: `PROJECT_PLAN.md`
- Modify: `CHANGELOG_MAINTENANCE.md`

- [ ] Run all available tests.
- [ ] Parse every modified Python file with the AST compiler.
- [ ] Run `git diff --check`.
- [ ] Inspect the final diff for unrelated changes.
- [ ] Document commands, actual results, missing runtime dependencies, and
  remaining experiment runs.
