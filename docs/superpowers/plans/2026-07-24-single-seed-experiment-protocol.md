# Single-Seed Experiment Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every formal SCAR rebuttal experiment use the single pre-specified model seed `42` while preserving CLI overrides, historical artifact readability, contamination folds, bootstrap resampling, and repeated timing measurements.

**Architecture:** Existing training entrypoints already default to seed 42, so the implementation centralizes the remaining formal runner defaults, removes fold-derived model/data sampling seeds, and updates all active protocol documentation. Collectors remain backward-compatible with historical `seed_*` directories.

**Tech Stack:** Python 3.12, argparse, unittest, Markdown, Git.

## Global Constraints

- Formal model seed is exactly `42`.
- CLI seed overrides remain available for debugging but are not formal paper results.
- Stochastic external baselines default to seed 42; deterministic methods run once.
- Contamination folds, block bootstrap draws, sensitivity values, and repeated timing measurements remain.
- Performance tables report fixed-seed results, not model-seed mean or standard deviation.
- Existing artifacts are never deleted or rewritten.
- Third-party source trees remain read-only.

---

### Task 1: Formal Seed Defaults And Regression Tests

**Files:**
- Modify: `scripts/tsb_ad/run_benchmark.py`
- Modify: `scripts/tsb_ad/collect_results.py`
- Modify: `scripts/rebuttal/muqn/baselines/run_paano.py`
- Modify: `scripts/rebuttal/muqn/run_contamination_sweep.py`
- Modify: `tests/test_tsb_ad.py`
- Create: `tests/test_single_seed_protocol.py`
- Modify: `tests/test_contamination_protocol.py`

**Interfaces:**
- Produces `scripts.tsb_ad.run_benchmark.FORMAL_SEED: int`.
- Produces `scripts.rebuttal.muqn.baselines.run_paano.FORMAL_SEED: int`.
- Preserves `--seed` and `--seeds` CLI compatibility.

- [x] **Step 1: Add failing assertions for formal defaults**

```python
def test_formal_seed_is_42():
    assert FORMAL_SEED == 42

def test_paano_default_seed_is_formal_seed():
    with patch.object(sys, "argv", ["run_paano.py", "--data_dir", "in", "--output_dir", "out"]):
        assert parse_args().seed == 42
```

- [x] **Step 2: Run focused tests and confirm the PaAno default fails**

Run:

```powershell
python tests/test_single_seed_protocol.py
```

Expected: failure showing PaAno currently defaults to `2000`.

- [x] **Step 3: Implement formal constants and remove fold-derived sampling seeds**

Use `FORMAL_SEED = 42` in the TSB-AD and PaAno runners. Resolve the TSB-AD implicit default through that constant. In contamination sweeps, use `args.seed` for every disjoint fold instead of `args.seed + fold_index`; the fold candidate pools are already disjoint.

- [x] **Step 4: Run focused tests**

Run:

```powershell
python tests/test_single_seed_protocol.py
python tests/test_tsb_ad.py
```

Expected: all tests pass.

### Task 2: Active Experiment Protocol Documentation

**Files:**
- Modify: `rebuttal执行计划.md`
- Modify: `Reviewer_ef7G_解决方案.md`
- Modify: `PROJECT_PLAN.md`
- Modify: `scripts/rebuttal/muqn/README.md`
- Modify: `scripts/tsb_ad/README.md`
- Modify: `docs/superpowers/specs/2026-07-24-baseline-efficiency-benchmark-design.md`
- Modify: `消融实验方案.md`

**Interfaces:**
- Consumes the formal seed constant `42`.
- Produces one unambiguous active protocol for commands, tables, and acceptance criteria.

- [x] **Step 1: Replace active three-seed requirements**

Change all formal commands from `--seeds 42 43 44` to `--seed 42`. Replace model-performance `mean +/- std` requirements with `fixed-seed result`. Preserve bootstrap confidence intervals and runtime repetitions.

- [x] **Step 2: Clarify formal versus debugging runs**

Add a concise rule that CLI overrides are allowed for diagnostics, but only seed 42 belongs in rebuttal and paper tables.

- [x] **Step 3: Correct efficiency repetition language**

Specify one model training run at seed 42 and three post-warm-up inference timing measurements. Timing variation is system measurement noise, not model-seed variance.

- [x] **Step 4: Scan active documents**

Run:

```powershell
rg -n "42/43/44|42 43 44|三随机种子|三个随机种子|三 seed|three seed|mean.?std|mean±std|Mean±Std|均值和标准差" rebuttal执行计划.md Reviewer_ef7G_解决方案.md PROJECT_PLAN.md scripts/rebuttal/muqn/README.md scripts/tsb_ad/README.md docs/superpowers/specs/2026-07-24-baseline-efficiency-benchmark-design.md
```

Expected: no stale model-seed repetition requirement; statistical normalizer and timing references may remain when explicitly unrelated to model seeds.

### Task 3: Maintenance Record And Verification

**Files:**
- Modify: `CHANGELOG_MAINTENANCE.md`
- Verify: first-party Python and Markdown files

**Interfaces:**
- Produces a dated maintenance entry superseding the earlier planned three-seed protocol.

- [x] **Step 1: Add the maintenance entry**

Record the formal seed 42 decision, code defaults changed, preserved repetition types, validation commands, and the fact that historical changelog entries describe superseded plans.

- [x] **Step 2: Run syntax and focused tests**

Run:

```powershell
python -m compileall -q coremad run.py scripts tests
python tests/test_single_seed_protocol.py
python tests/test_resource_monitor.py
python tests/test_tsb_ad.py
```

Expected: compilation succeeds and all runnable tests pass.

- [x] **Step 3: Run formal dry-runs**

Run:

```powershell
python scripts/tsb_ad/run_benchmark.py --edition M --split eval --limit 1 --dry-run --device cuda
python scripts/rebuttal/muqn/baselines/run_paano.py --data_dir input --output_dir output --dry_run
```

Expected: each generated command contains exactly `--seed 42` and schedules one model run.

- [x] **Step 4: Run final repository checks**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files plus the user's pre-existing worktree changes are present.
