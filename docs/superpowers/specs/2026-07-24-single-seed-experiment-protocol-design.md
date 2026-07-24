# Single-Seed Experiment Protocol Design

## Objective

Change SCAR's formal rebuttal and paper experiment protocol from three model
random seeds to one pre-specified seed. The formal seed is `42`.

The change applies project-wide to experiment documentation, batch runners,
examples, result schemas, and acceptance criteria. It must not remove the
existing CLI ability to override a seed for debugging or exploratory work.

## Protocol

- Every formal SCAR experiment uses `seed=42`.
- Stochastic external baselines use seed `42` when their official
  implementation exposes a seed option.
- Deterministic methods run once.
- Formal performance tables report a fixed-seed result, not `mean +/- std`.
- A manually overridden seed is allowed for debugging but is not a formal
  paper result.
- The Git commit, data manifest or hash, configuration, command, hardware, and
  environment must still be recorded for each formal run.

## Repetitions That Remain

The following are not model-seed repetitions and remain part of the protocol:

- contamination event folds, which measure coverage over disjoint events;
- block bootstrap resampling for retrieval-statistic confidence intervals;
- inference timing repetitions after warm-up, which measure system noise;
- multiple parameter values in sensitivity, ablation, and scaling studies.

Bootstrap intervals must be described as data/block uncertainty rather than
model-initialization uncertainty.

## Code And Documentation Scope

1. Replace formal seed lists such as `42 43 44` with the single seed `42`.
2. Change batch-runner defaults from multiple seeds to `[42]`.
3. Remove requirements to run key experiments three times for model variance.
4. Replace `mean +/- std` wording and fields with fixed-seed wording where the
   values refer to model performance.
5. Preserve repeated timing measurements and their timing mean/deviation.
6. Preserve CLI seed arguments and validation so exploratory runs remain
   possible.
7. Update `PROJECT_PLAN.md`, `rebuttal执行计划.md`, relevant READMEs, and
   `CHANGELOG_MAINTENANCE.md`.

## Compatibility

Existing multi-seed artifact directories remain readable. Collectors may
continue to discover arbitrary seed directories, but formal commands and
documentation only create and present seed 42. No existing experiment
artifact is deleted or rewritten.

## Verification

- Search the first-party repository for stale formal `42/43/44`,
  `42 43 44`, `three seed`, `3 seeds`, and model-performance `mean +/- std`
  requirements.
- Run syntax compilation for modified Python files.
- Run focused runner and collector tests.
- Run dry-run commands and confirm exactly one formal run is scheduled.
- Run `git diff --check`.

