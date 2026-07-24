# Sensitivity Analysis

This folder contains a reproducible sensitivity-analysis workflow for CoReM-AD on `MSL`, `SMAP`, `PSM`, `SWaT`, and `SMD`.

## What it does

- Sweeps one hyperparameter at a time while keeping all others at the dataset-specific defaults.
- Reuses a default Stage-A checkpoint for retrieval and memory sweeps, so those runs only execute Stage B + Test.
- Each single-dataset or multi-dataset run now gets a date tag by default, so repeated sensitivity batches are easy to distinguish later.
- The default result folders now also include an explicit `sensitivity` marker such as `msl_sensitivity`, so they do not get confused with main experiments or ablations.
- Stores a machine-readable summary in `csv/json`.
- The summary records `raw_max`, `zscore_mean`, `cdf_mean`, `cdf_max`, optional CDF variants, and every available diagnostic subscore for each sweep point.
- The cross-dataset collector discovers all score keys by default; `--score-keys` is only an explicit narrowing override for legacy analyses.
- Every discovered fusion score and diagnostic subscore carries the registered AUROC, AP, Point-F1, PA-F1, Aff-P, Aff-R, Aff-F1, VUS-ROC, and VUS-PR fields.
- Draws paper-style main/appendix figures and exports both `png` and `pdf`.

## Important design notes

- `seq_len=96` is intentionally not used. The current model requires `seq_len` to be divisible by every patch size, and the default patch sizes are `8` and `32`. The sweep therefore uses `[64, 128, 160, 192, 256]`.
- Retrieval and memory sweeps now work correctly with a reused Stage-A checkpoint because the trainer preserves runtime Stage-B/Test overrides such as `top_M`, `top_K`, `knn_k`, and `coreset_max_patches_per_scale`.
- Re-running an existing sensitivity point now recreates that experiment directory from scratch, which prevents stale memory banks or test metrics from being silently reused.
- `d_z` is now exposed through `run.py` and all dataset training scripts in `scripts/train_*.sh`.
- `PSM` keeps its dataset-specific defaults from the training script, notably `val_split_mode=interleaved` and `lr=5e-4`.

## Recommended workflow

Run the priority preset on one dataset first:

```bash
python scripts/sensitivity/run_dataset_sensitivity.py \
  --dataset MSL \
  --preset priority \
  --artifact-root ./artifacts/sensitivity/msl \
  --data-root ./dataset/anomaly_detect \
  --device cuda \
  --skip-existing
```

Then draw the figures:

```bash
python scripts/sensitivity/plot_dataset_sensitivity.py \
  --summary-json ./artifacts/sensitivity/msl_sensitivity/summary/msl_sensitivity_<timestamp>_summary.json
```

The plotting script keeps using the legacy top-level `cdf_max` metrics by default so the paper-style figures remain unchanged.

Run all five datasets in one shot:

```bash
python scripts/sensitivity/run_all_dataset_sensitivity.py \
  --preset priority \
  --artifact-root ./artifacts/sensitivity \
  --data-root ./dataset/anomaly_detect \
  --device cuda \
  --skip-existing \
  --plot-after-run
```

By default, the multi-dataset runner now also calls `collect_results.py` after all runs finish and writes:

- `./artifacts/sensitivity/sensitivity_summary_<timestamp>.csv`
- `./artifacts/sensitivity/sensitivity_summary_<timestamp>.json`

If you only want to run experiments and per-dataset outputs without the final combined summary:

```bash
python scripts/sensitivity/run_all_dataset_sensitivity.py \
  --preset priority \
  --artifact-root ./artifacts/sensitivity \
  --data-root ./dataset/anomaly_detect \
  --device cuda \
  --skip-existing \
  --plot-after-run \
  --skip-collect-results
```

## Presets

- `priority`: `top_M`, `top_K`, `knn_k`, `d_z`, `mask_ratio`
- `full`: all supported sweeps
- `retrieval`: `top_M`, `top_K`, `knn_k`
- `representation`: `d_z`, `seq_len`, `mask_ratio`
- `memory`: `coreset_max_patches_per_scale`

## Output layout

- Single dataset:
  - Default dataset root: `./artifacts/sensitivity/<dataset-lower>_sensitivity/`
  - Experiments: `<artifact-root>/<experiment-name>/`
  - Logs: `<artifact-root>/<experiment-name>/sensitivity_logs/`
  - Summary: `<artifact-root>/summary/`
  - Figures: `<artifact-root>/summary/figures/`
- Multi-dataset root:
  - Per-dataset artifacts: `<artifact-root>/<dataset-lower>_sensitivity/...`
  - Combined summary: `<artifact-root>/sensitivity_summary_<timestamp>.csv` and `<artifact-root>/sensitivity_summary_<timestamp>.json`
  - Manifest: `<artifact-root>/sensitivity_manifest_<timestamp>.json`
