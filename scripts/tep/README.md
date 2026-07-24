# TEP Mechanism Analysis

These scripts keep the main TEP train/test pipeline unchanged and move mechanism auditing offline.
Under the current TEP protocol, normal-mode mechanism audit is taken from validation normal windows, while fault mechanism analysis is taken from independent fault test sequences.
Two protocols are supported:

- `selected` (default): normal files from `M1/M3/M4` and eight fault files per mode.
- `full`: all normal files from `M1-M6` and all 28 fault files per mode (168 fault sequences).

Following the upstream official setting, files `d01` through `d28` map in reverse order to physical disturbances `IDV28` through `IDV1`. Metadata keeps both `file_fault_id` (`dXX`) and `fault_id` (official IDV).

Run the complete protocol:

```bash
DATA_ROOT=./Multi-mode-Fault-Diagnosis-Datasets-with-TE-process \
bash scripts/main/train_tep_full.sh full tep_full_mechanism
```

Resume an interrupted run by setting `RESUME=1` and the exact existing `EXP_NAME`.

`scripts/ablation/run_tep.sh` now calls this offline pipeline automatically after any TEP ablation run that reaches `test`.
Set `TEP_MECHANISM_POSTPROCESS=0` if you want to disable that automatic step.
For the paper-facing summary, `TEP` is reported separately from the point-wise main ablation table. The full protocol contains fault-only positive sequences, so it reports mechanism metrics instead of undefined `AUROC`, `AUPRC`, and `F1`.

## 1. Export mechanism logs

```bash
python scripts/tep/export_mechanism_logs.py --experiment_dir artifacts/tep_baseline_xxx
```

Selected-protocol outputs keep the legacy monolithic layout. Full-protocol outputs go to `artifacts/<exp>/tep_mechanism/`:

- `train_state_meta.npz`
- `audit_normal_window_logs.npz`
- `audit_normal_window_logs.csv`
- `fault_window_shards/*.npz` (all windows, compact per-sequence metric shards)
- `fault_detail_shards/*.npz` (bounded visualization samples)
- `fault_detail_sample.npz`
- `fault_window_shards_manifest.json`
- `fault_sequence_scores_with_meta.json`
- `sequence_scores_with_meta.json`
- `export_meta.json`

The full protocol does not write a multi-gigabyte fault-window CSV. Both sequence testing and mechanism export are resumable.

## 2. Compute mechanism metrics

Single or multiple experiments:

```bash
python scripts/tep/compute_mechanism_metrics.py --experiment_dirs artifacts/exp_a artifacts/exp_b
```

A4 simplified sequence-score comparison:

```bash
python scripts/tep/compute_mechanism_metrics.py ^
  --short_dir artifacts/tep_ablation_single_scale_short_xxx ^
  --long_dir artifacts/tep_ablation_single_scale_long_xxx ^
  --multi_dir artifacts/tep_ablation_multi_scale_xxx
```

Main metrics:

- `SMC@K`
- `SFR`
- `Mode-FPR-Std`
- `SMR@K`
- `SMR@K-sequence-balanced`
- `delta_mem_mode`
- `fault-normal-gap-sequence-balanced`
- `EE95`
- `Tail@0.99_error`
- `Fault-Consistency-Std`
- `Evidence-Dom-Consistency`
- `Proto-Purity`
- `Proto-Entropy`
- `Cross-mode-margin`
- `cross-mode-margin-sequence-balanced`
- `CG_mean`
- `%_multi_best`

## 3. Plot key figures

```bash
python scripts/tep/plot_mechanism_figures.py --experiment_dirs artifacts/exp_a artifacts/exp_b
```

A4 gain heatmap:

```bash
python scripts/tep/plot_mechanism_figures.py ^
  --short_dir artifacts/tep_ablation_single_scale_short_xxx ^
  --long_dir artifacts/tep_ablation_single_scale_long_xxx ^
  --multi_dir artifacts/tep_ablation_multi_scale_xxx
```

Key figures:

- `state_embedding.png`
- `normal_final_violin_by_mode.png`
- `retrieval_mode_confusion_heatmap.png`
- `exceedance_plot.png`
- `mode_fpr_and_fault_normal_gap.png`
- `fault_file_gain_heatmap.png`
- `fault_evidence_heatmap.png`
- `fault_triplet_consistency.png`

## Auto-run envs

Useful environment variables when calling `scripts/ablation/run_tep.sh`:

- `TEP_MECHANISM_POSTPROCESS=0|1`
- `TEP_MECHANISM_WINDOW_AGGREGATION=p95|top5_mean|max|mean`
- `TEP_MECHANISM_EMBEDDING_METHOD=umap|tsne`
- `TEP_MECHANISM_SMC_K=10`
- `TEP_MECHANISM_SMR_K=10`
- `TEP_MECHANISM_RUN_PLOTS=0|1`
- `TEP_MECHANISM_RUN_METRICS=0|1`
- `TEP_MECHANISM_RUN_A4_COMPARE=0|1`
- `TEP_A4_COMPARE_STRICT=0|1`

When the current ablation is `multi_scale`, the auto-runner will also try to build the A4 simplified comparison by resolving sibling experiment directories for `single_scale_short` and `single_scale_long`.
The default behavior is now strict: if those short/long/multi directories cannot all be resolved, the pipeline exits with an error instead of silently skipping, and it also writes `tep_mechanism/a4_compare_status.json` for auditability.
Set `TEP_A4_COMPARE_STRICT=0` if you want a best-effort run that records the missing-input status but does not fail the whole postprocess step.
If your experiment names are custom, provide:

- `TEP_A4_SHORT_EXP_DIR=/abs/or/relative/path`
- `TEP_A4_LONG_EXP_DIR=/abs/or/relative/path`
- `TEP_A4_MULTI_EXP_DIR=/abs/or/relative/path`

To collect the sequence-level TEP summary together with mechanism-metric and figure inventories, run:

```bash
python scripts/ablation/generate_paper_tables.py
```

For rebuttal tables T2 and T9:

```bash
python scripts/tep/generate_rebuttal_tables.py \
  --full-experiment artifacts/tep_full_mechanism_xxx \
  --selected-experiment TEP-abalation/tep_ablation_full_20260418_193550 \
  --output-dir artifacts/tep_full_mechanism_xxx/tep_rebuttal_tables
```

When `--selected-experiment` is omitted, the table generator discovers the latest
selected TEP baseline under `artifacts/` or `TEP-abalation/`. Historical schema-v2
metrics missing sequence-balanced fields are recomputed in memory without modifying
the old experiment directory.
