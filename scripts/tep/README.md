# TEP Mechanism Analysis

These scripts keep the main TEP train/test pipeline unchanged and move mechanism auditing offline.
Under the current TEP protocol, normal-mode mechanism audit is taken from validation normal windows, while fault mechanism analysis is taken from independent fault test sequences.
The current selected TEP subset uses normal files `m1d00`, `m3d00`, and `m4d00` (modes `M1`, `M3`, and `M4`, all with normal identifier `IDV0`) and fault files with identifiers `IDV1`, `IDV4`, `IDV6`, `IDV7`, `IDV10`, `IDV13`, `IDV14`, and `IDV27` under each selected mode.

`scripts/ablation/run_tep.sh` now calls this offline pipeline automatically after any TEP ablation run that reaches `test`.
Set `TEP_MECHANISM_POSTPROCESS=0` if you want to disable that automatic step.
For the paper-facing summary, `TEP` should be reported separately from the point-wise main ablation table, with the sequence-level main metrics fixed to `AUROC`, `AUPRC`, and `F1`.

## 1. Export mechanism logs

```bash
python scripts/tep/export_mechanism_logs.py --experiment_dir artifacts/tep_baseline_xxx
```

Outputs go to `artifacts/<exp>/tep_mechanism/`:

- `train_state_meta.npz`
- `audit_normal_window_logs.npz`
- `audit_normal_window_logs.csv`
- `fault_window_logs.npz`
- `fault_window_logs.csv`
- `fault_sequence_scores_with_meta.json`
- `sequence_scores_with_meta.json`
- `export_meta.json`

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
- `delta_mem_mode`
- `EE95`
- `Tail@0.99_error`
- `Fault-Consistency-Std`
- `Evidence-Dom-Consistency`
- `Proto-Purity`
- `Proto-Entropy`
- `Cross-mode-margin`
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
