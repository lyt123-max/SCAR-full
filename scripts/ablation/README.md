# Ablation Scripts

中文说明见 [README_zh.md](./README_zh.md)。

This directory contains the ablation runner for the `MSL`, `SMAP`, `PSM`, `SWAT`, `SMD`, and `TEP` experiments.
The default paper/reporting score is now fixed to `cdf_mean`.
For compatibility, `selected` is still written to `test_metrics.json`, and under the current official setup it follows `cdf_mean`.
For `PSM`, keep the validation split mode as `interleaved`; the runner already applies that dataset-specific default automatically.

## A1 split runners

The A1 representation-backbone ablations are now split into dedicated files instead of sharing one mixed switch block:

- `run_full_ablation.sh`
- `run_a1_wo_decomposition.sh`
- `run_a1_wo_state_aware_representation.sh`
- `run_a1_matrix.sh`

Recommended direct usage:

```bash
bash scripts/ablation/run_full_ablation.sh PSM
bash scripts/ablation/run_a1_wo_decomposition.sh PSM
bash scripts/ablation/run_a1_wo_state_aware_representation.sh PSM
bash scripts/ablation/run_a1_matrix.sh
```

`run_dataset_ablation.sh` still exists as a lightweight dispatcher, so the dataset wrappers such as `run_psm.sh` continue to work.

## A2 split runners

The A2 retrieval-design ablations are also split into dedicated files:

- `run_a2_global_retrieval.sh`
- `run_a2_state_only_retrieval.sh`
- `run_a2_context_only_retrieval.sh`
- `run_a2_dual_condition_retrieval.sh`
- `run_a2_matrix.sh`

Recommended direct usage:

```bash
bash scripts/ablation/run_a2_global_retrieval.sh PSM
bash scripts/ablation/run_a2_state_only_retrieval.sh PSM
bash scripts/ablation/run_a2_context_only_retrieval.sh PSM
bash scripts/ablation/run_a2_dual_condition_retrieval.sh PSM
bash scripts/ablation/run_a2_matrix.sh
```

## A3/A4 split runners

The A3 fusion-design and A4 temporal-scale ablations are also split into dedicated files:

- `run_a3_pit_fusion.sh`
- `run_a3_raw_max.sh`
- `run_a3_zscore_mean.sh`
- `run_a4_single_scale_short.sh`
- `run_a4_single_scale_long.sh`
- `run_a4_multi_scale.sh`
- `run_a3_matrix.sh`
- `run_a4_matrix.sh`

Recommended direct usage:

```bash
bash scripts/ablation/run_a3_pit_fusion.sh PSM
bash scripts/ablation/run_a3_raw_max.sh PSM
bash scripts/ablation/run_a3_zscore_mean.sh PSM
bash scripts/ablation/run_a4_single_scale_short.sh PSM
bash scripts/ablation/run_a4_single_scale_long.sh PSM
bash scripts/ablation/run_a4_multi_scale.sh PSM
bash scripts/ablation/run_a3_matrix.sh
bash scripts/ablation/run_a4_matrix.sh
```

## Official formal-study keys

- `full`
- `wo_decomposition`
- `wo_state_aware_representation`
- `global_retrieval`
- `state_only_retrieval`
- `context_only_retrieval`
- `dual_condition_retrieval`
- `pit_fusion`
- `raw_max`
- `zscore_mean`
- `single_scale_short`
- `single_scale_long`
- `multi_scale`

The default formal matrix runner `run_ablation_matrix.sh` uses the publication-ready set:

- `full`
- `wo_decomposition`
- `wo_state_aware_representation`
- `global_retrieval`
- `state_only_retrieval`
- `context_only_retrieval`
- `dual_condition_retrieval`
- `raw_max`
- `zscore_mean`
- `single_scale_short`
- `single_scale_long`
- `multi_scale`

`wo_state_aware_representation` remains the internal experiment key, but the paper-facing name should be interpreted as `w/o state-guided modulation`, because this ablation disables channel modulation rather than removing the whole state branch.
`pit_fusion` is kept as an A3-only comparison baseline and now corresponds to the `cdf_mean` fixed PIT/CDF fusion setting.

## Run examples

```bash
bash scripts/ablation/run_msl.sh full
bash scripts/ablation/run_msl.sh wo_decomposition
bash scripts/ablation/run_psm.sh wo_state_aware_representation
bash scripts/ablation/run_psm.sh global_retrieval
bash scripts/ablation/run_psm.sh state_only_retrieval
bash scripts/ablation/run_psm.sh context_only_retrieval
bash scripts/ablation/run_psm.sh dual_condition_retrieval
bash scripts/ablation/run_psm.sh raw_max
bash scripts/ablation/run_psm.sh zscore_mean
bash scripts/ablation/run_psm.sh single_scale_short
bash scripts/ablation/run_psm.sh single_scale_long
bash scripts/ablation/run_psm.sh multi_scale
```

If you want to run the A3 PIT/CDF baseline explicitly, use:

```bash
bash scripts/ablation/run_psm.sh pit_fusion
```

All ablation runs are now self-contained in their own experiment directories.
To make ablation outputs easy to distinguish from main experiments, the default folder name now includes an explicit `ablation` marker, for example `msl_ablation_full_<timestamp>` or `psm_ablation_raw_max_<timestamp>`.

Run the whole matrix on the default six datasets:

```bash
bash scripts/ablation/run_ablation_matrix.sh
```

Run the A1-only matrix:

```bash
bash scripts/ablation/run_a1_matrix.sh
```

Run the A2-only matrix:

```bash
bash scripts/ablation/run_a2_matrix.sh
```

Run the A3-only matrix:

```bash
bash scripts/ablation/run_a3_matrix.sh
```

For the paper-facing summary, use `SMD` and `SWAT` as the point-wise main ablation datasets, and report `TEP` in a separate sequence-level mechanism section.
The helper below generates the paper-ready tables:

```bash
python scripts/ablation/generate_paper_tables.py
```

It writes:

- `./artifacts/paper_tables/pointwise_main_ablation.{csv,json,md}`
- `./artifacts/paper_tables/a3_fusion_family.{csv,json,md}`
- `./artifacts/paper_tables/stability_analysis.{csv,json,md}`
- `./artifacts/paper_tables/tep_sequence_results.{csv,json,md}`

The paper-oriented defaults are now:

- Point-wise main table: `PR-AUC`, `ROC-AUC`, and `F1`
- A3 fusion-family table: `PR-AUC`, `ROC-AUC`, and `F1`
- Stability analysis: `PR-AUC`
- TEP sequence-level main metrics: `AUPRC`, `AUROC`, and `F1`

By default, the matrix runner now assigns one shared timestamp to the whole batch, the per-experiment directories include that date tag, and it automatically calls `collect_results.py` after all runs finish to write:

- `./artifacts/ablation_summary_<timestamp>.csv`
- `./artifacts/ablation_summary_<timestamp>.json`

If you only want to run experiments without the final collection step:

```bash
COLLECT_RESULTS=0 bash scripts/ablation/run_ablation_matrix.sh
```

If you want the old fixed-name resume behavior instead, set `RESUME=1`.

If you want to force a specific stage:

```bash
bash scripts/ablation/run_swat.sh single_scale_short stage_a
bash scripts/ablation/run_swat.sh single_scale_short stage_b
bash scripts/ablation/run_swat.sh single_scale_short test
```

If you want to run only `stage_b` or `test`, do that inside the same ablation experiment directory so the required earlier-stage artifacts come from that ablation run itself:

```bash
RESUME=1 bash scripts/ablation/run_swat.sh single_scale_short stage_a
RESUME=1 bash scripts/ablation/run_swat.sh single_scale_short stage_b
RESUME=1 bash scripts/ablation/run_swat.sh single_scale_short test
```

## Result collection

The scripts now default to `RESUME=0`, so experiment directories include both an explicit `ablation` marker and a timestamp by default. Set `RESUME=1` when you intentionally want to keep reusing one fixed ablation directory such as `msl_ablation_full`.

Collect the stable experiment directories with:

```bash
python scripts/ablation/collect_results.py
```

Render a Markdown table for the point-wise best F1 reported on `cdf_mean`:

```bash
python scripts/ablation/collect_results.py --markdown_metric_keys best_f1 --markdown_score_key cdf_mean
```

Render a Markdown table for point-adjust F1 instead:

```bash
python scripts/ablation/collect_results.py --markdown_metric_keys pa_best_f1 --markdown_score_key cdf_mean
```

The default ablation summary now also includes `Affiliated-Precision`, `Affiliated-Recall`, `Range-Precision`, and `Range-Recall`.
