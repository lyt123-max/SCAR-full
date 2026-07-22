# CoReM-AD 消融实验说明

本文档说明 `scripts/ablation/` 目录下的正式消融实验脚本、推荐运行方式、结果汇总方法，以及与 `TEP` 机制分析相关的注意事项。

当前消融脚本默认支持以下数据集：

- `MSL`
- `SMAP`
- `PSM`
- `SWAT`
- `SMD`
- `TEP`

当前正式论文口径下，默认主报告分数固定为 `cdf_mean`。
为了兼容已有产物，`test_metrics.json` 中仍会保留 `selected` 字段；在当前官方配置下，`selected` 与 `cdf_mean` 一致。

`PSM` 数据集默认仍使用 `interleaved` 验证划分，这个数据集特定设置已经在运行脚本中自动处理。

## 1. 设计目标

这套消融脚本的设计目标有三点：

- 将消融实验与主实验目录彻底分开，避免 checkpoint、测试结果和汇总文件互相污染。
- 仅保留正式论文版消融配置，默认跑批直接使用正式集合。
- 统一结果汇总口径，避免“实际测试分数”和“表格展示分数”不一致。

默认情况下，消融目录名会包含明确的 `ablation` 标记，例如：

```bash
artifacts/msl_ablation_full_20260410_120000
artifacts/psm_ablation_raw_max_20260410_120000
```

如果你希望持续复用同一个固定目录，可以设置：

```bash
RESUME=1
```

## 2. 实验分组

当前正式消融分为四组。

### A1 表征主干消融

- `full`
  说明：完整模型。
- `wo_decomposition`
  说明：去掉分解模块。
- `wo_state_aware_representation`
  说明：展示名应解释为“去掉状态引导调制（w/o state-guided modulation）”，因为该消融关闭的是通道调制，而不是整个状态分支。

对应脚本：

- `run_full_ablation.sh`
- `run_a1_wo_decomposition.sh`
- `run_a1_wo_state_aware_representation.sh`
- `run_a1_matrix.sh`

### A2 检索机制消融

- `global_retrieval`
  说明：全局检索。
- `state_only_retrieval`
  说明：仅状态检索。
- `context_only_retrieval`
  说明：仅上下文检索。
- `dual_condition_retrieval`
  说明：双条件检索。

对应脚本：

- `run_a2_global_retrieval.sh`
- `run_a2_state_only_retrieval.sh`
- `run_a2_context_only_retrieval.sh`
- `run_a2_dual_condition_retrieval.sh`
- `run_a2_matrix.sh`

### A3 融合方式消融

- `pit_fusion`
  说明：`cdf_mean` 作为固定 PIT/CDF 融合基线。
- `raw_max`
  说明：直接取原始子分数最大值。
- `zscore_mean`
  说明：对标准化后的子分数求均值。

对应脚本：

- `run_a3_pit_fusion.sh`
- `run_a3_raw_max.sh`
- `run_a3_zscore_mean.sh`

说明：

- `pit_fusion` 会保留在 A3 专项对比中。
- 但它不会进入默认的 overall 正式矩阵，因为它本质上对应完整模型使用的 PIT/CDF 融合设定，放进总表会和 `full` 产生语义重复。

### A4 时间尺度消融

- `single_scale_short`
  说明：仅使用短尺度。
- `single_scale_long`
  说明：仅使用长尺度。
- `multi_scale`
  说明：同时使用短尺度与长尺度。

对应脚本：

- `run_a4_single_scale_short.sh`
- `run_a4_single_scale_long.sh`
- `run_a4_multi_scale.sh`

虽然脚本文件名沿用了 `run_a3_*` 命名，但语义上这三项已经统一归入 A4。

## 3. 正式默认矩阵

执行：

```bash
bash scripts/ablation/run_ablation_matrix.sh
```

默认会在六个数据集上运行以下正式集合：

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

这套默认集合是“正式论文版/推荐汇报版”口径。
但论文主表不直接照搬六数据集总表，而是进一步整理为：

- 点级主消融：`SMD`、`SWaT`
- 序列级独立机制验证：`TEP`

如果你只想跑某一组矩阵，可以使用：

```bash
bash scripts/ablation/run_a1_matrix.sh
bash scripts/ablation/run_a2_matrix.sh
bash scripts/ablation/run_a3_matrix.sh
bash scripts/ablation/run_a4_matrix.sh
```

其中：

- `run_a1_matrix.sh` 默认跑 A1。
- `run_a2_matrix.sh` 默认跑 A2。
- `run_a3_matrix.sh` 默认跑 A3。
- `run_a4_matrix.sh` 默认跑 A4。

## 4. 单项实验运行方式

推荐优先使用“拆分后的专用脚本”，例如：

```bash
bash scripts/ablation/run_full_ablation.sh PSM
bash scripts/ablation/run_a1_wo_decomposition.sh PSM
bash scripts/ablation/run_a2_global_retrieval.sh PSM
bash scripts/ablation/run_a3_raw_max.sh PSM
bash scripts/ablation/run_a4_single_scale_short.sh PSM
bash scripts/ablation/run_a4_multi_scale.sh PSM
```

也可以继续使用数据集包装脚本：

```bash
bash scripts/ablation/run_psm.sh full
bash scripts/ablation/run_psm.sh wo_decomposition
bash scripts/ablation/run_psm.sh global_retrieval
bash scripts/ablation/run_psm.sh raw_max
bash scripts/ablation/run_psm.sh single_scale_short
bash scripts/ablation/run_psm.sh multi_scale
```

如果你希望显式运行 A3 的 PIT/CDF 基线：

```bash
bash scripts/ablation/run_psm.sh pit_fusion
```

## 5. 阶段控制

大多数脚本都支持按阶段执行。常见阶段包括：

- `stage_a`
- `stage_b`
- `test`
- `auto`

例如：

```bash
bash scripts/ablation/run_swat.sh single_scale_short stage_a
bash scripts/ablation/run_swat.sh single_scale_short stage_b
bash scripts/ablation/run_swat.sh single_scale_short test
```

重要建议：

- 如果你只打算单独重跑 `stage_b` 或 `test`，请确保仍然在同一个消融实验目录里继续执行。
- 不要让消融实验去复用主实验目录中的中间产物。
- 这样可以保证 checkpoint、缓存和测试结果都来自同一个消融配置，避免实验污染。

如果你要在固定目录上继续追加后续阶段，可以这样做：

```bash
RESUME=1 bash scripts/ablation/run_swat.sh single_scale_short stage_a
RESUME=1 bash scripts/ablation/run_swat.sh single_scale_short stage_b
RESUME=1 bash scripts/ablation/run_swat.sh single_scale_short test
```

## 6. 结果汇总

默认情况下，矩阵脚本会在所有实验结束后自动调用：

```bash
python scripts/ablation/collect_results.py
```

输出文件通常位于：

- `./artifacts/ablation_summary_<timestamp>.csv`
- `./artifacts/ablation_summary_<timestamp>.json`

如果只想跑实验、不想自动汇总，可以设置：

```bash
COLLECT_RESULTS=0 bash scripts/ablation/run_ablation_matrix.sh
```

### 6.1 汇总分数口径

当前默认的 Markdown 展示分数是：

- `cdf_mean`

这意味着正式论文表格默认以固定的 `cdf_mean` 口径展示，而不是再使用随配置切换的 `selected`。

例如，导出 `best_f1` 的 Markdown 表格：

```bash
python scripts/ablation/collect_results.py --markdown_metric_keys best_f1 --markdown_score_key cdf_mean
```

导出 `pa_best_f1` 的 Markdown 表格：

```bash
python scripts/ablation/collect_results.py --markdown_metric_keys pa_best_f1 --markdown_score_key cdf_mean
```

如果要直接生成论文主表、A3 融合策略表、稳定性分析表以及 TEP 序列级整理，可以运行：

```bash
python scripts/ablation/generate_paper_tables.py
```

默认输出包括：

- `./artifacts/paper_tables/pointwise_main_ablation.{csv,json,md}`
- `./artifacts/paper_tables/a3_fusion_family.{csv,json,md}`
- `./artifacts/paper_tables/stability_analysis.{csv,json,md}`
- `./artifacts/paper_tables/tep_sequence_results.{csv,json,md}`

当前论文专用汇总脚本的默认指标口径为：

- 点级主表：`PR-AUC`、`ROC-AUC`、`F1`
- A3 融合策略表：`PR-AUC`、`ROC-AUC`、`F1`
- 稳定性分析表：默认基于 `PR-AUC`
- `TEP` 序列级主指标：`AUPRC`、`AUROC`、`F1`

### 6.2 当前默认标签与数据集

`collect_results.py` 当前默认行为包括：

- 默认数据集包含 `TEP`
- 默认正式 ablation 集合即当前仓库支持的全部正式实验项
- `single_scale_short / single_scale_long / multi_scale` 在表格标签中分别显示为 A4.1 / A4.2 / A4.3
- `pit_fusion` 在表格中显示为 A3.1 基线

## 7. TEP 相关补充说明

`TEP` 已经纳入默认消融矩阵。

如果你运行的是 `TEP` 消融，并且实验执行到了 `test`，脚本还会触发离线机制分析后处理。相关说明在：

- [../tep/README.md](../tep/README.md)

特别是当当前消融为 `multi_scale` 时，后处理会尝试自动构建 A4 对比：

- `single_scale_short`
- `single_scale_long`
- `multi_scale`

当前默认行为为严格模式：

- 如果 short、long、multi 三个实验目录没有全部找到，流程会报错，而不是静默跳过。
- 同时会在 `tep_mechanism/a4_compare_status.json` 中写入状态，便于追踪和审计。

如果你希望改为“尽力执行，但不因为缺目录而失败”，可以设置：

```bash
TEP_A4_COMPARE_STRICT=0
```

如果实验目录名是自定义的，可以显式指定：

```bash
TEP_A4_SHORT_EXP_DIR=/abs/or/relative/path
TEP_A4_LONG_EXP_DIR=/abs/or/relative/path
TEP_A4_MULTI_EXP_DIR=/abs/or/relative/path
```

## 8. 常见建议

- 正式跑批时，优先使用矩阵脚本和默认正式集合。
- 做正式汇报表格时，优先使用固定的 `cdf_mean` 口径。
- 点级正文主指标建议优先展示 `PR-AUC` 与 `ROC-AUC`，`F1` 作为补充综合指标一并保留。
- 点级正文主表建议只使用 `SMD` 和 `SWaT`。
- `TEP` 建议单独写成序列级机制验证节，并保留机制指标与机制图。
- 做阶段续跑时，尽量在同一个 ablation 目录中继续，不要跨目录拼接产物。
- 如果只是对比 A3 融合方式，可以单独跑 `pit_fusion / raw_max / zscore_mean`。
- 如果只是对比 A4 时间尺度，可以单独跑 `single_scale_short / single_scale_long / multi_scale`。

## 9. 相关脚本一览

核心入口：

- `run_ablation_matrix.sh`
- `run_dataset_ablation.sh`
- `collect_results.py`

A1：

- `run_full_ablation.sh`
- `run_a1_wo_decomposition.sh`
- `run_a1_wo_state_aware_representation.sh`
- `run_a1_matrix.sh`

A2：

- `run_a2_global_retrieval.sh`
- `run_a2_state_only_retrieval.sh`
- `run_a2_context_only_retrieval.sh`
- `run_a2_dual_condition_retrieval.sh`
- `run_a2_matrix.sh`

A3 / A4：

- `run_a3_pit_fusion.sh`
- `run_a3_raw_max.sh`
- `run_a3_zscore_mean.sh`
- `run_a4_single_scale_short.sh`
- `run_a4_single_scale_long.sh`
- `run_a4_multi_scale.sh`
- `run_a3_matrix.sh`

数据集包装入口：

- `run_msl.sh`
- `run_smap.sh`
- `run_psm.sh`
- `run_swat.sh`
- `run_smd.sh`
- `run_tep.sh`
