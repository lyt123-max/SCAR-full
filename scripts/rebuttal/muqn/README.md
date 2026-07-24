# Reviewer muQn Rebuttal Experiments

本目录对应三类问题：检索是否真正满足条件兼容性、记忆净化是否稳健，以及近期基线是否
覆盖。所有正式模型实验都应使用相同数据边界和固定 seed `42`，并保留脚本生成的
manifest、原始 NPZ/CSV 和环境日志。CLI seed 覆盖仅用于调试，不得进入正式表格；
contamination folds 与 block bootstrap 仍按各自协议保留。

## 1. 检索条件兼容性与稳定性

现有 Stage-B 产物必须由当前代码重新构建，旧 `memory.pt` 没有精确 patch 来源，
加载后该字段只会标为 `-1`。

```bash
python scripts/rebuttal/muqn/export_retrieval_logs.py \
  --experiment_dir artifacts/<experiment> \
  --strategies full no_state no_context

python scripts/rebuttal/muqn/compute_retrieval_evidence.py \
  --log_dir artifacts/<experiment>/rebuttal_muqn/retrieval \
  --block_size 128 --n_bootstrap 2000
```

输出包括粗/细检索条件代理距离、相对完整策略的邻居集合 Jaccard、多个时间 lag 的
Jaccard，以及块 bootstrap 置信区间。

## 2. 净化与污染

`clean_ratio` 已加入通用 sensitivity runner：

```bash
python scripts/sensitivity/run_dataset_sensitivity.py \
  --dataset MSL --params clean_ratio \
  --param-values 0 0.01 0.02 0.05 0.10
```

事件级污染实验会把注入事件整段排除出该折评估：

```bash
python scripts/rebuttal/muqn/run_contamination_sweep.py \
  --base_experiment_dir artifacts/<experiment> \
  --contamination_ratios 0 0.01 0.05 0.10 \
  --n_folds 3 --seed 42

python scripts/rebuttal/muqn/analyze_purification_audit.py \
  --experiment_dir artifacts/rebuttal_muqn_contamination/<generated-experiment>
```

先使用 `--dry_run` 检查每折事件数、候选窗口数和实际污染比例。正式污染实验会重建
Stage B 和测试，不会重训 Stage A。

## 3. 外部基线

先导出共享数据：

```bash
python scripts/rebuttal/muqn/baselines/prepare_baseline_data.py \
  --experiment_dir artifacts/<experiment> \
  --output_dir artifacts/rebuttal_muqn_baselines/<dataset>
```

PaAno 使用其独立 Python 3.11/PyTorch 2.7.1 环境；DAMP 需要 MATLAB；GDFlex 的
官方入口限定 ML/UCR 单变量数据，不能直接作为五个多变量主数据集的原生结果。

```bash
python scripts/rebuttal/muqn/baselines/run_paano.py \
  --data_dir artifacts/rebuttal_muqn_baselines/<dataset>/paano \
  --output_dir artifacts/rebuttal_muqn_baselines/<dataset>/paano_results

python scripts/rebuttal/muqn/baselines/run_damp.py \
  --input_csv artifacts/rebuttal_muqn_baselines/<dataset>/damp/<dataset>.csv \
  --output_csv artifacts/rebuttal_muqn_baselines/<dataset>/damp_scores.csv

python scripts/rebuttal/muqn/baselines/audit_gdflex.py \
  --output artifacts/rebuttal_muqn_baselines/gdflex_scope.json
```

第三方目录始终只读。PaAno/DAMP 的原始分数再由
`baselines/evaluate_baseline_scores.py` 按统一逐点指标计算。
