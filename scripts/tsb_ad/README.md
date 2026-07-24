# SCAR TSB-AD Benchmark

该目录提供 SCAR 的独立 TSB-AD-U/M benchmark 轨道。每个 CSV 都使用自己的正常训练前缀独立完成归一化、Stage A、memory 构建和完整序列测试，不在不同序列之间共享模型或统计量。

## 协议

- 文件名中的 `_tr_N_` 决定正常训练前缀 `data[:N]`。
- 训练、memory、CDF 与 Z-score 统计量只使用该训练前缀。
- 推理和评估范围是完整 CSV，标签不参与训练、归一化、融合拟合或 VUS window 估计。
- VUS window 使用 TSB-AD 官方 rank-1 ACF 规则，由第一信号通道估计。
- 一次推理固定报告 `raw_max`、`zscore_mean`、`cdf_mean`、`cdf_max`，并报告全部 `completion_scale*`、`knn_distance`、`state_novelty` 和启用时的 `soft_support_score`；`cdf_mean` 是预先指定的 SCAR 主结果。
- 禁止按 CSV、来源数据集或指标选择表现最好的融合策略。

## 安装评估依赖

```bash
python -m pip install -r scripts/tsb_ad/requirements.txt
```

项目本身的 PyTorch、NumPy 等训练依赖仍需在 SCAR 运行环境中可用。

## 数据与清单校验

```bash
python scripts/tsb_ad/validate_setup.py --edition both --deep
```

默认识别以下当前目录结构：

```text
dataset/
  TSB-AD-M/TSB-AD-M/*.csv
  TSB-AD-U/TSB-AD-U/*.csv
```

官方清单位于 `scripts/tsb_ad/manifests/`。支持的规模为 M tuning 20、M eval 180、
U tuning 48、U eval 350 和 U eval-full 822。正式 rebuttal 使用官方 runner 默认的
M eval 180 与 U eval 350；U eval-full 822 仅作为可选扩展。

## 冒烟运行

先检查命令和路径，不启动训练：

```bash
python scripts/tsb_ad/run_benchmark.py --edition M --split eval --limit 1 --dry-run --device cuda
python scripts/tsb_ad/run_benchmark.py --edition U --split eval --limit 1 --dry-run --device cuda
```

实际运行一条序列：

```bash
python scripts/tsb_ad/run_benchmark.py --edition M --split eval --limit 1 --device cuda
python scripts/tsb_ad/run_benchmark.py --edition U --split eval --limit 1 --device cuda
```

## 正式运行

```bash
python scripts/tsb_ad/run_benchmark.py --edition M --split eval --seed 42 --device cuda --resume
python scripts/tsb_ad/run_benchmark.py --edition U --split eval --seed 42 --device cuda --resume
```

正式评估前可分别在 M tuning 20 和 U tuning 48 上验证预注册配置。若比较多组配置，
总运行量必须按配置数成倍登记，不能把多配置搜索计作一次 tuning。

`--split eval-full` 继续受支持，但不属于当前 rebuttal P0：

```bash
python scripts/tsb_ad/run_benchmark.py \
  --edition U --split eval-full --seed 42 --device cuda --resume
```

`--stage` 支持 `stage_a`、`stage_b`、`test` 和 `full`。每条序列由独立子进程执行；日志和状态写入该序列产物目录。`--resume` 只有在对应阶段要求的全部产物存在时才跳过。
正式 rebuttal/论文协议只使用 seed `42`。`--seed` 和 `--seeds` 仍保留用于调试及读取
历史实验，但其他 seed 的结果不得进入正式汇总表。

产物路径固定为：

```text
artifacts/tsb_ad/<edition>/<split>/seed_<seed>/<file_stem>/
```

## 结果汇总

完整运行或测试结束后会自动汇总，也可单独执行：

```bash
python scripts/tsb_ad/collect_results.py --edition M --split eval
```

输出包括：

- `results_long.csv`
- `results_per_series_wide.csv`
- `results_by_dataset_wide.csv`
- `results_official_average_wide.csv`
- `results_dataset_macro_average_wide.csv`
- `collection_failures.csv`

长表逐文件、逐 seed、逐融合策略/子分数保存 VUS-PR、VUS-ROC、AUROC、AP、运行时间和评分覆盖率。正式表只读取 seed `42`；collector 继续兼容历史 `seed_*` 目录。四张宽表分别用于逐序列、来源数据集平均、全部序列官方平均和来源数据集 macro-average。
