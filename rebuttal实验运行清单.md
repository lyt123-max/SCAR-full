# Rebuttal 实验运行清单

> 目标：以最少的重复训练覆盖 AC 和三位 reviewer 的实验问题。清单按“可复用产物”
> 组织，而不是按 reviewer 分开组织。
>
> 正式协议：远程服务器、模型 seed `42`、SCAR 主融合 `cdf_mean`、全程资源监控、
> rebuttal 只提交表格和文字，不生成或提交图片。

## 1. 正式实验前置门槛

- [ ] **G0 环境冻结**：保存 Git commit、未提交 diff、Python/PyTorch/CUDA/Faiss/
  GPU 驱动版本、CPU/GPU/RAM 型号、Conda/pip lock、数据根目录和磁盘余量。
- [x] **G1 主脚本审计参数**：`scripts/main/train_{msl,psm,smap,smd,swat}.sh`
  已默认透传 `MEMORY_AUDIT_MODE=full` 和资源监控参数。
- [x] **G2 E9 runner**：已把 `clean_ratio=0.005` 纳入 sensitivity runner，并让
  所有 E9 点使用 `memory_audit_mode=full`。
- [x] **G3 E10 runner**：每个数据集先冻结唯一的三折事件、候选顺序和嵌套污染
  前缀；`0%` 直接读取已有逐点分数，非零污染支持 no/default/stronger 三种设置。
- [x] **G4 E38 runner**：已增加 `coreset_keep_ratio={0.50,0.25,0.10,0.05}` 的
  Stage-B/Test 批处理和资源汇总；`1.00` 读取主运行，不另跑。
- [x] **G5 TSB 正式 split 防误跑**：正式 batch manifest 只允许
  `M/tuning`、`M/eval`、`U/tuning`、`U/eval`；`U/eval_full` 仅保留为论文修订
  扩展，不进入 rebuttal P0。
- [x] **G6 基线适配器**：已完成 PaAno、MEMTO、PUAD、PGRF-Net 的五数据集
  数据适配、统一逐点评估和资源包装，固定上游 commit 和独立环境；正式队列只运行
  PaAno、PUAD、PGRF-Net，MEMTO 与 CATCH adapter 仅保留作历史审计，不进入正式
  rebuttal 队列。
- [x] **G7 批量 manifest**：每行固定
  `run_id/method/dataset/seed/stage/config_hash/data_hash/command/artifact_dir/status`；
  runner 支持 dry-run、resume、失败清单和完整产物跳过。
- [ ] **G8 模型级冒烟**：每类 runner 只取一个最小样本，检查 checkpoint、score
  长度、metrics、memory audit 和 `resource_metrics.json`。冒烟结果不进入正式表。

## 2. 每次正式运行的必存产物

- [ ] 原始命令、工作目录、时间、退出码和 stdout/stderr；
- [ ] Git commit、dirty diff、环境清单、方法版本和数据 SHA-256；
- [ ] 完整配置与固定 seed `42`；
- [ ] Stage-A checkpoint、Stage-B memory/index 和校准产物；
- [ ] `memory_meta.json`；机制审计运行还须保存 full audit NPZ；
- [ ] 与测试标签等长的逐点 anomaly scores：`raw_max`、`zscore_mean`、`cdf_mean`、
  `cdf_max`，以及全部 `completion_scale*`、`knn_distance`、`state_novelty` 和启用时的
  `soft_support_score`；序列协议保存对应的逐序列分数；
- [ ] 上述每个分数各自配套输出 AUROC、AP、Point-F1、PA-F1、Aff-P、Aff-R、Aff-F1、
  VUS-ROC、VUS-PR；缺任一字段的任务不得标记完成或被 resume 复用；`cdf_mean` 固定为
  主结果，但不得只输出 selected 指标；
- [ ] 九项指标执行“字段强制、适用性显式”：有正负标签且时间轴连续的检测任务必须为
  有限值；TEP fault-only 单类机制序列保留字段并写 `NaN`/原因；E11/E12/E39 和纯资源
  汇总不产生新分数，只复用来源实验指标，不伪造九项数值；
- [ ] `resource_metrics.json`、模型大小、memory 大小和总产物大小；
- [ ] 失败时的 manifest、异常类型、最后 checkpoint 和可续跑状态。

缺少任意必存产物，该 `run_id` 不标记为完成。

## 3. 核心复用关系

| 锚点产物 | 运行量 | 直接解决 | 后续复用 |
| --- | ---: | --- | --- |
| 五数据集 SCAR 默认 full | 5 full | 主结果、checkpoint、训练资源、默认 memory/score | E1-E5；E9 `0.02`；E10 `0%+default`；E38 `1.00`；T3/T8 |
| 五数据集 E9 `clean_ratio=0` | 5 Stage-B/Test | 无净化对照 | E10 `0%+no purification` 的三折 mask 重评估 |
| E9 其余非默认点 | 20 Stage-B/Test | 净化敏感性 | E11 稀有正常误删统计 |
| E10 非零污染产物 | 135 Stage-B/Test | 污染鲁棒性 | E12 低误差异常存活统计 |
| E38 非默认点 | 20 Stage-B/Test | memory keep ratio | E39 bank size/latency/RAM 表 |
| 五数据集相关 baseline | 15 full | 性能比较 | 同一 checkpoint 做资源计时 |
| CATCH 官方发布结果 | 0 | 扩展数据集 CATCH 对照 | 标为 `reported`，不冒充统一协议重跑 |

## 4. P0-1 五数据集 SCAR 锚点

数据集为 MSL、PSM、SMAP、SMD、SWaT。

- [ ] 每个数据集只做一次 seed-42 `full`，共 5 次 Stage A/B/Test。
- [ ] 默认 `clean_ratio=0.02`、`coreset_keep_ratio=1.0`，保留当前
  `coreset_max_patches_per_scale=200000`，报告 cap 是否实际生效。
- [ ] 第一次正式运行即启用资源监控和 `memory_audit_mode=full`。
- [ ] checkpoint 丢失或任务中断时使用相同 `EXP_NAME` 和 `RESUME=1`。
- [ ] Test 后用同一 checkpoint 做 1 次预热和 3 次推理计时；不重新训练。

固定实验名：

```text
scar_main_msl_seed42
scar_main_psm_seed42
scar_main_smap_seed42
scar_main_smd_seed42
scar_main_swat_seed42
```

本组一次解决：主结果重建、E37 SCAR 效率、E9 默认点、E10 默认零污染分数、
E38 全量 memory 点，以及 E1-E5 的模型和默认 memory。

## 5. P0-2 复用锚点完成机制与鲁棒性

### 5.1 E1-E5 条件兼容检索

- [ ] 每个主数据集从同一 checkpoint/memory 导出 `full`、`no_state`、
  `no_context` 三种策略。
- [ ] 输出粗条件距离、细上下文距离、邻居 Jaccard、时间滞后 Jaccard、块 bootstrap
  区间和三策略性能。
- [ ] 不重新训练 Stage A，不运行绘图脚本。

```bash
python scripts/rebuttal/muqn/export_retrieval_logs.py \
  --experiment_dir artifacts/scar_main_msl_seed42 \
  --strategies full no_state no_context

python scripts/rebuttal/muqn/compute_retrieval_evidence.py \
  --log_dir artifacts/scar_main_msl_seed42/rebuttal_muqn/retrieval \
  --block_size 128 --n_bootstrap 2000
```

五数据集共 15 次策略推理，0 次完整训练。

### 5.2 E9 净化比例

正式取值为 `0/0.005/0.01/0.02/0.05/0.10`。

- [ ] `0.02` 直接读取五数据集主锚点。
- [ ] 只运行其余五个取值，共 `5 数据集 × 5 = 25` 次 Stage-B/Test。
- [ ] 所有点启用 full memory audit；E11 直接分析 audit，不另跑模型。
- [ ] 报告 AUROC、AP、净化前后 patch 数、稀有正常误删率和 cap 状态。

补齐 G2 后的单数据集命令模板：

```bash
python scripts/sensitivity/run_dataset_sensitivity.py \
  --dataset MSL --params clean_ratio \
  --param-values 0 0.005 0.01 0.05 0.10 \
  --base-exp-name scar_main_msl_seed42 \
  --run-tag rebuttal_seed42 --skip-existing
```

### 5.3 E10 污染率

污染率为 `0/0.01/0.03/0.05/0.10`，连续异常事件三折。比较 no purification
(`clean_ratio=0`)、default (`0.02`)，并在 `10%` 污染增加 stronger (`0.10`)。

- [x] 每个数据集先生成 `contamination_fold_manifest.json`，冻结三折事件、
  候选随机顺序和全部非零污染率的嵌套前缀；所有非零和零污染任务读取同一文件。
  唯一候选不足时使用固定 seed 的重复随机排列补齐，并披露
  `sampling_with_replacement`。
- [ ] `0% + no`：读取 E9 `clean_ratio=0` score，对三个 held-out masks 重评估。
- [ ] `0% + default`：读取主锚点 score，对三个 held-out masks 重评估。
- [ ] `1%/3%/5%/10% × no/default × 3 folds`：每数据集 24 次 Stage-B/Test。
- [ ] `10% × stronger × 3 folds`：每数据集 3 次 Stage-B/Test。
- [ ] 五数据集合计 135 次 Stage-B/Test，另做 30 次 mask-only 指标计算；0 次 Stage-A。
- [ ] E12 直接读取 contamination manifest、重构误差和 audit，不另跑模型。

正式任务由中央 manifest 生成；单独调试时必须先冻结协议，再显式传入：

```bash
python scripts/rebuttal/muqn/freeze_contamination_protocol.py \
  --base-experiment-dir artifacts/scar_main_msl_seed42 \
  --output-dir artifacts/scar_e10_msl_frozen_protocol \
  --contamination-ratios 0.01 0.03 0.05 0.10 --n-folds 3 --seed 42

python scripts/rebuttal/muqn/run_contamination_sweep.py \
  --base_experiment_dir artifacts/scar_main_msl_seed42 \
  --fold_manifest artifacts/scar_e10_msl_frozen_protocol/contamination_fold_manifest.json \
  --contamination_ratios 0.01 0.03 0.05 0.10 \
  --n_folds 3 --seed 42 --clean_ratio 0

python scripts/rebuttal/muqn/run_contamination_sweep.py \
  --base_experiment_dir artifacts/scar_main_msl_seed42 \
  --fold_manifest artifacts/scar_e10_msl_frozen_protocol/contamination_fold_manifest.json \
  --contamination_ratios 0.01 0.03 0.05 0.10 \
  --n_folds 3 --seed 42 --clean_ratio 0.02

python scripts/rebuttal/muqn/run_contamination_sweep.py \
  --base_experiment_dir artifacts/scar_main_msl_seed42 \
  --fold_manifest artifacts/scar_e10_msl_frozen_protocol/contamination_fold_manifest.json \
  --contamination_ratios 0.10 \
  --n_folds 3 --seed 42 --clean_ratio 0.10
```

### 5.4 E38-E39 memory keep ratio

- [ ] `1.00` 读取主锚点，保留并披露默认 200k per-scale cap。
- [ ] 只新增 `0.50/0.25/0.10/0.05`，共 `5 × 4 = 20` 次 Stage-B/Test。
- [ ] 每点保存 AUROC、AP、bank size、建库时间、推理 latency、CPU/GPU 内存和磁盘
  大小。
- [ ] E39 只汇总 E38，不另跑。

## 6. P0-3 五数据集相关基线与效率

方法固定为 PaAno、PUAD、PGRF-Net；SCAR 读取 P0-1。MEMTO 不运行，其既有失败
产物不得进入结果；CATCH 不重跑，仅使用官方发布结果作为 `reported from CATCH` 对照。

- [ ] 每种方法在五个主数据集只训练一次 seed `42`，共 15 次 full。
- [ ] full 同时保存性能、训练时间、峰值 GPU/CPU 内存和模型大小。
- [ ] 每个 checkpoint 做 1 次预热和 3 次推理计时，不重新训练。
- [ ] 使用相同测试标签、AUROC/AP evaluator 和数据集级汇总规则。
- [ ] 方法专属窗口、优化器和 epoch 使用官方推荐或训练侧 tuning，不看测试标签。
- [ ] DAMP/GDFlex 不运行非官方 Python 替代版，只做 MATLAB 环境与协议审计。

本组一次解决：最近相关方法缺失、SOTA 定位、五数据集性能、训练/推理成本、
GPU/CPU 内存和模型大小。

## 7. P0-4 补齐 CATCH 覆盖

当前五个主数据集已由 P0-1 覆盖，只运行缺失的 30 个 CSV：

- [ ] ASD dataset 1-12，共 12 次，最终按一个 ASD family 做 macro-average；
- [ ] CalIt2、CICIDS、Creditcard、GECCO、Genesis、NYC，共 6 次；
- [ ] 六类 synthetic anomaly 的两个比例版本，共 12 次；
- [ ] 合计 30 次 SCAR full，不重复 MSL/PSM/SMAP/SMD/SWaT。

对照规则：

- [ ] CATCH 官方全 benchmark 数字直接读取，标为 `reported from CATCH`。
- [ ] CATCH 不重跑：五主数据集和 30 个扩展 CSV 均不创建 CATCH 训练任务。
- [ ] 只运行 SCAR 在当前五主数据集之外缺失的 30 个 CSV，并与 CATCH 官方数字对照。

本组一次解决：为什么只选 5/18、真实/合成数据覆盖不足和选择性报告疑虑。

## 8. P0-5 MMFDD-TEP 六模式全故障

- [ ] 使用 M1-M6 正常数据联合训练一次 Stage A/Stage B。
- [ ] 测试全部 168 条故障序列，保存逐序列分片并支持断点续跑。
- [ ] 输出模式一致率、模式 margin、跨模式检索比例和序列等权汇总。
- [ ] 只运行日志导出、指标计算和表格生成，不运行绘图。

```bash
TEP_MECHANISM_POSTPROCESS=0 \
DATA_ROOT=./Multi-mode-Fault-Diagnosis-Datasets-with-TE-process \
EXP_NAME=tep_full_seed42 \
bash scripts/main/train_tep_full.sh full tep_full_seed42

python scripts/tep/export_mechanism_logs.py \
  --experiment_dir artifacts/tep_full_seed42 --resume
python scripts/tep/compute_mechanism_metrics.py \
  --experiment_dirs artifacts/tep_full_seed42
python scripts/tep/generate_rebuttal_tables.py \
  --full-experiment artifacts/tep_full_seed42 \
  --output-dir artifacts/tep_full_seed42/tep_rebuttal_tables
```

本组一次解决：TEP 子集质疑、ground-truth mode 机制证据和公共数据来源澄清。

## 9. P0-6 TSB-AD-M/U

TSB-AD 每个 CSV 独立训练，不能跨序列共享模型，是整个队列中耗时最大的部分。

### 9.1 SCAR

- [ ] 先运行 M tuning 20 和 U tuning 48，冻结统一配置；
- [ ] 正式运行 M-Eva 180；
- [ ] 正式运行官方默认 U-Eva 350；
- [ ] 不运行 U-Eva-Full 822；该官方扩展清单保留为论文修订 backlog；
- [ ] 固定主结果 `cdf_mean`，不逐文件或逐指标挑融合方式。

```bash
python scripts/tsb_ad/run_benchmark.py \
  --edition M --split tuning --seed 42 --device cuda --resume
python scripts/tsb_ad/run_benchmark.py \
  --edition U --split tuning --seed 42 --device cuda --resume
python scripts/tsb_ad/run_benchmark.py \
  --edition M --split eval --seed 42 --device cuda --resume
python scripts/tsb_ad/run_benchmark.py \
  --edition U --split eval --seed 42 --device cuda --resume
```

预注册一个固定配置并在 tuning 清单上验证时，SCAR 合计
`20 + 48 + 180 + 350 = 598` 次独立序列训练。若实际比较 \(H_M\) 组多变量配置和
\(H_U\) 组单变量配置，运行量必须按
`20×H_M + 48×H_U + 180 + 350` 重新登记；不得把多配置搜索仍写成 598 次。

### 9.2 汇总

- [ ] 对四种融合和全部子分数输出 AUROC、AP、Point-F1、PA-F1、Aff-P、Aff-R、
  Aff-F1、VUS-ROC、VUS-PR；
- [ ] 官方逐序列平均和来源数据集 macro-average；
- [ ] M 180、U 350 的有效序列数和失败数；
- [ ] 总运行时间、单序列时间分布和资源摘要；
- [ ] PaAno 不运行 TSB-AD；公开基线数字只作 `reported` 对照，不写成统一协议重跑；
- [ ] rebuttal P0 不运行 `U --split eval-full`。

本组一次解决：综合 benchmark、单变量适用性、多变量扩展和 TSB-AD 结果不足。

## 10. P1 只在 P0 完成后启动

- [ ] E29 长窗口对照；
- [ ] E30 长 patch 对照；
- [ ] E31 相同预算 global retrieval；
- [ ] 改变 `seq_len` 或 Stage-A 表征的点明确标记为新 full training；
- [ ] 不挤占 CATCH、TSB-AD、E9/E10、基线和 TEP 的 P0 队列。

## 11. 运行量核算

| 类型 | 数量 | 说明 |
| --- | ---: | --- |
| SCAR 五数据集锚点 | 5 full | 解锁机制、净化、污染、memory 和资源实验 |
| 五数据集相关 baseline | 15 full | 3 方法 × 5 数据集；MEMTO 不运行，CATCH 只引用官方结果 |
| CATCH 缺失 CSV 的 SCAR | 30 full | 不重复当前五数据集 |
| TEP full | 1 full | 一套模型测试 168 条故障序列 |
| TSB-AD SCAR | 598 full | tuning 68 + official eval 530；按一个预注册配置核算 |
| E9 非默认点 | 25 Stage-B/Test | 默认 `0.02` 复用锚点 |
| E10 非零污染 | 135 Stage-B/Test | 零污染只重评估已有 score |
| E38 非默认点 | 20 Stage-B/Test | `1.00` 复用锚点 |
| 检索策略 | 15 inference | 3 策略 × 5 数据集 |
| E11/E12/E39 | 0 training | 全部从已有产物派生 |

按 TSB-AD 每个 track 只验证一个预注册配置核算，完整 P0 为
**649 次 full model fit + 180 次 Stage-B/Test**，即 829 次重计算任务。其中 598 次
full 来自 SCAR 的官方 TSB-AD tuning/evaluation。PaAno 只在五个主数据集运行，不补
TSB-AD；若 TSB-AD 比较多组配置，须按上一节公式增加 tuning 运行量。

## 12. 推荐排队顺序

1. 完成 G0-G8，生成 dry-run manifest，人工检查预计运行数；
2. 先跑五数据集 SCAR 锚点，立即解锁最多后续任务；
3. 复用锚点并行跑 E1-E5、E9、E10、E38；
4. 同时跑五数据集相关 baseline，一次采集性能和资源；
5. 跑 CATCH 缺失 30 个 CSV 与 TEP full；
6. TSB-AD 冒烟通过后尽早启动长队列，按文件粒度断点续跑；
7. 所有 P0 表格齐全后再决定是否启动 P1。

### 12.1 三天 AC-core 双服务器队列

`--scope ac-core` 只包含 AC 明确要求且依赖闭环的证据链，共 102 项：

| 任务 | 数量 | 口径 |
| --- | ---: | --- |
| SCAR 五主集 anchor | 5 full | 同时保存 full audit 和资源记录 |
| PaAno/PUAD/PGRF-Net | 15 full | 三方法 × 五主集；MEMTO 不运行 |
| SCAR 补充 CATCH 缺失 CSV | 30 full | CATCH 方法本身不训练 |
| TEP full | 1 full | 六模式联合、168 条故障序列 |
| E9 | 25 Stage-B/Test | 默认点复用 anchor |
| 检索机制证据 | 20 analysis/inference | 三策略性能与代理指标 |
| 效率、数据审计 | 6 analysis | 不新增模型训练 |

E11/E12 需要完整 E10 污染折，保留在 P0，不进入三天 AC-core；本 scope 的净化
敏感性由五主数据集 E9 六档阈值表直接回答。

每台双卡服务器使用：

```bash
python scripts/experiments/rebuttal.py run \
  --scope ac-core --max-parallel 2 --gpu-devices 0 1 \
  --python-map environments/python-map.remote.json \
  --data-root-map environments/data-root-map.remote.json
```

正式分片必须再传 `--group`、`--method` 或 `--dataset`，保证两台服务器不生成
重复 run-id。`group:catch=31` 是 30 个 SCAR 补充实验加 1 个完整性审计，不是
CATCH baseline 重跑。

## 13. 最终验收

- [ ] manifest 中没有重复的
  `method + dataset/file + seed + config_hash + stage`；
- [ ] 五数据集 SCAR 各只有一个正式 Stage-A checkpoint；
- [ ] E9 默认点、E10 零污染点、E38 全量点都有明确复用来源；
- [ ] U 轨正式结果严格来自官方 `TSB-AD-U-Eva.csv` 的 350 条清单；
- [ ] rebuttal P0 manifest 中没有 `U/eval_full` 正式任务；
- [ ] 性能与资源记录可追溯到同一 checkpoint；
- [ ] 所有失败、OOM 和超时均有记录，没有静默删样本；
- [ ] CATCH 数字全部标为 `reported from CATCH`，manifest 中没有 CATCH 训练任务；
- [ ] 所有 rebuttal 结果均能整理成表格和文字，不依赖图片；
- [ ] `PROJECT_PLAN.md`、`rebuttal执行计划.md` 和
  `CHANGELOG_MAINTENANCE.md` 已同步更新。

## 14. 统一命令入口

不再手工拼接正式队列。远程环境先运行：

```bash
python scripts/experiments/preflight.py \
  --output artifacts/rebuttal_manifest/preflight.json --strict-runtime
python scripts/experiments/rebuttal.py plan --scope all
bash scripts/experiments/remote_smoke.sh
```

随后按 P0、P1 顺序使用 `rebuttal.py run|resume|status|validate|collect`。中央 manifest
是任务数量和复用关系的唯一口径；P0 固定为 649 个 full model fit 与 180 个
Stage-B/Test，P1 数量只读取 manifest，不手工累计。P0 完成后运行
`--group p0_tables`，严格生成主五集、baseline、效率、CATCH、E9/E10 和 TSB 的
CSV 与文字摘要，并单独生成 E1-E5 检索策略全分数表和 TEP 全分数表；缺项时该任务失败。
远程路径使用 `environments/python-map.example.json` 和
`environments/data-root-map.example.json` 为模板，正式命令同时传入
`--python-map` 与 `--data-root-map`。
