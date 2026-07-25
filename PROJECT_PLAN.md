# CoReM-AD 项目方案说明

> 维护约定：本文件是项目级方案入口。以后凡是模型结构、训练流程、数据协议、关键参数、脚本入口或实验口径发生变化，都需要同步更新本文对应章节，并在 `CHANGELOG_MAINTENANCE.md` 记录修改结果。

## 1. 项目介绍

本项目实现的是 CoReM-AD v2.0，一个面向多变量时间序列异常检测的自监督异常检测框架。项目以正常训练序列学习多尺度状态表征与记忆库，在测试阶段通过状态检索、上下文检索、补全误差和多分数融合生成逐点或序列级异常分数。

项目主要目标包括：

- 在仅使用正常训练数据的前提下学习稳定的时序状态表示。
- 通过 STSD 分解将慢变状态与残差扰动分开建模。
- 使用状态感知的多尺度 patch 表征描述不同时间尺度上的局部异常。
- 建立训练记忆库，在测试阶段用两级检索度量样本与正常状态/上下文的偏离程度。
- 结合 retrieval score 与 completion score，通过 CDF-PIT、raw max、z-score mean 等策略输出可比较的异常分数。
- 支持主实验、消融实验、敏感性分析、TEP 机制分析和论文图表导出。

## 2. 目录结构

| 路径 | 作用 |
| --- | --- |
| `run.py` | 统一命令行入口，负责解析参数并调用训练器的 `stage_a`、`stage_b`、`test` 或 `full` 流程。 |
| `coremad/` | 核心 Python 包，包含配置、数据加载、模型、记忆库、检索、分数融合、训练和可视化。 |
| `scripts/main/` | 主实验训练脚本、数据下载、诊断评分评估和论文主图/机制图绘制脚本。 |
| `scripts/ablation/` | 消融实验矩阵、单项消融脚本、结果汇总和论文表格生成脚本。 |
| `scripts/sensitivity/` | 超参数敏感性分析运行、汇总和绘图脚本。 |
| `scripts/tep/` | TEP 离线机制分析脚本，包括机制日志导出、指标计算和机制图绘制。 |
| `scripts/tsb_ad/` | TSB-AD-U/M 官方 split 清单、数据校验、逐 CSV 独立运行和多融合结果汇总脚本。 |
| `scripts/rebuttal/muqn/` | Reviewer muQn 专项实验：检索来源日志、条件兼容性/稳定性统计、净化与事件级污染实验，以及 PaAno/DAMP/GDFlex 项目侧适配器。 |
| `third_party/baselines/` | rebuttal 外部基线的只读上游代码；当前包含 PaAno、PUAD、PGRF-Net、GDFlex 四个独立 Git 工作树，以及从作者官方页面下载的 DAMP MATLAB 代码、说明和样例。精确来源、commit、Drive 文件 ID 与 SHA-256 见目录内 `README.md`，代码本体已被父项目 Git 忽略。 |
| `dataset/` | 本地 CATCH-style 异常检测数据目录，已被 Git 忽略。 |
| `TEP-DATA/` | 本地 TEP 选定 `.mat` 数据目录，已被 Git 忽略。 |
| `Multi-mode-Fault-Diagnosis-Datasets-with-TE-process/` | MMFDD-TEP 上游完整数据仓库，包含六种模式及全部 28 种故障；作为本地原始数据源使用，已被父项目 Git 忽略。 |
| `artifacts/` | 默认实验产物目录，包含 checkpoint、memory bank、分数、图和日志，已被 Git 忽略。 |
| `main-result/`、`GECCO-ablation/`、`TEP-abalation/` | 历史实验结果和论文产物目录，已被 Git 忽略。 |

## 3. 总体模型框架

CoReM-AD 的一次完整运行分为三阶段。

```text
原始多变量时间序列 X
        |
        v
Stage A: 自监督表示学习
  1. 标准化训练数据
  2. STSD 分解: X -> slow state S + residual R
  3. StateEncoder 编码慢变状态 S -> state vector
  4. ChannelModulation 用状态向量调制残差 patch
  5. MultiScalePatchEncoder 输出多尺度 patch 表征 z/u 和上下文键 c
  6. MaskedCompletionHead + NextPatchPredHead 做自监督训练
        |
        v
Stage B: 正常记忆库与融合器构建
  1. 冻结 Stage-A 模型
  2. 对完整正常训练段抽取 state bank、scale memory、context key
  3. 可选构建 Faiss 状态索引
  4. 计算训练诊断分数
  5. 拟合 CDF-PIT fusion 与 z-score fusion
        |
        v
Stage Test: 异常检测与评估
  1. 对测试窗口编码
  2. 两级检索获得 knn_distance、state_novelty、support score 等
  3. 计算多尺度 completion score
  4. 输出 raw/cdf/zscore 等多种分数
  5. 聚合到逐点或序列级别并计算指标
```

## 4. 方法说明

### 4.1 STSD 分解

`STSDDecomposer` 使用频域可学习低通掩码得到慢变分量初值，再通过卷积校正网络得到慢变状态 `slow`，残差定义为 `residual = x - slow`。慢变状态用于表达系统工况或运行模式，残差用于表达局部扰动和异常候选信号。

训练时额外加入二阶平滑损失，鼓励 `slow` 更平滑，避免慢变分量吞掉全部局部异常信息。

### 4.2 状态编码与通道调制

`StateEncoder` 将慢变状态编码为 `d_state` 维状态向量。`ChannelModulation` 根据状态向量生成每个通道的门控权重，用于调制残差 patch，使模型在不同状态下关注不同通道或局部残差模式。

### 4.3 多尺度 Patch 表征

`MultiScalePatchEncoder` 按 `patch_sizes` 将残差序列切成多个时间尺度的 patch。每个 patch 经共享卷积主干和尺度专属投影头得到 `d_z` 维表征。模型同时构建上下文键 `c`，即当前 patch 邻域内表征的局部平均，用于后续上下文检索。

默认尺度为 `[8, 32]`，表示短尺度和长尺度同时建模。

### 4.4 自监督预训练目标

Stage A 使用两个主要自监督任务：

- Masked completion：随机 mask 一部分 patch，用 Transformer completion head 预测被 mask 的原始 patch，误差形成 `mask_loss`。
- Next-patch prediction：用当前 patch 表征预测下一 patch 的原始值，误差形成 `pred_loss`。

总损失为：

```text
loss = mask_loss + lambda_pred * pred_loss + lambda_smooth * smooth_loss
```

### 4.5 记忆库与两级检索

Stage B 使用训练集正常窗口构建 `MemoryBank`：

- `state_bank`：窗口级状态向量集合。
- `scale memory`：每个尺度上的 patch 表征 `z`、上下文键 `c` 和所属窗口编号。
- `state prototypes`：可选的状态原型，用于支持度分数和机制分析。
- `StateIndex`：状态向量检索索引，优先使用 Faiss，不可用时回退到 torch。

测试阶段先用状态向量检索候选正常窗口，再在候选窗口对应的 patch 记忆中做上下文/表征检索，得到检索距离、状态新颖度和支持度等诊断分数。

原型支持度评分按整个输入 batch 一次性执行上下文近邻检索，并用有限距离掩码处理候选数少于 `top_K` 的补位项；该实现与逐样本检索数值等价，但避免每个 batch 内重复发起大量 CPU/GPU 小算子。正式远程运行应同时限制 OMP/MKL/OpenBLAS 线程数，防止多个 GPU 任务在共享 CPU 上过度订阅。

### 4.6 分数融合

项目保留多种评分口径：

| 分数 | 含义 |
| --- | --- |
| `knn_distance` | 测试 patch 与正常记忆中近邻的距离。 |
| `state_novelty` | 测试窗口状态与正常状态库的距离或新颖度。 |
| `completion_scale<size>` | 对应尺度的补全误差。 |
| `raw_max` | 多个原始子分数取最大值。 |
| `zscore_mean` | 子分数按训练分布标准化后求均值。 |
| `cdf_max` | 子分数经经验 CDF 校准后取最大值。 |
| `cdf_mean` | 子分数经经验 CDF 校准后求均值，是当前主报告默认口径。 |
| `cdf_mean_soft_support` | 将 soft support 替换或纳入后的 CDF mean 变体。 |
| `cdf_softmax` | CDF 校准后用 softmax 权重加权。 |

## 5. 各模块设计

| 模块 | 文件 | 设计说明 |
| --- | --- | --- |
| 配置模块 | `coremad/config.py` | `CoReMADConfig` 统一管理数据、训练、模型、检索、融合、输出路径和恢复运行参数，并在初始化时做合法性检查。 |
| 数据模块 | `coremad/data.py` | 支持 CATCH-style CSV、TEP `.mat` 和 TSB-AD 单文件协议；负责训练/验证/测试切分、标准化、滑窗采样和 DataLoader 构建。 |
| 模型模块 | `coremad/model.py` | 实现 STSD 分解、状态编码、通道调制、多尺度 patch encoder、masked completion head 和 next-patch prediction head。 |
| 记忆库模块 | `coremad/memory.py` | 构建、压缩、保存和加载状态记忆库与尺度记忆库；实现自清洗、coreset、原型构建和候选检索。 |
| 检索索引 | `coremad/faiss_index.py` | 封装 Faiss/torch 状态检索和上下文检索。 |
| 融合模块 | `coremad/scorer.py` | 实现 CDF-PIT fusion、ZScoreMeanFusion 和 raw max fusion。 |
| 训练模块 | `coremad/trainer.py` | 编排 Stage A、Stage B、Test、Full；负责 checkpoint、resume guard、分数导出、指标计算和可视化。 |
| 可视化模块 | `coremad/visualization.py` | 输出训练曲线、分数分布和时间线图。 |

## 6. 关键参数

### 6.1 数据与运行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `MSL` | 数据集名称；CATCH-style 数据可传 CSV stem 或文件名，TEP 使用 `TEP`。 |
| `--data_root` | `./dataset/anomaly_detect` | 数据根目录。TEP 可指向 `TEP-DATA/TEP_Selected_Data` 或其上级目录。 |
| `--data_format` | `auto` | 数据协议：`auto`、`detect`、`tep` 或 `tsb_ad`。TSB-AD 单文件运行必须使用 `tsb_ad`。 |
| `--tep_protocol` | `selected` | TEP 协议：`selected` 为历史三模式八故障子集，`full` 为六模式全部 28 种故障；Stage A/B 兼容检查会校验该字段。 |
| `--artifact_root` | `./artifacts` | 实验产物根目录。 |
| `--experiment_name` | `coremad_msl` | 实验目录名。最终路径为 `<artifact_root>/<experiment_name>`。 |
| `--stage` | `full` | 运行阶段：`stage_a`、`stage_b`、`test`、`full`。 |
| `--resume` | `0` | 是否复用/续跑已有产物。 |
| `--device` | 自动 | 为空时按 `torch.cuda.is_available()` 自动选择。 |
| `--seed` | `42` | 随机种子；正式 rebuttal/论文实验固定为 `42`，CLI 覆盖仅用于调试。 |

### 6.2 窗口与批量参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--seq_len` | `128` | 滑动窗口长度，必须能被所有 `patch_sizes` 整除。 |
| `--batch_size` | `128` | 默认批量大小。 |
| `--train_batch_size` | `None` | Stage A 训练批量；为空时使用 `batch_size`。 |
| `--val_batch_size` | `None` | Stage A 验证批量。 |
| `--memory_batch_size` | `None` | Stage B 建库批量。 |
| `--test_batch_size` | `None` | 测试批量。 |
| `--train_stride` | `1` | Stage A 训练滑窗步长。 |
| `--test_stride` | `1` | 测试滑窗步长。 |
| `--memory_build_stride` | `1` | Stage B 建库滑窗步长。 |
| `--max_train_windows` | `0` | 调试用训练/建库窗口上限，0 表示不限。 |
| `--max_test_windows` | `0` | 调试用测试窗口上限，0 表示不限。 |

### 6.3 模型参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--patch_sizes` | `8 32` | 多尺度 patch 长度列表。 |
| `--d_z` | `128` | patch 表征维度。 |
| `d_state` | `128` | 状态向量维度，配置类参数。 |
| `d_trunk` | `128` | 共享卷积主干输出维度，配置类参数。 |
| `--use_stsd_decomposition` | `1` | 是否使用 STSD 分解。 |
| `--use_channel_modulation` | `1` | 是否启用状态感知通道调制。 |
| `--use_completion_head` | `1` | 是否启用 masked completion head。 |
| `--mask_ratio` | `0.25` | Stage A 随机 mask patch 比例。 |
| `--n_mask_groups` | `4` | 测试时 deterministic completion 分组数。 |
| `--completion_dropout` | `0.1` | completion Transformer dropout。 |

### 6.4 训练参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--stage_a_epochs` | `100` | Stage A 最大训练轮数。 |
| `--early_stop_patience` | `15` | 验证指标无提升的早停耐心。 |
| `--lr` | `2e-3` | 学习率；部分脚本会对 PSM 等数据集设定专属默认值。 |
| `--weight_decay` | `1e-4` | AdamW 权重衰减。 |
| `--grad_clip_norm` | `1.0` | 梯度裁剪阈值。 |
| `--scheduler_eta_min_ratio` | `0.01` | CosineAnnealingLR 最小学习率比例。 |
| `--lambda_pred` | `0.5` | next-patch prediction loss 权重。 |
| `--lambda_smooth` | `0.01` | STSD 平滑损失权重。 |

### 6.5 检索、记忆库与融合参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--top_M` | `50` | Level-1 状态检索候选窗口数。 |
| `--top_K` | `20` | Level-2 上下文/patch 检索候选数。 |
| `--knn_k` | `5` | z-space memory score 的近邻数。 |
| `--clean_ratio` | `0.02` | completion self-cleaning 删除最高补全误差 patch 的比例；`0` 表示不执行净化。 |
| `--memory_seed` | `None` | Stage B 状态原型与 coreset 抽样种子；未设置时继承 `--seed`。 |
| `--memory_audit_mode` | `none` | 记忆库审计级别：`none`、`summary` 或 `full`；`full` 输出逐 patch 保留标记、来源起点和状态稀有度。 |
| `--state_prototype_count` | `0` | 状态原型数，0 表示自动选择。 |
| `--use_two_level_retrieval` | `1` | 是否使用两级检索。 |
| `--use_context_key_retrieval` | `1` | 是否使用上下文键检索。 |
| `--use_faiss` | `1` | 是否优先使用 Faiss。 |
| `--faiss_use_gpu` | `1` | Faiss 可用时是否使用 GPU。 |
| `--faiss_exact_threshold` | `100000` | 状态库小于该阈值时使用精确检索。 |
| `--faiss_ivf_nprobe` | `16` | IVF 检索时的 nprobe。 |
| `--coreset_keep_ratio` | `1.0` | coreset 保留比例。 |
| `--coreset_max_patches_per_scale` | `200000` | 单尺度最多保留 patch 数。 |
| `--evaluation_score_key` | `cdf_mean` | 主评估使用的分数键。 |

## 7. 数据集介绍

### 7.1 CATCH-style 异常检测数据

默认数据根目录为 `./dataset/anomaly_detect`，要求包含：

```text
dataset/anomaly_detect/
  DETECT_META.csv
  data/
    MSL.csv
    SMAP.csv
    PSM.csv
    SWAT.csv
    SMD.csv
    ...
```

`DETECT_META.csv` 至少需要为每个数据文件提供 `file_name` 和 `train_lens`。加载器会根据 `train_lens` 将长 CSV 拆为训练段和测试段，如果 CSV 末尾存在 label block，则用其生成测试标签。

当前脚本层重点支持的数据包括：

| 数据集 | 说明 |
| --- | --- |
| `MSL` | NASA MSL 多变量遥测异常检测数据。 |
| `SMAP` | NASA SMAP 多变量遥测异常检测数据。 |
| `PSM` | Server Machine/业务指标类异常检测数据；默认使用 `interleaved` 验证切分。 |
| `SWAT` | 工业控制水处理过程异常检测数据。 |
| `SMD` | 服务器机器数据；当前项目按联合版本处理。 |
| `GECCO`、`GENESIS`、`CICIDS`、`CreditCard`、`NYC`、`CalIt2`、`Synthetic_*`、`ASD_*` | 由 `scripts/main/train_*.sh` 提供训练入口，具体可用性取决于本地数据是否存在于 `dataset/anomaly_detect/data/`。 |

### 7.2 TEP 数据

TEP 使用 `.mat` 文件，并提供两套显式协议。`selected` 是历史兼容默认值，继续读取位于 `TEP-DATA/TEP_Selected_Data/` 的受控子集：

- 正常文件：`m1d00.mat`、`m3d00.mat`、`m4d00.mat`。
- 故障文件：`m1/m3/m4` 模式下的 `d01`、`d04`、`d06`、`d07`、`d10`、`d13`、`d14`、`d27`。
- 输入通道数：前 53 个通道。

`full` 协议从完整仓库自动发现 `M1-M6/mXdYY.mat`，由一个模型联合学习六个 `d00` 正常工况，并测试全部 `6 × 28 = 168` 条故障序列。每个正常序列按尾部 15% 作为审计段，并在训练段和审计段之间保留 128 点隔离区；当前完整数据对应 35,958 个正常训练点、35,196 个训练窗口、6,480 个正常审计点和 5,718 个审计窗口。故障测试使用 `stride=1`，共约 1,132,009 个窗口。

完整数据的 168 条故障全部纳入：158 条长度为 7,201 点，10 条为提前终止的短序列，但均不少于 `seq_len=128`。数据审计脚本会逐文件记录模式、文件号、官方 IDV、行列数、完整性和纳入状态，并生成 JSON、CSV 和 Markdown。

TEP 的正常模式机制审计来自验证正常窗口，故障机制分析来自独立故障测试序列。由于完整故障序列均为全正类，`full` 只定位为机制验证，不计算或报告无效的 `AUROC`、`AUPRC` 和 `F1`；主报告使用 `SMC@K`、`SFR`、`SMR@K`、`Mode-FPR-Std`、`delta_mem_mode`、尾部校准误差、跨模式距离边际和故障一致性，同时补充 `SMR@K-sequence-balanced`、`fault-normal-gap-sequence-balanced` 与 `cross-mode-margin-sequence-balanced`。

完整 MMFDD-TEP 上游仓库位于 `Multi-mode-Fault-Diagnosis-Datasets-with-TE-process/`，来源为 `THUFDD/Multi-mode-Fault-Diagnosis-Datasets-with-TE-process`，本地检出提交为 `4d7b8a7`。数据覆盖 `M1` 至 `M6`；每种模式包含 `d00` 正常序列和 `d01` 至 `d28` 故障序列，共 174 个核心 `.mat` 文件。上游说明称每段数据时长为 72 小时，包含 12 个输入变量、41 个测量变量和 28 个扰动变量，因此检测模型使用前 53 个过程变量。

项目统一遵循上游官方编号规则：文件 `d01` 至 `d28` 与物理扰动 `IDV28` 至 `IDV1` 反向对应，即 `IDV = 29 - d`；`d00` 表示正常数据。元数据同时保留 `file_fault_id`（原始 `dXX` 文件号）和 `fault_id`（官方 IDV），论文表格与机制图统一展示官方 IDV。由此，当前受控子集的八个文件编号对应 `IDV28`、`IDV25`、`IDV23`、`IDV22`、`IDV19`、`IDV16`、`IDV15`、`IDV2`。

### 7.3 TSB-AD-U/M

TSB-AD 作为独立扩展 benchmark，不与原 Table 1 的 CATCH-style 协议数字混合。当前本地目录为：

```text
dataset/
  TSB-AD-M/TSB-AD-M/*.csv
  TSB-AD-U/TSB-AD-U/*.csv
```

接入协议如下：

- 每个 CSV 是一个独立实验单元，不跨序列共享模型、normalizer、memory 或融合统计量。
- 文件名 `_tr_N_` 中的 `N` 是正常训练前缀长度；Stage A、Stage B、CDF 和 Z-score 拟合只能使用此前缀。
- 对完整 CSV 生成逐点分数和指标；标签只用于最终评估。
- 自动从 CSV 列数识别 U/M 通道数，并解析来源数据集、序列 ID、训练长度和首异常位置。
- VUS window 使用 TSB-AD 官方 rank-1 ACF 规则从第一信号通道估计，不使用测试标签估计。
- 官方清单规模为 M all 200、tuning 20、eval 180；U all 870、tuning 48、eval 350、eval-full 822。
- 正式默认使用 M-Eva 与 U-Eva；U-Eva-Full 只在显式指定时运行。

## 8. 训练命令

### 8.1 直接使用统一入口

Stage A：

```bash
python run.py --stage stage_a --dataset MSL --data_root ./dataset/anomaly_detect --experiment_name msl_baseline
```

Stage B：

```bash
python run.py --stage stage_b --dataset MSL --data_root ./dataset/anomaly_detect --experiment_name msl_baseline --resume 1
```

Test：

```bash
python run.py --stage test --dataset MSL --data_root ./dataset/anomaly_detect --experiment_name msl_baseline --resume 1
```

完整流程：

```bash
python run.py --stage full --dataset MSL --data_root ./dataset/anomaly_detect --experiment_name msl_baseline
```

限制窗口数的快速调试：

```bash
python run.py --stage full --dataset MSL --data_root ./dataset/anomaly_detect --experiment_name debug_msl --max_train_windows 200 --max_test_windows 200 --stage_a_epochs 2 --device cpu
```

### 8.2 使用主实验脚本

主脚本通常接受阶段和实验名前缀，具体细节以脚本内部为准：

```bash
bash scripts/main/train_msl.sh full msl_baseline
bash scripts/main/train_smap.sh full smap_baseline
bash scripts/main/train_psm.sh full psm_baseline
bash scripts/main/train_smd.sh full smd_baseline
bash scripts/main/train_swat.sh full swat_baseline
bash scripts/main/train_tep.sh full tep_baseline
```

完整 MMFDD-TEP 机制验证：

```bash
DATA_ROOT=./Multi-mode-Fault-Diagnosis-Datasets-with-TE-process \
bash scripts/main/train_tep_full.sh full tep_full_mechanism
```

续跑示例：

```bash
RESUME=1 bash scripts/main/train_msl.sh full msl_baseline_20260404_153000
```

### 8.3 消融实验

完整消融矩阵：

```bash
bash scripts/ablation/run_ablation_matrix.sh
```

单数据集单消融：

```bash
bash scripts/ablation/run_full_ablation.sh PSM
bash scripts/ablation/run_a1_wo_decomposition.sh PSM
bash scripts/ablation/run_a2_global_retrieval.sh PSM
bash scripts/ablation/run_a3_raw_max.sh PSM
bash scripts/ablation/run_a4_multi_scale.sh PSM
```

分组矩阵：

```bash
bash scripts/ablation/run_a1_matrix.sh
bash scripts/ablation/run_a2_matrix.sh
bash scripts/ablation/run_a3_matrix.sh
bash scripts/ablation/run_a4_matrix.sh
```

汇总论文表格：

```bash
python scripts/ablation/generate_paper_tables.py
```

### 8.4 敏感性分析

单数据集：

```bash
python scripts/sensitivity/run_dataset_sensitivity.py \
  --dataset MSL \
  --preset priority \
  --artifact-root ./artifacts/sensitivity/msl \
  --data-root ./dataset/anomaly_detect \
  --device cuda \
  --skip-existing
```

全数据集：

```bash
python scripts/sensitivity/run_all_dataset_sensitivity.py \
  --preset priority \
  --artifact-root ./artifacts/sensitivity \
  --data-root ./dataset/anomaly_detect \
  --device cuda \
  --skip-existing \
  --plot-after-run
```

绘图：

```bash
python scripts/sensitivity/plot_dataset_sensitivity.py \
  --summary-json ./artifacts/sensitivity/msl_sensitivity/summary/msl_sensitivity_<timestamp>_summary.json
```

### 8.5 TEP 机制分析

完整主实验入口按顺序执行数据审计、Stage A、Stage B、168 条故障序列测试、机制导出、指标计算、绘图和 rebuttal 表格生成。测试结果按序列原子写入 `test_sequence_shards/mXdYY.npz`，机制窗口写入 `tep_mechanism/fault_window_shards/`；设置 `RESUME=1` 后，仅跳过协议、模型、校准产物及全部原始 `.mat` 文件签名一致且产物完整的序列。默认不生成约 4 GB 的窗口级 CSV，只保留紧凑 NPZ、168 行序列汇总和数据审计表。

```bash
RESUME=1 DATA_ROOT=./Multi-mode-Fault-Diagnosis-Datasets-with-TE-process \
bash scripts/main/train_tep_full.sh full tep_full_mechanism
```

现有 TEP 消融仍使用 `selected`，不会扩展为全量消融矩阵。selected 手动机制分析方式如下：

```bash
python scripts/tep/export_mechanism_logs.py --experiment_dir artifacts/tep_baseline_xxx
python scripts/tep/compute_mechanism_metrics.py --experiment_dirs artifacts/tep_baseline_xxx
python scripts/tep/plot_mechanism_figures.py --experiment_dirs artifacts/tep_baseline_xxx
```

完整协议还会生成六模式 state embedding、`6 × 6` 检索混淆矩阵、各模式正常 FPR、fault-normal gap、`28 × 6` 官方 IDV 热力图，以及 T2 全故障可用性表和 T9 selected/full 机制指标对比表。T9 未显式给定 selected 实验时会从 `artifacts/` 与 `TEP-abalation/` 自动选择 selected 基线；历史 schema v2 日志会在只读加载时根据 `sequence_name` 恢复原始文件号并转换为官方 IDV，缺失的序列等权指标仅在内存中补算，不修改旧产物。

常用环境变量：

```bash
TEP_MECHANISM_POSTPROCESS=0
TEP_MECHANISM_WINDOW_AGGREGATION=p95
TEP_MECHANISM_RUN_PLOTS=1
TEP_MECHANISM_RUN_METRICS=1
TEP_A4_COMPARE_STRICT=1
```

### 8.6 TSB-AD benchmark

安装官方 VUS 指标依赖并校验数据：

```bash
python -m pip install -r scripts/tsb_ad/requirements.txt
python scripts/tsb_ad/validate_setup.py --edition both --deep
```

单文件配置冒烟与实际运行：

```bash
python scripts/tsb_ad/run_benchmark.py --edition M --split eval --limit 1 --dry-run --device cuda
python scripts/tsb_ad/run_benchmark.py --edition U --split eval --limit 1 --device cuda
```

正式固定 seed 运行：

```bash
python scripts/tsb_ad/run_benchmark.py --edition M --split eval --seed 42 --device cuda --resume
python scripts/tsb_ad/run_benchmark.py --edition U --split eval --seed 42 --device cuda --resume
```

也可直接运行单个 CSV：

```bash
python run.py --stage full --data_format tsb_ad \
  --dataset 005_MSL_id_4_Sensor_tr_855_1st_2700.csv \
  --data_root ./dataset/TSB-AD-M \
  --artifact_root ./artifacts/tsb_ad/M/eval/seed_42 \
  --experiment_name 005_MSL_id_4_Sensor_tr_855_1st_2700 \
  --evaluation_score_key cdf_mean --seed 42
```

## 9. 脚本功能

### 9.1 `scripts/main/`

| 脚本 | 功能 |
| --- | --- |
| `_train_detect_dataset.sh` | CATCH-style 数据集训练公共包装脚本。 |
| `train_msl.sh`、`train_smap.sh`、`train_psm.sh`、`train_smd.sh`、`train_swat.sh`、`train_tep.sh` | 主数据集训练入口；`train_tep.sh` 默认使用 selected 协议。 |
| `train_tep_full.sh` | 完整 MMFDD-TEP 六模式全故障机制验证入口，含审计、分片续跑、机制统计、绘图和 rebuttal 表格。 |
| `train_gecco.sh`、`train_genesis.sh`、`train_cicids.sh`、`train_creditcard.sh`、`train_nyc.sh`、`train_calit2.sh` | 其他检测数据集训练入口。 |
| `train_synthetic_*.sh` | 合成数据训练入口。 |
| `train_asd_dataset_*.sh` | ASD 系列数据训练入口。 |
| `download_tep_selected.py` | 下载或准备 TEP selected data。 |
| `eval_diagnostic_scores.py` | 对已导出的诊断分数重新评估。 |
| `inspect_memory_artifacts.py` | 检查 memory bank 和相关产物。 |
| `visualize_support_diagnostics.py` | 可视化 support score 与 completion score。 |
| `visualize_state_prototypes.py` | 可视化状态原型。 |
| `visualize_stsd_decomposition.py` | 可视化 STSD 分解效果。 |
| `visualize_stsd_evidence.py` | 生成 STSD 有效性证据图。 |
| `visualize_synthetic_radar.py` | 合成数据雷达/对比图。 |
| `visualize_switch_context_evidence.py` | 状态切换上下文证据图。 |
| `visualize_normal_transition_no_false_alarm.py` | 正常转换不误报案例图。 |
| `visualize_interpretability_case_panel.py` | 可解释性案例面板。 |
| `draw_framework_diagram.py` | 绘制模型框架图。 |

### 9.2 `scripts/ablation/`

| 脚本 | 功能 |
| --- | --- |
| `run_ablation_matrix.sh` | 正式消融矩阵总入口。 |
| `run_dataset_ablation.sh` | 单数据集消融公共入口。 |
| `run_full_ablation.sh` | 完整模型消融基线。 |
| `run_a1_*` | A1 表征主干消融。 |
| `run_a2_*` | A2 检索机制消融。 |
| `run_a3_*` | A3 融合策略消融。 |
| `run_a4_*` | A4 时间尺度消融。 |
| `run_msl.sh`、`run_smap.sh`、`run_psm.sh`、`run_smd.sh`、`run_swat.sh`、`run_tep.sh` | 数据集包装入口。 |
| `bootstrap_stage_a.py`、`bootstrap_stage_b.py` | 阶段产物复用/引导辅助脚本。 |
| `collect_results.py` | 汇总消融结果为 CSV/JSON/Markdown。 |
| `collect_global_auc_table.py` | 汇总全局 AUC 表。 |
| `generate_paper_tables.py` | 生成论文主表、A3 表、稳定性分析表和 TEP 序列级表。 |
| `materialize_a3_from_full.py` | 从完整实验产物物化 A3 对比所需结果。 |

### 9.3 `scripts/sensitivity/`

| 脚本 | 功能 |
| --- | --- |
| `run_dataset_sensitivity.py` | 单数据集超参数扫描。 |
| `run_all_dataset_sensitivity.py` | 多数据集敏感性分析总入口。 |
| `collect_results.py` | 汇总敏感性结果。 |
| `plot_dataset_sensitivity.py` | 单数据集敏感性绘图。 |
| `plot_cross_dataset_sensitivity*.py` | 跨数据集敏感性图与论文面板。 |
| `run_msl_sensitivity.py`、`plot_msl_sensitivity.py` | MSL 兼容包装入口。 |

### 9.4 `scripts/tep/`

| 脚本 | 功能 |
| --- | --- |
| `tep_common.py` | TEP 机制分析公共函数。 |
| `audit_tep_dataset.py` | 审计 selected/full 文件清单、尺寸、官方 IDV 映射、短序列和纳入状态。 |
| `run_mechanism_pipeline.sh` | 机制分析流水线入口。 |
| `export_mechanism_logs.py` | 导出正常/故障窗口机制日志；full 协议按序列分片并支持断点续跑。 |
| `compute_mechanism_metrics.py` | 计算 SMC、SFR、SMR、证据一致性及序列等权机制指标。 |
| `plot_mechanism_figures.py` | 绘制核心机制图。 |
| `export_paper_mechanism_figures.py` | 导出论文机制图。 |
| `export_full_mechanism_suite.py` | 导出完整机制图、表和案例套件。 |
| `generate_rebuttal_tables.py` | 生成 T2 六模式故障可用性表和 T9 selected/full 机制指标对比表。 |

### 9.5 `scripts/tsb_ad/`

| 脚本/目录 | 功能 |
| --- | --- |
| `run_benchmark.py` | 按 edition、split 和 seed 为每个 CSV 启动独立子进程，记录日志、耗时、失败状态并支持断点续跑；支持官方 manifest 互斥分片和延迟统一汇总，未指定 seed 时正式默认只运行 `42`，并自动关闭正式 rebuttal 可视化。 |
| `collect_results.py` | 固定汇总四种正式融合策略和全部诊断子分数，默认只收集 `seed_42` 并生成长表以及逐序列、来源数据集、官方总体和数据集 macro-average 四张宽表；`--all-seeds` 仅用于历史产物审计。 |
| `validate_setup.py` | 校验本地 M/U 文件清单、官方 split 数量和 CSV 通道结构。 |
| `common.py` | 维护文件名协议、官方清单映射和数据目录解析。 |
| `manifests/` | TSB-AD 官方 M/U all、tuning、eval 与 eval-full 文件清单。 |
| `requirements.txt` | 固定 TSB-AD 官方指标实现版本。 |

### 9.6 `scripts/rebuttal/muqn/`

| 脚本 | 功能 |
| --- | --- |
| `export_retrieval_logs.py` | 在同一 checkpoint 上导出 full、no-state、no-context 三种策略的粗检索窗口、细检索 patch 来源及距离。 |
| `compute_retrieval_evidence.py` | 计算条件代理距离、策略间 Jaccard、滞后 Jaccard，并给出块 bootstrap 置信区间。 |
| `run_contamination_sweep.py` | 按异常事件 K 折构造 Stage-B 污染；注入事件整体从评估中排除，避免直接数据泄漏。 |
| `analyze_purification_audit.py` | 汇总普通正常、稀有正常和注入异常 patch 的净化移除率、最终保留率及补全误差分布。 |
| `baselines/prepare_baseline_data.py` | 将项目数据导出为 PaAno 和 DAMP 适配格式，并记录训练边界。 |
| `baselines/run_paano.py` | 只读调用官方 PaAno 多变量实现。 |
| `baselines/run_damp.py`、`run_damp_multidim.m` | 调用官方多维 DAMP；在输出目录生成带分数返回值的临时函数副本，不修改第三方源码。 |
| `baselines/audit_gdflex.py` | 记录 GDFlex 官方实现的 ML/UCR 单变量输入约束，防止误报为原生多变量基线。 |
| `baselines/evaluate_baseline_scores.py` | 使用统一逐点 AUROC、AP 和 best-F1 口径评估外部基线分数。 |

## 10. 主要产物

每个实验目录通常包含：

| 产物 | 说明 |
| --- | --- |
| `config.json` | 实验配置快照。 |
| `stage_a.pt` | Stage A 最佳 checkpoint，包含模型、normalizer、优化器、scheduler、epoch 和 history。 |
| `stage_a_last.pt` | Stage A 最近 checkpoint，用于续训。 |
| `stage_a_loss_curve.png` | Stage A 训练/验证曲线。 |
| `memory.pt` | Stage B 正常记忆库。 |
| `memory_meta.json` | 记忆库元信息与兼容性检查字段。 |
| `faiss_state.index` | Faiss 状态索引，可用时生成。 |
| `cdf_fusion.npz/json` | CDF-PIT 融合器。 |
| `zscore_fusion.json` | z-score mean 融合器。 |
| `test_diagnostic_scores.npz` | 测试诊断子分数及标签。 |
| `test_scores*.npy/csv` | `raw_max`、`zscore_mean`、`cdf_mean`、`cdf_max` 及全部诊断子分数的逐点或逐序列结果。 |
| `test_metrics.json` | selected、四种正式融合以及全部诊断子分数的完整指标。 |
| `score_timeline*.png`、`*_distribution.png` | 分数时间线和分布图。 |

TSB-AD 每条序列的目录固定为
`artifacts/tsb_ad/<edition>/<split>/seed_<seed>/<file_stem>/`，并必须同时包含：

- `test_scores_raw_max.npy`
- `test_scores_zscore_mean.npy`
- `test_scores_cdf_mean.npy`
- `test_scores_cdf_max.npy`
- `test_scores_knn_distance.npy`
- `test_scores_state_novelty.npy`
- `test_scores_completion_scale8.npy`
- `test_scores_completion_scale32.npy`
- `test_diagnostic_scores.npz`
- `test_metrics.json`

split 目录额外生成 `results_long.csv` 和四张论文宽表；`run_record.json` 与阶段日志记录单序列运行时间和失败原因。

## 11. 评估指标

逐点异常检测主要指标包括：

- `ROC-AUC`
- `PR-AUC`
- `best_f1`
- `pa_best_f1`
- `event_best_f1`
- `VUS` 相关指标
- Affiliation / Range precision-recall 相关指标

TEP selected 历史序列结果可保留原评估字段；full 的故障序列均为全正类，因此不报告 `AUROC`、`AUPRC` 和 `F1`。full 重点报告窗口加权机制指标及对应的序列等权指标，避免 158 条完整序列与 10 条短序列的长度差异改变结论。

论文表格和脚本中当前默认主报告分数为 `cdf_mean`，但所有正式 SCAR 运行必须同时保存并汇总 `raw_max`、`zscore_mean`、`cdf_mean`、`cdf_max` 以及 `completion_scale*`、`knn_distance`、`state_novelty` 和启用时的 `soft_support_score`。每个融合分数和每个诊断子分数都必须配套输出九项注册指标：AUROC (`roc_auc`)、AP (`pr_auc`)、Point-F1 (`point_best_f1`)、PA-F1 (`pa_best_f1`)、Aff-P (`aff_precision`)、Aff-R (`aff_recall`)、Aff-F1 (`aff_f1`)、VUS-ROC (`vus_roc`) 和 VUS-PR (`vus_pr`)；缺少任一字段的正式产物不得复用。P0 额外生成五主集、E1-E5 检索策略和 TEP 全分数表；P1 的 E29/E30/E31 均保留 SCAR/global 的完整分数列，E30 macro 对九项指标逐分数聚合；TSB-AD 对每个可用分数生成逐序列、来源数据集、官方总体和 dataset-macro 表。`cdf_mean` 仍是预先固定的主结果，其余分数用于诊断和融合对照；不得逐 CSV、逐来源数据集或逐指标挑选最优策略。TEP 序列级单类或无连续时间邻接的协议仍保留全部字段，但不适用项必须写为 `NaN` 并记录原因，不能伪造数值。

九项指标契约采用“字段强制、适用性显式”的原则：具备正负标签且时间轴连续的逐点检测任务必须输出有限数值；单类 TEP fault-only 机制序列保留九个字段并写入 `NaN` 与原因；E11/E12/E39 及纯效率/资源汇总不产生新的异常分数，只复用其来源实验的指标，不单独套用九项分类指标。PA-F1 的最优阈值搜索使用与逐阈值扫描完全等价的向量化实现，以避免 SMD、SWaT 等长序列在全分数评估时出现不可接受的 Python 循环开销。

`coremad/temporal_metrics.py` 统一兼容 TSB-AD 1.5 的
`get_metrics(score, labels, slidingWindow, pred, ...)` 接口及旧版带
`metric="all"` 的接口，同时兼容 `VUS-ROC`/`VUS_ROC` 两类返回键。Aff-P、
Aff-R 和 Aff-F1 使用与 Point-F1 最优阈值相同的二值预测，通过 TSB-AD 官方
Affiliation event 实现计算；VUS-ROC/VUS-PR 仍由 TSB-AD 官方实现计算。这样九项
指标在 SCAR、四个 baseline 和全部融合/子分数之间保持同一阈值口径，且避免长序列
额外执行 100 次 Affiliation 阈值扫描。

正式 SCAR/TEP 任务通过 manifest 环境元数据固定
`SCAR_METRIC_WORKERS=8`，在 GPU 推理结束后使用 `spawn` 进程并行评估不同融合与
子分数。每个 worker 将 Torch CPU 线程限制为 1，只读取自己的 labels/scores 并调用
无共享状态的评估器，不 fork 已初始化的 CUDA 上下文，也不改变模型、分数或指标
定义；最终结果仍按注册表顺序写入。该设置利用远程多核 CPU，避免 SMD/SWaT 的官方
VUS 计算长时间阻塞下一项 GPU 任务。

## 12. 维护清单

以后修改项目时按以下规则维护文档：

- 修改模型结构：更新第 3、4、5、6 节。
- 修改数据加载或新增数据集：更新第 7 节和训练命令。
- 修改训练入口或参数：更新第 6、8、9 节。
- 修改实验产物或评估指标：更新第 10、11 节。
- 新增/删除/重命名脚本：更新第 9 节。
- 每次代码、配置、文档或实验口径变更：在 `CHANGELOG_MAINTENANCE.md` 新增一条记录。

## 13. Rebuttal 专项方案

Reviewer ef7G 的逐条证据方案、验收标准、执行顺序和缺失环境清单统一维护在
`Reviewer_ef7G_解决方案.md`。该专项固定以下口径：

- CATCH 扩展评估覆盖 12 个真实数据集族和 12 个合成文件（六类异常）；
- ASD 的 12 个子序列先做 dataset-level macro average，不作为 12 个独立数据集重复加权；
- 正式新增实验统一使用固定模型 seed `42`，SCAR 主融合固定为 `cdf_mean`；CLI 可覆盖 seed 进行调试，但结果不得进入正式表格；
- TEP 以 M1-M6、168 条全故障序列的 full 协议作为 rebuttal 主证据；
- 最近 retrieval 基线优先完成 PaAno，同步审计 DAMP、GDFlex、PUAD 和 PGRF-Net；
- Section 3.2-3.4 与 Figure 1 统一为 Representation Learning、Memory
  Construction、Retrieval and Anomaly Scoring 三阶段，Level 1/2 仅属于第三阶段；
- Related Work 保留 reviewer 认可的现有结构，在此基础上补齐 Table 1 映射、
  competitor selection rule 和各类方法相对 SCAR 的明确差异。

固定 seed 协议只约束模型随机初始化与随机训练流程。污染事件 folds、检索统计的 block
bootstrap、超参数扫描以及推理预热后的重复计时继续保留；这些结果不得表述为模型
seed 方差。确定性方法运行一次。

专项实验尚未在当前桌面环境启动；正式运行依赖完整 PyTorch/SciPy/Faiss/CUDA
环境，DAMP/GDFlex 还依赖 MATLAB 或兼容运行环境。

## 14. CATCH 主实验对齐审计

2026-07-24 对当前五数据集主实验与 CATCH 官方论文、代码和本地
`CATCH-master` 进行了逐项核对。结论是：**数据与主评测指标已经对齐，但随机种子、
验证协议和重复运行口径尚未完全对齐，因此当前论文中直接引用的 CATCH 官方单次结果
不能表述为与 SCAR 完全相同训练协议下的重跑结果。**

### 14.1 已对齐项目

| 项目 | SCAR 当前设置 | CATCH 官方设置 | 结论 |
| --- | --- | --- | --- |
| 数据文件 | `dataset/anomaly_detect/data/*.csv` | CATCH 发布的同名 CSV | 五个主数据集和 `DETECT_META.csv` 的 SHA-256 均一致 |
| 训练/测试边界 | 读取 `DETECT_META.csv` 的 `train_lens` | 读取同一 `train_lens` | 对齐 |
| 主数据集 | MSL、PSM、SMAP、SMD、SWaT | CATCH benchmark 对应数据集 | 对齐 |
| 标准化原则 | 仅在训练侧数据拟合 `StandardScaler` | 仅在训练侧数据拟合 `StandardScaler` | 原则对齐，拟合子集比例不同 |
| 测试标签使用 | 仅用于最终评估 | 仅用于最终评估 | 对齐 |
| 主指标 AR | `sklearn.metrics.roc_auc_score` | `sklearn.metrics.roc_auc_score` | 对齐 |
| 主指标 AP | `sklearn.metrics.average_precision_score` | `sklearn.metrics.average_precision_score` | 对齐 |
| 测试 DataLoader | `drop_last=False` | `drop_last=False` | 对齐 |

CATCH 论文 Table 4 的 anomaly ratio 使用全序列长度作为分母，而 SCAR 数据统计表明确
报告 test split anomaly ratio，因此数值不同但标签和数据并未不一致。例如 MSL 的
5.88% 与 10.53% 分别对应全序列口径和测试集口径。

### 14.2 尚未完全对齐项目

| 项目 | SCAR 当前设置 | CATCH 官方设置 | 影响与处理 |
| --- | --- | --- | --- |
| 随机种子 | 正式协议固定 `42` | 官方 benchmark 固定 `2021` | seed 不同；rebuttal 重跑统一使用预先固定的 `42` 并如实披露 |
| 重复次数 | 固定 seed 单次模型运行 | 官方发布表为固定 seed 单次结果 | 均不提供模型初始化方差；不得报告模型 seed mean/std |
| 验证比例 | 15% | 20% tail split | 模型选择可见的训练样本量不同 |
| 验证方式 | MSL/SMAP/SWaT 为 tail；PSM/SMD 为 interleaved | 五个数据集均为 80/20 tail | PSM/SMD 差异尤其明显 |
| 验证间隔 | tail split 启用 128 点 gap | 无显式 gap | 训练/验证隔离方式不同 |
| SCAR 窗口 | 五数据集统一 `seq_len=128` | CATCH 多数为 192，SWaT 为 2048 | 属于方法专属超参数，不要求强行相同，但必须如实披露 |
| CATCH 测试分块 | SCAR stride 1 重叠窗口并平均覆盖 | 官方 CATCH score 路径使用非重叠完整窗口，尾部不足部分补零 | 属于官方方法实现差异；统一 evaluator 时保留并披露 |
| 优化器 | AdamW + cosine scheduler | Adam + OneCycleLR | 属于方法专属训练配置，不要求相同 |

### 14.3 后续统一口径

1. 共享协议必须统一：完全相同的 CSV、`train_lens`、测试标签、AUROC/AP evaluator、
   数据集级汇总方式和正式随机种子 `42`。
2. 方法专属配置保留官方推荐值：窗口长度、patch、网络宽度、优化器、学习率和 epoch
   不强制与 SCAR 相同；这些参数分别通过各自训练侧验证集选择，不查看测试标签。
3. CATCH 不重跑，论文中的 CATCH 数字统一标记为 `reported from CATCH`，不得写成
   `reproduced under our unified protocol`。
4. 正式 rebuttal 只补跑 SCAR 相对 CATCH benchmark 缺失的 30 个 CSV；这些运行固定
   seed `42`，不得从调试 seed 或多个 score 版本中逐数据集挑选最优值。
5. “all methods are evaluated under the same protocol”应改为更精确的表述，明确
   SCAR 新结果与 CATCH 官方 reported 结果的来源差异。

## 15. Reviewer muQn 专项实现

### 15.1 记忆库可追溯性

Stage B 现在为每个尺度 patch 保存 `raw_starts`，检索详情同时返回粗候选窗口原始起点、
细邻居所属窗口、细邻居 patch 起点、上下文距离和有效掩码。旧版 `memory.pt` 仍可加载，
但未知来源统一标记为 `-1`，正式机制实验必须重建 Stage B。

`memory_meta.json` 记录每尺度净化前、净化后和 coreset 后数量、实际随机种子及 cap
是否生效。`--memory_audit_mode full` 额外生成逐 patch 审计 NPZ 和状态原型距离，
用于统计稀有正常误删率与注入异常保留率。

### 15.2 检索证据

`scripts/rebuttal/muqn/export_retrieval_logs.py` 在同一模型和测试窗口上运行：

- `full`：状态粗检索和上下文细检索均启用；
- `no_state`：关闭粗检索，在全局 patch memory 中检索；
- `no_context`：保留状态粗检索，以 patch 表征替代上下文键。

`compute_retrieval_evidence.py` 使用窗口均值、趋势和低频能量作为条件代理，报告粗/细
代理距离、策略间邻居集合 Jaccard、时间滞后 Jaccard 和块 bootstrap 区间。

### 15.3 净化与污染协议

`clean_ratio` 的正式扫描值为 `0/0.005/0.01/0.02/0.05/0.10`，并在
MSL、PSM、SMAP、SMD、SWaT 五个主数据集完整运行。污染率固定为
`0%/1%/3%/5%/10%`，每个污染率比较 no purification 与默认 `clean_ratio=0.02`，
并在 `10%` 污染率增加 stronger purification `clean_ratio=0.10`。污染实验按连续
异常事件做 3 折划分，只把指定折事件窗口注入 Stage-B memory，并在该折评估中整段
排除这些事件；因此不会将已注入的同一异常事件再次用于计分。每次运行保存事件列表、
固定候选顺序、注入窗口、实际污染率、评估掩码摘要和 held-out AUROC/AP/F1。若某折
唯一异常窗口不足以达到目标污染率，使用固定 seed 的重复随机排列补足并显式记录
`sampling_with_replacement`。稀有正常误删率与低误差异常存活率直接从上述 audit
产物计算，不启动额外训练。

### 15.4 基线边界

五个主数据集的正式相关基线固定运行 PaAno、PUAD 和 PGRF-Net，
并与 SCAR 在相同远程服务器上同步记录性能、时间、GPU/CPU 内存和产物大小。CATCH
不重跑，只引用官方 benchmark 结果并标记为 `reported from CATCH`。MEMTO 不进入
正式运行或结果汇总，其源码、adapter 和既有失败目录只保留作历史审计。PaAno
使用官方多变量实现。DAMP 官方 `DAMP_Multidim.m` 和 GDFlex 均依赖 MATLAB；当前没有
可用 MATLAB 环境，因此只保留只读适配和协议审计，不生成未经官方环境验证的替代实现
数字，也不把“缺少 MATLAB”描述成任务协议不兼容。GDFlex 还需单独说明其官方入口绑定
ML/UCR 单变量 MATLAB 数据协议。完整命令见 `scripts/rebuttal/muqn/README.md`。

## 16. 统一资源监控与效率实验入口

SCAR 的 `run.py` 默认对 Stage A、Stage B 和 Test 开启资源监控，每次运行使用同一个
`invocation_id`，并将阶段尝试追加写入
`artifacts/<experiment_name>/resource_metrics.json`。写入采用临时文件加原子替换；
resume 跳过的阶段记录为 `skipped`，不使用零耗时冒充有效结果，也不进入本次汇总。

关键参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--resource_monitor` | `1` | 是否生成统一资源记录；设为 `0` 可用于监控开销对照。 |
| `--resource_sample_interval` | `0.1` | CPU/GPU 后台采样周期，单位为秒，必须大于零。 |

SCAR 子区间包括 `train_loop`、`memory_build`、`fusion_fit`、batch 级纯
`inference`、端到端 `scoring`、`evaluation` 和 `result_export`。吞吐同时记录实际
窗口元素 `window_points`、实际覆盖唯一时间点 `covered_points` 和输入长度
`input_points`，避免窗口上限或分段协议造成口径高估。CPU 时间按 PID 累加，已退出子
进程的消耗不会丢失。

GPU 记录逻辑索引、物理索引/UUID 和映射来源；`CUDA_VISIBLE_DEVICES` 会参与 NVML
设备解析，SCAR 进程内监控可用时优先使用 PyTorch 设备 UUID。资源 JSON 的追加更新由
跨进程文件锁保护。不可用指标统一写入 `null` 和明确原因。

正式效率实验只训练固定 seed `42` 的模型一次。推理阶段在预热后独立计时 3 次并保留
原始计时、均值和标准差；该波动仅表示系统测量噪声，不是模型随机种子方差。

外部 baseline 通过只读命令包装器接入。以下命令仅展示包装器接口；CATCH 不进入
正式 rebuttal 队列：

```bash
python scripts/efficiency/monitor_command.py \
  --output artifacts/baseline_efficiency/CATCH/MSL/seed_42/resource_metrics.json \
  --method CATCH --dataset MSL --seed 42 \
  --stage inference --device cuda:0 \
  --points 73729 --windows 384 --timeout 3600 \
  -- python <baseline-entrypoint>
```

包装器从 `Popen` 前开始计时，启动失败同样写入 failed attempt；stdout/stderr 采用
追加模式并写 invocation 边界，透传非零退出码，超时返回 124，并以限时 terminate/kill
流程清理完整子进程树。安装固定监控依赖的命令为：

```bash
python -m pip install -r scripts/efficiency/requirements.txt
```

外部包装器默认执行严格依赖预检；CPU 运行缺少 `psutil`，或 CUDA 运行无法初始化目标
GPU 的 NVML 时，会在启动 baseline 前失败。`--strict-dependencies 0` 仅用于调试，
部分监控结果不得写入论文效率表。

完整字段定义和命令示例见 `scripts/efficiency/README.md`。五数据集 baseline 数据适配器、
批量调度和 SHA-256 完整性清单属于后续效率 runner，不在本次监控模块中修改第三方源码。

## 17. Rebuttal 正式执行范围

当前 rebuttal 不允许提交图片，所有新增结果以表格和文字分析呈现。Figure 1 重绘、
检索可视化、敏感性曲线和性能—效率曲线仅作为论文修订 backlog，不进入远程实验关键
路径。正式训练全部在远程服务器执行，并遵循以下共享运行原则：

1. 五个主数据集的 SCAR Stage-A 每个数据集只训练一次 seed `42`；
2. 检索机制、净化、污染和 memory keep ratio 复用相同 checkpoint，仅重建 Stage-B
   或重新推理；
3. SCAR 与 PaAno、PUAD、PGRF-Net 首次正式训练即启用资源监控，同一次
   运行同时产生性能和效率结果；CATCH 不重跑、不进入本次资源实测；
4. 所有任务保存配置、环境、checkpoint、逐点 score、memory provenance、失败日志和
   `resource_metrics.json`。

正式数据覆盖固定为：

- CATCH 覆盖：只运行 SCAR 补齐当前五个主数据集之外的全部兼容真实数据；ASD 12 个
  子数据集全部运行并以 family macro-average 汇总；六类合成异常的两个比例版本共
  12 个 CSV 全部运行；CATCH 方法本身不重跑，直接引用官方结果；
- TSB-AD-M：使用 tuning 20 条冻结配置，正式评估 M-Eva 180 条；
- TSB-AD-U：使用 tuning 48 条冻结配置，正式运行官方默认 U-Eva 350 条；
- U-Eva-Full 822 是官方仓库发布的扩展清单，仅保留为论文修订 backlog，不进入
  rebuttal P0；
- TSB-AD 只补充 SCAR 结果，不运行 PaAno；PaAno 仅保留在五个主数据集的相关基线与
  效率比较中，TSB-AD 公开基线数字统一标记为 `reported`；
- MMFDD-TEP：使用 M1-M6 和全部可用故障完成六模式全故障机制验证；
- 五个主数据集：E9 和 E10 必须全部覆盖，相关基线性能与效率也必须全部覆盖。

TSB-AD 对四类融合分数和全部诊断子分数统一报告 AUROC、AP、Point-F1、PA-F1、Aff-P、Aff-R、Aff-F1、VUS-ROC 和 VUS-PR；主融合固定为 `cdf_mean`，不得按
CSV、来源数据集或指标选择融合方式。TAB 保留为论文修订或后续工作，不作为本轮
rebuttal 必需项。完整任务矩阵和表格字段以 `rebuttal执行计划.md` 为准；远程服务器
的去重运行顺序、产物复用关系、运行量核算和逐项验收以
`rebuttal实验运行清单.md` 为准。

当前去重后的关键运行量为：五数据集 SCAR 只保留 5 个正式 Stage-A；E9 复用默认
`clean_ratio=0.02`，只新增 25 个 Stage-B/Test；E10 的零污染结果从主运行和 E9
`clean_ratio=0` 的逐点分数按三折 mask 重评估，只运行 135 个非零污染
Stage-B/Test；E38 复用 `coreset_keep_ratio=1.0`，只新增 20 个 Stage-B/Test；
按 M/U 各验证一个预注册配置核算，TSB-AD 为 tuning 68 加 official evaluation 530，
共 598 次 full。完整 P0 为 649 次 full model fit 和 180 次 Stage-B/Test；若在 tuning
清单比较多组配置，额外运行量按 `20×H_M + 48×H_U` 计算并单独登记。

## 18. Rebuttal 统一实验入口

正式实验由中央注册表 `scripts/experiments/protocol.py` 唯一生成任务，统一入口为：

```bash
python scripts/experiments/rebuttal.py plan --scope all
python scripts/experiments/rebuttal.py run --scope p0 --max-parallel 1
python scripts/experiments/rebuttal.py resume --scope p0 --failed-only
python scripts/experiments/rebuttal.py status --scope all
python scripts/experiments/rebuttal.py validate --scope p0
python scripts/experiments/rebuttal.py collect --scope p0
```

针对三天时限内优先完成 AC 要求的队列，可使用 `--scope ac-core`。该范围固定为
102 项：51 次 full model fit、25 次 Stage-B/Test 和 26 次 analysis。51 次完整训练
由 5 个 SCAR 主数据集锚点、15 个相关 baseline、30 个 SCAR 在 CATCH 缺失 CSV
上的补充结果和 1 个 TEP full 组成；**不包含任何 CATCH 方法训练**。计划输出中的
`group:catch=31` 专指 30 个 SCAR 补充任务和 1 个数据完整性审计，CATCH 对照数字
只使用论文发布结果并标为 `reported from CATCH`。

E11/E12 依赖完整 E10 污染折，因此保留在 P0、不得进入不含 E10 的三天 AC-core。
AC-core 强制检查内部依赖闭包；净化敏感性由五主数据集 E9 六档阈值直接回答。

双卡节点必须显式声明 GPU 槽位，调度器会为同批并发任务分别设置
`CUDA_VISIBLE_DEVICES`，并拒绝并发数超过槽位数：

```bash
python scripts/experiments/rebuttal.py run \
  --scope ac-core --max-parallel 2 --gpu-devices 0 1
```

两台服务器分片时使用 `--method`、`--dataset` 和 `--group` 限定任务集合；同一
dataset 的 anchor、E9、机制和效率任务应保留在同一节点，以减少 checkpoint 和
memory 传输。不同节点必须使用相同 Git commit、数据 hash、正式 seed `42` 和
环境锁文件。

每项任务使用稳定 `run_id`，保存完整命令、配置/数据/源码哈希、Git commit 与 dirty
状态、环境快照、日志、状态和产物路径。`resume` 只有在 `run_record.json` 的
`run_id`、配置哈希、当前源码哈希和当前数据哈希一致，且 JSON/NPY/NPZ 等全部必需
产物可读取时才跳过；只有文件存在但缺少匹配记录的旧目录会标为
`unverified_artifacts`。并发执行按 manifest 中的依赖关系分层，避免
Stage-B、策略分析或 baseline 早于 anchor 启动。

远程数据不在仓库默认目录时，用 JSON 映射显式覆盖物理根目录：

```json
{
  "anomaly_detect": "/data/anomaly_detect",
  "TSB-AD-M": "/data/TSB-AD-M",
  "TSB-AD-U": "/data/TSB-AD-U",
  "TEP": "/data/MMFDD-TEP"
}
```

并在所有 `plan/run/resume/status/validate/collect` 命令追加
`--data-root-map /path/to/data_roots.json`。覆盖值进入正式配置和 run-id，依赖 ID
会同步重映射，避免不同物理数据根目录误复用。可直接从
`environments/data-root-map.example.json` 和 `environments/python-map.example.json`
复制后修改，并分别传给 `--data-root-map`、`--python-map`。

远程服务器启动前执行：

```bash
python scripts/experiments/preflight.py \
  --output artifacts/rebuttal_manifest/preflight.json --strict-runtime
bash scripts/experiments/remote_smoke.sh
```

正式环境目标为 Python 3.11、PyTorch 2.7.1、CUDA 12.6。环境输入规格位于
`environments/`；创建环境后使用 `scripts/experiments/freeze_conda_env.py` 保存
平台相关 explicit lock、pip freeze、驱动版本和 GPU UUID。PaAno、PUAD、PGRF-Net
通过项目侧 adapter 使用只读上游源码，统一导出 `scores.npy`、`labels.npy`、
AUROC/AP、1 次预热加 3 次计时、资源 JSON 和运行 manifest。CATCH adapter 仅作为
论文修订复现工具保留，不进入正式 manifest。

`run_baseline.py --smoke` 仅用于远程模型级兼容性检查：PaAno 运行 1 iteration，
PUAD 运行 1 epoch，PGRF-Net 两阶段各运行 1 epoch 且 patience 为 1。正式
中央 manifest 不传该参数，仍使用各 adapter 的完整默认训练量。

PGRF-Net 上游模型在 `eval()` 中仍通过 `gumbel_softmax` 采样原型权重。项目侧
adapter 在每次预热/计时推理前重置 seed 42，使同一 checkpoint 的三次计时使用
同一采样结果；不修改上游源码，且在运行 manifest 中明确记录该兼容处理。

MEMTO 的历史 adapter 固定使用官方两阶段协议：每阶段最大 100 epoch、10 个 memory item，第一/
第二阶段学习率分别为 `1e-4`/`5e-5`。上游 batch 256 由四卡 DataParallel 分摊，
单卡 adapter 使用等效的每卡 batch 64；第二阶段显式设为 `second_train`，推理前
切换为 `test` 以冻结 memory，并将非 parameter 的 memory tensor 随 checkpoint
保存。k-means 与官方一致只读取训练窗口的 10%。该 adapter 仅供历史审计，MEMTO
不进入正式 manifest、资源比较或 rebuttal 结果。

SCAR/PaAno 环境固定 NumPy `1.26.4`，以满足 `TSB-AD==1.5` 声明的
`numpy>=1.24.3,<2.0` 约束；PaAno 所需 statsmodels 固定为 `0.14.5`。若使用 pip
构建兼容环境，必须明确配对 torch `2.7.1`、torchvision `0.22.1` 和 torchaudio
`2.7.1` 的 cu126 wheel，禁止让未固定的 torchaudio 升级并破坏 Torch ABI。

PaAno 正式适配遵循其官方 multivariate 启动脚本的 `patch_size=96`、
`num_iters=100`、`batch_size=512`、`lr=1e-4` 和 RevIN 设置；仅随机种子按本项目
预注册协议统一为 `42`，不采用上游示例的 `2027`。

当前中央 manifest 共 1220 项：P0 为 893 项，P1 为 327 项；其中实际重计算口径为
P0 的 649 次 full model fit 与 180 次 Stage-B/Test，新增的 E10 冻结协议和 P0
表格收口均为 analysis，不增加训练。TEP full 按序列级产物验收，正式要求
`test_sequence_scores_selected.npy`、序列表、指标、资源、checkpoint、memory、
fusion 和 full memory audit；`max_test_sequences` 仅供远程 smoke 显式限为 2，
正式任务保持 `0` 并评估全部 168 条故障序列。中央 manifest 通过环境变量显式传入
正式 `ARTIFACT_ROOT`，禁止 TEP shell 回退到仓库内默认 `./artifacts`。
