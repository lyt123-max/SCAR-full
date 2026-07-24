# Reviewer ef7G 逐条解决方案

> 分析依据：Reviewer ef7G 于 2026-06-25 提交、2026-07-23 修订的 Official
> Review 原文，`rebuttal执行计划.md`、当前论文
> `Formatting_Instructions_For_NeurIPS_2026/main.tex` 与 `main.pdf`、现有数据、
> 训练脚本、TEP 全量协议和外部基线代码。

## 1. 总体判断

ef7G 的五组问题不是同一优先级：

| 问题 | 实际风险 | 优先级 | 必须交付的证据 |
| --- | --- | --- | --- |
| 只使用 CATCH 中 5 个数据集 | 容易被判断为选择性报告，且与 AC 意见重合 | P0 | 完整纳入/排除审计、扩展真实数据集结果、六类合成结果 |
| TEP 只用 M1/M3/M4 和 8 个故障 | 容易被判断为人为挑选有利子集，且与 AC 意见重合 | P0 | 六模式全故障审计和全量机制实验 |
| Table 1 与 Related Work 不一致 | 削弱 novelty 和 empirical positioning | P0 | 最近邻基线定量结果、逐方法兼容性说明、映射表 |
| Figure 1 太密集且层级不一致 | 影响可读性，但不依赖新增实验 | P2 | 新主图、检索细节图、全文术语统一 |
| 方法直觉、符号和分数不清楚 | 影响理解与可复现性 | P2 | 符号表、模块解释、三类分数的异常语义 |

当前执行清单的方向基本正确，但建议作三项调整：

1. TEP 以“六模式全故障”作为主方案，不再把三组替代子集作为同等候选。
2. 基线对齐提升到 P0，因为它与 AC 和其他 reviewer 的意见重合。
3. Figure 1 只保留总体流程，校准细节放正文或附录，不再默认新增第三张图。

### 1.1 原始评分所反映的决策重点

ef7G 给出 Quality 2、Clarity 2、Significance 3、Originality 3 和 Reject 2，
同时明确承认 operating-regime 视角相关、机制验证有价值、Related Work 结构良好。
这说明该 reviewer 并未否定问题价值和原创性，拒稿主要来自三件可修复事项：

- 评估覆盖和 competitor selection 不能支撑当前经验结论；
- Figure 1 与 Method 的组织让方法难以理解；
- 核心定义被放到 Appendix，正文不足以独立复现方法。

因此回复不能把篇幅主要花在重新解释 motivation。最有效的策略是优先给新增实验表，
随后展示章节/图的结构性修改，并把核心定义直接移回正文。

原文把 TEP 称为 “a custom dataset”。这不是正面认可，而是论文没有清楚说明数据来源
造成的误解。回复中必须明确 TEP 来自公开 MMFDD-TEP，并提供引用、下载来源、模式/故障
清单和受控子集规则，不能继续只写 “controlled subset”。

## 2. 问题一：为什么只使用 CATCH 中的 5 个数据集

### 2.1 审稿人的真实关切

审稿人不是只想听“五个数据集很常用”，而是在判断：

- 五个数据集是否在看到 SCAR 结果后被挑选；
- 论文是否把“使用 CATCH 的部分结果”写成了“使用完整 CATCH benchmark”；
- “五个数据集上 SOTA”是否被外推成了广泛的 MTSAD SOTA；
- 未纳入数据集是否会改变平均排名和结论。

只补选择理由不足以消除该质疑，必须增加无选择性的扩展结果。

### 2.2 项目中的现有条件

项目已经具备完整执行基础：

- `dataset/anomaly_detect/data/` 已包含 CATCH 发布表对应的 12 个真实数据集族：
  CICIDS、CalIt2、SWaT、Creditcard、GECCO、Genesis、MSL、NYC、PSM、SMD、
  SMAP 和 ASD；ASD 由 12 个子序列组成。
- 同目录包含 12 个合成 CSV，构成 contextual、global、seasonal、shapelet、
  trend、mixture 六类异常，每类两个异常比例。
- `scripts/main/` 已有全部真实数据和合成数据的训练脚本。
- 已有五个主数据集、GECCO、Genesis、ASD 和合成数据的部分历史结果，但这些结果
  不是当前代码版本、固定 seed `42` 和统一 evaluator 下的正式 rebuttal 结果，不能直接拼成最终表。

正式论文应把口径写准确：官方发布表是 12 个真实数据集族；合成部分是 12 个文件，
按六种异常类型汇总。不要混用“6 个 synthetic datasets”和“12 个 synthetic files”。

### 2.3 固定实验协议

在查看新增结果前，将以下规则写入实验清单并固定：

1. 纳入全部 12 个真实数据集族，不按 SCAR 性能排除数据集。
2. ASD 的 12 个子序列分别训练和评分，再做 dataset-level macro average；不能把
   ASD 的 12 个子序列当作 12 个独立真实数据集与其他数据集等权。
3. 主指标固定为 point-wise AUROC 和 AP；附加指标可以报告，但不能替换主指标。
4. SCAR 主融合固定为 `cdf_mean`，不允许逐数据集挑选最佳融合。
5. 正式模型随机种子固定为 `42`；随机方法运行一次并报告固定 seed 结果，确定性方法运行一次。
6. 训练/验证只使用训练段；测试标签不得参与模型、融合或阈值选择。
7. 对 12 个合成文件全部运行；先对同一异常类型的两个异常比例求均值，再报告六类
   macro average，同时保留 12 个文件的明细。
8. CATCH、PaAno 等随机模型同样固定使用 seed `42`；确定性方法注明 deterministic。

### 2.4 需要运行的实验

真实数据主扩展：

- 保留 MSL、PSM、SMAP、SMD、SWaT；
- 新增 CICIDS、CalIt2、Creditcard、GECCO、Genesis、NYC、ASD；
- 对历史上已有结果的数据也按同一代码版本、固定 seed `42` 和统一 evaluator 重跑。

最低可接受基线组合：

- CATCH：同一数据版本、划分和 AUROC/AP evaluator；
- PaAno：最接近的 patch retrieval 基线，官方代码明确支持多变量 TSB-AD；
- H-PAD 或 MEMTO：memory/prototype 代表；
- AE 或 TranAD：reconstruction/forecasting 代表。

最终主表可以保留五个经典数据集以控制宽度，但必须新增一张“完整 12 数据集”表，
并报告：

- 每数据集 AUROC/AP；
- 12 数据集 macro average；
- 平均排名；
- SCAR 对最强基线的 win/tie/loss；
- 固定 seed `42` 的正式结果。

合成实验需要输出：

- 12 个文件的 AUROC/AP 明细；
- 六类异常的 AUROC/AP 均值；
- 六类 macro average；
- 六类雷达图只能作为辅助，不能替代表格。

### 2.5 验收标准

- 数据审计表中 12 个真实数据集族均有纳入状态和理由；
- 不存在“符合预设规则但未运行”的数据集；
- 全部正式结果来自同一 evaluator；
- 平均排名按数据集族计算，ASD 不被重复加权；
- 若扩展结果不再支持广泛 SOTA，摘要和贡献改为
  “on the evaluated benchmarks”或“competitive performance across 12 datasets”。

### 2.6 建议回复结构

正式 rebuttal 不应先辩解，应先承认表述不清，再给新增证据：

> Thank you for identifying this ambiguity. The original paper used five
> widely adopted multivariate benchmarks and used CATCH only as one source of
> baseline results; it did not evaluate the full CATCH suite. To remove any
> concern about selective reporting, we have now evaluated SCAR on all 12
> real-world dataset families in the released suite and all 12 synthetic files
> grouped into six anomaly types under a pre-specified protocol. We report
> dataset-level AUROC/AP, mean ranks, and win/tie/loss statistics in Table XX.

其中结果句必须等实验完成后再填写，不能预写“结论保持不变”。

## 3. 问题二：TEP 为什么只使用子集

### 3.1 当前论文中的具体问题

当前论文只写“controlled subset”，但没有给出可复核选择规则，因此无法排除挑选。
此外，论文旧表仍把文件号 `d01/d04/d06/d07/d10/d13/d14/d27` 直接写成
`IDV1/4/6/7/10/13/14/27`。上游 MMFDD-TEP 的官方规则是反向映射：

```text
IDV = 29 - d
```

所以当前八个文件实际对应：

```text
IDV28, IDV25, IDV23, IDV22, IDV19, IDV16, IDV15, IDV2
```

这是需要与 rebuttal 同时修正的事实错误。

### 3.2 项目中的现有条件

- 完整上游仓库已经在
  `Multi-mode-Fault-Diagnosis-Datasets-with-TE-process/`。
- 数据包含 M1-M6，每模式 `d00-d28`，共 174 个核心 `.mat` 文件。
- 已审计出 168 条故障序列，其中 158 条为 7,201 点，10 条较短但均不少于
  `seq_len=128`，因此全部可纳入。
- `run.py`、`coremad/data.py`、`scripts/main/train_tep_full.sh` 和
  `scripts/tep/` 已支持 full 协议、断点续跑、序列分片与正式表格导出。

因此最有说服力的方案不是再设计多个小子集，而是直接做六模式全故障实验。

### 3.3 主实验设计

使用全部 M1-M6 正常数据训练并构建记忆库，使用全部 168 条故障序列做机制审计：

```bash
DATA_ROOT=./Multi-mode-Fault-Diagnosis-Datasets-with-TE-process \
bash scripts/main/train_tep_full.sh full tep_full_mechanism
```

固定模型随机种子 `42`，正式运行输出：

- SMC@K：state embedding 的同模式邻域纯度；
- SFR：模式间/模式内距离比；
- SMR@K：检索参考与 query 同模式的比例；
- query-mode vs retrieved-mode 混淆矩阵；
- fault-normal memory/state/final-score gap；
- mode-wise FPR 标准差；
- tail calibration error；
- sequence-equal 和 window-weighted 两种汇总。

由于 full TEP 的每条故障序列都是全故障序列，不能对单条全正类序列报告 AUROC/AP。
TEP full 的主要作用是验证机制和跨模式稳定性，不应被包装成与五个逐点数据集相同的
检测主表。

### 3.4 selected 与 full 的关系

- full 六模式结果是 rebuttal 主证据；
- 原 M1/M3/M4 子集仅作为历史协议对照；
- 报告 selected 与 full 的机制指标差异，证明结论是否依赖原子集；
- leave-one-mode-out 只适合解释未见工况边界，不是解决“为何挑选子集”的主证据；
- 如果 full 结果下降，应据实写成“原结论在部分模式更强”，不能只展示 selected。

### 3.5 验收标准

- 提供 6 模式 × 29 文件的完整可用性表；
- 同时显示 `file_fault_id` 与官方 `IDV`，避免再次混淆；
- 168 条故障序列全部进入审计，没有按结果删除；
- seed `42` 的完整运行完成，配置、日志和产物可追溯；
- 论文数据集表、图标签、正文和 appendix 使用同一官方 IDV 映射。

### 3.6 建议回复结构

> We agree that the original subset description was insufficient and could
> raise a selection concern. TEP is not a custom dataset; it is a controlled
> subset of the public MMFDD-TEP benchmark. We have therefore replaced the
> subset-only evidence with a full audit and experiment covering all six
> operating modes and all 168 available fault sequences. Table XX lists every
> mode/fault file and the official reverse mapping from d01-d28 to IDV28-IDV1.
> The original M1/M3/M4 result is retained only as a historical protocol
> comparison.

## 4. 问题三：Table 1 与 Related Work 不一致

### 4.1 解决原则

不能简单把 Related Work 中所有方法都塞进 Table 1。需要做到：

1. 每个进入 Table 1 的方法在 Related Work 有介绍；
2. 每个被称为“最相关”的方法要么有同协议定量结果，要么有逐项技术不兼容说明；
3. 结构差异和性能差异分开回答；
4. 不能把不同数据划分或不同指标的公开数字直接混入同一排名表。

### 4.2 建议的四类结构

| 类别 | 代表方法 | 与 SCAR 的比较点 |
| --- | --- | --- |
| Global reconstruction/forecasting | AE、LSTM-VAE、TranAD | 全局模型是否按 query 改变正常参考 |
| Unified representation/association | Anomaly Transformer、DCdetector、CATCH | 统一表示空间是否先限制候选集合 |
| Memory/prototype | MEMTO、PUAD、H-PAD、PGRF-Net | 记忆项是否为 query-specific 条件集合 |
| Instance/subsequence retrieval | kNN、LOF、DAMP、GDFlex、PaAno | 是否同时进行状态过滤和上下文过滤 |

### 4.3 外部基线的具体处理

项目已有 PaAno、PUAD、PGRF-Net、GDFlex 和 DAMP 官方代码。

优先顺序建议：

1. **PaAno 必跑。** 它是 patch representation + normal patch retrieval，且官方实现
   支持多变量 TSB-AD，是对 SCAR 最直接的替代解释。
2. **DAMP 尽量跑。** 使用官方 `DAMP_Multidim.m`，但需在项目外层做只返回
   `Left_MP` 的薄包装，并记录相对官方文件的唯一改动。窗口长度只能由训练/验证规则
   确定，不能根据测试结果挑选。
3. **PUAD/PGRF-Net 至少选一个完成同协议复现。** 它们补足近期 prototype/memory
   对照。
4. **GDFlex 做严格兼容性审计。** 当前官方入口面向单变量 ML/UCR 数据和其专用
   `.mat` 协议。不要把逐通道运行后取 max/mean 的自定义版本冒充官方多变量结果。
   若无法保持算法定义，应在结构表中说明不纳入多变量主表，并可在单变量附录中报告。
5. **DMemAD** 若没有公开官方实现，不伪造复现；明确记录检索日期、无公开代码和
   无法统一复现的原因。

### 4.4 必须新增的两张表

表 A：Related Work - Baseline 映射。

字段至少包括方法、年份、类别、正常参考、候选池是否全局、是否 query-specific、
是否状态过滤、是否上下文检索、是否进入主表、未进入原因。

表 B：最近相关方法结构差异。

建议只比较 SCAR、PaAno、DAMP、GDFlex、MEMTO/H-PAD，避免大而空的勾选表。
关键列为：

- reference unit；
- candidate pool；
- condition filtering；
- context retrieval；
- content distance；
- multi-scale；
- multivariate support；
- official protocol compatibility。

### 4.5 验收标准

- 至少新增一个最近 retrieval 基线的五数据集或十二数据集定量结果，首选 PaAno；
- 所有 Table 1 方法都能在 Related Work 找到明确定位；
- DAMP/GDFlex/DMemAD 的纳入或排除都有技术原因，而非“protocol mismatch”一句话；
- 所有排名只使用同一数据划分和同一 evaluator。

## 5. 问题四：Figure 1 太密集且与 Section 3 不一致

### 5.1 当前图的实际问题

已按论文页面尺寸检查当前 Figure 1。它在单页上同时放入：

- STSD、state encoder、gate、patch encoder 和辅助任务；
- 两级检索的候选窗口、patch pool、公式和局部参考；
- 三类 raw diagnostics、两张 ECDF 小图、fusion 公式和时间线。

在论文 100% 缩放下，公式、图例和说明文字不可读。正文称“Stage A、Stage B、
Inference”，图中却按 `(a)/(b)/(c)` 组织；memory construction 也没有成为独立阶段。

### 5.2 先重组 Section 3

Reviewer 点名的是 Figure 1 三块与 Section 3.2-3.4 不一致，因此不能只给图换 Stage
名称。正文应先重组为：

1. **3.1 Overview**
2. **3.2 Representation Learning**
3. **3.3 Memory Construction**
4. **3.4 Condition-Aware Retrieval and Anomaly Scoring**

其中：

- 原 3.2 STSD 改为 3.2.1；
- 原 3.3 state-conditioned patch representation 改为 3.2.2；
- masked reconstruction 和 next-patch prediction 改为 3.2.3；
- 原 3.4 开头的 memory purification/coreset 独立成新的 3.3；
- Level 1、Level 2、memory score、reconstruction diagnostics 和 fusion 留在 3.4。

这样 Figure 1 的三块、Section 3.2-3.4、算法伪代码和代码 Stage 才能一一对应。

### 5.3 新 Figure 1

全文统一为：

1. Stage I: Representation Learning
2. Stage II: Memory Construction
3. Stage III: Condition-Aware Retrieval and Anomaly Scoring

Figure 1 只保留以下主链路：

```text
Input X
  -> STSD (S, R)
  -> Representations (h, z, c)
  -> Purified Memory
  -> State Filtering
  -> Context Retrieval
  -> Three Diagnostics
  -> Calibrated Score A_t
```

设计约束：

- 不放完整公式；
- 不放 ECDF 小曲线；
- 不放 patch 邻域均值的索引细节；
- 每阶段最多 2-3 个关键节点；
- 正文最终排版下最小文字不低于约 7 pt；
- 图标题、正文小节名和伪代码使用完全相同的 Stage/Level 名称。

### 5.4 新 Figure 2

只解释 Stage III 内部：

```text
query h -> Level 1 top-M windows
query c -> Level 2 top-K context-compatible patches
query z -> kNN content distance -> memory score
```

State novelty 与 reconstruction score 用两条侧支路表示即可。CDF 校准保留一个
“ECDF calibration + mean”方框，不单独再画第三张主图，除非最终版面确有余量。

### 5.5 验收标准

- Figure 1 在 letter/A4 页面 100% 缩放下可读；
- 图中只出现 Stage I/II/III，Level 1/2 只属于 Stage III；
- Figure 1 三块与 Section 3.2、3.3、3.4 同名、同序；
- 正文、算法、图注和 appendix 不再出现相互冲突的 Stage A/B；
- 新 PDF 需重新渲染并人工检查，不只检查绘图源文件。

## 6. 问题五：方法直觉、符号和分数解释不清楚

### 6.1 编码器解释

在公式

```text
z_i^(p) = f_scale^(p)(f_trunk(R_i^(p)))
```

后直接补充：

- `f_trunk` 是所有 patch 尺度共享的 1D convolutional feature extractor，用于学习
  跨尺度可复用的局部残差模式；
- `f_scale^(p)` 是尺度 `p` 专用的 projection head，将不同长度 patch 的 trunk
  特征映射到统一的 `d_z` 维空间；
- 共享 trunk 控制参数量并鼓励跨尺度共性，尺度头保留不同持续时间的特异性。

同时新增一张紧凑符号表，至少包含：

```text
X, S, R, h, g, z, c, M, K, k_nn, k_s, e, s_mem, a_w, A_t
```

### 6.2 三类分数的异常语义

**State novelty**

- 输入：query window 的状态向量 `h_w`；
- 参照：训练正常状态库；
- 数值：到 `k_s` 个最近正常状态的平均平方距离；
- 高分含义：当前慢变工况缺少正常训练支持；
- 限制：它提示 unseen/rare state，不单独等价于故障。

**Patch-level memory score**

- 输入：query patch 的 content 表征 `z`；
- 参照：经过 state 和 context 两级筛选后的局部正常 patch；
- 数值：到局部参考中 `k_nn` 个 content-nearest patch 的平均平方距离；
- 高分含义：即使在相容工况和上下文中，局部快速动态仍不匹配；
- 必须说明 context 用于选参考，content 用于算异常距离，二者不是重复 kNN。

**Per-scale reconstruction score**

- 输入：尺度 `p` 下被 grouped masking 的原始 patch；
- 参照：同一编码器从未遮挡上下文恢复出的 patch；
- 数值：确定性 masked reconstruction error；
- 高分含义：当前 patch 难以由正常上下文恢复；
- 短尺度偏向短促异常，长尺度偏向持续异常和趋势偏移。

### 6.3 推荐的模块写作模板

每个方法小节按同一顺序写四句话：

1. 输入和张量维度；
2. 模块做什么；
3. 为什么该模块对 condition-aware reference 必要；
4. 输出进入哪个后续步骤。

不要只在 appendix 补定义。`f_trunk`、`f_scale` 和三类分数的直觉必须进入正文。

### 6.4 从 Appendix 移回正文的最低内容

原评审明确指出 core information 被放在 Appendix，因此以下内容必须进入主文：

- STSD 的频率初始化、可学习修正和 `X=S+R` 的作用各用一句话说明；
- `f_trunk`、`f_scale`、state gate 的输入输出维度；
- masked reconstruction 与 next-patch prediction 的目标对象；
- memory purification 的分位数定义；
- Level 1、Level 2 和最终 content distance 分别使用 `h/c/z` 的原因；
- 三类 raw score 如何扩展到 time-step；
- ECDF 只使用训练正常分数拟合，测试标签不参与；
- 最终平均融合包含哪些诊断项。

附录继续保留完整推导、伪代码和实现细节，但正文应使读者无需翻附录也能回答：
“输入是什么、参考集合如何形成、三个分数各自测量什么、最终分数如何得到”。

## 7. 推荐执行顺序

1. 冻结 CATCH 12 真实数据集、12 合成文件、正式 seed `42`、主融合和 evaluator。
2. 修正论文 TEP IDV 映射，生成六模式全故障审计表。
3. 并行启动 CATCH 扩展 SCAR 实验与 TEP full 固定 seed `42` 实验。
4. 优先完成 PaAno 适配和正式运行，再处理 DAMP 与一个 prototype/memory 基线。
5. 生成 12 数据集表、六类合成表、TEP selected/full 对照表和基线结构表。
6. 先重组 Section 3.2-3.4，再按同名三阶段重绘 Figure 1/2。
7. 在现有 Related Work 结构上补方法映射和明确差异，同时重写 Dataset、Method 和 Limitations。
8. 最后根据真实结果收窄或保留摘要、贡献和结论中的 SOTA 表述。

## 8. Rebuttal 最小交付包

篇幅有限时，回复中优先放：

1. CATCH 12 个真实数据集的平均排名和 win/tie/loss；
2. 新增 7 个真实数据集的 AUROC/AP 汇总；
3. TEP 六模式全故障的 SMC@K、SMR@K 和校准稳定性；
4. PaAno 与 SCAR 的同协议结果；
5. 一句话说明新 Figure 1 已重绘并统一术语；
6. 一句话定义三类分数。

完整纳入/排除表、168 条 TEP 明细、六类合成明细和结构映射表放 appendix。

## 9. 当前仍缺少的材料或环境

数据本身目前基本不缺。为了正式完成 ef7G 回复，仍需要：

1. **可运行项目的 CUDA/Conda 环境。**
   当前桌面可调用 Python 缺少完整 `PyTorch/SciPy/Faiss/CUDA`，因此不能启动正式
   CATCH 扩展和 TEP full 训练。需要提供已有环境名称/启动方式，或允许按项目依赖安装。
2. **MATLAB 或兼容运行环境。**
   DAMP 和 GDFlex 官方实现为 MATLAB；若机器上没有 MATLAB 许可证/命令行入口，
   需提供可用环境。没有 MATLAB 时，PaAno 应作为必跑的最近 retrieval 基线，
   DAMP/GDFlex 只做严格兼容性审计。
3. **原始 TEP 子集选择依据。**
   如果 M1/M3/M4 与八个故障来自先验文献、导师约定、公开协议或实验前记录，请提供
   对应论文、链接或历史文档。full 实验可以解决选择性担忧，但不能替代对原协议来源的
   诚实说明。
4. **若已有未归档结果，请提供其完整实验目录。**
   特别是新增真实数据集、固定 seed `42`、PaAno/DAMP 或 TEP full 的 checkpoint、
   `config.json`、逐点 score 和日志；只有表格数字而无配置与 score 不应进入正式结果。

不需要用户额外提供 CATCH 数据、TSB-AD 数据或 MMFDD-TEP 数据；这些已经在项目中。
