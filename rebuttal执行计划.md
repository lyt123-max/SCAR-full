# SCAR Rebuttal 执行计划

> 目标：覆盖 Reviewer muQn、Reviewer ef7G、Reviewer dk9H 以及 Area Chair（AC）提出的全部意见与质疑。
> 总原则：凡是可以通过新增实验、定量分析、表格或可复核结果回答的问题，优先用证据回答；只有属于论文结构、表达、符号说明、数学直觉或理论推导的问题，才主要通过文字修改解决。
> 文件中的“预期结论”仅用于说明实验要回答什么问题，不能在实验完成前预设结果。若结果与预期不一致，应据实报告并相应收窄论文结论。
>
> 当前 rebuttal 阶段不允许提交图片。所有新增证据统一以表格和文字分析呈现；Figure 1
> 重绘、机制可视化和曲线图仅保留为论文修订 backlog，不占用 rebuttal 实验关键路径。
> 所有正式训练均在远程服务器执行。本地已具备 CATCH、TSB-AD-U/M 和完整
> MMFDD-TEP 数据，不需要再次下载数据集。
>
> 所有正式 SCAR 结果统一输出 `raw_max`、`zscore_mean`、`cdf_mean`、`cdf_max` 和
> 全部诊断子分数的原始数组及逐分数指标。`cdf_mean` 是预注册主结果，但任何实验汇总
> 都不得只保留 selected 指标。每个融合分数和子分数必须配套包含 AUROC、AP、
> Point-F1、PA-F1、Aff-P、Aff-R、Aff-F1、VUS-ROC、VUS-PR；正式完整性检查拒绝缺项。
> TEP 序列级协议不适用的时序指标保留字段并写 `NaN` 与原因，不以伪造数值补齐。
>
> 面向远程服务器的去重运行顺序、复用关系、运行量核算和逐项勾选入口见
> `rebuttal实验运行清单.md`。本文件负责“需要回答什么”，运行清单负责“怎样只跑一次并
> 复用到多个问题”。

---

## 0. 总体执行原则

### 0.1 证据优先级

1. **补齐 CATCH 缺失数据集，并完成 TSB-AD-U/M 综合基准**
2. **新增最相关外部基线，并在五个主数据集上统一比较**
3. **新增五个主数据集上的条件相容检索证据**
4. **在五个主数据集上完成净化阈值与污染鲁棒性实验**
5. **利用本轮重训同步完成五数据集效率与内存比较**
6. **完成六模式全故障 TEP 机制验证**
7. **补充控制实验、文字澄清和理论推导**

### 0.2 公平实验统一要求

所有新增实验必须尽量遵循以下统一协议：

- 使用与 SCAR 主实验相同的数据划分、预处理和标准化方式；
- 模型选择和超参数调节只能使用训练集中的验证划分；
- 测试集仅用于最终评估；
- 统一报告 point-wise AUROC 和 AP；
- 不使用 point-adjustment、test-label threshold tuning 或测试集异常比例；
- 正式实验统一固定并记录模型随机种子 `42`；命令行可为调试临时覆盖，但覆盖结果不得进入 rebuttal 或论文正式表格；
- 随机方法使用固定 seed `42` 运行一次，确定性方法运行一次；性能结果报告固定 seed 数值，不报告模型 seed 均值和标准差；
- contamination event folds、block bootstrap、敏感性参数扫描和推理预热后的重复计时不属于模型 seed 重复，仍按各自协议保留；
- 所有效率结果在同一硬件、同一 batch size 或明确说明 batch size 的条件下测量；
- 所有远程正式运行从第一次启动起就开启统一资源监控，性能与资源结果来自同一次训练，
  不为效率表另行重训；
- 五个主数据集的 SCAR Stage-A 每个数据集只训练一次；E1-E12、E38 及内部检索变体
  复用相同 seed-42 checkpoint，仅按协议重建 Stage-B 或重新推理；
- 所有新增基线优先采用官方代码；若必须自行适配，应在补充材料中公开适配方式；
- 若某方法无法公平比较，必须逐项说明不兼容原因，不能只写 “protocol mismatch”。
- rebuttal 中禁止放图；所有曲线、散点图、热力图和案例图必须转换为预先定义字段的
  数值表格及文字分析。

### 0.3 任务优先级

- **P0：直接影响 AC 最终决策，必须完成**
- **P1：多位审稿人共同提出，强烈建议完成**
- **P2：单位审稿人提出，但可以显著增强说服力**
- **P3：主要为写作和可读性修改**

---

# 1. Reviewer muQn

## 1.1 问题：主数据集上缺少“条件相容检索”的机制证据

### 审稿人的问题

TEP 具有真实 operating-mode 标签，因此可以验证 same-mode retrieval；但 MSL、SMAP、PSM、SMD 和 SWaT 没有工况标签。当前论文尚未证明主数据集上的性能提升也来自相同的 condition-aware retrieval 机制。

审稿人明确希望看到：

- clustering；
- retrieval visualization；
- temporal stability；
- retrieval consistency；
- 或其他能够说明主数据集上检索参考具有条件相容性的分析。

### 我们如何解决

**P0：在五个主数据集上新增不依赖真实工况标签、且尽量独立于 SCAR 检索表示的代理机制验证。**

核心目标不是声称“主数据集检索到了真实同工况样本”，而是证明：

1. SCAR 检索到的参考在独立的慢状态统计空间中更加接近查询；
2. SCAR 检索到的局部 patch 在原始波形空间中更加匹配查询上下文；
3. 两级检索比 global、state-only 和 context-only 更稳定；
4. 轻微局部扰动不会破坏 Level-1 状态候选集合。

### 具体步骤

#### 实验 E1：独立状态代理距离

1. 对每个查询窗口和被检索参考窗口，从原始窗口或慢变成分中提取**非学习统计特征**：
   - 每通道均值；
   - 每通道标准差；
   - 一阶线性趋势斜率；
   - 低频能量比例；
   - 可选：一阶差分均值、长期幅值范围。
2. 对特征进行训练集统计量标准化。
3. 定义查询与参考之间的状态代理距离：
   \[
   D_{\text{state}}(w)=
   \frac{1}{|\mathcal R_w|}
   \sum_{r\in\mathcal R_w}
   \|\phi(S_w)-\phi(S_r)\|_2.
   \]
4. 比较以下检索策略：
   - Random reference；
   - Global retrieval；
   - Context-only retrieval；
   - State-only retrieval；
   - Full SCAR。
5. 在五个主数据集上报告固定 seed `42` 的均值、block bootstrap 置信区间和相对下降比例。
6. 对正常查询和异常查询分别报告，避免只在某一类样本上成立。
7. 使用 block bootstrap 报告数据/时间块层面的置信区间；不得将其解释为模型初始化方差。
8. 解释该指标是 **proxy condition consistency**，不是 ground-truth operating-mode accuracy。

#### 实验 E2：原始局部上下文距离

1. 对查询 patch 和 Level-2 检索 patch 进行每通道去均值与幅值归一化。
2. 计算以下独立于 \(z\)、\(c\) 的原始空间匹配指标：
   - normalized L1 distance；
   - Pearson correlation；
   - 可选：DTW distance，仅用于少量代表性数据集。
3. 定义：
   \[
   D_{\text{context}}=
   \frac{1}{K}
   \sum_{r\in \mathcal N_i}
   d(\operatorname{Norm}(X_i),\operatorname{Norm}(X_r)).
   \]
4. 比较 Global、State-only、Context-only、Full SCAR。
5. 重点检验：
   - State-only 是否状态相近但局部形态不够匹配；
   - Context-only 是否局部形态相近但状态代理距离较大；
   - Full SCAR 是否同时取得低状态距离和低上下文距离。

#### 实验 E3：状态—上下文联合权衡表

E3 不再启动独立实验，直接复用 E1/E2 结果：

1. 每行对应一个数据集和一种检索策略；
2. 同时报告标准化状态代理距离与上下文距离；
3. 增加两项相对 Full SCAR 的差值；
4. 报告跨数据集平均排名；
5. 用文字判断 Full SCAR 是否同时取得较低的状态与上下文距离，不使用二维散点图。

#### 实验 E4：检索稳定性

1. 对正常查询窗口施加不改变慢状态的小幅扰动：
   - 高频高斯噪声；
   - 小幅 masking；
   - 局部幅值扰动。
2. 扰动幅度必须通过验证，确保慢状态统计量变化很小。
3. 计算原查询与扰动查询的 Level-1 Top-M 候选集合 Jaccard：
   \[
   \text{Stability@M}=
   \frac{|\mathcal J_M(X)\cap\mathcal J_M(\tilde X)|}
   {|\mathcal J_M(X)\cup\mathcal J_M(\tilde X)|}.
   \]
4. 计算 Level-2 检索集合重合率。
5. 比较 SCAR 与不分解版本、context-only 版本。
6. 对固定扰动强度逐档报告 Top-M/Top-K overlap、均值和 block bootstrap 区间，
   不绘制稳定性曲线。

#### 实验 E5：时间一致性分析

1. 对相邻正常窗口 \(w\) 与 \(w+1\) 计算 Level-1 候选集合重合率；
2. 区分平稳区段和状态变化区段；
3. 在平稳区段，检索集合应较稳定；
4. 在明显变化区段，检索集合允许发生变化；
5. 按数据集报告平稳区段与状态变化区段的候选集合重合率、状态代理变化和异常分数
   变化摘要，不输出时间序列案例图。
6. 该分析只用于说明“检索随状态平稳而稳定、随状态变化而调整”，不声称恢复真实工况。

### 需要新增的表格

- 表：五个数据集上的状态代理距离、上下文距离和跨策略排名；
- 表：各扰动强度下的 Level-1/Level-2 retrieval stability；
- 表：平稳区段与状态变化区段的 temporal consistency；
- 所有表均区分正常/异常查询，并给出 block bootstrap 区间。

### Rebuttal 中的回应重点

- 承认只有 TEP 能直接计算 ground-truth same-mode retrieval；
- 主数据集新增的是无标签代理机制证据；
- 强调这些指标在独立原始/统计空间计算，避免循环论证；
- 不将代理指标表述为真实工况标签恢复。

---

## 1.2 问题：缺少 DAMP、GDFlex、PaAno 等最相关检索方法

### 审稿人的问题

论文 Related Work 讨论了 DAMP、GDFlex、PaAno 等 retrieval/subsequence-reference 方法，但主实验没有纳入这些方法，导致 SCAR 的 empirical positioning 和 originality 不够清楚。

### 我们如何解决

**P0：对最接近方法做兼容性审计，并完成固定 Python 基线集合的五数据集结果。**

正式五数据集定量比较固定优先级：

1. PaAno：必须完成，作为近期 patch/reference 方法；
2. PUAD、PGRF-Net：使用公开 Python 官方实现完成 memory/prototype 对比；
3. CATCH：直接引用官方 benchmark 结果，不在 rebuttal 阶段重跑；
4. DAMP、GDFlex：完成逐项协议审计。当前没有 MATLAB 环境，不把“缺少 MATLAB”
   伪装成协议不兼容；GDFlex 另需明确其官方 ML/UCR 单变量协议边界。
5. MEMTO：不运行；只保留已完成的协议审计、adapter 和失败诊断记录，不纳入正式
   结果、效率比较或任务计数。

### 具体步骤

#### 实验 E6：基线兼容性审计

对 DAMP、GDFlex、PaAno、MEMTO、PUAD、PGRF-Net 和 CATCH 分别核查并形成表格：

1. 原始任务是单变量还是多变量；
2. 是否需要使用测试序列自身构建参考；
3. 是否为 transductive setting；
4. 是否输出 point-wise 分数；
5. 是否依赖测试标签、异常比例或阈值；
6. 是否支持统一窗口化；
7. 是否有官方代码；
8. 是否能够在五个数据集上运行；
9. 运行复杂度是否可承受；
10. 需要哪些适配。

#### 实验 E7：新增最相关外部基线

1. 五个主数据集固定运行 PaAno、PUAD 和 PGRF-Net；MEMTO 不运行；CATCH 不重跑，使用
   官方发布结果并明确标记为 `reported from CATCH`；
2. 保持与 SCAR 相同的数据划分与预处理；
3. 若方法仅支持单变量：
   - 首选寻找其官方多变量扩展；
   - 若无扩展，不应简单逐通道运行后取最大值，除非原论文允许；
   - 如必须做逐通道适配，要明确标为 adaptation。
4. 对每个新增基线调节窗口长度、子序列长度和关键参数；
5. 只用验证集选择超参数；
6. 报告五个数据集上的 AUROC/AP；
7. 随机方法使用固定 seed `42` 运行一次并报告固定 seed 结果；确定性方法运行一次；
8. 首次正式训练即接入统一资源监控，同时输出 E37 所需性能、时间和内存字段；
9. DAMP/GDFlex 不产生未经官方环境验证的替代实现数字，也不把第三方协议数字混入
   统一五数据集结果表。

#### 实验 E8：机制对应的内部公平对照

除了外部基线，再保留：

- Global retrieval；
- State-only retrieval；
- Context-only retrieval；
- Full SCAR。

说明外部方法回答“与已有工作相比”，内部变体回答“SCAR 的条件筛选究竟带来什么”。

### 需要新增的表格

- 表：最相关方法协议兼容性；
- 表：新增检索/子序列基线结果；
- 表：方法是否使用全局候选池、是否先做状态筛选、是否使用局部上下文；
- 表：SCAR 与最接近基线的性能、推理吞吐和内存联合比较。

### Rebuttal 中的回应重点

- 不声称 SCAR 首次使用 nearest-neighbor 或 retrieval；
- 核心区别是“先决定哪些样本可比较，再在条件相容集合中构造局部参考”；
- 若某方法无法公平比较，必须给出具体技术原因；
- 若补充结果不占优，应如实报告并收窄“优于最接近方法”的表述。

---

## 1.3 问题：重构误差净化可能删除稀有正常瞬态，保留低误差异常

### 审稿人的问题

memory purification 依据重构误差删除高误差 patch。审稿人担心：

1. 稀有但正常的 transient 可能被误删；
2. subtle、low-reconstruction-error anomaly 可能残留；
3. 当前没有 contamination 和 purification threshold 敏感性分析。

### 我们如何解决

**P0：增加净化比例敏感性、训练污染鲁棒性、稀有正常覆盖和低误差异常存活分析。**

### 具体步骤

#### 实验 E9：净化比例敏感性

1. 固定 Stage-A；
2. 仅重新构建 memory bank；
3. 测试：
   \[
   r\in\{0,0.005,0.01,0.02,0.05,0.10\}.
   \]
4. 报告五个数据集的 AUROC/AP；
5. 报告 memory bank 中保留 patch 数量；
6. 报告推理时间和内存变化；
7. 以数据集 × 净化比例表格报告完整结果；
8. 观察默认 \(r=0.02\) 是否处于稳定区间，而非孤立最优点。

#### 实验 E10：训练污染率鲁棒性

1. 从测试集异常段或预定义合成异常类型中抽取异常 patch；
2. 仅向 memory-construction training split 注入异常，不污染 Stage-A 预训练；五个数据集
   均复用各自固定 seed-42 Stage-A checkpoint；
3. 测试污染率：
   \[
   \alpha\in\{0\%,1\%,3\%,5\%,10\%\}.
   \]
4. 在每个污染率比较 no purification 与默认 \(r=0.02\)；
5. 在最高污染率 \(\alpha=10\%\) 额外加入 stronger purification \(r=0.10\)，避免
   污染率 × 全部净化率的重复笛卡尔积；E9 已独立覆盖完整净化阈值扫描；
6. 五个数据集均使用 3 个 contamination event folds，并明确 folds 不是模型 seed 重复；
7. 每个数据集预先生成唯一 `contamination_fold_manifest.json`，冻结三折事件、
   候选随机顺序及 `1%/3%/5%/10%` 的嵌套前缀；零污染重评估和所有非零任务只读取
   这份协议文件；若某折唯一异常窗口不足以达到目标污染率，则按固定 seed 的重复
   随机排列补足，并在表格披露 `sampling_with_replacement`；
8. 报告 AUROC/AP、异常 patch 删除率、正常 patch 误删率和实际污染率；
9. 至少加入低幅值 anomaly，专门模拟 low-error anomaly。

#### 实验 E11：稀有正常瞬态保留分析

E11 不再启动独立训练，直接从 E9 的 full memory audit 产物计算：

1. 在训练正常数据中定义“稀有正常 patch”：
   - 基于原始统计空间密度的低密度正常样本；
   - 或 TEP 中真实模式切换/过渡片段；
   - 不得使用测试标签定义训练正常稀有性。
2. 计算净化前后：
   - 稀有正常 patch 保留率；
   - 正常统计空间覆盖率；
   - 最近邻覆盖半径；
   - 各模式/簇的保留比例。
3. 检验净化是否集中删除某一类正常状态。
4. 若发现明显偏差，考虑改进净化：
   - 按状态簇分层分位数净化；
   - 每个状态簇单独保留最低比例；
   - 设置最小覆盖约束。

#### 实验 E12：低误差异常存活率

E12 不再启动独立训练，直接从 E10 的污染 manifest、重构误差和 memory audit 产物计算：

1. 在有标签的合成异常或 TEP 故障中统计重构误差；
2. 将异常按重构误差分位数分组；
3. 报告低误差异常进入 memory bank 的比例；
4. 对比进入 memory bank 后对最终 AUROC/AP 的影响；
5. 说明 purification 只是过滤明显高误差污染，而不是完整 anomaly-removal mechanism。

### 需要新增的表格

- 表：五数据集净化比例—AUROC/AP、bank size、时间和内存；
- 表：五数据集污染率—AUROC/AP，比较 no/default purification；
- 表：异常删除率与正常误删率；
- 表：净化前后稀有正常覆盖与最近邻覆盖半径；
- 表：不同重构误差区间异常的存活率。

### 文字修改

在 Limitations 中明确加入：

- quantile purification 是轻量防护，不保证完全清除异常；
- 稀有正常瞬态可能被删除；
- 低误差异常可能残留；
- 分层或状态感知净化是未来方向。

---

# 2. Reviewer ef7G

## 2.1 问题：为什么只使用 CATCH 中的 5 个数据集

### 审稿人的问题

审稿人认为 CATCH benchmark 包含 12 个真实数据集和 6 个合成数据集，但当前只使用 5 个，怀疑数据集覆盖不足或存在选择性报告。

### 我们如何解决

**P0：澄清 CATCH 在论文中的角色，并补齐相对 CATCH 欠缺的全部兼容数据集结果。**

### 具体步骤

#### 任务 E13：数据集来源与选择审计

1. 列出 CATCH benchmark 中全部 18 个数据集；
2. 对每个数据集记录：
   - 单变量/多变量；
   - point-wise/sequence-level 标签；
   - 是否有标准训练/测试划分；
   - 是否适用于无监督训练；
   - 是否支持统一 AUROC/AP 协议；
   - 数据规模和计算可行性；
   - 是否已被当前五个数据集覆盖。
3. 明确区分：
   - CATCH 作为部分基线结果来源；
   - 当前五个数据集作为 SCAR 主实验数据集；
   - 我们并未声称采用完整 CATCH benchmark。
4. 将选择规则写成预先定义、与 SCAR 结果无关的客观标准。
5. 检查是否存在符合标准却未纳入的数据集；若存在，应优先补跑，不能仅用文字解释。

#### 实验 E14：扩展 CATCH 数据集结果

1. 本地 CATCH 数据已完整具备：12 个真实数据集族和六类合成异常，共 35 个 CSV；
2. 对当前五个主数据集之外的全部兼容真实数据集运行 SCAR，不按结果选择；
3. ASD 的 12 个子数据集全部运行，并按预先固定的 family macro-average 汇总；
4. 若某些数据集规模过大，使用官方推荐配置并保留完整资源记录；
5. CATCH 不重跑；新增数据集只补 SCAR 结果，并与 CATCH 官方发布结果对照；
6. 报告新增数据集上的 AUROC/AP；
7. 报告 12 个真实数据集族的平均排名；
8. 若任何数据因技术原因失败，保留失败日志并在排除表中给出具体原因。

#### 实验 E15：六类合成异常

1. 使用本地 CATCH 官方六类合成异常文件，每类包含两个异常比例版本，共 12 个 CSV；
2. 补充生成协议、异常比例、强度和持续时间；
3. 报告每个原始 CSV 的 AUROC/AP，并按异常类型汇总两个比例版本；
4. 与 CATCH 官方发布结果比较，不为这些合成文件重新训练 CATCH；
5. 增加总体平均和各异常类型结果，不制作雷达图。

### 需要新增的表格

- 表：CATCH 18 个数据集的纳入/排除审计；
- 表：12 个真实数据集族的完整结果，ASD 使用 family macro-average；
- 表：12 个合成 CSV 及六类异常汇总 AUROC/AP；
- 表：跨数据集平均排名和 win/tie/loss 统计。

---

## 2.2 问题：TEP 为什么只使用子集

### 审稿人的问题

审稿人不清楚：

- 为什么只使用 M1、M3、M4；
- 为什么只使用 8 个故障；
- 这是否是作者人为挑选的有利子集；
- TEP 是否是 custom dataset。

### 我们如何解决

**P0：审计 TEP 数据可用性，并以六模式全故障实验替代仅依赖 selected 子集的主证据。**

### 具体步骤

#### 任务 E16：TEP 数据审计

1. 列出 extended TEP 中全部可用 operating modes；
2. 列出每个 mode 下：
   - 正常训练序列数量；
   - 正常验证序列数量；
   - 每种 fault 的序列数量；
   - 通道定义是否一致；
   - 标签粒度；
   - 序列长度。
3. 记录 M1、M3、M4 和 8 个故障的真实选择依据；
4. 核对选择是否在查看 SCAR 结果前确定；
5. 若依据来自数据提供论文或公开设置，补充引用和原始协议说明；
6. 明确 TEP 不是自建数据集，而是 established extended TEP 的 controlled subset。
7. 修正文件号与官方故障号的映射：`d01-d28` 反向对应 `IDV28-IDV1`，
   即 `IDV = 29 - d`；论文表格不得再把 `dXX` 直接写成同号 `IDVXX`。

#### 实验 E17：TEP 全可用模式与故障验证

项目现已具备 M1-M6、每模式 `d00-d28` 的完整 MMFDD-TEP 数据，因此固定执行
方案 A 作为 rebuttal 主证据。方案 B/C 移入论文修订 backlog，不占用 rebuttal
远程训练关键路径。

**方案 A：全可用模式与故障**

1. 使用所有满足样本完整性要求的 mode；
2. 使用所有在这些 mode 下可用的 fault；
3. 重新计算：
   - SMC@K；
   - SFR；
   - SMR@K；
   - fault-normal gap；
   - mode-wise FPR std；
   - tail calibration error。

**方案 B：多组替代子集**

1. 构建至少 3 组不同 mode/fault 子集；
2. 子集选择规则提前固定；
3. 报告各子集机制指标均值与标准差；
4. 检验结果是否依赖 M1/M3/M4。

**方案 C：Leave-one-mode-out**

1. 每次使用其中若干模式训练/构建记忆；
2. 对剩余模式仅做状态 novelty 或 out-of-regime 分析；
3. 不将未见模式当作正常同工况检索任务；
4. 用于说明模型在模式变化时的行为边界。

### 需要新增的表格

- 表：TEP 全部模式与故障的数据可用性；
- 表：六模式全故障的 SMC@K、SFR、SMR@K、fault-normal gap、FPR std 和 tail errors；
- 表：query mode × retrieved mode 的数值混淆矩阵，不制作热力图。

---

## 2.3 问题：Table 1 与 Related Work 不一致，基线选择缺少依据

### 审稿人的问题

- Related Work 中出现但 Table 1 未比较的方法较多；
- Table 1 中部分方法没有在 Related Work 中充分介绍；
- 缺少明确的 competitor selection rule；
- Related Work 没有清楚说明 SCAR 与相关方法的差异和局限。

### 我们如何解决

**P0：建立 Related Work—Baseline 一一对应关系，并用最近 retrieval 基线的新增结果支撑。**

### 具体步骤

#### 任务 E18：基线映射表

为所有方法建立以下字段：

- 方法；
- 年份/会议；
- 方法类别；
- 正常参考形式；
- 是否全局共享候选池；
- 是否查询特定；
- 是否先做状态筛选；
- 是否做 patch/subsequence retrieval；
- 是否在 Table 1；
- 未纳入原因。

#### 任务 E19：重写 Related Work 结构

按以下四组组织：

1. Global reconstruction/forecasting references；
2. Unified association/representation references；
3. Memory/prototype references；
4. Instance/subsequence retrieval references。

每组最后补一句：

- 该类方法如何定义正常参考；
- 与 SCAR 的差异；
- 该类中哪些方法进入主实验。

#### 任务 E20：基线选择原则

正文新增明确说明：

- 覆盖不同技术家族；
- 优先加入当前公开代码可复现的近期方法；
- 对最接近的 memory/retrieval 方法给予额外比较；
- 所有方法采用统一评估协议；
- 对不兼容方法给出具体原因。

### 需要新增的表格

- 表：Related Work 与主实验基线映射；
- 表：最相关方法与 SCAR 的结构差异。

---

## 2.4 问题：Figure 1 太密集，且与 Section 3 的组织不一致

### 审稿人的问题

- Figure 1 在 100% 缩放下不可读；
- 图中三块与正文的 Stage A、Stage B、Level 1、Level 2 不一致；
- 读者无法理解整个流程的层级。

### 我们如何解决

**P3：rebuttal 中用文字明确承认问题并给出三阶段重组方案；图形重绘移入论文修订
backlog，不占用当前实验关键路径。**

### 具体步骤

#### 任务 E21：统一流程层级

先将正文重组为与主图一一对应的三阶段：

1. **Section 3.2 / Stage I: Representation Learning**
2. **Section 3.3 / Stage II: Memory Construction**
3. **Section 3.4 / Stage III: Retrieval and Anomaly Scoring**

原 STSD、state-conditioned patch representation 和 self-supervised objectives
统一作为 Section 3.2 的子小节；memory purification/coreset 独立为 Section 3.3；
Level 1/2、三类诊断和融合统一置于 Section 3.4。

其中 Stage III 内部包含：

- Level 1: State-level candidate filtering；
- Level 2: Context-level local reference retrieval。

不再混用：

- Stage A / Stage B；
- 图中的 (a)/(b)/(c)；
- 章节 3.2/3.3/3.4；
- Level 1/Level 2。

#### 任务 E22：重绘主图

本任务不进入 rebuttal 交付，仅在允许提交修订论文或后续 camera-ready/resubmission 时执行。

主图只保留：

\[
X\rightarrow(S,R)\rightarrow(h,z,c)
\rightarrow\text{Memory}
\rightarrow\text{L1/L2 Retrieval}
\rightarrow\text{Diagnostics}
\rightarrow A_t.
\]

删除：

- 小字号公式；
- ECDF 小曲线；
- 过多 patch 图标；
- 所有训练细节；
- 复杂图例。

#### 任务 E23：拆分检索细节图

本任务不进入 rebuttal 交付，仅保留为论文修订 backlog。

论文修订阶段可新增一个单独图：

- Query window；
- Level-1 Top-M windows；
- Level-2 Top-K context patches；
- Content-nearest neighbors；
- Local normal reference；
- Memory score。

### Rebuttal 交付

- 表：旧术语与统一 Stage I-III / Level 1-2 术语映射；
- 文字：说明修订稿将如何重组 Section 3.2-3.4；
- rebuttal 中不提交 Figure 1、检索细节图或融合图。

---

## 2.5 问题：方法部分缺少直觉，符号和分数解释不清楚

### 审稿人的问题

点名不清楚的内容：

- \(f_{\text{trunk}}\)；
- \(f_{\text{scale}}\)；
- state-novelty score；
- patch-level memory score；
- per-scale reconstruction score。

### 我们如何解决

**P2：把核心定义从 Appendix 移回正文，并在公式后增加“作用—输入—输出—异常含义”解释。**

### 具体步骤

#### 任务 E24：模块解释模板

每个模块统一增加四句话：

1. 输入是什么；
2. 做了什么；
3. 为什么需要；
4. 输出用于哪一步。

#### 任务 E25：三类分数解释

- **State novelty**：查询的慢变状态是否得到训练正常状态支持；
- **Memory score**：在状态与上下文相容的参考中，局部残差内容仍有多大差异；
- **Reconstruction score**：当前 patch 是否能由周围正常上下文恢复。

#### 任务 E26：多尺度解释

明确说明：

- 短 patch 对短促异常敏感；
- 长 patch 对持续异常和趋势偏移敏感；
- 多尺度不是简单重复，而是覆盖不同持续时间。

#### 任务 E26b：正文自包含检查

以下内容必须进入正文而不能只引用 Appendix：

- STSD 的初始化、可学习修正和分解作用；
- `f_trunk`、`f_scale` 和 gate 的输入输出；
- 两个自监督目标；
- memory purification 的定义；
- `h/c/z` 分别用于 state filtering、context retrieval 和 content distance 的原因；
- 三类分数的 patch-to-time 展开；
- 训练 ECDF 与最终平均融合。

---

# 3. Reviewer dk9H

## 3.1 问题：理论依据不够清楚

### 审稿人的问题

审稿人认为 motivation 合理，但没有足够理论直觉说明：

- 为什么 global reference 会产生偏差；
- 为什么状态筛选和上下文筛选能够降低偏差；
- 为什么两级设计必要。

### 我们如何解决

**P2：增加轻量统计推导，不追求复杂定理，但明确展示跨工况参考引入的额外距离项。**

### 具体步骤

#### 理论任务 T1：工况混合偏差推导

1. 假设正常样本满足：
   \[
   X=\mu_R+\epsilon,
   \]
   其中 \(R\) 是 operating regime。
2. 推导全局参考下：
   \[
   \mathbb E[\|X-X'\|^2\mid R=r]
   =
   \text{within-regime variation}
   +
   \mathbb E_{R'}\|\mu_r-\mu_{R'}\|^2.
   \]
3. 解释第二项是跨工况正常差异；
4. 当参考限制到 \(R'=r\) 时，该偏差项消失；
5. 再将局部上下文 \(C\) 加入条件：
   \[
   p(X\mid R,C),
   \]
   说明同一工况内不同阶段仍会产生上下文偏差；
6. 明确该推导只是 mechanism intuition，不是完整误差界或一致性定理。

#### 理论任务 T2：两级检索必要性说明

1. Level 1 近似条件于 \(R\)；
2. Level 2 近似条件于局部上下文 \(C\)；
3. Content distance 估计剩余局部偏差；
4. 将状态筛选、上下文筛选和异常距离建立清晰对应关系。

### 需要新增的文字/公式

- Introduction 或 Method 开头增加一段统计直觉；
- Appendix 增加完整推导；
- 正文只保留核心公式和解释。

---

## 3.2 问题：缺少真实工业例子

### 审稿人的问题

希望看到同一局部波形在一个工况正常、在另一个工况异常的直观现实案例。

### 我们如何解决

**P3：增加一个不依赖实验结果的动机示例；真实案例图移入论文修订 backlog。**

### 具体步骤

#### 任务 E27：文字案例

在 Introduction 增加：

- 电机启动后的短暂电流尖峰可能正常；
- 稳态低负荷下出现同样尖峰可能异常。

或：

- 升载阶段压力持续上升正常；
- 稳态阶段相同趋势可能表示控制偏移。

注明这是 motivating example，不是 SCAR 的故障诊断结论。

#### 实验 E28：真实数据案例图

E28 不进入 rebuttal 交付。若后续修订论文允许加图，再执行：

1. 在 TEP 或 SWaT 中寻找波形相似、状态不同的正常/异常片段；
2. 采用独立波形相似度进行筛选；
3. 图中展示：
   - 两个局部 patch；
   - 前后较长上下文；
   - 所属 mode 或运行阶段；
   - SCAR 检索到的参考；
   - global retrieval 检索到的参考；
   - 异常分数。
4. 若主数据集没有真实工况标签，只描述为“different slow-state contexts”，不声称真实 mode。

---

## 3.3 问题：增大 look-back window 或 patch size 是否可替代 SCAR

### 审稿人的问题

审稿人怀疑 SCAR 的优势可能主要来自 patch 太短；普通方法通过增加窗口或 patch 长度也可能看到工况信息。

### 我们如何解决

**P1：新增长窗口/长 patch 控制实验，直接排除替代解释。**

### 具体步骤

#### 实验 E29：窗口长度控制

1. 完整协议测试：
   \[
   L\in\{64,128,256,512\}
   \]
   或根据数据采样率设置等价范围；
2. 对每个 \(L\) 比较：
   - Global retrieval；
   - Context-only；
   - Full SCAR。
3. 保持 latent dimension 和 memory size 尽量一致；
4. 对不同窗口长度重新训练，避免不公平复用；
5. 报告 AUROC/AP 与运行时间。

三天轻量协议只保留五主集的端点 `L={128,512}`，比较 full SCAR 与 global
retrieval。`L=128` 的 full 直接复用 anchor；每个数据集只新增一个 L512 模型。

#### 实验 E30：patch size 控制

1. 完整协议仅在五个主数据集测试：
   \[
   p\in\{8,16,32,64\}
   \]
   或 short/medium/long；
2. 比较：
   - single-scale global retrieval；
   - single-scale SCAR；
   - multi-scale SCAR。
3. 报告：
   - AUROC/AP；
   - 检索状态距离和上下文距离。

三天轻量协议保留 `p={8,64}` 两个端点，并复用默认多尺度 `8+32`。每个单尺度
模型同时产生 full 与 global 结果；不在 E30 中运行 synthetic 文件，避免与 CATCH
扩展数据实验混淆。

#### 实验 E31：匹配计算预算控制

1. 让长窗口 global retrieval 与 SCAR 具有近似：
   - 参数量；
   - memory bank 大小；
   - 推理时间。
2. 避免 SCAR 因计算预算更大而占优。
3. 若长窗口 global retrieval 接近 SCAR，应收窄结论；
4. 若仍明显低于 SCAR，说明“看到更多历史”不等于“按状态改变参考集合”。

三天轻量协议复用 E29 的 L512 模型作为 global 候选，仅测试
`q={1,0.25,0.10}` 和 `top_K={10,20,40}`。参数量、bank bytes、推理延迟均以
±15% 为注册容差；若无候选同时满足，按三轴绝对对数相对误差和最小选择并披露实际
偏差。候选选择不得读取测试性能。

### 需要新增的表格

- 表：窗口长度 × 检索策略的 AUROC/AP、运行时间和 bank size；
- 表：五主集 patch size × 检索策略的完整九指标；
- 表：相同计算预算下的对比。

---

## 3.4 问题：SCAR 是否适用于单变量时间序列

### 审稿人的问题

论文只研究多变量设置，未验证 univariate setting。

### 我们如何解决

**P0：进行架构适配说明，并在完整 TSB-AD-U 官方评估清单上补充单变量实验。**

### 具体步骤

#### 任务 E32：架构适配

1. 说明当 \(C=1\) 时：
   - slow-fast decomposition 仍成立；
   - channel-wise gate 退化为 scalar gate；
   - \(h,z,c\) 和两级检索仍可定义。
2. 说明单变量下缺少跨通道互补信息，因此不保证效果与多变量相同。

#### 实验 E33：单变量基准

1. 本地 TSB-AD-U 已完整具备 870 条序列；
2. tuning 48 条仅用于预先选择统一配置，不进入正式测试汇总；
3. 正式运行官方默认 U-Eva 350 条，不根据 SCAR 表现筛选；
4. U-Eva-Full 822 仅保留为论文修订扩展，不进入 rebuttal P0；
5. 与 TSB-AD benchmark 公开强基线比较，公开数字明确标记为 `reported`；
6. 对每个融合和子分数统一报告 AUROC、AP、Point-F1、PA-F1、Aff-P、Aff-R、Aff-F1、
   VUS-ROC、VUS-PR，以及官方逐序列平均和来源数据集 macro-average；
7. 分析 scalar gate 是否有贡献；
8. 若结果一般，将单变量定位为已验证适用但性能仍有边界，不夸大普适性。

### 需要新增的表格

- 表：TSB-AD-U Eva 350 汇总结果；
- 表：SCAR 与 w/o scalar gate；
- 表：单变量来源数据集 macro-average 和平均排名。

---

## 3.5 问题：评估数据集过少，缺少 TSB-AD/TAB 等综合基准

### 审稿人的问题

当前五个数据集不足以支持广泛 SOTA 主张，希望看到更全面 benchmark。

### 我们如何解决

**P0：完成本地已具备的 TSB-AD-M 和 TSB-AD-U 官方综合基准。TAB 不作为本轮
rebuttal 必需项。**

### 具体步骤

#### 实验 E34：TSB-AD 协议与清单审计

1. 本地固定使用官方 manifests：
   - M-Tuning 20；
   - M-Eva 180；
   - U-Tuning 48；
   - U-Eva 350；
   - U-Eva-Full 822（官方扩展清单，仅审计，不作为 rebuttal P0）。
2. tuning 清单只用于统一配置选择；
3. 多变量正式运行 M-Eva 180；
4. 单变量正式运行官方默认 U-Eva 350；U-Eva-Full 822 留作修订扩展；
5. 每个 CSV 仅使用文件名 `_tr_N_` 指定的正常前缀训练、归一化、建库和校准；
6. 标签不参与训练、归一化、融合选择或 VUS window 估计；
7. TAB 保留为论文修订/后续工作，不与本轮 TSB-AD 同时扩张范围。

#### 实验 E35：综合基准运行

固定运行：

1. TSB-AD-M Eva 180 条；
2. TSB-AD-U Eva 350 条；
3. SCAR 主融合固定为 `cdf_mean`，不得按 CSV 或指标选择融合方式；
4. TSB-AD 只新增 SCAR 结果，不运行 PaAno；官方 TSB-AD 基线数字明确标记为
   `reported`，不写成统一协议重跑。

报告：

- 每个融合和子分数的 AUROC、AP、Point-F1、PA-F1、Aff-P、Aff-R、Aff-F1、VUS-ROC、VUS-PR；
- 平均排名；
- win/tie/loss；
- 完整失败清单和失败原因；
- 数据规模分层结果。

#### 实验 E36：结论范围调整

无论是否完成全集，都修改：

- 将“state-of-the-art on multivariate TSAD”收窄为“on the evaluated benchmarks”；
- 若新增综合基准支持更广泛结论，再据实恢复更强表述。

---

## 3.6 问题：缺少计算成本、内存、可扩展性分析

### 审稿人的问题

审稿人希望知道：

- 训练成本；
- 推理成本；
- GPU/CPU 内存；
- memory bank 大小；
- 随数据规模增长的扩展性；
- 与 SOTA 的效率比较。

### 我们如何解决

**P0：利用五个主数据集和相关基线的必要重训，同步完成复杂度、运行时间、内存占用
和 memory-bank 影响分析。**

### 具体步骤

#### 理论任务 T3：复杂度分析

定义：

- \(N_w\)：训练窗口数；
- \(N_p^{(s)}\)：尺度 \(s\) 的 patch 数；
- \(d_s\)：状态维度；
- \(d_z\)：内容/上下文维度；
- \(M\)：Level-1 候选窗口数；
- \(K\)：Level-2 检索数。

推导：

- state bank memory；
- patch memory；
- exact Level-1 search；
- Level-2 candidate-pool search；
- FAISS approximate search；
- coreset 后的复杂度。

#### 实验 E37：运行时间与内存表

1. 在 MSL、PSM、SMAP、SMD、SWaT 五个主数据集全部测量，不再只选小/中/大代表集；
2. 所有方法在同一远程服务器、同一 GPU 型号和固定 CPU/RAM 环境测量；
3. 记录：
   - Stage-I 训练时间；
   - memory construction 时间；
   - 每 10k time steps 推理时间；
   - 吞吐量；
   - 峰值 GPU 显存；
   - 峰值 CPU 内存；
   - memory bank 磁盘/内存占用；
   - 参数量。
4. 固定重跑比较：
   - SCAR；
   - PaAno；
   - PUAD；
   - PGRF-Net。
   CATCH 仅作为官方 `reported` 性能参照，不纳入本次资源实测；
5. 每个模型仅使用 seed `42` 训练一次；
6. 性能和资源数据必须来自同一次正式训练，禁止为效率表另行重训；
7. 推理在充分预热后重复计时 3 次，报告原始计时、均值和标准差，性能指标仅报告
   固定 seed 结果；
8. 若方法需要不同 batch size，报告实际 batch size，并同时提供每时间点/每窗口吞吐，
   不用 OOM 方法的更小 batch size 冒充同批量公平比较；
9. 保存 GPU UUID、CUDA/PyTorch 版本、CPU 型号、RAM、驱动版本和 Git commit。

#### 实验 E38：memory keep ratio

测试：

\[
q\in\{1.0,0.5,0.25,0.10,0.05\}.
\]

报告：

- AUROC/AP；
- memory size；
- inference latency；
- candidate retrieval recall；
- 五个主数据集逐项结果及跨数据集平均；
- 不绘制 Pareto 曲线，以性能、latency、RAM 和 bank size 联合表呈现。

#### 实验 E39：规模扩展性

rebuttal 仅保留可由 E38 直接得到的 bank size、latency 和 RAM 缩放表。以下独立扩展
实验移入论文修订 backlog：

1. 按变量数构建通道子集；
2. 按序列长度构建不同规模数据；
3. candidate M/K 全网格；
4. 额外 10%/25%/50%/75%/100% 训练窗口子集。

### 需要新增的表格

- 表：五数据集 × 五个实测方法的参数、训练时间、建库时间、推理吞吐、显存、CPU 内存和 bank size；CATCH 仅列官方性能，不填本次资源实测值；
- 表：五数据集 memory keep ratio 的 AUROC/AP、latency、RAM 和 bank size；
- 表：理论复杂度。

---

## 3.7 问题：与近期 SOTA 的比较不足

### 审稿人的问题

Reviewer dk9H 与另外两位审稿人一致，认为当前虽然基线数量多，但缺少最接近、最新、最能挑战 SCAR 的方法。

### 我们如何解决

与 E6–E8 联动：

1. 增加最近 retrieval/subsequence 方法；
2. 检查 2025–2026 的 memory/prototype、patch 和 retrieval 方法；
3. 优先加入官方代码可用且协议兼容的方法；
4. 在 Related Work 和 Table 1 中保持一致；
5. 报告相同协议下结果；
6. 不将来自不同协议的公开数字直接混入同一表格。

---

# 4. Area Chair（AC）

## 4.1 问题：数据集和 TEP 子集选择依据不清楚

### AC 的要求

希望看到：

1. 主实验五个数据集的清晰选择依据；
2. TEP 三个模式和八个故障的选择依据；
3. 更广泛 benchmark；
4. 避免选择性评估的证据。

### 我们如何解决

对应任务：

- E13 数据集选择审计；
- E14 扩展 CATCH 数据集；
- E15 六类合成异常；
- E16 TEP 数据审计；
- E17 TEP 六模式全故障验证；
- E34–E35 TSB-AD-M/U 综合基准。

### AC 回应中必须提供的证据

- 一张完整数据集纳入/排除表；
- 一张 TEP 模式—故障可用性表；
- CATCH 当前五数据集之外的全部兼容真实数据集结果；
- TSB-AD-M Eva 180 和 TSB-AD-U Eva 350 结果；
- TEP 六模式全故障结果；
- 明确说明所有筛选规则在查看新增结果前固定。

---

## 4.2 问题：缺少最接近 retrieval-based 方法

### AC 的要求

希望看到：

- 与 DAMP、GDFlex、PaAno 等最接近方法的结果；
- 或技术上有说服力的不可比说明。

### 我们如何解决

对应任务：

- E6 兼容性审计；
- E7 新增外部检索基线；
- E8 内部公平对照；
- E18–E20 Related Work—Baseline 对齐。

### AC 回应中必须提供的证据

- 至少一个新增最接近方法的定量结果；
- 对其余方法的逐项协议说明；
- 方法结构差异表；
- 不再仅用“baseline suite is broad”回应。

---

## 4.3 问题：TEP 之外缺少 condition-compatible retrieval 证据

### AC 的要求

希望看到：

- 主数据集上的 retrieval visualization；
- clustering；
- temporal consistency；
- retrieval consistency；
- 或其他无标签机制分析。

### 我们如何解决

对应任务：

- E1 独立状态代理距离；
- E2 原始局部上下文距离；
- E3 状态—上下文联合权衡表；
- E4 扰动稳定性；
- E5 时间一致性；
- E28 仅保留为论文修订 backlog。

### AC 回应中必须提供的证据

- 五个主数据集上的代理机制表；
- 五个主数据集上的稳定性与时间一致性表；
- 明确代理指标不等于 ground-truth mode accuracy；
- 说明指标独立于用于检索的 \(h/c\) 表示。

---

## 4.4 问题：缺少计算和内存成本比较

### AC 的要求

希望看到：

- computation；
- memory cost；
- inference scalability；
- memory-bank size impact。

### 我们如何解决

对应任务：

- T3 复杂度分析；
- E37 运行时间与内存表；
- E38 memory keep ratio；
- E39 由 E38 派生的 bank size 缩放表。

### AC 回应中必须提供的证据

- 同硬件实测表；
- memory bank 额外开销；
- 五数据集、同硬件、同次训练得到的性能—效率折中表；
- 不仅报告参数量。

---

## 4.5 问题：净化鲁棒性仍不足

### AC 的要求

希望看到：

- contamination sensitivity；
- purification threshold sensitivity；
- rare normal transient removal；
- subtle low-error anomaly survival。

### 我们如何解决

对应任务：

- E9 净化比例敏感性；
- E10 训练污染率；
- E11 稀有正常保留；
- E12 低误差异常存活。

### AC 回应中必须提供的证据

- 五数据集净化比例完整表；
- 五数据集污染率完整表；
- 正常误删/异常删除统计；
- 修改 limitation。

---

## 4.6 问题：论文表达、图1和 Appendix 依赖过重

### AC 的要求

指出：

- 关键定义和直觉主要放在 Appendix；
- 正文方法太粗；
- Figure 1 太密集；
- Section 3 组织不一致；
- patch size、计算、memory bank 影响缺少分析。

### 我们如何解决

对应任务：

- E21 统一流程层级，E22–E23 移入论文修订 backlog；
- E24–E26 补模块与分数直觉；
- E29–E31 patch/window 控制；
- T1–T2 理论直觉；
- E37–E39 五数据集效率、内存和 bank size 影响。

由于 rebuttal 阶段不允许提交图片，对 Figure 1 的回应只包含问题承认、统一后的三阶段
术语和明确修订方案，不在本轮回复中提交重绘图片。

---

# 5. 跨审稿人统一实验任务清单

## 5.1 P0：必须优先完成

### 实验组 A：CATCH 与综合基准完整覆盖

- E13 CATCH 全部 18 个数据集/类型审计；
- E14 补齐当前五数据集之外的全部兼容真实数据；
- E15 CATCH 六类合成异常、12 个原始 CSV；
- E34-E35 TSB-AD-M Eva 180 与 TSB-AD-U Eva 350；
- U-Eva-Full 822 只保留为论文修订扩展。

### 实验组 B：五数据集最近相关基线与资源比较

- E6 全部相关方法兼容性审计；
- E7 PaAno、PUAD、PGRF-Net 五数据集正式运行；MEMTO 不运行，CATCH 不重跑；
- E37 同次训练采集性能、时间、GPU/CPU 内存、吞吐和产物大小；
- E8 SCAR 内部 retrieval 对照；
- DAMP/GDFlex 只做诚实协议与运行环境说明，不生成未经验证的替代数字。

### 实验组 C：主数据集条件相容机制证据

- E1 状态代理距离；
- E2 原始上下文距离；
- E3 状态—上下文联合权衡表；
- E4 扰动稳定性表；
- E5 时间一致性表。

### 实验组 D：净化鲁棒性

- E9 五数据集净化比例完整扫描；
- E10 五数据集污染率、no/default purification 和最高污染率 stronger purification；
- E11 从 E9 audit 产物派生稀有正常覆盖；
- E12 从 E10 audit 产物派生低误差异常存活。

### 实验组 E：TEP 全量机制与 memory-bank 影响

- E16 TEP 数据审计；
- E17 六模式全故障机制验证；
- E38 五数据集 memory keep ratio；
- E39 从 E38 派生 bank size、latency 和 RAM 缩放表；
- T3 理论复杂度。

---

## 5.2 P1：强烈建议完成

### 实验组 F：长窗口与 patch 替代解释

- E29 窗口长度；
- E30 patch size；
- E31 相同计算预算控制。

仅在 P0 远程任务已稳定排队且不会挤占 CATCH、TSB-AD、E9/E10 和相关基线时启动。

---

## 5.3 P2：资源允许时完成

### 写作与论文修订 backlog

- E21、E24-E26：流程层级和方法解释，rebuttal 以文字回应；
- T1-T2：工况混合偏差与两级检索直觉；
- E22-E23、E28：Figure 1、检索细节图和真实案例图，待允许修订论文时执行；
- E39 中独立通道数、序列长度和 M/K 网格；
- TAB 完整 benchmark。

---

# 6. 需要新增的实验表格

除纯资源/机制字段外，所有含 SCAR 性能的表统一保留 `raw_max`、`zscore_mean`、
`cdf_mean`、`cdf_max` 及全部可用诊断子分数的指标列；正文可突出预注册主结果
`cdf_mean`，附表/补充文字不得丢弃其余分数。P0 收口额外生成 E1-E5 三策略全分数表
和 TEP 全分数表，P1 的 E31 同时列目标 SCAR 与预算匹配 global 的全分数结果。

## 表 T1：数据集纳入/排除审计

字段：

- Dataset；
- Source benchmark；
- Uni/Multi；
- Label granularity；
- Train/test split；
- Protocol compatibility；
- Included；
- Exclusion reason。

## 表 T2：TEP 模式—故障可用性

字段：

- Mode；
- Normal train/val；
- Available faults；
- Sequence count；
- Sequence length；
- Included reason。

## 表 T3：新增最近相关基线

字段：

- Method；
- MSL/PSM/SMAP/SMD/SWaT AUROC/AP；
- Avg rank；
- Fixed-seed result (`seed=42`)；
- Reported/Reproduced。

## 表 T4：方法结构差异

字段：

- Method；
- Global/shared reference；
- Query-specific candidate set；
- State filtering；
- Context retrieval；
- Content scoring；
- Transductive/inductive。

## 表 T5：主数据集代理机制验证

字段：

- Dataset；
- Strategy；
- State proxy distance；
- Context distance；
- Retrieval stability；
- Temporal consistency。

## 表 T6：净化敏感性

字段：

- Purification ratio；
- AUROC/AP；
- Bank size；
- Normal removal rate；
- Anomaly removal rate。

## 表 T7：污染鲁棒性

字段：

- Contamination rate；
- No purification；
- Default purification；
- Strong purification；
- Low-error anomaly survival。

## 表 T8：效率比较

字段：

- Dataset；
- Method；
- Parameters；
- Training time；
- Memory construction time；
- Inference throughput；
- GPU memory；
- CPU memory；
- Bank size；
- Batch size；
- Hardware/Software environment。

## 表 T9：TEP 六模式全故障机制验证

字段：

- Mode subset；
- Fault subset；
- SMC@K；
- SFR；
- SMR@K；
- Fault-normal gap；
- FPR std；
- Tail errors。

## 表 T10：窗口与 patch 控制

字段：

- Window length；
- Patch size；
- Retrieval strategy；
- AUROC/AP；
- Runtime；
- Bank size。

## 表 T11：CATCH 完整真实数据集结果

字段：

- Dataset；
- SCAR；
- Closest baseline；
- CATCH；
- Best published/reproduced baseline；
- Rank；
- ASD family macro-average 标记。

## 表 T12：TSB-AD-U/M 综合基准

字段：

- Edition/Split；
- Sequence count；
- SCAR；
- Best reported/reproduced baseline；
- VUS-PR/VUS-ROC/AUROC/AP；
- Official per-series average；
- Source-dataset macro-average；
- Failure count。

## 表 T13：CATCH 六类合成异常

字段：

- Anomaly type；
- Contamination ratio/file；
- SCAR AUROC/AP；
- CATCH AUROC/AP；
- Closest baseline AUROC/AP；
- Two-ratio type average；
- Overall average。

---

# 7. 论文修订阶段的图形 Backlog

> 本节所有图均禁止放入当前 rebuttal，也不进入远程实验关键路径。当前 rebuttal 只提交
> 第 6 节定义的数值表格和文字分析。本节仅用于后续允许修改论文 PDF、camera-ready
> 或重新投稿时维护。

## 图 F1：重做 SCAR Overall Pipeline

- 三阶段；
- 大字号；
- 仅保留核心变量；
- 与 Section 3 一致。

## 图 F2：Two-Level Retrieval Detail

- Query；
- Level-1 state filtering；
- Level-2 context retrieval；
- Content nearest references；
- Memory score。

## 图 F3：主数据集状态—上下文二维机制图

- 横轴状态代理距离；
- 纵轴上下文距离；
- 比较四种 retrieval strategy。

## 图 F4：Retrieval Stability 曲线

- 横轴扰动强度；
- 纵轴 Top-M/Top-K overlap。

## 图 F5：时间一致性案例

- 原始信号；
- 状态表示；
- 参考集合变化；
- 异常分数。

## 图 F6：净化比例敏感性

- 净化比例—AUROC/AP；
- 可增加 bank size 次坐标或单独图。

## 图 F7：污染率鲁棒性

- 污染率—AUROC/AP；
- 比较不同 purification。

## 图 F8：正常覆盖变化

- 净化前后低密度正常 patch 的覆盖；
- 可用 embedding 散点或簇保留率柱状图。

## 图 F9：memory keep ratio 的性能—效率曲线

- AUROC/AP vs latency；
- AUROC/AP vs RAM。

## 图 F10：bank size 扩展性

- bank size—latency；
- bank size—memory。

## 图 F11：窗口长度与 patch size

- window length—AUROC/AP；
- patch size—六类异常性能。

## 图 F12：六类 synthetic anomaly 雷达图

- 必须配套数值表；
- 明确生成协议；
- 与相关基线对比。

## 图 F13：真实世界动机案例

- 同类局部波形；
- 不同慢状态上下文；
- 不同参考与异常分数。

---

# 8. 需要修改的文字和公式

## 8.1 Abstract

- 收窄未经更广泛 benchmark 支撑的 SOTA 表述；
- 明确 TEP 是机制验证，不是主 benchmark；
- 若补充新数据集，可更新数据集数量；
- 不将代理机制分析称为真实工况恢复。

## 8.2 Introduction

新增：

1. 真实工业例子；
2. global reference 的跨工况偏差直觉；
3. 数据集适用范围；
4. SCAR 的核心创新不是 nearest-neighbor 本身，而是 query-specific reference-set construction；
5. 不再使用过强的普适性表述。

## 8.3 Related Work

修改：

1. 按正常参考类型重组；
2. 确保每个 Table 1 方法都有对应介绍；
3. 对最接近 retrieval 方法展开具体比较；
4. 增加基线选择原则；
5. 对未比较方法说明技术原因。

## 8.4 Method

修改：

1. 统一 Stage I–III 与 Level 1–2；
2. 在正文解释 \(f_{\text{trunk}}\)、\(f_{\text{scale}}\)；
3. 解释 \(h,z,c\) 的不同职责；
4. 解释三类分数各自检测什么；
5. 增加两级检索统计直觉；
6. 将重要定义从 Appendix 移入正文；
7. 清楚说明 STSD 是 retrieval-oriented representation，而非物理解耦。

## 8.5 Experiments

新增：

1. 数据集客观选择规则；
2. CATCH 角色澄清；
3. TEP 子集选择依据；
4. 新增基线协议；
5. 代理机制指标定义；
6. 净化、污染、窗口、patch、效率设置；
7. 所有新增结果均明确标注固定模型 seed `42`；bootstrap 区间和计时波动需与模型 seed 方差区分；
8. 计算资源与测量方法。

## 8.6 Limitations

新增：

1. 数据集覆盖仍有限；
2. 单变量仅在新增实验支持后才能声称；
3. purification 无法完全识别污染；
4. 稀有正常可能被删除；
5. 低误差异常可能残留；
6. memory-bank 成本随规模增长；
7. 频域先验不适用于所有信号；
8. 主数据集缺少真实工况标签，代理指标不能代替 ground truth；
9. 更大 benchmark 和更强理论保证仍是未来工作。

## 8.7 数学公式与理论

新增：

- 工况混合偏差推导；
- 条件于 regime/context 后的距离分解；
- 理论复杂度；
- memory size 与检索复杂度；
- 说明推导属于直觉性分析，不声称一般性定理。

---

# 9. 推荐执行顺序

## 阶段 0：远程服务器与协议冻结

1. 记录服务器 GPU/CPU/RAM、驱动、CUDA、PyTorch、Python 和 Git commit；
2. 建立 SCAR 与 PaAno/PUAD/PGRF-Net 的独立可复现环境；保留 MEMTO/CATCH
   环境说明，但不把它加入正式队列；
3. 安装并验证统一资源监控；
4. 对每条流水线先执行一个小样本 dry-run 和一个模型级 smoke test；
5. 冻结数据 manifests、seed `42`、指标、融合方式和失败处理规则；
6. 确认 DAMP/GDFlex 没有可用 MATLAB 环境，只做协议审计。

## 阶段 1：五个主数据集一次性重训

1. 在 MSL、PSM、SMAP、SMD、SWaT 各训练一次 SCAR seed-42 Stage-A；
2. 构建默认 Stage-B、测试并全程采集 E37 资源数据；
3. 保存 checkpoint、config、逐点 scores、memory provenance、resource JSON 和环境清单；
4. 对 PaAno、PUAD、PGRF-Net 做五数据集正式训练；MEMTO 不运行，CATCH 只读取官方结果；
5. 同次生成性能表 T3 和效率表 T8，不为效率单独重跑。

## 阶段 2：复用五数据集 checkpoint 完成机制与鲁棒性

1. E1-E5：导出 full/global/state-only/context-only 检索证据并生成表 T5；
2. E9：六个净化比例、五数据集，仅重建 Stage-B；
3. E10：五个污染率、五数据集、3 event folds；
4. E11/E12：直接分析 E9/E10 audit 产物；
5. E38：五个 memory keep ratios，生成 bank size、latency、RAM 联合表；
6. 所有失败任务保留 manifest 和日志后使用相同实验名断点续跑。

## 阶段 3：补齐 CATCH 与完整 TEP

1. E13-E14：仅运行 SCAR 在当前五数据集之外的全部兼容 CATCH 真实数据；
2. SCAR 在 ASD 12 个子数据集全部运行并汇总 family macro-average；
3. E15：SCAR 运行六类、两个异常比例版本的 12 个 synthetic CSV；
4. E16-E17：运行 MMFDD-TEP 六模式全故障并生成 T2/T9；
5. CATCH 方法不创建训练任务，仅录入论文发布值并标记
   `reported from CATCH`；
6. 所有结果只以表格和文字摘要进入 rebuttal。

## 阶段 4：TSB-AD-M/U 综合基准

1. 使用 tuning manifests 冻结统一配置；
2. 运行 M-Eva 180 条；
3. 运行官方默认 U-Eva 350 条；
4. U-Eva-Full 822 不进入 rebuttal P0；
5. 不在 TSB-AD 上运行 PaAno；
6. 输出逐序列长表、官方平均、来源数据集 macro-average 和失败清单，公开基线只标为
   `reported`。

## 阶段 5：P1 控制实验与统一写作

1. 若 P0 队列完成且资源允许，再执行 E29-E31；
2. 根据真实结果调整结论；
3. 写 AC 四点主回应，再逐位 reviewer 补充；
4. 每条回复都使用“问题—新增表格证据—数值结果—文字解释—论文修改承诺”结构；
5. Figure 1、机制图、敏感性曲线和真实案例图不放入 rebuttal；
6. 禁止承诺无法在修订稿中兑现的实验。

---

# 10. 最终分类总结

## 10.1 必须补的实验

1. 最接近 retrieval/subsequence 基线；
2. 五个主数据集上的状态代理距离与上下文距离；
3. 五个主数据集上的检索稳定性与时间一致性；
4. 五个主数据集上的净化比例敏感性；
5. 五个主数据集上的训练污染率鲁棒性；
6. TEP 六模式全故障；
7. 五数据集 × SCAR/PaAno/PUAD/PGRF-Net 的性能和资源比较，并列出 CATCH
   官方 reported 性能作为参照；
8. 五数据集 memory keep ratio 的性能—效率—bank size 折中；
9. CATCH 当前五数据集之外的全部兼容真实数据集；
10. CATCH 六类合成异常、12 个原始 CSV；
11. TSB-AD-M Eva 180；
12. TSB-AD-U Eva 350。

## 10.2 强烈建议补的实验

1. 长窗口/长 patch 与 SCAR 的控制对比；
2. 相同计算预算的 global retrieval 控制；
3. scalar gate 单变量消融；
4. 更完整的 M/K 检索规模控制。

## 10.3 必须新增的表格

1. 数据集纳入/排除表；
2. TEP 模式—故障可用性表；
3. 最相关基线结果表；
4. 方法结构差异表；
5. 主数据集代理机制表；
6. 净化敏感性表；
7. 污染鲁棒性表；
8. 效率与内存表；
9. TEP 六模式全故障机制表；
10. CATCH 完整真实数据集表；
11. CATCH 六类合成异常表；
12. TSB-AD-M/U 综合基准表；
13. memory keep ratio 的性能、latency、RAM 和 bank size 表。

## 10.4 Rebuttal 图形约束

1. rebuttal 不提交任何图；
2. 所有结果统一用表格和文字分析；
3. Figure 1、检索细节、敏感性曲线、效率曲线和真实案例图保留在第 7 节论文修订 backlog；
4. 不因不能放图而删除对应定量实验，只改变结果呈现方式。

## 10.5 必须修改的文字

1. 澄清 CATCH 只是部分基线结果来源；
2. 说明五个数据集的客观选择规则；
3. 说明 TEP 子集的真实选择依据；
4. 说明 TEP 不是 custom dataset；
5. 重写 Related Work 与 baseline 对齐关系；
6. 解释 SCAR 与 global retrieval 的核心区别；
7. 统一 Stage/Level；
8. 解释 \(f_{\text{trunk}}\)、\(f_{\text{scale}}\)、\(h,z,c\)；
9. 解释 state novelty、memory score、reconstruction score；
10. 增加真实工业例子；
11. 增加工况混合偏差推导；
12. 增加复杂度分析；
13. 扩充 Limitations；
14. 根据新增 benchmark 结果收窄或调整 SOTA 主张；
15. 明确主数据集代理指标不是 ground-truth mode labels。

---

# 11. Rebuttal 完成验收清单

- [ ] Reviewer muQn 的 3 个问题全部有实验或明确限制说明；
- [ ] Reviewer ef7G 的数据集、TEP、基线、图1、方法解释全部回应；
- [ ] Reviewer dk9H 的理论、案例、窗口、单变量、综合 benchmark、效率、SOTA 全部回应；
- [ ] AC 点名的 4 个决定性事项均有新增证据；
- [ ] CATCH 当前五数据集之外的全部兼容真实数据和六类合成异常均有结果或明确失败记录；
- [ ] TSB-AD-M Eva 180 和 U Eva 350 均完成；
- [ ] E9 和 E10 均覆盖 MSL、PSM、SMAP、SMD、SWaT；
- [ ] SCAR 与 PaAno/PUAD/PGRF-Net 在五个主数据集上完成性能和资源比较；
  CATCH 只使用官方 reported 结果，且正式 manifest 中不存在 CATCH 训练任务；
- [ ] 所有新增结果均来自统一协议；
- [ ] 所有正式随机实验均使用固定模型 seed `42`，未把调试 seed、bootstrap 或计时重复误写成模型 seed 重复；
- [ ] 所有无法比较的方法均给出逐项技术原因；
- [ ] DAMP/GDFlex 的说明区分运行环境限制与任务协议不兼容，不伪造替代实现结果；
- [ ] rebuttal 中未放入任何图片，所有证据均为表格和文字分析；
- [ ] 所有远程正式运行从第一次训练起保存资源监控、环境、配置、checkpoint 和逐点 score；
- [ ] 所有强结论均有对应实验支撑；
- [ ] 所有不被实验支持的结论均已收窄；
- [ ] Rebuttal 不只重复论文已有内容，而是明确展示新增证据；
- [ ] 每条回复都包含“问题—新增表格证据—结果—文字分析—论文修改位置”五部分。

---

# 12. 已实现的统一运行协议

当前代码已提供 `scripts/experiments/rebuttal.py
plan|run|resume|status|validate|collect`。正式任务由中央配置注册表生成，固定 seed
`42`、禁止 TSB `U/eval_full` 和绘图，按依赖关系复用 Stage-A、memory 与已有分数。
三天轻量队列可用 `--lite-anchor-root` 指向经过审计的五主集 Stage-A 目录；该入口
只复用 `stage_a.pt`，随后在当前 artifact 根目录重新执行 memory、fusion、full audit
和测试，并产生当前源码哈希对应的新 run record，禁止直接把旧记录改写成当前结果。
五主集 anchor 从首次运行起启用 full memory audit 和资源监控；E9、E10、E38 只执行
必要的 Stage-B/Test，E11、E12、E39 只分析已有 audit。

P1 中 E29 每个窗口长度只训练一个模型并共享三策略；E30 每个 single-scale 只训练
一次并共享 global/full，默认 multi-scale 复用 anchor；E31 依次匹配参数量、bank
bytes 与 1+3 推理延迟，并在无候选同时落入 ±15% 时按三轴绝对对数误差和既定平局
规则选择。所有正式结果只汇总为 CSV/JSON 表格与文字，失败或未验证记录不会进入
rebuttal 结论。
