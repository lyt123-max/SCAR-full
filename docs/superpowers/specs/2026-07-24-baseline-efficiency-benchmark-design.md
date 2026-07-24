# SCAR 全量新增 Baseline 效率与可扩展性实验设计

日期：2026-07-24

## 1. 目标

本设计用于回答审稿人关于 SCAR 训练成本、推理成本、内存占用和可扩展性的质疑。
由于 SCAR 与历史 baseline 的权重文件已经遗失，正式 rebuttal 不再从历史运行中推断
效率，而是在同一环境中重新训练 SCAR 和全部可运行的新增 baseline，并在运行过程中
自动记录精度、时间、显存、CPU 内存和磁盘产物。

成功标准如下：

1. 所有方法使用相同的五个数据文件、训练/验证/测试边界、固定模型 seed `42` 和 AUROC/AP
   evaluator。
2. 每个方法保留其官方模型结构、窗口长度、patch、损失函数和推荐训练超参数。
3. 每个“方法 × 数据集 × seed 42”运行生成完整 checkpoint、逐点分数、指标、资源记录、
   环境信息、日志和 SHA-256 清单。
4. GPU 方法进入统一主表；MATLAB/CPU 方法进入独立表，不进行跨硬件运行时间排名。
5. 可扩展性曲线能够展示时间长度、通道数量和 SCAR memory-bank 规模变化时的时间、
   内存和精度趋势。

## 2. 方法范围

### 2.1 GPU 主表

- SCAR
- CATCH
- PaAno
- PUAD
- PGRF-Net
- GDFlex（仅在官方实现可在统一 GPU 环境运行时纳入）

### 2.2 CPU/MATLAB 独立表

- DAMP，正式入口固定为官方 `DAMP_Multidim.m`
- GDFlex（若其正式复现只能使用 MATLAB/CPU）

DMemAD 不纳入，因为当前没有可核验的公开官方实现。任何不能在五个数据集之一运行的
组合必须记录为 `unsupported`，并保存明确原因，不能静默跳过或用第三方实现替代。

## 3. 公平实验协议

### 3.1 共享协议

- 数据集：MSL、PSM、SMAP、SMD、SWaT。
- 数据源：`dataset/anomaly_detect/` 中与 CATCH 发布版本 SHA-256 一致的文件。
- 训练/测试边界：`DETECT_META.csv` 中的 `train_lens`。
- 正式模型随机种子：固定为 `42`；CLI 覆盖仅用于调试，覆盖结果不得进入正式表格。
- 标准化：只能在训练侧数据上拟合，不得观察测试数据和测试标签。
- 验证集：使用 SCAR 固定验证索引；baseline 适配器读取同一份可复现 split manifest。
- 测试范围：完整测试序列，禁止丢弃最后一个不完整 batch 或未覆盖尾部。
- 主指标：逐点 AUROC 和 average precision。
- 标签限制：测试标签只能进入统一 evaluator，不得用于调参、阈值、窗口选择、分数方向
  选择或多个 score head 之间的择优。
- 聚合：论文报告固定 seed `42` 的性能结果，不报告模型 seed mean 或 standard deviation。

### 3.2 方法专属设置

以下项目沿用各官方实现或原论文推荐值，不强制与 SCAR 相同：

- 模型结构与参数规模；
- sequence length、patch size 和 patch stride；
- 训练 epoch、优化器、scheduler 和学习率；
- 官方异常分数定义；
- 方法必需的索引、memory、prototype 或矩阵轮廓设置。

如果官方提供多个候选设置，只能在统一训练侧验证集上选择一次，并将候选集合、选择指标
和最终配置写入运行记录。不得使用调试 seed 或逐测试指标选择配置。

## 4. 系统设计

实现位于主仓库，外部官方仓库保持只读。统一效率实验分成六个组件。

### 4.1 协议清单生成器

生成版本化 JSON manifest，记录：

- 数据文件绝对/相对路径和 SHA-256；
- `train_lens`、验证索引、测试长度、通道数；
- 方法、数据集、seed、设备和运行阶段；
- 官方仓库路径、commit 和配置摘要。

manifest 是所有 runner 的唯一共享协议输入。runner 不得自行重新划分数据。

### 4.2 Baseline 适配器

每个方法实现统一生命周期：

```text
prepare -> train -> build_index_or_memory -> infer -> export_scores
```

适配器负责数据格式转换、调用官方入口、恢复 checkpoint 和将官方输出转换为与完整测试
序列等长的逐点 anomaly score。转换规则必须预先固定，不允许读取测试标签决定分数方向。

SCAR 适配器将 `train` 拆为 Stage A，将 `build_index_or_memory` 映射为 Stage B，将
`infer` 映射为正式 test。DAMP 没有训练阶段，其预处理和搜索时间分别记入 build 与
infer。

### 4.3 统一资源监控器

资源监控器包装每个阶段并输出同一 schema：

- wall-clock 秒数，使用 monotonic 高精度时钟；
- GPU peak allocated/reserved memory；
- GPU 总显存峰值，优先使用 NVML；
- 当前进程及子进程 CPU RSS 峰值；
- CPU user/system time；
- 输入点数、窗口数、batch 数和吞吐率；
- 阶段退出码、超时、OOM 和异常摘要。

GPU 计时前后必须执行同步。模型仅使用 seed `42` 训练一次；推理先预热，再进行三次
计时，正式值保存三次原始记录、均值和标准差。计时波动表示系统测量噪声，不是模型
seed 方差。

### 4.4 批量运行器

批量运行器枚举方法、数据集和 seed，使用独立子进程启动每个运行单元，并支持：

- `--methods`、`--datasets`、`--seeds`；
- `--stage` 和 `--resume`；
- 单运行超时；
- 完整产物检查后才允许 resume 跳过；
- 状态原子写入；
- GPU 和 CPU/MATLAB 队列分离；
- dry-run 输出最终命令、配置和预计运行矩阵。

默认正式矩阵为 6 个 GPU 方法 × 5 个数据集 × 1 个固定 seed `42`；实际不支持的组合保留状态
记录。DAMP 使用独立 MATLAB 队列。

### 4.5 结果汇总器

汇总器只读取标准化产物，不解析控制台文本。输出：

- 逐运行长表；
- 逐方法/数据集固定 seed `42` 结果表；
- GPU 主效率表；
- CPU/MATLAB 独立效率表；
- 训练、建库、推理阶段分解表；
- 精度—时间、精度—显存关系表；
- 失败、OOM、超时和 unsupported 审计表。

正式 seed `42` 缺失时必须显示为 incomplete；历史或调试 seed 不能代替正式结果。

### 4.6 产物与防丢失机制

目录固定为：

```text
artifacts/baseline_efficiency/<method>/<dataset>/seed_<seed>/
```

每个运行单元至少包含：

- `best_checkpoint` 和 `last_checkpoint`，名称可由适配器映射；
- `config.json`；
- `protocol_manifest.json`；
- `environment.json`；
- `resource_metrics.json`；
- `metrics.json`；
- `scores.npy`；
- `stdout.log`、`stderr.log`；
- `run_status.json`；
- `artifact_manifest.sha256`。

checkpoint 采用“临时文件写完后原子重命名”的方式保存。完成状态只有在 checkpoint、
分数、指标、资源记录和 SHA-256 清单均通过检查后才能写为 `complete`。

## 5. 指标定义

### 5.1 精度

- AUROC；
- average precision；
- 固定 seed `42` 的 AUROC/AP。

### 5.2 时间

- `train_seconds`；
- `build_seconds`；
- `inference_seconds`；
- `end_to_end_seconds`；
- `milliseconds_per_window`；
- `points_per_second`。

SCAR 额外报告 Stage A、Stage B、fusion fitting 和 test；其他方法按统一生命周期映射。

### 5.3 内存与存储

- `gpu_peak_allocated_bytes`；
- `gpu_peak_reserved_bytes`；
- `gpu_peak_device_used_bytes`；
- `cpu_peak_rss_bytes`；
- checkpoint bytes；
- index/memory/prototype bytes；
- 全部正式产物 bytes。

### 5.4 模型规模

- trainable parameters；
- total parameters；
- checkpoint 大小。

FLOPs 仅在能够对所有 GPU 深度方法采用同一可靠计数工具时报告，否则不放入主表，
避免不同算子支持程度造成伪比较。

## 6. 可扩展性实验

可扩展性使用固定 seed `42`，与主效率矩阵保持一致。

### 6.1 时间长度扩展

对统一训练侧正常数据使用前缀比例：

```text
25%, 50%, 75%, 100%
```

测试集保持不变，记录训练、建库、推理时间，峰值内存、索引大小、AUROC 和 AP。

### 6.2 通道数量扩展

使用预先固定、与标签无关的通道顺序，测试：

```text
25%, 50%, 75%, 100%
```

训练和测试使用同一通道子集。无法改变通道数的官方实现必须记录为 `unsupported`。

### 6.3 SCAR memory-bank 扩展

SCAR 使用固定的 coreset keep ratio：

```text
25%, 50%, 75%, 100%
```

记录状态窗口数、各尺度 patch 数、索引/内存大小、建库时间、推理延迟、CPU/GPU 内存和
AUROC/AP。该实验用于直接验证两级候选限制、coreset 和 FAISS 对成本的控制效果。

### 6.4 可扩展性输出

- 数据长度—训练时间；
- 数据长度—峰值内存；
- 通道数—推理延迟；
- 通道数—峰值显存；
- memory size—推理延迟；
- memory size—CPU RAM/磁盘；
- AUROC/AP—推理延迟折中。

## 7. 错误处理

- OOM：记录发生阶段、原始 batch size 和错误，不自动降低 batch size后覆盖正式结果。
  如需降低 batch size，必须生成新的显式配置并在表中注明。
- 超时：保留日志、资源记录和部分产物，状态为 `timeout`。
- NaN/Inf score：运行失败，不进入汇总。
- 分数长度不等于测试长度：运行失败，不允许 evaluator 自动截断。
- 单类标签或缺失标签：运行失败并记录协议错误。
- 外部仓库版本变化：manifest 与固定 commit 不一致时拒绝正式运行。

## 8. 测试与验收

### 8.1 单元测试

- manifest 生成稳定且 split 不读取测试标签；
- 资源监控器能够捕获当前进程和子进程峰值；
- GPU 不可用时输出明确的 null/reason，而不是伪造零值；
- score 长度、方向和有限值检查；
- checkpoint 原子写入与 SHA-256 校验；
- incomplete/failed/unsupported 和非正式 seed 不进入正式汇总。

### 8.2 集成测试

- SCAR CPU 小样本完成 train/build/infer 全生命周期；
- 每个 Python baseline 至少完成一个小数据 dry-run；
- MATLAB 可用时 DAMP 完成官方样例 smoke test；
- resume 只跳过产物完整且 manifest 一致的运行；
- 汇总器能同时处理 complete、failed、OOM、timeout 和 unsupported。

### 8.3 正式验收

1. 五个数据集均完成 SCAR 固定 seed `42`。
2. 每个可运行 GPU baseline 的支持组合均完成固定 seed `42`。
3. DAMP 独立赛道完成支持组合并明确硬件与运行环境。
4. 所有正式结果均能从标准产物重新生成，不依赖手工抄录。
5. GPU 主表和 CPU/MATLAB 表不进行跨设备时间排名。
6. 可扩展性图的每个点均可追溯到唯一 manifest、配置、日志和资源记录。

## 9. 文档维护

实现时同步更新：

- `PROJECT_PLAN.md`：新增效率 benchmark 入口、参数、产物和运行命令；
- `CHANGELOG_MAINTENANCE.md`：记录每次适配、测试、正式运行和结果生成；
- `third_party/baselines/README.md`：仅在上游版本或正式入口变化时更新；
- rebuttal 执行文档：将效率实验范围固定为本设计中的全量可运行 baseline。
