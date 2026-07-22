# CoReM-AD v2.0

按 [框架方案.md](/E:/physguard-mamba-new/框架方案.md)、[代码细节.md](/E:/physguard-mamba-new/代码细节.md) 和 [修改方案.md](/E:/physguard-mamba-new/修改方案.md) 重建的异常检测项目。

当前版本已经落实的关键修复：

- Stage A checkpoint 现在保存并恢复 `normalizer + optimizer + scheduler + epoch + history`
- `normalizer.fit()` 只在 Stage A 首次数据准备时发生，Stage B / Test 全部改为复用 checkpoint 并只做 `transform`
- Stage B 结束后会直接拟合并保存 CDF-PIT 融合器，测试阶段固定输出 `cdf_max`、`cdf_mean` 和各子分数结果
- Level-2 context retrieval 现在返回 `valid mask`，z-space kNN 会屏蔽 padded neighbors
- Stage A 增加 `CosineAnnealingLR`、梯度裁剪和更完整的 checkpoint
- 多尺度 completion/prediction loss 改为跨尺度平均
- 数据加载和 memory 建库加入了更严格的断言与 sanity check
- Stage A 验证集改为“原始训练序列尾部分割 + `seq_len` gap + 仅训练段 fit normalizer”

## 项目结构

```text
physguard-mamba-new/
├─ coremad/
│  ├─ config.py
│  ├─ data.py
│  ├─ faiss_index.py
│  ├─ memory.py
│  ├─ model.py
│  ├─ scorer.py
│  └─ trainer.py
├─ data/
├─ artifacts/
├─ run.py
├─ 框架方案.md
├─ 代码细节.md
├─ 自检清单.md
├─ 修改方案.md
└─ 拍板回复.md
```

## 运行方式

Stage A:

```bash
python run.py --stage stage_a --dataset MSL --data_root ./data --experiment_name coremad_msl
```

Stage B:

```bash
python run.py --stage stage_b --dataset MSL --data_root ./data --experiment_name coremad_msl
```

Test:

```bash
python run.py --stage test --dataset MSL --data_root ./data --experiment_name coremad_msl
```

完整流程:

```bash
python run.py --stage full --dataset MSL --data_root ./data --experiment_name coremad_msl
```

也可以直接用现成脚本跑 MSL：

```bash
bash scripts/train_msl.sh full msl_baseline
```

单独跑某一阶段：

```bash
bash scripts/train_msl.sh stage_a msl_stage_a
```

脚本特性：

- 首次运行会自动把日期时间拼到实验名后面，例如 `msl_baseline_20260404_153000`
- 训练 stdout/stderr 会自动落盘到 `artifacts/<exp_name>/logs/`
- 支持断点续训；续训时把已有实验名作为第二个参数，并设置 `RESUME=1`

```bash
RESUME=1 bash scripts/train_msl.sh full msl_baseline_20260404_153000
```

## 常用参数

- `--use_faiss`
  是否优先使用 Faiss 做 Level-1 state retrieval；无 Faiss 时自动回退到 `torch`
- `--faiss_exact_threshold`
  state bank 小于该阈值时使用精确检索
- `--faiss_ivf_nprobe`
  IVF 检索时的 `nprobe`
- `--top_M`
  Level-1 粗筛窗口数
- `--top_K`
  Level-2 精排邻居数
- `--knn_k`
  z-space memory score 的 kNN 邻居数
- `--coreset_keep_ratio`
  coreset 保留比例
- `--coreset_max_patches_per_scale`
  每个尺度保留的 patch 上限
- `--coreset_fps_threshold`
  超过该规模后从近似 farthest-point 切到分层随机采样
- `--grad_clip_norm`
  Stage A 梯度裁剪阈值
- `--scheduler_eta_min_ratio`
  Stage A cosine scheduler 的最小学习率比例
- `--val_ratio`
  Stage A 尾部验证集比例，默认 `0.15`
- `--val_gap`
  是否在 Stage A 训练段和验证段之间留一个 `seq_len` 的 gap，默认开启
- `--val_min_train_windows`
  允许切验证集时，训练段至少要保留的窗口数，默认 `10`
- `--max_train_windows`
  调试时限制训练/建库窗口数
- `--max_test_windows`
  调试时限制测试窗口数

## 数据集支持

当前稳定支持：

- `MSL`
- `PSM`
- 聚合版 `SMD`
- `SWAT`（按 `swat_train2.csv / swat2.csv` 读取）

说明：

- `SMD` 的 28 台机器独立训练/评估版本还没拆开，当前仍是聚合版
- 这不影响现阶段框架联调，但如果你后面要严格对论文表格，建议下一步把 SMD 独立脚本补上

## 依赖建议

远程服务器推荐：

```bash
pip install torch scikit-learn faiss-cpu
```

如果服务器上的 CUDA 与 Faiss GPU 版本兼容，也可以安装：

```bash
pip install faiss-gpu
```

项目里保留了 `faiss` 不可用时的 `torch` fallback，所以本地或轻量环境也能跑通。
