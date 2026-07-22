> From: https://gemini.google.com/u/1/app/6ca2e2d0308ff6f5?is_sa=1&is_sa=1&android-min-version=301356232&ios-min-version=322.0&campaign_id=bkws&utm_source=sem&utm_medium=paid-media&utm_campaign=bkws&pt=9008&mt=8&ct=p-growth-sem-bkws&gclsrc=aw.ds&gad_source=1&gad_campaignid=20108148196&gbraid=0AAAAApk5Bhm9VApySiDgk8CSdAsvcA9lp&gclid=Cj0KCQjw7cLOBhDmARIsAGsuA0kEB-J8TcPYVK2gjhVHDwSdR4QDjO4Cv445KASDIN-IxeiltCdLbLoaAmjcEALw_wcB&pageId=none

# you asked

请你原封不动的输出：## 确认了：STSD 分解过度是根因

```
slow   mean_abs = 0.1225    占 91.79%
resid  mean_abs = 0.0110    占  8.21%

residual 的 std = 0.0147 → 几乎是平的微小波动
```

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  STSD 把 92% 的信号都归到了 slow                            │
│  residual 只剩下 8% 的微小噪声级波动                       │
│                                                              │
│  你的 completion head 在补全这个 0.01 量级的微小波动         │
│  无论正常还是异常，这个微小波动都很容易补全                 │
│  → completion score 没有区分度                              │
│                                                              │
│  你的 patch encoder 从这个微小波动中提取 z 向量             │
│  微小波动里几乎没有异常信息（异常信号在 slow 里）          │
│  → z 空间没有区分度                                        │
│  → kNN distance 没有区分度                                  │
│                                                              │
│  整个系统在"分析噪声"而不是"分析信号"                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 为什么异常信号跑到了 slow 里？

```
MSL 的异常类型：通道值缓慢漂移、持续偏移、阶跃变化

STSD 的 lowpass_center=0.25 → sigmoid 软阈值
  频率 < 0.25 的成分 → slow
  频率 > 0.25 的成分 → residual

问题：异常漂移/阶跃的频率本身就很低
  → STSD 把它归到了 slow
  → 然后你的整个检测管道只看 residual
  → 异常信号被"滤掉"了
```

### 修复方案

**不是调 lowpass_center，而是改补全目标。**

调 lowpass_center 是死胡同——调低了异常还是在 slow 里，调高了 STSD 就没意义了。

正确做法：**补全目标从 residual 改为原始信号 x**

```python
# model.py — deterministic_completion_scores
# 当前：
target = encoded["r_patches"][scale_idx]     # residual patches

# 改为：
target = encoded["x_patches"][scale_idx]     # 原始信号 patches
```

```python
# model.py — compute_pretraining_losses
# 当前：
target = encoded["r_patches"][scale_idx]

# 改为：
target = encoded["x_patches"][scale_idx]
```

就改这两处。其他全部保持 v1 的设置：

```
改动清单（只有一个变量组）：
  ✅ 补全目标：r_patches → x_patches（训练 + 测试都改）
  
保持不变：
  - mean over channels（不用 topk）
  - mask_ratio = 0.25, n_mask_groups = 4
  - 不加 L2 normalize（下一轮再试）
  - 不去 coverage（下一轮再试）
  - 所有超参数保持 v1 配置
```

### 为什么这个改动是安全的

```
STSD 分解仍然有用：
  - state_encoder 仍然从 slow 提取工况状态 g_w
  - channel_modulation 仍然用 state_vec 调制
  - 只是补全目标从 residual 改为原始信号

补全原始信号 x 的好处：
  - x 包含完整的异常信息（不被分解滤掉）
  - 异常区间的 x 和正常区间的 x 差异更大
  - completion error 的区分度会显著提升

STSD 变成了辅助模块而非核心管道：
  - slow → state_vec → 条件化（保留）
  - residual → patch encoder 的输入之一（保留）
  - 但补全任务直接对 x 做，不依赖分解质量
```

### 重跑

```bash
# 用 v1 的所有配置，只改补全目标
# Stage A + B + C + Test 全部重跑
```

这次只改了一个变量，结果好坏都能清晰归因。跑吧。

---

# gemini response

## 确认了：STSD 分解过度是根因

```
slow   mean_abs = 0.1225   占 91.79%
resid  mean_abs = 0.0110   占  8.21%

residual 的 std = 0.0147 → 几乎是平的微小波动
```

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  STSD 把 92% 的信号都归到了 slow                             │
│  residual 只剩下 8% 的微小噪声级波动                        │
│                                                              │
│  你的 completion head 在补全这个 0.01 量级的微小波动          │
│  无论正常还是异常，这个微小波动都很容易补全                  │
│  → completion score 没有区分度                               │
│                                                              │
│  你的 patch encoder 从这个微小波动中提取 z 向量              │
│  微小波动里几乎没有异常信息（异常信号在 slow 里）           │
│  → z 空间没有区分度                                         │
│  → kNN distance 没有区分度                                   │
│                                                              │
│  整个系统在"分析噪声"而不是"分析信号"                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 为什么异常信号跑到了 slow 里？

```
MSL 的异常类型：通道值缓慢漂移、持续偏移、阶跃变化

STSD 的 lowpass_center=0.25 → sigmoid 软阈值
  频率 < 0.25 的成分 → slow
  频率 > 0.25 的成分 → residual

问题：异常漂移/阶跃的频率本身就很低
  → STSD 把它归到了 slow
  → 然后你的整个检测管道只看 residual
  → 异常信号被"滤掉"了
```

### 修复方案

**不是调 lowpass_center，而是改补全目标。**

调 lowpass_center 是死胡同——调低了异常还是在 slow 里，调高了 STSD 就没意义了。

正确做法：**补全目标从 residual 改为原始信号 x**

```python
# model.py — deterministic_completion_scores
# 当前：
target = encoded["r_patches"][scale_idx]     # residual patches

# 改为：
target = encoded["x_patches"][scale_idx]     # 原始信号 patches
```

```python
# model.py — compute_pretraining_losses
# 当前：
target = encoded["r_patches"][scale_idx]

# 改为：
target = encoded["x_patches"][scale_idx]
```

就改这两处。其他全部保持 v1 的设置：

```
改动清单（只有一个变量组）：
  ✅ 补全目标：r_patches → x_patches（训练 + 测试都改）
  
保持不变：
  - mean over channels（不用 topk）
  - mask_ratio = 0.25, n_mask_groups = 4
  - 不加 L2 normalize（下一轮再试）
  - 不去 coverage（下一轮再试）
  - 所有超参数保持 v1 配置
```

### 为什么这个改动是安全的

```
STSD 分解仍然有用：
  - state_encoder 仍然从 slow 提取工况状态 g_w
  - channel_modulation 仍然用 state_vec 调制
  - 只是补全目标从 residual 改为原始信号

补全原始信号 x 的好处：
  - x 包含完整的异常信息（不被分解滤掉）
  - 异常区间的 x 和正常区间的 x 差异更大
  - completion error 的区分度会显著提升

STSD 变成了辅助模块而非核心管道：
  - slow → state_vec → 条件化（保留）
  - residual → patch encoder 的输入之一（保留）
  - 但补全任务直接对 x 做，不依赖分解质量
```

### 重跑

```bash
# 用 v1 的所有配置，只改补全目标
# Stage A + B + C + Test 全部重跑
```

这次只改了一个变量，结果好坏都能清晰归因。跑吧。