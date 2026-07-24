# 统一资源监控

## SCAR

SCAR 默认监控 Stage A、Stage B 和 Test，并将结果原子写入实验目录下的
`resource_metrics.json`：

```bash
python run.py --stage full --dataset MSL --experiment_name msl_seed42
```

如需测量监控本身的额外开销，可显式关闭：

```bash
python run.py --stage full --dataset MSL --experiment_name msl_no_monitor \
  --resource_monitor 0
```

采样周期默认为 0.1 秒，可通过 `--resource_sample_interval` 调整。正式效率实验前安装：

```bash
python -m pip install -r scripts/efficiency/requirements.txt
```

缺少 `psutil` 时仍会记录时间，但 CPU RSS 和子进程树数据为 `null` 并附带原因；缺少
NVML 或 CUDA 时 GPU 字段同样使用 `null/reason`，不会伪造为零。

`inference` span 只覆盖 batch 级模型前向、检索和诊断分数传输；`scoring` 额外包含
滑窗聚合和融合。吞吐同时区分实际处理的 `window_points`、至少被一个窗口覆盖的唯一
`covered_points` 和原始 `input_points`，避免窗口上限或分段协议下高估吞吐。

CPU 时间按 PID 累加采样增量，已经退出的 DataLoader/子进程不会从总时间中消失。
GPU 记录逻辑索引、物理索引/UUID 和 `mapping_source`；设置
`CUDA_VISIBLE_DEVICES` 时按可见设备条目解析，进程内 SCAR 可用时优先使用 PyTorch
设备 UUID 映射 NVML。

## 外部方法

包装器不修改第三方仓库。默认日志为输出目录中的 `stdout.log` 和 `stderr.log`：

```bash
python scripts/efficiency/monitor_command.py \
  --output artifacts/baseline_efficiency/CATCH/MSL/seed_42/resource_metrics.json \
  --method CATCH --dataset MSL --seed 42 \
  --stage inference --device cuda:0 \
  --points 73729 --windows 384 --timeout 3600 \
  -- python third_party/baselines/CATCH/run.py
```

包装器透传正常或失败退出码；超时返回 124。外部进程的 GPU 内存由 NVML 按进程树
采样，PyTorch allocator 字段明确保持为空，因为包装器不能读取另一个进程的
allocator 状态。包装器默认启用严格依赖预检：CPU 运行必须有 `psutil`，CUDA 运行还
必须能够初始化目标 GPU 的 NVML。仅在调试缺失依赖时使用
`--strict-dependencies 0`，这种部分监控产物不得进入正式效率表。

计时从启动子进程前开始，启动失败也会写入 failed attempt。超时使用限时
terminate/kill 流程并返回 124。日志采用追加模式，每个 invocation 写入边界，因此
重复阶段不会覆盖已有日志。

## 主要字段

- `attempts`：追加保存每次阶段尝试，使用 `attempt_id` 和 `invocation_id` 区分；
  read-modify-write 由跨进程文件锁保护。
- `timing`：阶段端到端时间及 CPU user/system 时间。
- `spans`：`train_loop`、`memory_build`、`fusion_fit`、`inference`、`scoring`、
  `evaluation` 和 `result_export` 等核心区间。
- `cpu` / `gpu`：采样可用性、峰值、采样次数和不可用原因。
- `workload` / `throughput`：point、window、batch、epoch 数及相应吞吐率。
- `model` / `artifacts`：参数量及 checkpoint、memory、index、metrics 等文件大小。
- `status`：`running`、`completed`、`skipped`、`failed`、`oom`、`timeout` 或
  `interrupted`。
