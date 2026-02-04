---
name: rocprof-hip-profiling
description: Automated rocprofiler-systems profiling workflow for AMD GPU with HIP kernel analysis
version: 1.0.0
author: jixiong
tags: [hip, rocm, profiling, amd, performance, optimization, pytorch]
---

# ROCm HIP 自动化性能分析

本 Skill 提供完整的自动化 rocprofiler-systems 性能分析流程，支持 **HIP kernel 分析**、**PyTorch 模型分析**、**内存传输追踪** 和 **性能诊断**。

---

## 🚀 快速开始

### 方式 1: 分析 HIP 可执行文件

```bash
./examples/auto_rocprof.sh ./your_hip_program my_report
```

### 方式 2: 分析 PyTorch 模型

```bash
# 单个模型
./examples/auto_rocprof.sh --pytorch model.py model_profile

# 目录中所有模型
./examples/auto_rocprof.sh --pytorch-dir ./models all_models
```

### 方式 3: 直接使用 rocprof-sys

```bash
rocprof-sys-run --trace --profile --use-rocm \
    --rocm-domains kernel_dispatch \
    -o ./results \
    -- python scripts/kernel_runner.py --file model.py
```

---

## 📋 AI 分析流程

当用户提供 ROCm/HIP 性能数据时，AI 按以下流程处理：

### Phase 1: 数据获取

**情况 A: 分析已有 trace**
```bash
# 直接分析 Perfetto trace
cat <dir>/perfetto-trace-*.proto
# 上传到 https://ui.perfetto.dev/ 查看
```

**情况 B: 采集新数据**
```bash
# HIP 程序
./examples/auto_rocprof.sh ./program my_report

# PyTorch 模型
./examples/auto_rocprof.sh --pytorch model.py my_report
```

### Phase 2: 数据持久化

分析数据自动保存到：

```
rocprof_reports/
├── <prefix>/
│   ├── <timestamp>/
│   │   ├── perfetto-trace-*.proto
│   │   ├── wall_clock-*.txt
│   │   ├── sampling_*.txt
│   │   ├── metadata-*.json
│   │   └── README.md
│   └── <prefix>_rocprof.log
```

### Phase 3: 自动诊断

```python
def auto_diagnose_hip(metrics):
    duration_ms = metrics.get('duration_ms', 0)
    kernel_count = metrics.get('kernel_count', 0)
    avg_kernel_time = metrics.get('avg_kernel_time_us', 0)
    memory_transfer_time = metrics.get('memory_transfer_ms', 0)
    total_time = metrics.get('total_time_ms', 0)
    
    gpu_busy_ratio = (total_time - memory_transfer_time) / total_time if total_time > 0 else 0
    
    if memory_transfer_time > total_time * 0.5:
        return "MEMORY_TRANSFER_BOUND"
    elif avg_kernel_time < 10:  # < 10 微秒
        return "KERNEL_LAUNCH_OVERHEAD_BOUND"
    elif kernel_count > 1000 and avg_kernel_time < 100:
        return "TOO_MANY_SMALL_KERNELS"
    elif gpu_busy_ratio < 0.5:
        return "LOW_GPU_UTILIZATION"
    else:
        return "COMPUTE_BOUND"
```

---

## 🔧 rocprof-sys 参数参考

### 核心参数

```bash
rocprof-sys-run [选项] -- <program> [args]

--trace                    # 启用 tracing (生成 Perfetto trace)
--profile                  # 启用 profiling (生成统计报告)
--use-rocm                 # 启用 ROCm/HIP 追踪
--rocm-domains <domain>    # 指定追踪域 (可多次使用)
-o, --output <dir>         # 输出目录
```

### 可用的 --rocm-domains

| Domain | 说明 |
|--------|------|
| `kernel_dispatch` | GPU kernel 执行 ✅ **推荐** |
| `hip_api` | HIP API 调用 |
| `hip_runtime_api` | HIP 运行时 API |
| `memory_copy` | 内存拷贝 (Host-Device) |
| `memory_allocation` | 内存分配 |
| `hsa_api` | HSA API |
| `hsa_core_api` | HSA 核心 API |
| `marker_api` | 用户标记 (roctx) |

### 示例命令

```bash
# 基础 kernel 分析
rocprof-sys-run --trace --use-rocm --rocm-domains kernel_dispatch \
    -o ./results -- ./program

# 完整分析 (kernel + memory)
rocprof-sys-run --trace --profile --use-rocm \
    --rocm-domains kernel_dispatch \
    --rocm-domains memory_copy \
    --rocm-domains hip_api \
    -o ./results -- ./program

# PyTorch 模型
rocprof-sys-run --trace --profile --use-rocm \
    --rocm-domains kernel_dispatch \
    -o ./results -- python model.py
```

---

## 📊 输出模板

```markdown
# ROCm 性能分析报告

## 📁 报告信息
- **程序**: {program_name}
- **采集时间**: {timestamp}
- **GPU 设备**: {gpu_device}
- **ROCm 版本**: {rocm_version}

## 📈 执行摘要

| 项目 | 数值 |
|------|------|
| **主要瓶颈** | {bottleneck_type} |
| **Kernel 总数** | {kernel_count} |
| **总 GPU 时间** | {total_gpu_time} ms |
| **内存传输时间** | {memory_time} ms |
| **GPU 利用率** | {gpu_utilization}% |

## 📊 Top 10 耗时 Kernel

| 排名 | Kernel 名称 | 执行时间 | 调用次数 | 占比 |
|------|-------------|----------|----------|------|
| 1 | {kernel_1} | {time_1} ms | {count_1} | {pct_1}% |
| 2 | {kernel_2} | {time_2} ms | {count_2} | {pct_2}% |

## 💡 优化建议

{optimization_suggestions}
```

---

## 📖 诊断规则详解

### MEMORY_TRANSFER_BOUND

```
IF memory_transfer_time > 50% of total_time:
    诊断: MEMORY_TRANSFER_BOUND
    
    优化策略:
    1. 使用 hipMemcpyAsync 异步传输
    2. 使用 Pinned Memory (hipHostMalloc)
    3. 减少 Host-Device 数据传输次数
    4. 使用 hipMemPrefetchAsync 预取
```

### KERNEL_LAUNCH_OVERHEAD_BOUND

```
IF avg_kernel_time < 10us AND kernel_count > 100:
    诊断: KERNEL_LAUNCH_OVERHEAD_BOUND
    
    优化策略:
    1. 合并小 kernel 为大 kernel
    2. 使用 HIP Graph 减少启动开销
    3. 增加每个 kernel 的工作量
```

### TOO_MANY_SMALL_KERNELS

```
IF kernel_count > 1000 AND avg_kernel_time < 100us:
    诊断: TOO_MANY_SMALL_KERNELS
    
    优化策略:
    1. Kernel Fusion
    2. 使用 HIP Graph
    3. 批处理多个操作
```

### LOW_GPU_UTILIZATION

```
IF gpu_busy_ratio < 50%:
    诊断: LOW_GPU_UTILIZATION
    
    优化策略:
    1. 增加并行度
    2. 使用多 Stream 并发
    3. 检查 CPU-GPU 同步点
```

### COMPUTE_BOUND

```
IF gpu_busy_ratio > 80% AND no obvious bottleneck:
    诊断: COMPUTE_BOUND
    
    优化策略:
    1. 优化 kernel 内部算法
    2. 使用向量化指令
    3. 优化内存访问模式
    4. 调整 workgroup size
```

---

## 🎯 优化策略速查

| 瓶颈类型 | 立即行动 | 代码示例 | 预期收益 |
|---------|---------|---------|---------|
| **MEMORY_TRANSFER_BOUND** | 异步传输 | `hipMemcpyAsync()` | 1.5-3x |
| **KERNEL_LAUNCH_OVERHEAD** | HIP Graph | `hipGraphLaunch()` | 2-5x |
| **TOO_MANY_SMALL_KERNELS** | Kernel Fusion | 合并多个 kernel | 1.5-3x |
| **LOW_GPU_UTILIZATION** | 多 Stream | `hipStreamCreate()` | 1.3-2x |
| **COMPUTE_BOUND** | 向量化 | 使用 `float4` | 1.2-1.5x |

---

## ⚠️ 常见问题

### 1. 找不到 rocprof-sys 命令

```bash
export PATH=$PATH:/opt/rocm/bin
```

### 2. PyTorch 初始化错误

如果遇到 `libcaffe2_nvrtc.so` 错误，这是 PyTorch ROCm 与 rocprof-sys 的兼容性问题。
确保使用正确版本的 PyTorch ROCm。

### 3. 权限问题

```bash
sudo usermod -a -G video $USER
# 重新登录后生效
```

### 4. Trace 文件过大

```bash
# 限制采集时间
timeout 10 rocprof-sys-run -- ./program

# 只采集 kernel
rocprof-sys-run --trace --use-rocm --rocm-domains kernel_dispatch -- ./program
```

---

## 🔗 相关资源

- [AMD ROCm 文档](https://rocm.docs.amd.com/)
- [rocprofiler-systems GitHub](https://github.com/ROCm/rocprofiler-systems)
- [Perfetto UI](https://ui.perfetto.dev/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)

---

*本 Skill 支持完整的自动化 ROCm/HIP 性能分析工作流*
