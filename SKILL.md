---
name: rocprof-hip-profiling
description: Automated rocprofiler-systems profiling workflow for AMD GPU with HIP kernel analysis
version: 1.0.0
author: jixiong
tags: [hip, rocm, profiling, amd, performance, optimization, pytorch, triton]
---

# ROCm HIP Automated Performance Analysis Skill

This Skill enables AI Agents with comprehensive **AMD GPU performance analysis** capabilities, supporting HIP kernel analysis, PyTorch model analysis, Triton kernel analysis, memory transfer tracing, and automated performance bottleneck diagnosis.

---

## Trigger Conditions

This Skill is automatically activated when the user makes requests such as:

- Analyze HIP/ROCm program performance
- Profile PyTorch model operator execution on AMD GPUs
- Analyze Triton kernel performance on ROCm
- View or interpret rocprof-sys / Perfetto trace data
- Optimize GPU kernel performance
- Diagnose low GPU utilization, memory transfer bottlenecks, etc.

---

## Toolset

### 1. PyTorch Operator Runner

```bash
# Run a single PyTorch operator file
python scripts/kernel_runner.py --file <operator.py>

# Run an entire operator directory
python scripts/kernel_runner.py --dir <operators_dir> --summary

# Operator file interface (three required components):
#   class Model(nn.Module):      forward(self, x) -> operator logic
#   def get_inputs():            -> returns list of input tensors
#   def get_init_inputs():       -> returns list of Model constructor arguments
```

### 2. Triton Operator Runner

```bash
# Run PyTorch operator files with auto-matched built-in Triton kernels
python scripts/triton_runner.py --pytorch-file <operator.py>

# Run entire directory (auto Triton / PyTorch fallback)
python scripts/triton_runner.py --pytorch-dir <operators_dir> --summary

# Run native Triton kernel files
python scripts/triton_runner.py --file <triton_kernel.py>

# List supported built-in Triton operators
python scripts/triton_runner.py --list-ops
```

### 3. rocprof-sys Automated Analysis Script

```bash
# HIP executable
./examples/auto_rocprof.sh ./your_hip_program my_report

# Single PyTorch model
./examples/auto_rocprof.sh --pytorch model.py model_profile

# Batch PyTorch directory
./examples/auto_rocprof.sh --pytorch-dir ./models all_models
```

### 4. Direct rocprof-sys Commands

```bash
# Basic kernel analysis
rocprof-sys-run --trace --profile --use-rocm \
    --rocm-domains kernel_dispatch \
    -o ./results -- <program>

# Full analysis (kernel + memory + HIP API)
rocprof-sys-run --trace --profile --use-rocm \
    --rocm-domains kernel_dispatch \
    --rocm-domains memory_copy \
    --rocm-domains hip_api \
    -o ./results -- <program>

# PyTorch model
rocprof-sys-run --trace --profile --use-rocm \
    --rocm-domains kernel_dispatch \
    -o ./results -- python scripts/kernel_runner.py --file model.py

# Triton kernel
rocprof-sys-run --trace --profile --use-rocm \
    --rocm-domains kernel_dispatch \
    -o ./results -- python scripts/triton_runner.py --pytorch-file model.py
```

---

## AI Analysis Flow

When the user provides ROCm/HIP performance data, follow this process:

### Phase 1: Data Acquisition

**Case A: Analyze existing trace**
```bash
# Perfetto trace -> upload to https://ui.perfetto.dev/ for viewing
ls <dir>/perfetto-trace-*.proto

# View time statistics
cat <dir>/wall_clock-*.txt

# View sampling data
cat <dir>/sampling_percent-*.txt
```

**Case B: Collect new data**
```bash
# Choose the appropriate collection method based on the user's program type
./examples/auto_rocprof.sh [--pytorch | --pytorch-dir] <target> <prefix>
```

### Phase 2: Data Persistence

Analysis data is automatically saved to:
```
rocprof_reports/
├── <prefix>/
│   ├── <timestamp>/
│   │   ├── perfetto-trace-*.proto    # Visual trace
│   │   ├── wall_clock-*.txt          # Time statistics
│   │   ├── sampling_*.txt            # Sampling data
│   │   ├── metadata-*.json           # Environment metadata
│   │   └── functions-*.json          # Function mapping
│   └── <prefix>_rocprof.log          # Run log
```

### Phase 3: Auto-Diagnosis

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
    elif avg_kernel_time < 10:  # < 10 microseconds
        return "KERNEL_LAUNCH_OVERHEAD_BOUND"
    elif kernel_count > 1000 and avg_kernel_time < 100:
        return "TOO_MANY_SMALL_KERNELS"
    elif gpu_busy_ratio < 0.5:
        return "LOW_GPU_UTILIZATION"
    else:
        return "COMPUTE_BOUND"
```

### Phase 4: Output Analysis Report

```markdown
# ROCm Performance Analysis Report

## Report Information
- **Program**: {program_name}
- **Collection Time**: {timestamp}
- **GPU Device**: {gpu_device}
- **ROCm Version**: {rocm_version}

## Executive Summary
| Item | Value |
|------|-------|
| **Primary Bottleneck** | {bottleneck_type} |
| **Total Kernels** | {kernel_count} |
| **Total GPU Time** | {total_gpu_time} ms |
| **Memory Transfer Time** | {memory_time} ms |
| **GPU Utilization** | {gpu_utilization}% |

## Top 10 Time-Consuming Kernels
| Rank | Kernel Name | Execution Time | Call Count | Percentage |
|------|-------------|----------------|------------|------------|
| 1 | {kernel_1} | {time_1} ms | {count_1} | {pct_1}% |

## Optimization Recommendations
{optimization_suggestions}
```

---

## Diagnostic Rules

| Bottleneck Type | Condition | Optimization Strategy | Expected Improvement |
|----------------|-----------|----------------------|---------------------|
| **MEMORY_TRANSFER_BOUND** | Memory transfer > 50% total time | hipMemcpyAsync / Pinned Memory / Reduce transfers | 1.5-3x |
| **KERNEL_LAUNCH_OVERHEAD_BOUND** | avg kernel < 10us, count > 100 | HIP Graph / Merge kernels | 2-5x |
| **TOO_MANY_SMALL_KERNELS** | count > 1000, avg < 100us | Kernel Fusion / HIP Graph / Batching | 1.5-3x |
| **LOW_GPU_UTILIZATION** | GPU busy ratio < 50% | Multi-stream concurrency / Reduce sync points | 1.3-2x |
| **COMPUTE_BOUND** | GPU busy ratio > 80% | Vectorize / Optimize algorithms / Tune workgroup | 1.2-1.5x |

---

## rocprof-sys Parameter Reference

### Core Parameters

```bash
rocprof-sys-run [options] -- <program> [args]

--trace                    # Enable tracing (generates Perfetto trace)
--profile                  # Enable profiling (generates statistical reports)
--use-rocm                 # Enable ROCm/HIP tracing
--rocm-domains <domain>    # Specify tracing domain (can be used multiple times)
-o, --output <dir>         # Output directory
```

### Available --rocm-domains

| Domain | Description |
|--------|-------------|
| `kernel_dispatch` | GPU kernel execution ✅ **Recommended** |
| `hip_api` | HIP API calls |
| `hip_runtime_api` | HIP runtime API |
| `memory_copy` | Memory copy (Host-Device) |
| `memory_allocation` | Memory allocation |
| `hsa_api` | HSA API |
| `hsa_core_api` | HSA core API |
| `marker_api` | User markers (roctx) |

---

## FAQ

1. **Cannot find rocprof-sys**: `export PATH=$PATH:/opt/rocm/bin`
2. **Permission issues**: Run `sudo usermod -a -G video $USER` and re-login
3. **Trace file too large**: Use `timeout 10` to limit or only collect `kernel_dispatch`
4. **PyTorch initialization error**: Ensure PyTorch ROCm version matches the system ROCm version

---

## Related Resources

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [rocprofiler-systems GitHub](https://github.com/ROCm/rocprofiler-systems)
- [Perfetto UI](https://ui.perfetto.dev/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
