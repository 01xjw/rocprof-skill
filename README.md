# 🚀 ROCprof Skill

> AMD ROCm GPU 性能分析工具集，专为 HIP 应用和 PyTorch 模型设计

[![ROCm](https://img.shields.io/badge/ROCm-7.x-red.svg)](https://rocm.docs.amd.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 功能特性

- ✅ **HIP 可执行文件分析** - 支持原生 HIP 应用的 kernel 级别分析
- ✅ **PyTorch 模型分析** - 自动化 PyTorch 算子性能采集
- ✅ **批量分析** - 支持目录级批量模型分析
- ✅ **Perfetto 可视化** - 生成标准 Perfetto trace 文件
- ✅ **自动报告生成** - 自动生成分析摘要和使用说明

## 🔧 环境要求

- AMD GPU (MI100/MI200/MI300 系列或 Radeon RX 7000 系列)
- ROCm 7.x
- Python 3.8+
- PyTorch with ROCm (可选，用于 PyTorch 分析)

## 📦 安装

```bash
git clone https://github.com/your-username/rocprof-skill.git
cd rocprof-skill

# 添加执行权限
chmod +x scripts/*.sh examples/*.sh

# 可选: 安装 Python 依赖
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 分析 HIP 可执行文件

```bash
./examples/auto_rocprof.sh ./your_hip_program my_report
```

### 2. 分析 PyTorch 模型

```bash
# 单个模型
./examples/auto_rocprof.sh --pytorch model.py my_model

# 目录中所有模型
./examples/auto_rocprof.sh --pytorch-dir ./models all_models
```

### 3. 使用 Python 脚本

```bash
# 单个模型
python scripts/kernel_runner.py --file 22_Tanh.py

# 结合 rocprof-sys
rocprof-sys-run --trace --use-rocm --rocm-domains kernel_dispatch \
    -o ./results -- python scripts/kernel_runner.py --file 22_Tanh.py
```

## 📁 项目结构

```
rocprof-skill/
├── README.md                    # 项目说明
├── LICENSE                      # MIT 许可证
├── requirements.txt             # Python 依赖
├── ROCPROF_SKILL.md            # 详细技术文档
│
├── scripts/                     # 核心脚本
│   ├── kernel_runner.py        # PyTorch 模型运行器
│   └── rocprof_pytorch.sh      # PyTorch 专用分析脚本
│
├── examples/                    # 示例脚本
│   ├── auto_rocprof.sh         # 自动化分析脚本 (通用)
│   └── sample_models/          # 示例模型
│
└── docs/                        # 文档
    ├── USAGE.md                # 使用指南
    └── TROUBLESHOOTING.md      # 常见问题
```

## 📊 输出文件说明

| 文件 | 说明 |
|------|------|
| `perfetto-trace-*.proto` | Perfetto 格式 trace 文件，可在 [ui.perfetto.dev](https://ui.perfetto.dev) 查看 |
| `wall_clock-*.txt` | 函数级时间统计 |
| `sampling_percent-*.txt` | 采样百分比分布 |
| `sampling_wall_clock-*.txt` | 采样墙钟时间 |
| `metadata-*.json` | 运行元数据 (CPU、GPU、ROCm 版本等) |
| `functions-*.json` | 函数地址映射 |

## 🔍 可视化分析

### 使用 Perfetto UI

1. 打开 [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. 点击 "Open trace file"
3. 上传生成的 `perfetto-trace-*.proto` 文件
4. 浏览时间线，分析 kernel 执行时间

### 命令行查看

```bash
# 查看时间统计
cat results/wall_clock-*.txt

# 查看采样分布
cat results/sampling_percent-*.txt | head -50
```

## ⚙️ rocprof-sys 常用参数

```bash
rocprof-sys-run [选项] -- <program>

# 常用选项
--trace                    # 启用 tracing
--profile                  # 启用 profiling
--use-rocm                 # 启用 ROCm 追踪
--rocm-domains <domains>   # 指定追踪域

# 可用 domains
kernel_dispatch            # GPU kernel 执行 ✅ (推荐)
hip_api                    # HIP API 调用
hip_runtime_api           # HIP 运行时 API
memory_copy               # 内存拷贝
memory_allocation         # 内存分配
hsa_api                   # HSA API
```

## 🐛 常见问题

### 1. 找不到 rocprof-sys

```bash
export PATH=$PATH:/opt/rocm/bin
```

### 2. PyTorch 初始化错误

如果遇到 `libcaffe2_nvrtc.so` 错误，这是 PyTorch ROCm 版本与 rocprof-sys 的兼容性问题。
解决方法：确保使用正确的 PyTorch ROCm 版本。

### 3. 权限问题

```bash
sudo usermod -a -G video $USER
# 重新登录后生效
```

## 📚 相关资源

- [AMD ROCm 文档](https://rocm.docs.amd.com/)
- [rocprofiler-systems GitHub](https://github.com/ROCm/rocprofiler-systems)
- [Perfetto UI](https://ui.perfetto.dev/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

Made with ❤️ for AMD GPU developers
