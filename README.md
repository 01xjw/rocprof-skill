# 🚀 ROCprof Skill

> AMD ROCm GPU performance analysis toolkit, designed for HIP applications and PyTorch models

[![ROCm](https://img.shields.io/badge/ROCm-7.x-red.svg)](https://rocm.docs.amd.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Features

- ✅ **HIP Executable Analysis** - Kernel-level profiling for native HIP applications
- ✅ **PyTorch Model Analysis** - Automated PyTorch operator performance collection
- ✅ **Triton Kernel Analysis** - Built-in Triton kernel matching and profiling
- ✅ **Batch Analysis** - Directory-level batch model analysis
- ✅ **Perfetto Visualization** - Generates standard Perfetto trace files
- ✅ **Auto Report Generation** - Automatically generates analysis summaries and usage instructions
- ✅ **Multi-Agent Support** - One-click install to Kimi Code CLI / Claude Code / Cursor / Codex / DeepSeek

## 🔧 Requirements

- AMD GPU (MI100/MI200/MI300 series or Radeon RX 7000 series)
- ROCm 7.x
- Python 3.8+
- PyTorch with ROCm (optional, for PyTorch analysis)
- Triton (optional, for Triton kernel analysis)

## 📦 Installation

```bash
git clone https://github.com/your-username/rocprof-skill.git
cd rocprof-skill

# Add execution permissions
chmod +x install.sh scripts/*.sh examples/*.sh

# Optional: Install Python dependencies
pip install -r requirements.txt
```

## 🤖 AI Agent Integration

The `install.sh` script deploys this skill to various AI coding agents so they can automatically perform GPU profiling analysis.

### Supported Agents

| Agent | Flag | Install Path |
|-------|------|-------------|
| **Kimi Code CLI** | `--kimi` | `~/.config/agents/skills/rocprof-hip-profiling/` |
| **Claude Code** | `--claude` | `~/.claude/skills/rocprof-hip-profiling/` |
| **Cursor** | `--cursor` | `~/.cursor/rules/rocprof-hip-profiling.md` |
| **Codex (OpenAI)** | `--codex` | `~/.codex/skills/rocprof-hip-profiling/` |
| **DeepSeek** | `--deepseek` | `~/.deepseek/skills/rocprof-hip-profiling/` |

### Install to a Single Agent

```bash
# Install to Kimi Code CLI (default)
./install.sh

# Install to Claude Code
./install.sh --claude

# Install to Cursor (user-level rules)
./install.sh --cursor

# Install to DeepSeek
./install.sh --deepseek

# Install to Codex
./install.sh --codex
```

### Install to Multiple / All Agents

```bash
# Install to specific agents
./install.sh --kimi --claude --deepseek

# Install to ALL supported agents at once
./install.sh --all
```

### Project-Level Cursor Rules

```bash
# Install to current project's .cursor/rules/ directory
./install.sh --project
```

### Custom Install Path

```bash
# Install to any custom directory
./install.sh -t ~/my-skills/rocprof
```

### Management Commands

```bash
# Check ROCm environment dependencies
./install.sh --check

# View installation status across all agents
./install.sh --status

# Uninstall from all installed locations
./install.sh --uninstall
```

### What Gets Installed

For directory-based agents (Kimi / Claude / Codex / DeepSeek):

```
~/.config/agents/skills/rocprof-hip-profiling/   # (Kimi example)
├── SKILL.md              # Skill entry file (auto-loaded by agent)
├── scripts/
│   ├── kernel_runner.py  # PyTorch model runner
│   ├── triton_runner.py  # Triton kernel runner
│   └── rocprof_pytorch.sh
└── examples/
    ├── auto_rocprof.sh
    ├── sample_models/
    └── sample_triton/
```

For Cursor: a single `rocprof-hip-profiling.md` rules file is copied to `~/.cursor/rules/`.

### After Installation

Once installed, the AI agent will automatically recognize GPU profiling requests. Just ask naturally:

- *"Profile my PyTorch model on AMD GPU"*
- *"Analyze the HIP kernel performance of this program"*
- *"Run Triton kernel profiling on my operators"*
- *"Help me find GPU performance bottlenecks"*

The agent will use the skill's tools (`kernel_runner.py`, `triton_runner.py`, `auto_rocprof.sh`, `rocprof-sys`) to collect and analyze data.

---

## 🚀 Quick Start

### 1. Analyze HIP Executables

```bash
./examples/auto_rocprof.sh ./your_hip_program my_report
```

### 2. Analyze PyTorch Models

```bash
# Single model
./examples/auto_rocprof.sh --pytorch model.py my_model

# All models in a directory
./examples/auto_rocprof.sh --pytorch-dir ./models all_models
```

### 3. Use Python Scripts

```bash
# Single model
python scripts/kernel_runner.py --file 22_Tanh.py

# With rocprof-sys
rocprof-sys-run --trace --use-rocm --rocm-domains kernel_dispatch \
    -o ./results -- python scripts/kernel_runner.py --file 22_Tanh.py
```

### 4. Triton Kernel Analysis

```bash
# Run PyTorch op with auto Triton matching
python scripts/triton_runner.py --pytorch-file 22_Tanh.py

# Run native Triton kernel
python scripts/triton_runner.py --file examples/sample_triton/tanh_triton.py

# List supported built-in Triton ops
python scripts/triton_runner.py --list-ops
```

## 📁 Project Structure

```
rocprof-skill/
├── README.md                    # Project description
├── SKILL.md                     # AI Agent skill entry file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── install.sh                   # Multi-agent installation script
│
├── scripts/                     # Core scripts
│   ├── kernel_runner.py         # PyTorch model runner
│   ├── triton_runner.py         # Triton kernel runner
│   └── rocprof_pytorch.sh       # PyTorch profiling script
│
├── examples/                    # Example scripts
│   ├── auto_rocprof.sh          # Automated analysis script (general)
│   ├── sample_models/           # Sample PyTorch models
│   │   ├── tanh_model.py
│   │   └── matmul_model.py
│   └── sample_triton/           # Sample Triton kernels
│       ├── tanh_triton.py
│       └── matmul_triton.py
│
└── docs/                        # Documentation
    ├── USAGE.md                 # Usage guide
    └── TROUBLESHOOTING.md       # Troubleshooting
```

## 📊 Output Files

| File | Description |
|------|-------------|
| `perfetto-trace-*.proto` | Perfetto format trace file, viewable at [ui.perfetto.dev](https://ui.perfetto.dev) |
| `wall_clock-*.txt` | Function-level time statistics |
| `sampling_percent-*.txt` | Sampling percentage distribution |
| `sampling_wall_clock-*.txt` | Sampling wall clock time |
| `metadata-*.json` | Run metadata (CPU, GPU, ROCm version, etc.) |
| `functions-*.json` | Function address mapping |

## 🔍 Visual Analysis

### Using Perfetto UI

1. Open [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. Click "Open trace file"
3. Upload the generated `perfetto-trace-*.proto` file
4. Browse the timeline to analyze kernel execution times

### Command Line Viewing

```bash
# View time statistics
cat results/wall_clock-*.txt

# View sampling distribution
cat results/sampling_percent-*.txt | head -50
```

## ⚙️ rocprof-sys Common Parameters

```bash
rocprof-sys-run [options] -- <program>

# Common options
--trace                    # Enable tracing
--profile                  # Enable profiling
--use-rocm                 # Enable ROCm tracing
--rocm-domains <domains>   # Specify tracing domains

# Available domains
kernel_dispatch            # GPU kernel execution ✅ (Recommended)
hip_api                    # HIP API calls
hip_runtime_api           # HIP runtime API
memory_copy               # Memory copy
memory_allocation         # Memory allocation
hsa_api                   # HSA API
```

## 🐛 Troubleshooting

### 1. Cannot find rocprof-sys

```bash
export PATH=$PATH:/opt/rocm/bin
```

### 2. PyTorch Initialization Error

If you encounter a `libcaffe2_nvrtc.so` error, this is a compatibility issue between PyTorch ROCm version and rocprof-sys.
Solution: Ensure you are using the correct PyTorch ROCm version.

### 3. Permission Issues

```bash
sudo usermod -a -G video $USER
# Re-login for changes to take effect
```

## 📚 Related Resources

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [rocprofiler-systems GitHub](https://github.com/ROCm/rocprofiler-systems)
- [Perfetto UI](https://ui.perfetto.dev/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🤝 Contributing

Issues and Pull Requests are welcome!

---
