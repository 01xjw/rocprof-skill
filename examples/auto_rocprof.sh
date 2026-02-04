#!/bin/bash
# ============================================================================
# ROCm/HIP 自动化性能分析脚本
# 支持 HIP 可执行文件和 PyTorch 模型的 profiling
# ============================================================================
#
# 使用方法:
#   ./auto_rocprof.sh <executable_or_script> [output_prefix]
#   ./auto_rocprof.sh --pytorch <model.py> [output_prefix]
#   ./auto_rocprof.sh --pytorch-dir <models_dir> [output_prefix]
#
# 示例:
#   ./auto_rocprof.sh ./hip_matmul my_report
#   ./auto_rocprof.sh --pytorch 22_Tanh.py tanh_profile
#   ./auto_rocprof.sh --pytorch-dir ./kernel_test/level1 level1_profile

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${CYAN}$1${NC}"; }

show_help() {
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 ROCm/HIP 自动化性能分析脚本                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

使用方法:
    ./auto_rocprof.sh <executable> [output_prefix]
    ./auto_rocprof.sh --pytorch <model.py> [output_prefix]
    ./auto_rocprof.sh --pytorch-dir <models_dir> [output_prefix]

选项:
    --pytorch, -p       分析 PyTorch 模型文件
    --pytorch-dir, -pd  分析目录中的所有 PyTorch 模型
    --warmup, -w        PyTorch 预热迭代次数 (默认: 3)
    --iterations, -i    PyTorch 正式迭代次数 (默认: 10)
    --domains           rocprof-sys domains (默认: kernel_dispatch)
    --help, -h          显示此帮助信息

示例:
    # HIP 可执行文件
    ./auto_rocprof.sh ./hip_matmul my_analysis

    # PyTorch 单个模型
    ./auto_rocprof.sh --pytorch 22_Tanh.py tanh_profile

    # PyTorch 模型目录
    ./auto_rocprof.sh --pytorch-dir ./models all_models

输出文件:
    📁 rocprof_reports/<prefix>/
    ├── perfetto-trace-*.proto    # Perfetto trace (上传到 ui.perfetto.dev)
    ├── wall_clock-*.txt          # 时间统计
    ├── sampling_*.txt            # 采样数据
    ├── metadata-*.json           # 运行元数据
    └── functions-*.json          # 函数信息

EOF
}

# ============================================================================
# 参数解析
# ============================================================================

MODE="executable"  # executable | pytorch | pytorch-dir
TARGET=""
PREFIX=""
WARMUP=3
ITERATIONS=10
DOMAINS="kernel_dispatch"
REPORT_DIR="rocprof_reports"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --pytorch|-p)
            MODE="pytorch"
            TARGET="$2"
            shift 2
            ;;
        --pytorch-dir|-pd)
            MODE="pytorch-dir"
            TARGET="$2"
            shift 2
            ;;
        --warmup|-w)
            WARMUP="$2"
            shift 2
            ;;
        --iterations|-i)
            ITERATIONS="$2"
            shift 2
            ;;
        --domains)
            DOMAINS="$2"
            shift 2
            ;;
        -*)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$TARGET" ]; then
                TARGET="$1"
            elif [ -z "$PREFIX" ]; then
                PREFIX="$1"
            fi
            shift
            ;;
    esac
done

# 验证参数
if [ -z "$TARGET" ]; then
    print_error "请指定目标文件或目录"
    show_help
    exit 1
fi

# 设置默认前缀
if [ -z "$PREFIX" ]; then
    PREFIX="rocprof_$(date +%Y%m%d_%H%M%S)"
fi

# ============================================================================
# 环境检查
# ============================================================================

# 检查 rocprof-sys
if ! command -v rocprof-sys-run &> /dev/null; then
    if [ -x "/opt/rocm/bin/rocprof-sys-run" ]; then
        export PATH="/opt/rocm/bin:$PATH"
    else
        print_error "rocprof-sys-run 未找到"
        print_info "请确保 ROCm 已正确安装: export PATH=\$PATH:/opt/rocm/bin"
        exit 1
    fi
fi

# 检查 Python (PyTorch 模式)
if [[ "$MODE" == "pytorch"* ]]; then
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_error "Python 未找到"
        exit 1
    fi
    PYTHON_CMD=$(command -v python3 || command -v python)
fi

# 创建报告目录
mkdir -p "$REPORT_DIR"

# ============================================================================
# 主逻辑
# ============================================================================

print_header "
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 ROCm/HIP 自动化性能分析                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"

echo "配置信息:"
echo "  模式:        $MODE"
echo "  目标:        $TARGET"
echo "  输出前缀:    $PREFIX"
echo "  报告目录:    $REPORT_DIR"
if [[ "$MODE" == "pytorch"* ]]; then
    echo "  Python:      $PYTHON_CMD"
    echo "  预热次数:    $WARMUP"
    echo "  迭代次数:    $ITERATIONS"
fi
echo "  Domains:     $DOMAINS"
echo ""

OUTPUT_DIR="${REPORT_DIR}/${PREFIX}"

# ============================================================================
# 执行 Profiling
# ============================================================================

case $MODE in
    "executable")
        # HIP 可执行文件模式
        if [ ! -f "$TARGET" ]; then
            print_error "文件不存在: $TARGET"
            exit 1
        fi
        
        print_info "📊 开始 HIP 可执行文件分析..."
        
        rocprof-sys-run \
            --trace \
            --profile \
            --device \
            --include rocm \
            -o "$OUTPUT_DIR" \
            -- "$TARGET" 2>&1 | tee "${REPORT_DIR}/${PREFIX}_rocprof.log"
        ;;
        
    "pytorch")
        # PyTorch 单文件模式
        if [ ! -f "$TARGET" ]; then
            print_error "文件不存在: $TARGET"
            exit 1
        fi
        
        TARGET_PATH=$(realpath "$TARGET")
        TARGET_DIR=$(dirname "$TARGET_PATH")
        
        print_info "📊 开始 PyTorch 模型分析: $TARGET"
        
        # 创建临时运行脚本
        TEMP_SCRIPT=$(mktemp /tmp/rocprof_pytorch_XXXXXX.py)
        
        cat > "$TEMP_SCRIPT" << PYTHON_EOF
import os
import sys
import torch

os.chdir("$TARGET_DIR")
sys.path.insert(0, "$TARGET_DIR")

from importlib.util import spec_from_file_location, module_from_spec

print(f"PyTorch: {torch.__version__}")
print(f"HIP available: {torch.cuda.is_available()}")
print(f"Devices: {torch.cuda.device_count()}")

spec = spec_from_file_location('m', "$TARGET_PATH")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.Model(*mod.get_init_inputs()).cuda().eval()
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in mod.get_inputs()]

print(f"Model: {type(model).__name__}")
if inputs and isinstance(inputs[0], torch.Tensor):
    print(f"Input: {inputs[0].shape}, {inputs[0].dtype}")

print(f"Warmup ({$WARMUP}x)...")
with torch.no_grad():
    for _ in range($WARMUP):
        _ = model(*inputs)
        torch.cuda.synchronize()

print(f"Profile ({$ITERATIONS}x)...")
with torch.no_grad():
    for _ in range($ITERATIONS):
        _ = model(*inputs)
        torch.cuda.synchronize()

print("Done!")
PYTHON_EOF
        
        rocprof-sys-run \
            --trace \
            --profile \
            --use-rocm \
            --rocm-domains "$DOMAINS" \
            -o "$OUTPUT_DIR" \
            -- "$PYTHON_CMD" "$TEMP_SCRIPT" 2>&1 | tee "${REPORT_DIR}/${PREFIX}_rocprof.log"
        
        rm -f "$TEMP_SCRIPT"
        ;;
        
    "pytorch-dir")
        # PyTorch 目录模式
        if [ ! -d "$TARGET" ]; then
            print_error "目录不存在: $TARGET"
            exit 1
        fi
        
        TARGET_DIR=$(realpath "$TARGET")
        
        print_info "📊 开始 PyTorch 目录分析: $TARGET"
        
        # 创建临时运行脚本
        TEMP_SCRIPT=$(mktemp /tmp/rocprof_pytorch_dir_XXXXXX.py)
        
        cat > "$TEMP_SCRIPT" << PYTHON_EOF
import os
import sys
import glob
import torch

os.chdir("$TARGET_DIR")
sys.path.insert(0, "$TARGET_DIR")

from importlib.util import spec_from_file_location, module_from_spec

print(f"PyTorch: {torch.__version__}")
print(f"HIP available: {torch.cuda.is_available()}")
print(f"Devices: {torch.cuda.device_count()}")

files = sorted([f for f in glob.glob("*.py") if not f.startswith(('run_', 'test_', '__'))])
print(f"Found {len(files)} models")

for f in files:
    print(f"\\n{'='*60}")
    print(f"Model: {f}")
    print('='*60)
    
    try:
        spec = spec_from_file_location('m', f)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        model = mod.Model(*mod.get_init_inputs()).cuda().eval()
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in mod.get_inputs()]
        
        if inputs and isinstance(inputs[0], torch.Tensor):
            print(f"Input: {inputs[0].shape}")
        
        with torch.no_grad():
            for _ in range($WARMUP):
                _ = model(*inputs)
                torch.cuda.synchronize()
            
            for _ in range($ITERATIONS):
                _ = model(*inputs)
                torch.cuda.synchronize()
        
        print("OK")
    except Exception as e:
        print(f"Error: {e}")

print("\\nAll done!")
PYTHON_EOF
        
        rocprof-sys-run \
            --trace \
            --profile \
            --use-rocm \
            --rocm-domains "$DOMAINS" \
            -o "$OUTPUT_DIR" \
            -- "$PYTHON_CMD" "$TEMP_SCRIPT" 2>&1 | tee "${REPORT_DIR}/${PREFIX}_rocprof.log"
        
        rm -f "$TEMP_SCRIPT"
        ;;
esac

# ============================================================================
# 生成报告
# ============================================================================

echo ""
print_header "
╔══════════════════════════════════════════════════════════════════════════════╗
║                            📊 分析完成                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"

# 查找输出子目录
RESULT_SUBDIR=$(ls -td "$OUTPUT_DIR"/*/ 2>/dev/null | head -1)

if [ -n "$RESULT_SUBDIR" ]; then
    print_success "结果目录: $RESULT_SUBDIR"
    echo ""
    echo "生成的文件:"
    ls -lh "$RESULT_SUBDIR" 2>/dev/null | grep -v "^total" | head -20 | while read line; do
        echo "  $line"
    done
    
    # 查找 Perfetto trace
    PERFETTO_FILE=$(find "$RESULT_SUBDIR" -name "perfetto*.proto" 2>/dev/null | head -1)
    METADATA_FILE=$(find "$RESULT_SUBDIR" -name "metadata*.json" 2>/dev/null | head -1)
    
    # 生成摘要
    if [ -n "$METADATA_FILE" ]; then
        echo ""
        print_info "📋 Metadata 摘要:"
        python3 -c "
import json
with open('$METADATA_FILE') as f:
    data = json.load(f)
    
rs = data.get('rocprofiler-systems', {})
info = rs.get('metadata', {}).get('info', {})

print(f\"  ROCm Version: {info.get('ROCPROFSYS_ROCM_VERSION', 'N/A')}\")
print(f\"  CPU: {info.get('CPU_MODEL', 'N/A')}\")
" 2>/dev/null || true
    fi
    
    echo ""
    if [ -n "$PERFETTO_FILE" ]; then
        print_info "📊 可视化方法:"
        echo "   1. 打开 https://ui.perfetto.dev/"
        echo "   2. 点击 'Open trace file'"
        echo "   3. 上传: $PERFETTO_FILE"
    fi
    
    # 生成 README
    cat > "${RESULT_SUBDIR}/README.md" << EOF
# ROCm Profiling 结果

**生成时间**: $(date)
**目标**: $TARGET
**模式**: $MODE

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| \`perfetto-trace-*.proto\` | Perfetto 格式 trace 文件 |
| \`metadata-*.json\` | 运行元数据 |
| \`wall_clock-*.txt\` | 时间统计 |
| \`sampling_*.txt\` | 采样数据 |
| \`functions-*.json\` | 函数信息 |

## 📊 查看方法

### Perfetto UI (推荐)
1. 打开 https://ui.perfetto.dev/
2. 上传 \`$(basename "$PERFETTO_FILE" 2>/dev/null || echo "perfetto-trace-*.proto")\`

### 命令行
\`\`\`bash
# 查看时间统计
cat wall_clock-*.txt

# 查看采样数据
cat sampling_percent-*.txt
\`\`\`

## 🔧 重新采集
\`\`\`bash
$(basename "$0") $TARGET $PREFIX
\`\`\`
EOF
    
    print_success "README 已生成: ${RESULT_SUBDIR}/README.md"
fi

echo ""
print_info "日志: ${REPORT_DIR}/${PREFIX}_rocprof.log"
echo ""
print_success "🎉 完成!"
