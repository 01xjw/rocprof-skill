#!/bin/bash
# ROCm PyTorch Kernel Profiler
# 专门用于 PyTorch 模型的 rocprof-sys 性能分析
#
# 使用方法:
#   ./rocprof_pytorch.sh <model_file.py> [output_dir]
#   ./rocprof_pytorch.sh --dir <models_dir> [output_dir]
#
# 示例:
#   ./rocprof_pytorch.sh 22_Tanh.py ./results
#   ./rocprof_pytorch.sh --dir ./kernel_test/level1 ./results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${CYAN}$1${NC}"; }

show_help() {
    cat << EOF
🚀 ROCm PyTorch Kernel Profiler

使用方法:
    $0 <model_file.py> [output_dir]
    $0 --dir <models_dir> [output_dir]
    $0 --help

选项:
    --dir, -d       指定包含模型文件的目录
    --warmup, -w    预热迭代次数 (默认: 3)
    --iterations, -i  正式运行迭代次数 (默认: 10)
    --domains       rocprof-sys domains (默认: kernel_dispatch)
    --help, -h      显示此帮助信息

示例:
    # 单个模型
    $0 22_Tanh.py ./results

    # 目录中所有模型
    $0 --dir ./models ./results

    # 自定义参数
    $0 --warmup 5 --iterations 20 model.py ./results

输出:
    - perfetto-trace-*.proto    Perfetto 格式 trace (可在 ui.perfetto.dev 查看)
    - wall_clock-*.txt          时间统计
    - sampling_*.txt            采样数据
    - metadata-*.json           运行元数据

EOF
}

# 默认参数
WARMUP=3
ITERATIONS=10
DOMAINS="kernel_dispatch"
OUTPUT_DIR=""
MODEL_FILE=""
MODEL_DIR=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --dir|-d)
            MODEL_DIR="$2"
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
            if [ -z "$MODEL_FILE" ] && [ -z "$MODEL_DIR" ]; then
                MODEL_FILE="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

# 验证参数
if [ -z "$MODEL_FILE" ] && [ -z "$MODEL_DIR" ]; then
    print_error "请指定模型文件或目录"
    show_help
    exit 1
fi

# 设置默认输出目录
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./rocprof_results_$(date +%Y%m%d_%H%M%S)"
fi

# 检查 rocprof-sys
if ! command -v rocprof-sys-run &> /dev/null; then
    if [ -x "/opt/rocm/bin/rocprof-sys-run" ]; then
        export PATH="/opt/rocm/bin:$PATH"
    else
        print_error "rocprof-sys-run 未找到"
        print_info "请确保 ROCm 已正确安装，或运行: export PATH=\$PATH:/opt/rocm/bin"
        exit 1
    fi
fi

# 检查 Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    print_error "Python 未找到"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)

print_header "
╔══════════════════════════════════════════════════════════════╗
║          🚀 ROCm PyTorch Kernel Profiler                     ║
╚══════════════════════════════════════════════════════════════╝
"

echo "配置信息:"
echo "  Python:      $PYTHON_CMD"
echo "  rocprof-sys: $(which rocprof-sys-run)"
echo "  输出目录:    $OUTPUT_DIR"
echo "  预热次数:    $WARMUP"
echo "  迭代次数:    $ITERATIONS"
echo "  Domains:     $DOMAINS"
echo ""

# 创建临时运行脚本
TEMP_SCRIPT=$(mktemp /tmp/rocprof_runner_XXXXXX.py)

if [ -n "$MODEL_FILE" ]; then
    # 单个文件模式
    if [ ! -f "$MODEL_FILE" ]; then
        print_error "文件不存在: $MODEL_FILE"
        exit 1
    fi
    
    MODEL_PATH=$(realpath "$MODEL_FILE")
    MODEL_NAME=$(basename "$MODEL_FILE" .py)
    WORK_DIR=$(dirname "$MODEL_PATH")
    
    print_info "Profiling: $MODEL_FILE"
    
    cat > "$TEMP_SCRIPT" << PYTHON_EOF
import os
import sys
import torch

# 切换到模型文件所在目录
os.chdir("$WORK_DIR")
sys.path.insert(0, "$WORK_DIR")

from importlib.util import spec_from_file_location, module_from_spec

print(f"PyTorch version: {torch.__version__}")
print(f"HIP available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

spec = spec_from_file_location('model_module', "$MODEL_PATH")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.Model(*mod.get_init_inputs()).cuda().eval()
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in mod.get_inputs()]

print(f"Model: {type(model).__name__}")
if inputs and isinstance(inputs[0], torch.Tensor):
    print(f"Input shape: {inputs[0].shape}, dtype: {inputs[0].dtype}")

# Warmup
print(f"\\nWarmup ({$WARMUP} iterations)...")
with torch.no_grad():
    for i in range($WARMUP):
        _ = model(*inputs)
        torch.cuda.synchronize()

# Profile
print(f"Profiling ({$ITERATIONS} iterations)...")
with torch.no_grad():
    for i in range($ITERATIONS):
        _ = model(*inputs)
        torch.cuda.synchronize()

print("Done!")
PYTHON_EOF

else
    # 目录模式
    if [ ! -d "$MODEL_DIR" ]; then
        print_error "目录不存在: $MODEL_DIR"
        exit 1
    fi
    
    MODEL_DIR_PATH=$(realpath "$MODEL_DIR")
    
    print_info "Profiling directory: $MODEL_DIR"
    
    cat > "$TEMP_SCRIPT" << PYTHON_EOF
import os
import sys
import glob
import torch

os.chdir("$MODEL_DIR_PATH")
sys.path.insert(0, "$MODEL_DIR_PATH")

from importlib.util import spec_from_file_location, module_from_spec

print(f"PyTorch version: {torch.__version__}")
print(f"HIP available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

model_files = sorted([
    f for f in glob.glob("*.py")
    if not f.startswith(('run_', 'test_', '__'))
])

print(f"Found {len(model_files)} model files")

for model_file in model_files:
    print(f"\\n{'='*60}")
    print(f"Profiling: {model_file}")
    print('='*60)
    
    try:
        spec = spec_from_file_location('model_module', model_file)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        model = mod.Model(*mod.get_init_inputs()).cuda().eval()
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in mod.get_inputs()]
        
        print(f"Model: {type(model).__name__}")
        if inputs and isinstance(inputs[0], torch.Tensor):
            print(f"Input shape: {inputs[0].shape}, dtype: {inputs[0].dtype}")
        
        # Warmup
        with torch.no_grad():
            for i in range($WARMUP):
                _ = model(*inputs)
                torch.cuda.synchronize()
        
        # Profile
        with torch.no_grad():
            for i in range($ITERATIONS):
                _ = model(*inputs)
                torch.cuda.synchronize()
        
        print("OK")
        
    except Exception as e:
        print(f"Error: {e}")
        continue

print("\\nAll done!")
PYTHON_EOF
fi

# 运行 rocprof-sys
print_info "Starting rocprof-sys profiling..."
echo ""

mkdir -p "$OUTPUT_DIR"

rocprof-sys-run \
    --trace \
    --profile \
    --use-rocm \
    --rocm-domains "$DOMAINS" \
    -o "$OUTPUT_DIR" \
    -- "$PYTHON_CMD" "$TEMP_SCRIPT" 2>&1 | tee "${OUTPUT_DIR}/rocprof.log"

# 清理临时文件
rm -f "$TEMP_SCRIPT"

# 输出结果
echo ""
print_header "
╔══════════════════════════════════════════════════════════════╗
║                    📊 Profiling 完成                          ║
╚══════════════════════════════════════════════════════════════╝
"

# 查找输出文件
RESULT_SUBDIR=$(ls -td "$OUTPUT_DIR"/*/ 2>/dev/null | head -1)
if [ -n "$RESULT_SUBDIR" ]; then
    print_success "结果目录: $RESULT_SUBDIR"
    echo ""
    echo "生成的文件:"
    ls -lh "$RESULT_SUBDIR" 2>/dev/null | grep -v "^total" | while read line; do
        echo "  $line"
    done
    
    # 查找 perfetto trace
    PERFETTO_FILE=$(find "$RESULT_SUBDIR" -name "perfetto*.proto" 2>/dev/null | head -1)
    if [ -n "$PERFETTO_FILE" ]; then
        echo ""
        print_info "📊 可视化: 打开 https://ui.perfetto.dev/ 上传 $PERFETTO_FILE"
    fi
else
    print_warning "未找到输出子目录，请检查 $OUTPUT_DIR"
fi

echo ""
print_info "日志文件: ${OUTPUT_DIR}/rocprof.log"
