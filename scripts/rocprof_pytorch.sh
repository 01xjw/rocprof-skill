#!/bin/bash
# ROCm PyTorch Kernel Profiler
# Dedicated rocprof-sys profiling for PyTorch models
#
# Usage:
#   ./rocprof_pytorch.sh <model_file.py> [output_dir]
#   ./rocprof_pytorch.sh --dir <models_dir> [output_dir]
#
# Examples:
#   ./rocprof_pytorch.sh 22_Tanh.py ./results
#   ./rocprof_pytorch.sh --dir ./kernel_test/level1 ./results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color definitions
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
ROCm PyTorch Kernel Profiler

Usage:
    $0 <model_file.py> [output_dir]
    $0 --dir <models_dir> [output_dir]
    $0 --help

Options:
    --dir, -d       Specify directory containing model files
    --warmup, -w    Number of warmup iterations (default: 3)
    --iterations, -i  Number of profiling iterations (default: 10)
    --domains       rocprof-sys domains (default: kernel_dispatch)
    --help, -h      Show this help message

Examples:
    # Single model
    $0 22_Tanh.py ./results

    # All models in a directory
    $0 --dir ./models ./results

    # Custom parameters
    $0 --warmup 5 --iterations 20 model.py ./results

Output:
    - perfetto-trace-*.proto    Perfetto format trace (viewable at ui.perfetto.dev)
    - wall_clock-*.txt          Time statistics
    - sampling_*.txt            Sampling data
    - metadata-*.json           Run metadata

EOF
}

# Default parameters
WARMUP=3
ITERATIONS=10
DOMAINS="kernel_dispatch"
OUTPUT_DIR=""
MODEL_FILE=""
MODEL_DIR=""

# Parse arguments
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
            print_error "Unknown option: $1"
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

# Validate arguments
if [ -z "$MODEL_FILE" ] && [ -z "$MODEL_DIR" ]; then
    print_error "Please specify a model file or directory"
    show_help
    exit 1
fi

# Set default output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./rocprof_results_$(date +%Y%m%d_%H%M%S)"
fi

# Check rocprof-sys
if ! command -v rocprof-sys-run &> /dev/null; then
    if [ -x "/opt/rocm/bin/rocprof-sys-run" ]; then
        export PATH="/opt/rocm/bin:$PATH"
    else
        print_error "rocprof-sys-run not found"
        print_info "Please ensure ROCm is properly installed, or run: export PATH=\$PATH:/opt/rocm/bin"
        exit 1
    fi
fi

# Check Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    print_error "Python not found"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)

print_header "
╔══════════════════════════════════════════════════════════════╗
║          ROCm PyTorch Kernel Profiler                        ║
╚══════════════════════════════════════════════════════════════╝
"

echo "Configuration:"
echo "  Python:      $PYTHON_CMD"
echo "  rocprof-sys: $(which rocprof-sys-run)"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Warmup:      $WARMUP"
echo "  Iterations:  $ITERATIONS"
echo "  Domains:     $DOMAINS"
echo ""

# Create temporary run script
TEMP_SCRIPT=$(mktemp /tmp/rocprof_runner_XXXXXX.py)

if [ -n "$MODEL_FILE" ]; then
    # Single file mode
    if [ ! -f "$MODEL_FILE" ]; then
        print_error "File not found: $MODEL_FILE"
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

# Switch to model file directory
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
print(f"\nWarmup ({$WARMUP} iterations)...")
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
    # Directory mode
    if [ ! -d "$MODEL_DIR" ]; then
        print_error "Directory not found: $MODEL_DIR"
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
    print(f"\n{'='*60}")
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

print("\nAll done!")
PYTHON_EOF
fi

# Run rocprof-sys
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

# Clean up temporary files
rm -f "$TEMP_SCRIPT"

# Output results
echo ""
print_header "
╔══════════════════════════════════════════════════════════════╗
║                    Profiling Complete                         ║
╚══════════════════════════════════════════════════════════════╝
"

# Find output files
RESULT_SUBDIR=$(ls -td "$OUTPUT_DIR"/*/ 2>/dev/null | head -1)
if [ -n "$RESULT_SUBDIR" ]; then
    print_success "Results directory: $RESULT_SUBDIR"
    echo ""
    echo "Generated files:"
    ls -lh "$RESULT_SUBDIR" 2>/dev/null | grep -v "^total" | while read line; do
        echo "  $line"
    done

    # Find perfetto trace
    PERFETTO_FILE=$(find "$RESULT_SUBDIR" -name "perfetto*.proto" 2>/dev/null | head -1)
    if [ -n "$PERFETTO_FILE" ]; then
        echo ""
        print_info "Visualization: Open https://ui.perfetto.dev/ and upload $PERFETTO_FILE"
    fi
else
    print_warning "Output subdirectory not found, please check $OUTPUT_DIR"
fi

echo ""
print_info "Log file: ${OUTPUT_DIR}/rocprof.log"
