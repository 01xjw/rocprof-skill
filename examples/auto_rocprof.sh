#!/bin/bash
# ============================================================================
# ROCm/HIP Automated Performance Analysis Script
# Supports profiling for HIP executables and PyTorch models
# ============================================================================
#
# Usage:
#   ./auto_rocprof.sh <executable_or_script> [output_prefix]
#   ./auto_rocprof.sh --pytorch <model.py> [output_prefix]
#   ./auto_rocprof.sh --pytorch-dir <models_dir> [output_prefix]
#
# Examples:
#   ./auto_rocprof.sh ./hip_matmul my_report
#   ./auto_rocprof.sh --pytorch 22_Tanh.py tanh_profile
#   ./auto_rocprof.sh --pytorch-dir ./kernel_test/level1 level1_profile

set -e

# Color definitions
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
║                    ROCm/HIP Automated Performance Analysis Script            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    ./auto_rocprof.sh <executable> [output_prefix]
    ./auto_rocprof.sh --pytorch <model.py> [output_prefix]
    ./auto_rocprof.sh --pytorch-dir <models_dir> [output_prefix]

Options:
    --pytorch, -p       Profile a PyTorch model file
    --pytorch-dir, -pd  Profile all PyTorch models in a directory
    --warmup, -w        PyTorch warmup iterations (default: 3)
    --iterations, -i    PyTorch profiling iterations (default: 10)
    --domains           rocprof-sys domains (default: kernel_dispatch)
    --help, -h          Show this help message

Examples:
    # HIP executable
    ./auto_rocprof.sh ./hip_matmul my_analysis

    # Single PyTorch model
    ./auto_rocprof.sh --pytorch 22_Tanh.py tanh_profile

    # PyTorch model directory
    ./auto_rocprof.sh --pytorch-dir ./models all_models

Output files:
    rocprof_reports/<prefix>/
    ├── perfetto-trace-*.proto    # Perfetto trace (upload to ui.perfetto.dev)
    ├── wall_clock-*.txt          # Time statistics
    ├── sampling_*.txt            # Sampling data
    ├── metadata-*.json           # Run metadata
    └── functions-*.json          # Function information

EOF
}

# ============================================================================
# Argument Parsing
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
            print_error "Unknown option: $1"
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

# Validate arguments
if [ -z "$TARGET" ]; then
    print_error "Please specify a target file or directory"
    show_help
    exit 1
fi

# Set default prefix
if [ -z "$PREFIX" ]; then
    PREFIX="rocprof_$(date +%Y%m%d_%H%M%S)"
fi

# ============================================================================
# Environment Check
# ============================================================================

# Check rocprof-sys
if ! command -v rocprof-sys-run &> /dev/null; then
    if [ -x "/opt/rocm/bin/rocprof-sys-run" ]; then
        export PATH="/opt/rocm/bin:$PATH"
    else
        print_error "rocprof-sys-run not found"
        print_info "Please ensure ROCm is properly installed: export PATH=\$PATH:/opt/rocm/bin"
        exit 1
    fi
fi

# Check Python (PyTorch mode)
if [[ "$MODE" == "pytorch"* ]]; then
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    PYTHON_CMD=$(command -v python3 || command -v python)
fi

# Create report directory
mkdir -p "$REPORT_DIR"

# ============================================================================
# Main Logic
# ============================================================================

print_header "
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ROCm/HIP Automated Performance Analysis                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"

echo "Configuration:"
echo "  Mode:        $MODE"
echo "  Target:      $TARGET"
echo "  Prefix:      $PREFIX"
echo "  Report dir:  $REPORT_DIR"
if [[ "$MODE" == "pytorch"* ]]; then
    echo "  Python:      $PYTHON_CMD"
    echo "  Warmup:      $WARMUP"
    echo "  Iterations:  $ITERATIONS"
fi
echo "  Domains:     $DOMAINS"
echo ""

OUTPUT_DIR="${REPORT_DIR}/${PREFIX}"

# ============================================================================
# Execute Profiling
# ============================================================================

case $MODE in
    "executable")
        # HIP executable mode
        if [ ! -f "$TARGET" ]; then
            print_error "File not found: $TARGET"
            exit 1
        fi

        print_info "Starting HIP executable analysis..."

        rocprof-sys-run \
            --trace \
            --profile \
            --device \
            --include rocm \
            -o "$OUTPUT_DIR" \
            -- "$TARGET" 2>&1 | tee "${REPORT_DIR}/${PREFIX}_rocprof.log"
        ;;

    "pytorch")
        # PyTorch single file mode
        if [ ! -f "$TARGET" ]; then
            print_error "File not found: $TARGET"
            exit 1
        fi

        TARGET_PATH=$(realpath "$TARGET")
        TARGET_DIR=$(dirname "$TARGET_PATH")

        print_info "Starting PyTorch model analysis: $TARGET"

        # Create temporary run script
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
        # PyTorch directory mode
        if [ ! -d "$TARGET" ]; then
            print_error "Directory not found: $TARGET"
            exit 1
        fi

        TARGET_DIR=$(realpath "$TARGET")

        print_info "Starting PyTorch directory analysis: $TARGET"

        # Create temporary run script
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
    print(f"\n{'='*60}")
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

print("\nAll done!")
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
# Generate Report
# ============================================================================

echo ""
print_header "
╔══════════════════════════════════════════════════════════════════════════════╗
║                            Analysis Complete                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"

# Find output subdirectory
RESULT_SUBDIR=$(ls -td "$OUTPUT_DIR"/*/ 2>/dev/null | head -1)

if [ -n "$RESULT_SUBDIR" ]; then
    print_success "Results directory: $RESULT_SUBDIR"
    echo ""
    echo "Generated files:"
    ls -lh "$RESULT_SUBDIR" 2>/dev/null | grep -v "^total" | head -20 | while read line; do
        echo "  $line"
    done

    # Find Perfetto trace
    PERFETTO_FILE=$(find "$RESULT_SUBDIR" -name "perfetto*.proto" 2>/dev/null | head -1)
    METADATA_FILE=$(find "$RESULT_SUBDIR" -name "metadata*.json" 2>/dev/null | head -1)

    # Generate summary
    if [ -n "$METADATA_FILE" ]; then
        echo ""
        print_info "Metadata summary:"
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
        print_info "Visualization:"
        echo "   1. Open https://ui.perfetto.dev/"
        echo "   2. Click 'Open trace file'"
        echo "   3. Upload: $PERFETTO_FILE"
    fi

    # Generate README
    cat > "${RESULT_SUBDIR}/README.md" << EOF
# ROCm Profiling Results

**Generated**: $(date)
**Target**: $TARGET
**Mode**: $MODE

## Files

| File | Description |
|------|-------------|
| \`perfetto-trace-*.proto\` | Perfetto format trace file |
| \`metadata-*.json\` | Run metadata |
| \`wall_clock-*.txt\` | Time statistics |
| \`sampling_*.txt\` | Sampling data |
| \`functions-*.json\` | Function information |

## Viewing Results

### Perfetto UI (Recommended)
1. Open https://ui.perfetto.dev/
2. Upload \`$(basename "$PERFETTO_FILE" 2>/dev/null || echo "perfetto-trace-*.proto")\`

### Command Line
\`\`\`bash
# View time statistics
cat wall_clock-*.txt

# View sampling data
cat sampling_percent-*.txt
\`\`\`

## Re-collect
\`\`\`bash
$(basename "$0") $TARGET $PREFIX
\`\`\`
EOF

    print_success "README generated: ${RESULT_SUBDIR}/README.md"
fi

echo ""
print_info "Log: ${REPORT_DIR}/${PREFIX}_rocprof.log"
echo ""
print_success "Done!"
