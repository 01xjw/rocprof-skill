#!/usr/bin/env python3
"""
ROCm Triton Kernel Profiling Runner
Runs Triton kernels to enable rocprof-sys to capture HIP kernel information

Supports two modes:
  1. Triton kernel file mode - file must provide standard interface (triton_kernel, get_inputs, get_init_args)
  2. PyTorch operator file compatibility mode - auto-converts PyTorch ops to equivalent Triton kernels

Usage:
    # Run Triton kernel file
    python triton_runner.py --file my_triton_kernel.py

    # Run PyTorch operator file (auto-generates equivalent Triton kernel)
    python triton_runner.py --pytorch-file 22_Tanh.py

    # Run all operators in a directory
    python triton_runner.py --pytorch-dir ./kernel_test/KernelBench/level1 --summary

    # Use with rocprof-sys
    rocprof-sys-run --trace --use-rocm --rocm-domains kernel_dispatch \\
        -o ./results -- python triton_runner.py --pytorch-file 22_Tanh.py
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# ============================================================================
# Triton Kernel File Interface Definition
# ============================================================================
#
# Triton kernel files must provide the following interface:
#
#   1. triton_kernel   - @triton.jit decorated kernel function
#   2. get_inputs()    - returns list of input tensors (already on GPU)
#   3. get_grid()      - returns grid configuration (lambda or tuple)
#   4. get_init_args() - returns additional kernel arguments (optional, default [])
#   5. run(inputs)     - custom run function (optional, if provided it will be called directly)
#
# Example Triton kernel file:
#
#   import torch
#   import triton
#   import triton.language as tl
#
#   @triton.jit
#   def tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
#       pid = tl.program_id(0)
#       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#       mask = offsets < n_elements
#       x = tl.load(x_ptr + offsets, mask=mask)
#       out = tl.math.tanh(x)
#       tl.store(out_ptr + offsets, out, mask=mask)
#
#   def get_inputs():
#       x = torch.randn(4096 * 393216, device='cuda', dtype=torch.float16)
#       return [x]
#
#   def get_grid():
#       return lambda meta: (triton.cdiv(4096 * 393216, meta['BLOCK_SIZE']),)
#
#   def get_init_args():
#       return {'BLOCK_SIZE': 1024}
#
# ============================================================================


def load_module_from_file(file_path: str):
    """Dynamically load a Python module from file"""
    spec = importlib.util.spec_from_file_location("triton_module", file_path)
    module = importlib.util.module_from_spec(spec)

    # Add the module's directory to sys.path
    module_dir = os.path.dirname(os.path.abspath(file_path))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec.loader.exec_module(module)
    return module


# ============================================================================
# Built-in Triton Kernel Library (for auto-converting PyTorch ops to Triton kernels)
# ============================================================================

def _get_builtin_triton_kernels() -> Dict[str, dict]:
    """
    Returns a mapping of built-in Triton kernel implementations
    key: PyTorch operator name (inferred from filename or forward method)
    value: dict with 'kernel_code' and 'run_fn'
    """
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return {}

    kernels = {}

    # ---- Elementwise operators ----

    @triton.jit
    def _tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = tl.math.tanh(x)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = tl.where(x > 0, x, 0.0)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = tl.sigmoid(x)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _leaky_relu_kernel(x_ptr, out_ptr, n_elements,
                           negative_slope: tl.constexpr,
                           BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = tl.where(x > 0, x, x * negative_slope)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        out = x * cdf
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _selu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        out = tl.where(x > 0, scale * x, scale * alpha * (tl.math.exp(x) - 1.0))
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _elu_kernel(x_ptr, out_ptr, n_elements,
                    alpha: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = tl.where(x > 0, x, alpha * (tl.math.exp(x) - 1.0))
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _swish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = x * tl.sigmoid(x)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _softplus_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # softplus(x) = log(1 + exp(x)), numerically stable version
        out = tl.where(x > 20.0, x, tl.math.log(1.0 + tl.math.exp(x)))
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _hard_sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = tl.minimum(tl.maximum(x / 6.0 + 0.5, 0.0), 1.0)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _hard_tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = tl.minimum(tl.maximum(x, -1.0), 1.0)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _softsign_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = x / (1.0 + tl.abs(x))
        tl.store(out_ptr + offsets, out, mask=mask)

    def _make_elementwise_runner(kernel, extra_kwargs=None):
        """Create a run function for elementwise kernels"""
        import torch
        def run_fn(inputs):
            x = inputs[0]
            n_elements = x.numel()
            output = torch.empty_like(x)
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            kwargs = {'BLOCK_SIZE': BLOCK_SIZE}
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            kernel[grid](x, output, n_elements, **kwargs)
            return output
        return run_fn

    # Register elementwise kernels
    kernels['tanh'] = {'kernel': _tanh_kernel, 'run': _make_elementwise_runner(_tanh_kernel)}
    kernels['relu'] = {'kernel': _relu_kernel, 'run': _make_elementwise_runner(_relu_kernel)}
    kernels['sigmoid'] = {'kernel': _sigmoid_kernel, 'run': _make_elementwise_runner(_sigmoid_kernel)}
    kernels['leaky_relu'] = {'kernel': _leaky_relu_kernel, 'run': _make_elementwise_runner(_leaky_relu_kernel, {'negative_slope': 0.01})}
    kernels['gelu'] = {'kernel': _gelu_kernel, 'run': _make_elementwise_runner(_gelu_kernel)}
    kernels['selu'] = {'kernel': _selu_kernel, 'run': _make_elementwise_runner(_selu_kernel)}
    kernels['elu'] = {'kernel': _elu_kernel, 'run': _make_elementwise_runner(_elu_kernel, {'alpha': 1.0})}
    kernels['swish'] = {'kernel': _swish_kernel, 'run': _make_elementwise_runner(_swish_kernel)}
    kernels['softplus'] = {'kernel': _softplus_kernel, 'run': _make_elementwise_runner(_softplus_kernel)}
    kernels['hard_sigmoid'] = {'kernel': _hard_sigmoid_kernel, 'run': _make_elementwise_runner(_hard_sigmoid_kernel)}
    kernels['hard_tanh'] = {'kernel': _hard_tanh_kernel, 'run': _make_elementwise_runner(_hard_tanh_kernel)}
    kernels['softsign'] = {'kernel': _softsign_kernel, 'run': _make_elementwise_runner(_softsign_kernel)}
    kernels['new_gelu'] = {'kernel': _gelu_kernel, 'run': _make_elementwise_runner(_gelu_kernel)}  # MinGPT NewGelu

    return kernels


def _detect_op_from_pytorch_file(file_path: str) -> Optional[str]:
    """
    Infer operator type from PyTorch operator filename or content
    Returns the lowercase operator name
    """
    basename = os.path.basename(file_path).lower()

    # Filename pattern matching
    op_patterns = {
        'tanh': ['tanh'],
        'relu': ['relu', '_relu'],
        'leaky_relu': ['leakyrelu', 'leaky_relu'],
        'sigmoid': ['sigmoid'],
        'gelu': ['gelu', 'newgelu'],
        'new_gelu': ['mingptnewgelu', 'newgelu'],
        'selu': ['selu'],
        'elu': ['elu'],
        'swish': ['swish'],
        'softplus': ['softplus'],
        'hard_sigmoid': ['hardsigmoid', 'hard_sigmoid'],
        'hard_tanh': ['hardtanh', 'hard_tanh'],
        'softsign': ['softsign'],
    }

    for op_name, patterns in op_patterns.items():
        for pattern in patterns:
            if pattern in basename.replace('_', '').replace('.py', ''):
                return op_name

    return None


# ============================================================================
# Mode 1: Run native Triton kernel files
# ============================================================================

def run_triton_file(
    file_path: str,
    warmup: int = 3,
    iterations: int = 10,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Run a native Triton kernel file

    The file must provide:
      - triton_kernel or kernel: @triton.jit function
      - get_inputs(): returns input list
      - get_grid(): returns grid configuration
      - get_init_args(): returns additional kernel arguments (optional)
      - run(inputs): custom run function (optional, if provided the above interfaces are ignored)
    """
    import torch

    if verbose:
        print(f"=" * 60)
        print(f"[Triton] Loading kernel from: {file_path}")

    module = load_module_from_file(file_path)

    # Get inputs
    inputs = module.get_inputs()

    # Check for custom run function
    has_custom_run = hasattr(module, 'run')

    if has_custom_run:
        run_fn = module.run
        if verbose:
            print(f"Mode: Custom run function")
    else:
        # Get kernel
        kernel = getattr(module, 'triton_kernel', None) or getattr(module, 'kernel', None)
        if kernel is None:
            raise ValueError(f"Triton kernel file must define 'triton_kernel' or 'kernel' or 'run': {file_path}")

        grid = module.get_grid()
        extra_args = module.get_init_args() if hasattr(module, 'get_init_args') else {}

        def run_fn(inputs):
            output = torch.empty_like(inputs[0])
            kernel[grid](*inputs, output, inputs[0].numel(), **extra_args)
            return output

        if verbose:
            print(f"Mode: Triton kernel with grid")

    if verbose:
        if inputs and isinstance(inputs[0], torch.Tensor):
            print(f"Input shape: {inputs[0].shape}, dtype: {inputs[0].dtype}")

    # Warmup
    if verbose:
        print(f"\n--- Warmup ({warmup} iterations) ---")

    for i in range(warmup):
        _ = run_fn(inputs)
        torch.cuda.synchronize()
        if verbose:
            print(f"  Warmup {i+1}/{warmup}")

    # Profile
    if verbose:
        print(f"\n--- Profiling ({iterations} iterations) ---")

    start_time = time.perf_counter()
    for i in range(iterations):
        _ = run_fn(inputs)
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / iterations

    if verbose:
        print(f"\n--- Results ---")
        print(f"Total time: {total_time*1000:.2f} ms")
        print(f"Average time per iteration: {avg_time*1000:.2f} ms")
        print(f"Throughput: {iterations/total_time:.2f} iter/s")

    return total_time, avg_time


# ============================================================================
# Mode 2: Auto-convert PyTorch operator files to Triton kernels
# ============================================================================

def run_pytorch_as_triton(
    file_path: str,
    warmup: int = 3,
    iterations: int = 10,
    verbose: bool = True,
    fallback_pytorch: bool = True
) -> Tuple[float, float, str]:
    """
    Read a PyTorch operator file and auto-match a built-in Triton kernel to run

    Args:
        file_path: Path to the PyTorch operator file
        warmup: Number of warmup iterations
        iterations: Number of profiling iterations
        verbose: Whether to print detailed information
        fallback_pytorch: Whether to fall back to PyTorch if no matching Triton kernel exists

    Returns:
        (total_time, avg_time, mode): mode is "triton" or "pytorch_fallback"
    """
    import torch

    if verbose:
        print(f"=" * 60)
        print(f"[Triton] Loading PyTorch op: {file_path}")

    # 1. Load PyTorch operator file
    module = load_module_from_file(file_path)

    # 2. Get inputs
    inputs = module.get_inputs()
    inputs = [x.cuda() if isinstance(x, torch.Tensor) and not x.is_cuda else x for x in inputs]

    # 3. Detect operator type
    op_name = _detect_op_from_pytorch_file(file_path)

    # 4. Find corresponding Triton kernel
    builtin_kernels = _get_builtin_triton_kernels()

    mode = "triton"
    run_fn = None

    if op_name and op_name in builtin_kernels:
        triton_info = builtin_kernels[op_name]
        run_fn = triton_info['run']
        if verbose:
            print(f"Matched Triton kernel: {op_name}")
            print(f"Mode: Triton (built-in)")
    elif fallback_pytorch:
        # Fall back to PyTorch
        mode = "pytorch_fallback"
        init_inputs = module.get_init_inputs()
        model = module.Model(*init_inputs)
        model = model.cuda().eval()

        def run_fn(inputs):
            with torch.no_grad():
                return model(*inputs)

        if verbose:
            op_hint = op_name if op_name else "unknown"
            print(f"No Triton kernel for op '{op_hint}', falling back to PyTorch")
            print(f"Mode: PyTorch fallback")
    else:
        raise ValueError(
            f"No built-in Triton kernel for op detected from '{file_path}'. "
            f"Detected op: {op_name}. Available: {list(builtin_kernels.keys())}"
        )

    if verbose:
        if inputs and isinstance(inputs[0], torch.Tensor):
            print(f"Input shape: {inputs[0].shape}, dtype: {inputs[0].dtype}")
            print(f"Input numel: {inputs[0].numel():,}")

    # Warmup
    if verbose:
        print(f"\n--- Warmup ({warmup} iterations) ---")

    for i in range(warmup):
        _ = run_fn(inputs)
        torch.cuda.synchronize()
        if verbose:
            print(f"  Warmup {i+1}/{warmup}")

    # Profile
    if verbose:
        print(f"\n--- Profiling ({iterations} iterations) ---")

    start_time = time.perf_counter()
    for i in range(iterations):
        _ = run_fn(inputs)
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / iterations

    if verbose:
        print(f"\n--- Results ({mode}) ---")
        print(f"Total time: {total_time*1000:.2f} ms")
        print(f"Average time per iteration: {avg_time*1000:.2f} ms")
        print(f"Throughput: {iterations/total_time:.2f} iter/s")

    return total_time, avg_time, mode


# ============================================================================
# Batch execution
# ============================================================================

def run_pytorch_dir_as_triton(
    directory: str,
    warmup: int = 3,
    iterations: int = 10,
    verbose: bool = True,
    fallback_pytorch: bool = True,
    exclude_patterns: Optional[List[str]] = None
) -> dict:
    """
    Run all PyTorch operator files in a directory, auto-matching Triton kernels

    Returns:
        {filename: (total_time, avg_time, mode)} dictionary
    """
    import torch

    if exclude_patterns is None:
        exclude_patterns = ['kernel_runner', 'triton_runner', 'run_', '__pycache__', 'test_']

    model_files = sorted([
        f for f in os.listdir(directory)
        if f.endswith('.py') and not any(p in f for p in exclude_patterns)
    ])

    if verbose:
        print(f"Found {len(model_files)} operator files in {directory}")
        print(f"PyTorch version: {torch.__version__}")
        try:
            import triton
            print(f"Triton version: {triton.__version__}")
        except ImportError:
            print("Triton: NOT INSTALLED (will fallback to PyTorch)")
        print(f"HIP available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device count: {torch.cuda.device_count()}")
        print()

    results = {}
    triton_count = 0
    fallback_count = 0
    error_count = 0

    for model_file in model_files:
        file_path = os.path.join(directory, model_file)
        try:
            total_time, avg_time, mode = run_pytorch_as_triton(
                file_path, warmup, iterations, verbose, fallback_pytorch
            )
            results[model_file] = (total_time, avg_time, mode)
            if mode == "triton":
                triton_count += 1
            else:
                fallback_count += 1
        except Exception as e:
            print(f"Error running {model_file}: {e}")
            results[model_file] = (None, None, "error")
            error_count += 1
            continue

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Batch Summary: {triton_count} Triton, {fallback_count} PyTorch fallback, {error_count} errors")

    return results


def print_summary(results: dict):
    """Print results summary"""
    print("\n" + "=" * 90)
    print("TRITON RUNNER SUMMARY")
    print("=" * 90)
    print(f"{'Model':<45} {'Avg (ms)':<12} {'Mode':<20} {'Status'}")
    print("-" * 90)

    for model, values in sorted(results.items()):
        if len(values) == 3:
            total, avg, mode = values
        else:
            total, avg = values
            mode = "unknown"

        if avg is not None:
            mode_icon = "Triton" if mode == "triton" else "PyTorch"
            print(f"{model:<45} {avg*1000:<12.2f} {mode_icon:<20} OK")
        else:
            print(f"{model:<45} {'N/A':<12} {'Error':<20} FAILED")

    total_count = len(results)
    triton_count = sum(1 for v in results.values() if len(v) >= 3 and v[2] == "triton")
    fallback_count = sum(1 for v in results.values() if len(v) >= 3 and v[2] == "pytorch_fallback")
    error_count = sum(1 for v in results.values() if (len(v) >= 3 and v[2] == "error") or v[0] is None)

    print("-" * 90)
    print(f"Total: {total_count} operators | "
          f"Triton: {triton_count} | "
          f"PyTorch fallback: {fallback_count} | "
          f"Error: {error_count}")
    print("=" * 90)


def list_supported_ops():
    """List all supported built-in Triton operators"""
    print("\nSupported Built-in Triton Kernels:")
    print("=" * 50)

    try:
        kernels = _get_builtin_triton_kernels()
        for i, op_name in enumerate(sorted(kernels.keys()), 1):
            print(f"  {i:2d}. {op_name}")
        print(f"\nTotal: {len(kernels)} kernels")
    except ImportError:
        print("  Triton not installed. Install with:")
        print("     pip install triton")

    print("\nFor ops without built-in Triton kernels, PyTorch fallback is used.")
    print("=" * 50)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Triton kernels for ROCm profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a native Triton kernel file
  python triton_runner.py --file my_triton_kernel.py

  # Run a PyTorch op file with auto Triton conversion
  python triton_runner.py --pytorch-file 22_Tanh.py

  # Run all ops in a directory
  python triton_runner.py --pytorch-dir ./kernel_test/KernelBench/level1 --summary

  # List supported built-in Triton kernels
  python triton_runner.py --list-ops

  # Use with rocprof-sys
  rocprof-sys-run --trace --use-rocm --rocm-domains kernel_dispatch \\
      -o ./results -- python triton_runner.py --pytorch-file 22_Tanh.py

  # Compare PyTorch vs Triton (disable fallback)
  python triton_runner.py --pytorch-file 22_Tanh.py --no-fallback
        """
    )

    parser.add_argument(
        "--file", "-f", type=str,
        help="Native Triton kernel file to run"
    )
    parser.add_argument(
        "--pytorch-file", "-pf", type=str,
        help="PyTorch op file to run with Triton (auto-convert)"
    )
    parser.add_argument(
        "--pytorch-dir", "-pd", type=str,
        help="Directory of PyTorch op files to run with Triton"
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=3,
        help="Number of warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=10,
        help="Number of profile iterations (default: 10)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--summary", "-s", action="store_true",
        help="Print summary table at the end"
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Disable PyTorch fallback for unsupported ops"
    )
    parser.add_argument(
        "--list-ops", action="store_true",
        help="List all supported built-in Triton ops"
    )

    args = parser.parse_args()

    if args.list_ops:
        list_supported_ops()
        return

    if not args.file and not args.pytorch_file and not args.pytorch_dir:
        parser.print_help()
        print("\nError: Please specify --file, --pytorch-file, or --pytorch-dir")
        sys.exit(1)

    verbose = not args.quiet
    fallback = not args.no_fallback

    if args.file:
        # Mode 1: Native Triton kernel file
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        run_triton_file(args.file, args.warmup, args.iterations, verbose)

    elif args.pytorch_file:
        # Mode 2: PyTorch operator -> Triton
        if not os.path.exists(args.pytorch_file):
            print(f"Error: File not found: {args.pytorch_file}")
            sys.exit(1)
        run_pytorch_as_triton(
            args.pytorch_file, args.warmup, args.iterations, verbose, fallback
        )

    elif args.pytorch_dir:
        # Mode 3: PyTorch operator directory -> Triton batch
        if not os.path.isdir(args.pytorch_dir):
            print(f"Error: Directory not found: {args.pytorch_dir}")
            sys.exit(1)
        results = run_pytorch_dir_as_triton(
            args.pytorch_dir, args.warmup, args.iterations, verbose, fallback
        )
        if args.summary:
            print_summary(results)


if __name__ == "__main__":
    main()
