"""
Sample: Triton Tanh Kernel
Demonstrates the native Triton kernel file interface for triton_runner.py

Usage:
    python scripts/triton_runner.py --file examples/sample_triton/tanh_triton.py

    # With rocprof-sys
    rocprof-sys-run --trace --use-rocm --rocm-domains kernel_dispatch \
        -o ./results -- python scripts/triton_runner.py --file examples/sample_triton/tanh_triton.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def triton_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton tanh kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.math.tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)


# Parameters
batch_size = 4096
dim = 393216
n_elements = batch_size * dim
BLOCK_SIZE = 1024


def get_inputs():
    """Returns input tensor list (must be on GPU)"""
    x = torch.randn(n_elements, device='cuda', dtype=torch.float16)
    return [x]


def get_grid():
    """Returns grid configuration"""
    return (triton.cdiv(n_elements, BLOCK_SIZE),)


def get_init_args():
    """Returns additional kernel arguments"""
    return {'BLOCK_SIZE': BLOCK_SIZE}


def run(inputs):
    """
    Custom run function (optional)
    If run() is provided, triton_runner will call it directly, ignoring get_grid/get_init_args
    """
    x = inputs[0]
    output = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    triton_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
