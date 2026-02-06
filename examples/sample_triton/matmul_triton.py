"""
Sample: Triton Matrix Multiplication Kernel
Demonstrates the run() interface for triton_runner.py

Usage:
    python scripts/triton_runner.py --file examples/sample_triton/matmul_triton.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Triton matmul kernel"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K-dimension loop
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offs = k_start + offs_k

        # Load A and B tiles
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=a_mask, other=0.0
        )
        b = tl.load(
            b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=b_mask, other=0.0
        )

        acc += tl.dot(a, b)

    # Write back results
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.float16),
        mask=c_mask
    )


# Parameters
M = N = K = 2048
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32


def get_inputs():
    """Returns input tensor list"""
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    return [A, B]


def run(inputs):
    """Custom run function"""
    A, B = inputs
    C = torch.empty(M, N, device='cuda', dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C
