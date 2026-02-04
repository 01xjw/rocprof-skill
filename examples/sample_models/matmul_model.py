"""
示例模型: 矩阵乘法
用于演示 rocprof-skill 的使用
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)


# 模型参数
N = 4096


def get_inputs():
    """返回模型输入"""
    A = torch.rand(N, N, dtype=torch.float16)
    B = torch.rand(N, N, dtype=torch.float16)
    return [A, B]


def get_init_inputs():
    """返回模型初始化参数"""
    return []  # No special initialization inputs needed
