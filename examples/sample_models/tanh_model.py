"""
示例模型: Tanh 激活函数
用于演示 rocprof-skill 的使用
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a Tanh activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return torch.tanh(x)


# 模型参数
batch_size = 4096
dim = 393216


def get_inputs():
    """返回模型输入"""
    x = torch.rand(batch_size, dim, dtype=torch.float16)
    return [x]


def get_init_inputs():
    """返回模型初始化参数"""
    return []  # No special initialization inputs needed
