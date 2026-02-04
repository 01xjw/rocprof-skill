#!/usr/bin/env python3
"""
ROCm Kernel Profiling Runner
用于运行 PyTorch 模型并使 rocprof-sys 能够捕获 HIP kernel 信息

使用方法:
    python kernel_runner.py --file model.py --warmup 3 --iterations 10
    python kernel_runner.py --dir ./models --warmup 3 --iterations 10
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Any


def load_module_from_file(file_path: str):
    """动态加载 Python 模块"""
    spec = importlib.util.spec_from_file_location("model_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_model(
    file_path: str,
    device: str = "cuda",
    warmup: int = 3,
    iterations: int = 10,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    运行指定的模型文件并返回性能指标
    
    Args:
        file_path: 模型文件路径
        device: 运行设备 (cuda/cpu)
        warmup: 预热迭代次数
        iterations: 正式运行迭代次数
        verbose: 是否打印详细信息
    
    Returns:
        (total_time, avg_time): 总时间和平均时间（秒）
    """
    import torch
    
    if verbose:
        print(f"=" * 60)
        print(f"Loading model from: {file_path}")
    
    # 加载模块
    module = load_module_from_file(file_path)
    
    # 获取初始化参数并创建模型
    init_inputs = module.get_init_inputs()
    model = module.Model(*init_inputs)
    
    if device == "cuda":
        model = model.cuda()
    model.eval()
    
    # 获取输入并移动到设备
    inputs = module.get_inputs()
    if device == "cuda":
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
    
    if verbose:
        print(f"Model: {type(model).__name__}")
        print(f"Device: {device}")
        if inputs and isinstance(inputs[0], torch.Tensor):
            print(f"Input shape: {inputs[0].shape}, dtype: {inputs[0].dtype}")
    
    # Warmup
    if verbose:
        print(f"\n--- Warmup ({warmup} iterations) ---")
    
    with torch.no_grad():
        for i in range(warmup):
            _ = model(*inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            if verbose:
                print(f"  Warmup {i+1}/{warmup}")
    
    # Profile iterations
    if verbose:
        print(f"\n--- Profiling ({iterations} iterations) ---")
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for i in range(iterations):
            _ = model(*inputs)
            if device == "cuda":
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


def run_all_models(
    directory: str,
    device: str = "cuda",
    warmup: int = 3,
    iterations: int = 10,
    verbose: bool = True,
    exclude_patterns: Optional[List[str]] = None
) -> dict:
    """
    运行目录下所有模型文件
    
    Args:
        directory: 模型文件目录
        device: 运行设备
        warmup: 预热迭代次数
        iterations: 正式运行迭代次数
        verbose: 是否打印详细信息
        exclude_patterns: 要排除的文件名模式列表
    
    Returns:
        results: {filename: (total_time, avg_time)} 字典
    """
    import torch
    
    if exclude_patterns is None:
        exclude_patterns = ['kernel_runner.py', 'run_', '__pycache__', 'test_']
    
    model_files = sorted([
        f for f in os.listdir(directory) 
        if f.endswith('.py') and not any(p in f for p in exclude_patterns)
    ])
    
    if verbose:
        print(f"Found {len(model_files)} model files in {directory}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"HIP available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device count: {torch.cuda.device_count()}")
    
    results = {}
    
    for model_file in model_files:
        file_path = os.path.join(directory, model_file)
        try:
            total_time, avg_time = run_model(
                file_path, device, warmup, iterations, verbose
            )
            results[model_file] = (total_time, avg_time)
        except Exception as e:
            print(f"Error running {model_file}: {e}")
            results[model_file] = (None, None)
            continue
    
    return results


def print_summary(results: dict):
    """打印结果摘要"""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<50} {'Avg Time (ms)':<15} {'Status'}")
    print("-" * 80)
    
    for model, (total, avg) in sorted(results.items()):
        if avg is not None:
            print(f"{model:<50} {avg*1000:<15.2f} OK")
        else:
            print(f"{model:<50} {'N/A':<15} FAILED")
    
    successful = sum(1 for _, (t, _) in results.items() if t is not None)
    print("-" * 80)
    print(f"Total: {len(results)} models, {successful} successful, {len(results)-successful} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Run PyTorch models for ROCm kernel profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile a single model
  python kernel_runner.py --file 22_Tanh.py
  
  # Profile all models in a directory
  python kernel_runner.py --dir ./models --warmup 5 --iterations 20
  
  # Use with rocprof-sys
  rocprof-sys-run --trace --use-rocm --rocm-domains kernel_dispatch \\
      -o ./results -- python kernel_runner.py --file 22_Tanh.py
        """
    )
    
    parser.add_argument(
        "--file", "-f", type=str,
        help="Single model file to run"
    )
    parser.add_argument(
        "--dir", "-d", type=str,
        help="Directory containing model files to run"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
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
    
    args = parser.parse_args()
    
    if not args.file and not args.dir:
        parser.print_help()
        print("\nError: Please specify --file or --dir")
        sys.exit(1)
    
    verbose = not args.quiet
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        run_model(args.file, args.device, args.warmup, args.iterations, verbose)
    
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            sys.exit(1)
        results = run_all_models(
            args.dir, args.device, args.warmup, args.iterations, verbose
        )
        if args.summary:
            print_summary(results)


if __name__ == "__main__":
    main()
