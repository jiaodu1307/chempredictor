"""
性能监控模块
"""

import time
import functools
import logging
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

# 尝试导入memory_profiler，如果不可用则使用空装饰器
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    logger.warning("Memory profiler not installed, memory profiling will be disabled")
    MEMORY_PROFILER_AVAILABLE = False
    
    def memory_profile(func):
        return func

class Timer:
    """计时器类，用于测量代码块执行时间"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} execution time: {duration:.4f} seconds")
        
def profile_memory(func: Callable) -> Callable:
    """
    内存使用量分析装饰器
    
    如果memory_profiler可用，则使用其进行内存分析；
    否则返回原函数
    """
    if MEMORY_PROFILER_AVAILABLE:
        return memory_profile(func)
    return func

def profile_gpu_memory() -> Dict[str, Any]:
    """获取GPU内存使用情况"""
    if not torch.cuda.is_available():
        return {}
        
    return {
        'allocated': torch.cuda.memory_allocated(),
        'cached': torch.cuda.memory_reserved(),
        'max_allocated': torch.cuda.max_memory_allocated()
    }
    
@contextmanager
def profile_section(name: str, profile_memory: bool = False):
    """
    性能分析上下文管理器
    
    Args:
        name: 分析区块名称
        profile_memory: 是否进行内存分析（需要安装memory_profiler）
    """
    logger.info(f"Starting execution of {name}")
    start_time = time.perf_counter()
    start_memory = profile_gpu_memory() if profile_memory and MEMORY_PROFILER_AVAILABLE else None
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"{name} execution time: {duration:.4f} seconds")
        
        if profile_memory and MEMORY_PROFILER_AVAILABLE and start_memory:
            end_memory = profile_gpu_memory()
            memory_diff = {
                k: end_memory[k] - start_memory[k]
                for k in start_memory
            }
            logger.info(f"{name} GPU memory change: {memory_diff}")
            
def log_performance(func: Callable) -> Callable:
    """性能日志装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with profile_section(func.__name__):
            return func(*args, **kwargs)
    return wrapper 