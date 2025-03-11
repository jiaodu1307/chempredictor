"""
性能监控模块
"""

import time
import functools
import logging
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

import torch
from memory_profiler import profile as memory_profile

logger = logging.getLogger(__name__)

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
        logger.info(f"{self.name} 执行时间: {duration:.4f} 秒")
        
def profile_memory(func: Callable) -> Callable:
    """内存使用量分析装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiled_func = memory_profile(func)
        return profiled_func(*args, **kwargs)
    return wrapper

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
def profile_section(name: str):
    """性能分析上下文管理器"""
    logger.info(f"开始执行 {name}")
    start_time = time.perf_counter()
    start_memory = profile_gpu_memory()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = profile_gpu_memory()
        
        duration = end_time - start_time
        logger.info(f"{name} 执行时间: {duration:.4f} 秒")
        
        if start_memory and end_memory:
            memory_diff = {
                k: end_memory[k] - start_memory[k]
                for k in start_memory
            }
            logger.info(f"{name} GPU内存变化: {memory_diff}")
            
def log_performance(func: Callable) -> Callable:
    """性能日志装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with profile_section(func.__name__):
            return func(*args, **kwargs)
    return wrapper 