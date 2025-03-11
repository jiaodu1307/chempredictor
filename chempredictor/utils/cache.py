"""
缓存管理模块
"""

import os
import pickle
import hashlib
import logging
from typing import Any, Callable, Dict, Optional
from pathlib import Path
from functools import wraps

logger = logging.getLogger(__name__)

class Cache:
    """缓存管理类"""
    
    def __init__(self, cache_dir: str = '.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        # 将参数转换为字符串并计算哈希值
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.pkl"
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
                return None
        return None
        
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"写入缓存失败: {e}")
            
    def clear(self) -> None:
        """清除所有缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"删除缓存文件失败: {e}")
                
def cache_result(cache_dir: str = '.cache'):
    """结果缓存装饰器"""
    cache = Cache(cache_dir)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = cache._get_cache_key(
                func.__name__,
                *args,
                **kwargs
            )
            
            # 尝试从缓存获取结果
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"从缓存获取结果: {func.__name__}")
                return result
                
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
            
        return wrapper
    return decorator
    
class MemoryCache:
    """内存缓存类"""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache: Dict[str, Any] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        return self.cache.get(key)
        
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        # 如果缓存已满，删除最早的项
        if len(self.cache) >= self.maxsize:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[key] = value
        
    def clear(self) -> None:
        """清除所有缓存"""
        self.cache.clear()
        
def memory_cache(maxsize: int = 128):
    """内存缓存装饰器"""
    cache = MemoryCache(maxsize)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            # 尝试从缓存获取结果
            result = cache.get(cache_key)
            if result is not None:
                return result
                
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
            
        return wrapper
    return decorator 