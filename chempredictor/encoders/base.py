"""
编码器基类和注册系统
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Type, List, Optional, Union
import pandas as pd
import numpy as np

# 编码器注册表
ENCODER_REGISTRY: Dict[str, Type["BaseEncoder"]] = {}

def register_encoder(name: str):
    """
    编码器注册装饰器
    
    参数:
        name: 编码器名称
        
    返回:
        装饰器函数
    """
    def decorator(cls):
        if name in ENCODER_REGISTRY:
            logging.getLogger(__name__).warning(f"编码器 '{name}' 已存在，将被覆盖")
        ENCODER_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator

class BaseEncoder(ABC):
    """
    编码器基类
    
    所有特征编码器必须继承此类并实现其抽象方法
    """
    
    def __init__(self, **kwargs):
        """
        初始化编码器
        
        参数:
            **kwargs: 编码器参数
        """
        self.params = kwargs
        self.is_fitted = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def fit(self, data: Union[pd.Series, np.ndarray, List]) -> "BaseEncoder":
        """
        拟合编码器
        
        参数:
            data: 输入数据
            
        返回:
            拟合后的编码器实例
        """
        pass
    
    @abstractmethod
    def transform(self, data: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        转换数据
        
        参数:
            data: 输入数据
            
        返回:
            编码后的数据
        """
        pass
    
    def fit_transform(self, data: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        拟合并转换数据
        
        参数:
            data: 输入数据
            
        返回:
            编码后的数据
        """
        return self.fit(data).transform(data)
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        返回:
            特征名称列表
        """
        return [f"{self.__class__.__name__}_{i}" for i in range(self.get_output_dim())]
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        返回:
            输出特征的维度
        """
        raise NotImplementedError("子类必须实现get_output_dim方法")
    
    def save(self, path: str) -> None:
        """
        保存编码器到文件
        
        参数:
            path: 保存路径
        """
        import joblib
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str) -> "BaseEncoder":
        """
        从文件加载编码器
        
        参数:
            path: 文件路径
            
        返回:
            加载的编码器实例
        """
        import joblib
        return joblib.load(path)

def get_encoder(name: str, **params) -> BaseEncoder:
    """
    获取编码器实例
    
    参数:
        name: 编码器名称
        **params: 编码器参数
        
    返回:
        编码器实例
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"未知的编码器: {name}，可用的编码器: {list(ENCODER_REGISTRY.keys())}")
    
    return ENCODER_REGISTRY[name](**params) 