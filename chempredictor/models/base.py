"""
模型基类和注册系统
"""

from abc import ABC, abstractmethod
import logging
import os
import numpy as np
from typing import Dict, Any, Type, List, Optional, Union, Tuple

# 模型注册表
MODEL_REGISTRY: Dict[str, Type["BaseModel"]] = {}

def register_model(name: str):
    """
    模型注册装饰器
    
    参数:
        name: 模型名称
        
    返回:
        装饰器函数
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            logging.getLogger(__name__).warning(f"模型 '{name}' 已存在，将被覆盖")
        MODEL_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator

class BaseModel(ABC):
    """
    模型基类
    
    所有预测模型必须继承此类并实现其抽象方法
    """
    
    def __init__(self, task_type: str = "regression", **kwargs):
        """
        初始化模型
        
        参数:
            task_type: 任务类型，'regression'或'classification'
            **kwargs: 模型参数
        """
        self.task_type = task_type
        self.params = kwargs
        self.is_fitted = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.feature_importances_ = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            
        返回:
            训练后的模型实例
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数:
            X: 特征矩阵
            
        返回:
            预测结果
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（仅适用于分类任务）
        
        参数:
            X: 特征矩阵
            
        返回:
            预测概率
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba仅适用于分类任务")
        raise NotImplementedError("子类必须实现predict_proba方法")
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        获取特征重要性
        
        返回:
            特征重要性数组
        """
        return self.feature_importances_
    
    def save(self, path: str) -> None:
        """
        保存模型到文件
        
        参数:
            path: 保存路径
        """
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """
        从文件加载模型
        
        参数:
            path: 文件路径
            
        返回:
            加载的模型实例
        """
        import joblib
        return joblib.load(path)

def get_model(name: str, **params) -> BaseModel:
    """
    获取模型实例
    
    参数:
        name: 模型名称
        **params: 模型参数
        
    返回:
        模型实例
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知的模型: {name}，可用的模型: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[name](**params) 