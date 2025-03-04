"""
MLP模型模块 - 提供多层感知机模型
"""

import numpy as np
import logging
from typing import List, Optional, Union, Dict, Any

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler

from chempredictor.models.base import BaseModel, register_model

@register_model("mlp")
class MLPModel(BaseModel):
    """
    多层感知机模型
    
    基于scikit-learn的MLPRegressor和MLPClassifier实现
    """
    
    def __init__(self, task_type: str = "regression", **kwargs):
        """
        初始化MLP模型
        
        参数:
            task_type: 任务类型，'regression'或'classification'
            **kwargs: 传递给MLPRegressor或MLPClassifier的参数
        """
        super().__init__(task_type=task_type)
        self.logger = logging.getLogger(__name__)
        
        # 特征标准化器
        self.scaler = StandardScaler()
        
        # 设置默认参数
        default_params = {
            "hidden_layer_sizes": [100, 50],
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 1000,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
            "random_state": 42
        }
        
        # 更新参数
        self.params = default_params.copy()
        self.params.update(kwargs)
        
        # 将列表转换为元组（scikit-learn要求）
        if isinstance(self.params["hidden_layer_sizes"], list):
            self.params["hidden_layer_sizes"] = tuple(self.params["hidden_layer_sizes"])
        
        # 初始化模型
        if self.task_type == "regression":
            self.model = MLPRegressor(**self.params)
        else:
            self.model = MLPClassifier(**self.params)
            
        self.logger.info(f"初始化MLP模型，任务类型: {task_type}")
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPModel":
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 目标值
            
        返回:
            训练后的模型实例
        """
        self.logger.info("训练MLP模型")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model.fit(X_scaled, y)
        
        # 设置状态
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数:
            X: 特征矩阵
            
        返回:
            预测值
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 预测
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（仅用于分类任务）
        
        参数:
            X: 特征矩阵
            
        返回:
            预测概率
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba方法仅适用于分类任务")
        
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 预测概率
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        获取特征重要性
        
        返回:
            None，因为MLP模型不直接提供特征重要性
        """
        return None 