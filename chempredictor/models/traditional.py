"""
传统机器学习模型模块
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging

from chempredictor.models.base import BaseModel, register_model

# 检查scikit-learn是否可用
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.getLogger(__name__).warning("scikit-learn未安装，传统机器学习模型将不可用")

# 检查XGBoost是否可用
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.getLogger(__name__).warning("XGBoost未安装，XGBoost模型将不可用")

# 检查LightGBM是否可用
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.getLogger(__name__).warning("LightGBM未安装，LightGBM模型将不可用")

@register_model("random_forest")
class RandomForestModel(BaseModel):
    """
    随机森林模型
    
    支持回归和分类任务
    """
    
    def __init__(self, task_type: str = "regression", n_estimators: int = 100, 
                 max_depth: Optional[int] = None, random_state: int = 42, **kwargs):
        """
        初始化随机森林模型
        
        参数:
            task_type: 任务类型，'regression'或'classification'
            n_estimators: 树的数量
            max_depth: 树的最大深度
            random_state: 随机种子
            **kwargs: 其他参数
        """
        super().__init__(task_type=task_type, **kwargs)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn未安装，无法使用RandomForestModel")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # 根据任务类型创建模型
        if task_type == "regression":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs
            )
        elif task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs
            )
        else:
            raise ValueError(f"未知的任务类型: {task_type}，支持的类型: ['regression', 'classification']")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            
        返回:
            训练后的模型实例
        """
        self.logger.info(f"训练随机森林模型，特征形状: {X.shape}")
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importances_ = self.model.feature_importances_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数:
            X: 特征矩阵
            
        返回:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict(X)
    
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
        
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict_proba(X)

@register_model("xgboost")
class XGBoostModel(BaseModel):
    """
    XGBoost模型
    
    支持回归和分类任务
    """
    
    def __init__(self, task_type: str = "regression", n_estimators: int = 100, 
                 max_depth: int = 6, learning_rate: float = 0.1, 
                 objective: Optional[str] = None, random_state: int = 42, **kwargs):
        """
        初始化XGBoost模型
        
        参数:
            task_type: 任务类型，'regression'或'classification'
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            objective: 目标函数，如果为None则根据task_type自动选择
            random_state: 随机种子
            **kwargs: 其他参数
        """
        super().__init__(task_type=task_type, **kwargs)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost未安装，无法使用XGBoostModel")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # 如果未指定目标函数，根据任务类型自动选择
        if objective is None:
            if task_type == "regression":
                self.objective = "reg:squarederror"
            elif task_type == "classification":
                self.objective = "binary:logistic"
            else:
                raise ValueError(f"未知的任务类型: {task_type}，支持的类型: ['regression', 'classification']")
        else:
            self.objective = objective
        
        # 创建模型
        self.model = xgb.XGBRegressor if task_type == "regression" else xgb.XGBClassifier
        self.model = self.model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=self.objective,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            
        返回:
            训练后的模型实例
        """
        self.logger.info(f"训练XGBoost模型，特征形状: {X.shape}")
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importances_ = self.model.feature_importances_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数:
            X: 特征矩阵
            
        返回:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict(X)
    
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
        
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict_proba(X)

@register_model("lightgbm")
class LightGBMModel(BaseModel):
    """
    LightGBM模型
    
    支持回归和分类任务
    """
    
    def __init__(self, task_type: str = "regression", n_estimators: int = 100, 
                 max_depth: int = -1, learning_rate: float = 0.1, 
                 objective: Optional[str] = None, random_state: int = 42, **kwargs):
        """
        初始化LightGBM模型
        
        参数:
            task_type: 任务类型，'regression'或'classification'
            n_estimators: 树的数量
            max_depth: 树的最大深度，-1表示无限制
            learning_rate: 学习率
            objective: 目标函数，如果为None则根据task_type自动选择
            random_state: 随机种子
            **kwargs: 其他参数
        """
        super().__init__(task_type=task_type, **kwargs)
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM未安装，无法使用LightGBMModel")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # 如果未指定目标函数，根据任务类型自动选择
        if objective is None:
            if task_type == "regression":
                self.objective = "regression"
            elif task_type == "classification":
                self.objective = "binary"
            else:
                raise ValueError(f"未知的任务类型: {task_type}，支持的类型: ['regression', 'classification']")
        else:
            self.objective = objective
        
        # 创建模型
        self.model = lgb.LGBMRegressor if task_type == "regression" else lgb.LGBMClassifier
        self.model = self.model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=self.objective,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            
        返回:
            训练后的模型实例
        """
        self.logger.info(f"训练LightGBM模型，特征形状: {X.shape}")
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importances_ = self.model.feature_importances_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数:
            X: 特征矩阵
            
        返回:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict(X)
    
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
        
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict_proba(X) 