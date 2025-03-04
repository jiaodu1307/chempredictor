"""
传统机器学习模型模块
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging

from chempredictor.models.base import BaseModel, register_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler

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
            "random_state": 42,
            "verbose": True  # 启用详细输出
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
        
        # 初始化训练历史记录
        self.history = {
            'loss': [],
            'val_loss': [],
            'n_iter': 0,
            'best_loss': float('inf'),
            'best_iter': 0
        }
        
    def _log_progress(self, model):
        """记录训练进度"""
        if not hasattr(model, 'loss_curve_'):
            return
        
        current_loss = model.loss_curve_[-1]
        self.history['loss'].append(current_loss)
        
        if hasattr(model, 'validation_scores_'):
            val_loss = -model.validation_scores_[-1]  # 验证分数是负的损失
            self.history['val_loss'].append(val_loss)
            
            if val_loss < self.history['best_loss']:
                self.history['best_loss'] = val_loss
                self.history['best_iter'] = len(self.history['loss'])
        
        self.history['n_iter'] = len(self.history['loss'])
        
        # 打印训练进度
        if self.history['n_iter'] % 10 == 0:  # 每10次迭代打印一次
            msg = f"迭代 {self.history['n_iter']}: loss={current_loss:.4f}"
            if hasattr(model, 'validation_scores_'):
                msg += f", val_loss={val_loss:.4f}"
            if self.history['best_iter'] == self.history['n_iter']:
                msg += " (最佳模型)"
            self.logger.info(msg)
        
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
        self.logger.info(f"网络结构: 输入层({X.shape[1]}) -> " + 
                        " -> ".join(str(size) for size in self.params['hidden_layer_sizes']) +
                        f" -> 输出层({1 if self.task_type == 'regression' else len(np.unique(y))})")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 重置训练历史
        self.history = {
            'loss': [],
            'val_loss': [],
            'n_iter': 0,
            'best_loss': float('inf'),
            'best_iter': 0
        }
        
        # 训练模型
        self.model.fit(X_scaled, y)
        
        # 记录最终训练状态
        self._log_progress(self.model)
        
        # 打印训练总结
        self.logger.info(f"\n训练完成:")
        self.logger.info(f"- 总迭代次数: {self.history['n_iter']}")
        self.logger.info(f"- 最终损失: {self.history['loss'][-1]:.4f}")
        if self.history['val_loss']:
            self.logger.info(f"- 最佳验证损失: {self.history['best_loss']:.4f} (迭代 {self.history['best_iter']})")
        
        # 设置状态
        self.is_fitted = True
        
        return self
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        获取训练历史记录
        
        返回:
            包含训练历史的字典
        """
        return self.history
    
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