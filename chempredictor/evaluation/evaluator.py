"""
评估器模块 - 提供模型评估功能
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union, Callable

from chempredictor.evaluation.metrics import get_metrics

# 检查SHAP是否可用
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.getLogger(__name__).warning("SHAP未安装，特征重要性分析功能将不可用")

class Evaluator:
    """
    评估器
    
    用于评估模型性能和分析特征重要性
    """
    
    def __init__(self, task_type: str, metrics: Optional[List[str]] = None,
                 feature_importance: bool = True, shap_analysis: bool = False):
        """
        初始化评估器
        
        参数:
            task_type: 任务类型，'regression'或'classification'
            metrics: 评估指标列表，如果为None则使用默认指标
            feature_importance: 是否计算特征重要性
            shap_analysis: 是否进行SHAP分析
        """
        self.task_type = task_type
        self.metrics = get_metrics(task_type, metrics)
        self.feature_importance = feature_importance
        self.shap_analysis = shap_analysis
        
        if shap_analysis and not SHAP_AVAILABLE:
            logging.getLogger(__name__).warning("SHAP未安装，无法进行SHAP分析")
            self.shap_analysis = False
            
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, model: Any, X: np.ndarray, y: np.ndarray, 
                 feature_names: Optional[List[str]] = None,
                 y_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        评估模型性能
        
        参数:
            model: 模型实例
            X: 特征矩阵
            y: 真实目标值
            feature_names: 特征名称列表
            y_train: 训练集目标值，用于计算Q²
            
        返回:
            包含评估结果的字典
        """
        self.logger.info(f"评估{model.__class__.__name__}模型")
        
        # 预测
        if self.task_type == 'regression':
            y_pred = model.predict(X)
            y_score = None
        else:  # classification
            y_pred = model.predict(X)
            try:
                y_score = model.predict_proba(X)
            except (AttributeError, NotImplementedError):
                self.logger.warning("模型不支持predict_proba方法，无法计算概率")
                y_score = None
        
        # 计算指标
        results = {}
        for name, metric_fn in self.metrics.items():
            try:
                if name == 'roc_auc' and y_score is not None:
                    # 对于ROC AUC，使用预测概率
                    if y_score.shape[1] == 2:
                        # 二分类问题，只需要正类的概率
                        score = metric_fn(y, y_score[:, 1])
                    else:
                        # 多分类问题
                        score = metric_fn(y, y_score)
                elif name == 'q2' and y_train is not None:
                    # 对于Q²，需要训练集目标值
                    score = metric_fn(y, y_pred, y_train)
                else:
                    # 其他指标
                    score = metric_fn(y, y_pred)
                
                results[name] = score
                self.logger.info(f"{name}: {score:.4f}")
            except Exception as e:
                self.logger.warning(f"计算{name}指标时出错: {e}")
        
        # 计算特征重要性
        if self.feature_importance:
            try:
                importances = self._get_feature_importance(model, X, feature_names)
                results['feature_importance'] = importances
            except Exception as e:
                self.logger.warning(f"计算特征重要性时出错: {e}")
        
        # 进行SHAP分析
        if self.shap_analysis:
            try:
                shap_values = self._get_shap_values(model, X, feature_names)
                results['shap_values'] = shap_values
            except Exception as e:
                self.logger.warning(f"进行SHAP分析时出错: {e}")
        
        return results
    
    def _get_feature_importance(self, model: Any, X: np.ndarray, 
                               feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        获取特征重要性
        
        参数:
            model: 模型实例
            X: 特征矩阵
            feature_names: 特征名称列表
            
        返回:
            特征名称到重要性的映射
        """
        # 尝试从模型获取特征重要性
        if hasattr(model, 'get_feature_importances'):
            importances = model.get_feature_importances()
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            self.logger.warning("模型不支持特征重要性计算")
            return {}
        
        # 如果没有提供特征名称，使用索引
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # 确保特征名称和重要性长度一致
        if len(feature_names) != len(importances):
            self.logger.warning(f"特征名称长度({len(feature_names)})与特征重要性长度({len(importances)})不一致")
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # 创建特征重要性字典
        importance_dict = dict(zip(feature_names, importances))
        
        # 按重要性降序排序
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def _get_shap_values(self, model: Any, X: np.ndarray, 
                        feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取SHAP值
        
        参数:
            model: 模型实例
            X: 特征矩阵
            feature_names: 特征名称列表
            
        返回:
            包含SHAP分析结果的字典
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP未安装，无法进行SHAP分析")
        
        # 创建SHAP解释器
        try:
            # 尝试使用TreeExplainer（适用于基于树的模型）
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # 对于XGBoost多分类模型，shap_values是一个列表
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # 使用平均绝对值作为总体重要性
                shap_values_abs = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                shap_values_abs = np.abs(shap_values).mean(axis=0)
        except Exception:
            # 如果TreeExplainer不适用，尝试使用KernelExplainer
            try:
                # 使用数据子集以提高效率
                background = shap.kmeans(X, 10)
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X[:100])  # 使用前100个样本
                shap_values_abs = np.abs(shap_values).mean(axis=0)
            except Exception as e:
                self.logger.warning(f"SHAP分析失败: {e}")
                return {}
        
        # 如果没有提供特征名称，使用索引
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(shap_values_abs))]
        
        # 确保特征名称和SHAP值长度一致
        if len(feature_names) != len(shap_values_abs):
            self.logger.warning(f"特征名称长度({len(feature_names)})与SHAP值长度({len(shap_values_abs)})不一致")
            feature_names = [f"feature_{i}" for i in range(len(shap_values_abs))]
        
        # 创建SHAP值字典
        shap_dict = dict(zip(feature_names, shap_values_abs))
        
        # 按重要性降序排序
        return dict(sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)) 