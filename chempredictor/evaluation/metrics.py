"""
评估指标模块 - 提供各种评估指标
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
import logging

# 检查scikit-learn是否可用
try:
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.getLogger(__name__).warning("scikit-learn未安装，评估指标功能将不可用")

# 回归指标
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    均方根误差
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        
    返回:
        RMSE值
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn未安装，无法计算RMSE")
    
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    平均绝对误差
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        
    返回:
        MAE值
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn未安装，无法计算MAE")
    
    return mean_absolute_error(y_true, y_pred)

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    决定系数
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        
    返回:
        R²值
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn未安装，无法计算R²")
    
    return r2_score(y_true, y_pred)

def q2(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """
    预测决定系数
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        y_train: 训练集目标值
        
    返回:
        Q²值
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn未安装，无法计算Q²")
    
    # 计算预测残差平方和
    press = np.sum((y_true - y_pred) ** 2)
    
    # 计算总平方和
    tss = np.sum((y_true - np.mean(y_train)) ** 2)
    
    return 1 - press / tss

# 分类指标
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    准确率
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        
    返回:
        准确率
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn未安装，无法计算准确率")
    
    return accuracy_score(y_true, y_pred)

def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """
    F1分数
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        average: 平均方式，'micro'、'macro'、'weighted'或'binary'
        
    返回:
        F1分数
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn未安装，无法计算F1分数")
    
    return f1_score(y_true, y_pred, average=average)

def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """
    精确率
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        average: 平均方式，'micro'、'macro'、'weighted'或'binary'
        
    返回:
        精确率
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn未安装，无法计算精确率")
    
    return precision_score(y_true, y_pred, average=average)

def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """
    召回率
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        average: 平均方式，'micro'、'macro'、'weighted'或'binary'
        
    返回:
        召回率
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn未安装，无法计算召回率")
    
    return recall_score(y_true, y_pred, average=average)

def roc_auc(y_true: np.ndarray, y_score: np.ndarray, average: str = 'weighted') -> float:
    """
    ROC曲线下面积
    
    参数:
        y_true: 真实值
        y_score: 预测概率或分数
        average: 平均方式，'micro'、'macro'、'weighted'或'binary'
        
    返回:
        AUC值
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn未安装，无法计算ROC AUC")
    
    # 对于二分类问题
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_score)
    
    # 对于多分类问题
    try:
        return roc_auc_score(y_true, y_score, average=average, multi_class='ovr')
    except ValueError:
        # 如果y_score不是概率形式，尝试使用OvO方法
        return roc_auc_score(y_true, y_score, average=average, multi_class='ovo')

# 指标注册表
REGRESSION_METRICS = {
    'rmse': rmse,
    'mae': mae,
    'r2': r2
}

CLASSIFICATION_METRICS = {
    'accuracy': accuracy,
    'f1': f1,
    'precision': precision,
    'recall': recall,
    'roc_auc': roc_auc
}

def get_metrics(task_type: str, metric_names: Optional[List[str]] = None) -> Dict[str, Callable]:
    """
    获取指定任务类型的评估指标
    
    参数:
        task_type: 任务类型，'regression'或'classification'
        metric_names: 指标名称列表，如果为None则返回所有指标
        
    返回:
        指标名称到函数的映射
    """
    if task_type == 'regression':
        metrics = REGRESSION_METRICS
    elif task_type == 'classification':
        metrics = CLASSIFICATION_METRICS
    else:
        raise ValueError(f"未知的任务类型: {task_type}，支持的类型: ['regression', 'classification']")
    
    if metric_names is None:
        return metrics
    
    # 过滤指定的指标
    return {name: metrics[name] for name in metric_names if name in metrics} 