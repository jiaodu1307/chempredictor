"""
评估模块 - 提供模型评估功能和各种评估指标
"""

# 导入回归指标
from chempredictor.evaluation.metrics import rmse, mae, r2, q2

# 导入分类指标
from chempredictor.evaluation.metrics import accuracy, f1, precision, recall, roc_auc

# 导入指标获取函数
from chempredictor.evaluation.metrics import get_metrics

# 导入评估器
from chempredictor.evaluation.evaluator import Evaluator

# 定义公开的API
__all__ = [
    # 回归指标
    'rmse', 'mae', 'r2', 'q2',
    
    # 分类指标
    'accuracy', 'f1', 'precision', 'recall', 'roc_auc',
    
    # 工具函数
    'get_metrics',
    
    # 评估器
    'Evaluator'
] 