from dataclasses import dataclass
from typing import List, Union, Optional

@dataclass
class TaskConfig:
    """任务配置类"""
    task_type: str  # 'regression' 或 'classification'
    target_column: str  # 目标列名
    num_classes: Optional[int] = None  # 分类任务的类别数
    class_weights: Optional[List[float]] = None  # 分类任务的类别权重
    loss_function: str = 'mse'  # 损失函数类型
    metrics: List[str] = None  # 评估指标
    output_activation: str = None  # 输出层激活函数
    
    def __post_init__(self):
        if self.metrics is None:
            if self.task_type == 'regression':
                self.metrics = ['mse', 'mae', 'r2']
            else:
                self.metrics = ['accuracy', 'f1']
        
        if self.output_activation is None:
            if self.task_type == 'regression':
                self.output_activation = 'sigmoid'
            else:
                self.output_activation = 'softmax' 