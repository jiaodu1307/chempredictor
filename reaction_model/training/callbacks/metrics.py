from pytorch_lightning.callbacks import Callback
from typing import Dict, Any
import torch
from torchmetrics import MeanSquaredError, R2Score, Accuracy, F1Score

class MetricsCallback(Callback):
    """指标计算回调"""
    def __init__(self, task_config):
        super().__init__()
        self.task_config = task_config
        self.metrics = self._setup_metrics()
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """设置评估指标"""
        metrics = {}
        if self.task_config.task_type == 'regression':
            metrics.update({
                'mse': MeanSquaredError(),
                'r2': R2Score()
            })
        else:  # classification
            metrics.update({
                'accuracy': Accuracy(
                    task='multiclass' if self.task_config.num_classes > 2 else 'binary',
                    num_classes=self.task_config.num_classes
                ),
                'f1': F1Score(
                    task='multiclass' if self.task_config.num_classes > 2 else 'binary',
                    num_classes=self.task_config.num_classes
                )
            })
        return metrics
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """验证批次结束时更新指标"""
        y_hat, y = outputs['y_hat'], outputs['y']
        for name, metric in self.metrics.items():
            metric(y_hat, y)
            pl_module.log(f'val_{name}', metric, on_step=False, on_epoch=True) 