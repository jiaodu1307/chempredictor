from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Optional

class TrainingVisualizer(Callback):
    """训练过程可视化回调"""
    def __init__(self, log_dir: Optional[str] = None):
        super().__init__()
        self.log_dir = log_dir
        
    def on_fit_end(self, trainer, pl_module):
        """训练结束时绘制训练曲线"""
        if not pl_module.plot_training_curves:
            return
            
        metrics_path = os.path.join(trainer.logger.log_dir, 'metrics.csv')
        if not os.path.exists(metrics_path):
            return
            
        # 读取训练日志
        metrics_df = pd.read_csv(metrics_path)
        
        # 创建训练曲线图
        self._plot_training_curves(metrics_df, trainer.logger.log_dir)
    
    def _plot_training_curves(self, metrics_df: pd.DataFrame, log_dir: str):
        """绘制训练曲线
        
        参数:
            metrics_df: 指标数据
            log_dir: 日志目录
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制损失曲线
        self._plot_metric(metrics_df, 'loss', 'Loss Curves')
        plt.savefig(os.path.join(log_dir, 'loss_curves.png'))
        plt.close()
        
        # 绘制其他指标
        for metric in ['accuracy', 'r2']:
            if f'val_{metric}' in metrics_df.columns:
                plt.figure(figsize=(12, 6))
                self._plot_metric(metrics_df, metric, f'{metric.capitalize()} Curves')
                plt.savefig(os.path.join(log_dir, f'{metric}_curves.png'))
                plt.close()
    
    def _plot_metric(self, df: pd.DataFrame, metric: str, title: str):
        """绘制单个指标的曲线"""
        train_col = f'train_{metric}'
        val_col = f'val_{metric}'
        
        if train_col in df.columns:
            plt.plot(df['epoch'], df[train_col], 'b-', label=f'Train {metric}')
        if val_col in df.columns:
            plt.plot(df['epoch'], df[val_col], 'r-', label=f'Val {metric}')
            
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(title)
        plt.legend()
        plt.grid(True) 