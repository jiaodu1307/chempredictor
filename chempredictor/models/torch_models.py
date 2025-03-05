import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Dict, Any, Optional
import numpy as np

class LightningMLP(pl.LightningModule):
    """使用PyTorch Lightning实现的MLP模型"""
    
    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: List[int] = [1024, 512, 256, 128],
        activation: str = "relu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        task_type: str = "regression",
        **kwargs
    ):
        """
        初始化MLP模型
        
        Args:
            input_size: 输入特征维度
            hidden_layer_sizes: 隐藏层神经元数量列表
            activation: 激活函数类型，支持 'relu', 'tanh', 'sigmoid'
            learning_rate: 学习率
            weight_decay: L2正则化系数
            task_type: 任务类型，'regression' 或 'classification'
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        # 激活函数映射
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        activation_fn = activation_map.get(activation.lower(), nn.ReLU())
        
        # 构建隐藏层
        for size in hidden_layer_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                activation_fn,
                nn.BatchNorm1d(size),
                nn.Dropout(0.2)
            ])
            prev_size = size
        
        # 输出层
        output_size = 1 if task_type == "regression" else 2
        layers.append(nn.Linear(prev_size, output_size))
        
        # 如果是分类任务，添加Sigmoid激活
        if task_type == "classification":
            layers.append(nn.Sigmoid())
            
        self.network = nn.Sequential(*layers)
        
        # 损失函数
        self.loss_fn = (
            nn.MSELoss() if task_type == "regression"
            else nn.BCELoss()
        )
        
        # 记录训练历史
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'best_loss': float('inf'),
            'best_iter': 0,
            'n_iter': 0
        }
        
    def forward(self, x):
        return self.network(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # 记录训练损失
        self.training_history['loss'].append(loss.item())
        self.training_history['n_iter'] += 1
        
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        
        # 记录验证损失
        self.training_history['val_loss'].append(val_loss.item())
        
        # 更新最佳模型
        if val_loss < self.training_history['best_loss']:
            self.training_history['best_loss'] = val_loss.item()
            self.training_history['best_iter'] = self.training_history['n_iter']
        
        self.log('val_loss', val_loss)
        return val_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
        
    def get_training_history(self) -> Dict[str, Any]:
        """获取训练历史"""
        return self.training_history 