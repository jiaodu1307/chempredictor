import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, R2Score
from typing import Dict, List, Union

from .encoders import BaseEncoder, MultiModalEncoder


class FlexibleMLP(pl.LightningModule):
    def __init__(
        self,
        encoders: Dict[str, BaseEncoder],
        mlp_dims: List[int] = [512, 256],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
    ):
        """用于反应产率预测的灵活MLP模型
        
        参数:
            encoders: 编码器字典 {特征名称: 编码器}
            mlp_dims: MLP隐藏层的维度
            dropout: 丢弃率
            learning_rate: 初始学习率
            weight_decay: 权重衰减系数
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoders'])
        
        # 初始化特征编码器
        self.encoder = MultiModalEncoder(encoders)
        input_dim = self.encoder.output_dim
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        for dim in mlp_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        # 指标
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()
        
        self.val_outputs = []
        self.test_outputs = []
        
        # 添加train_loss_epoch指标
        self.train_loss_epoch = MeanSquaredError()
        
        # 添加画图相关的属性
        self.plot_training_curves = True  # 控制是否绘制训练曲线
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        参数:
            inputs: 输入特征字典 {特征名称: 特征数据}
        返回:
            预测的反应产率
        """
        x = self.encoder(inputs)
        return self.mlp(x)
    
    def _shared_step(self, batch, batch_idx, stage='train'):
        inputs, y = batch
        y_hat = self(inputs)
        
        if stage == 'train':
            loss = self.train_mse(y_hat, y)
            self.log('train_loss', loss, prog_bar=True)
        elif stage == 'val':
            loss = self.val_mse(y_hat, y)
            self.log('val_loss', loss, prog_bar=True)
        else:  # test
            loss = self.test_mse(y_hat, y)
            self.log('test_mse', loss, on_epoch=True)
            return {'y_hat': y_hat, 'y': y}
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, batch_idx, 'test')
        self.test_outputs.append(outputs)
    
    def on_test_epoch_end(self):
        y_hat = torch.cat([x['y_hat'] for x in self.test_outputs])
        y = torch.cat([x['y'] for x in self.test_outputs])
        r2 = self.test_r2(y_hat, y)
        self.log('test_r2', r2)
        self.test_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=10,
            threshold=1e-2,
            threshold_mode='rel',
            min_lr=1e-5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)
    
    def on_train_epoch_end(self):
        # 计算并记录整个epoch的训练损失
        train_loss_epoch = self.train_loss_epoch.compute()
        self.log('train_loss_epoch', train_loss_epoch, prog_bar=True)
        self.train_loss_epoch.reset()  # 重置指标
    
    def on_fit_end(self):
        """训练结束后绘制训练曲线"""
        if not self.plot_training_curves:
            return
        
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        
        # 获取日志目录
        log_dir = self.logger.log_dir
        metrics_path = os.path.join(log_dir, 'metrics.csv')
        
        if os.path.exists(metrics_path):
            # 读取训练日志
            metrics_df = pd.read_csv(metrics_path)
            
            # 分别获取训练损失和验证损失
            train_loss = metrics_df[['epoch', 'train_loss_epoch']].dropna()
            val_loss = metrics_df[['epoch', 'val_loss']].dropna()
            
            # 创建图形
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss['epoch'], train_loss['train_loss_epoch'], 'b-', label='train_loss')
            plt.plot(val_loss['epoch'], val_loss['val_loss'], 'r-', label='val_loss')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # 保存图形
            save_path = os.path.join(log_dir, 'training_curves.png')
            plt.savefig(save_path)
            plt.close()
