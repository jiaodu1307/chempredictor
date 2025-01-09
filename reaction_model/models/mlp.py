from typing import Dict, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, R2Score
from ..encoders.base import BaseEncoder

from .base import BaseModel
from ..utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register('mlp')
class MLPModel(BaseModel, pl.LightningModule):
    """MLP模型实现"""
    def __init__(
        self,
        encoders: Dict[str, BaseEncoder],
        hidden_dims: List[int],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        **kwargs
    ):
        super().__init__(encoders=encoders)
        self.save_hyperparameters(ignore=['encoders'])
        
        # 构建MLP层
        layers = []
        prev_dim = sum(encoder.output_dim for encoder in encoders.values())
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # 指标
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.encode_features(features)
        return self.mlp(x)
    
    def training_step(self, batch, batch_idx):
        features, y = batch
        y_hat = self(features)
        loss = self.train_mse(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, y = batch
        y_hat = self(features)
        loss = self.val_mse(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return {'y_hat': y_hat, 'y': y}
    
    def test_step(self, batch, batch_idx):
        features, y = batch
        y_hat = self(features)
        self.test_mse(y_hat, y)
        self.test_r2(y_hat, y)
        self.log('test_mse', self.test_mse)
        self.log('test_r2', self.test_r2)
        return {'y_hat': y_hat, 'y': y}
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        ) 