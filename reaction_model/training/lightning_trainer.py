import os
from typing import Dict, Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .base_trainer import BaseTrainer
from .callbacks.visualization import TrainingVisualizer
from .callbacks.metrics import MetricsCallback
from ..data_utils import load_and_split_data, load_split_data

class LightningModelTrainer(BaseTrainer):
    """Lightning模型训练器"""
    def __init__(
        self,
        model: pl.LightningModule,
        config: Dict[str, Any],
        callbacks: Optional[list] = None
    ):
        super().__init__(model, config)
        self.callbacks = callbacks or []
        
    def _setup(self):
        """设置训练器"""
        # 创建日志目录
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # 设置logger
        self.logger = CSVLogger(
            save_dir=self.config['log_dir'],
            name=self.config.get('experiment_name', 'experiment')
        )
        
        # 设置checkpoint回调
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.logger.log_dir, 'checkpoints'),
            filename='best-{epoch:02d}-{val_loss:.3f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        # 添加可视化回调
        self.callbacks.extend([
            self.checkpoint_callback,
            TrainingVisualizer(),
            MetricsCallback(self.model.task_config)
        ])
        
        # 创建训练器
        self.trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            accelerator='cuda' if self.config.get('use_gpu', True) else 'cpu',
            devices=1,
            logger=self.logger,
            callbacks=self.callbacks,
            **self.config.get('trainer_kwargs', {})
        )
    
    def train(self, data_path: str) -> Dict[str, Any]:
        """训练模型"""
        # 加载数据
        if data_path.endswith('.csv'):
            train_loader, val_loader, test_loader = load_and_split_data(
                data_path,
                self.config['feature_configs'],
                self.config['batch_size']
            )
        else:
            train_loader, val_loader, test_loader = load_split_data(
                data_path,
                self.config['feature_configs'],
                self.config['batch_size']
            )
        
        # 训练模型
        self.trainer.fit(self.model, train_loader, val_loader)
        
        # 测试模型
        results = {}
        for name, loader in [
            ('test', test_loader),
            ('val', val_loader),
            ('train', train_loader)
        ]:
            results[name] = self.trainer.test(self.model, loader)[0]
        
        return results
    
    def evaluate(self, data_path: str) -> Dict[str, float]:
        """评估模型"""
        # 实现评估逻辑
        pass
    
    def save_model(self, save_path: str):
        """保存模型"""
        self.trainer.save_checkpoint(save_path)
    
    def load_model(self, model_path: str):
        """加载模型"""
        self.model = self.model.load_from_checkpoint(model_path) 