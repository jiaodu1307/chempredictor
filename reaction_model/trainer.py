import os
import yaml
import torch
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .model import FlexibleMLP
from .data_utils import create_dataloaders, FeatureConfig
from .encoders import (
    MorganFingerprintEncoder, 
    OneHotEncoder,
    NumericalEncoder,
    MPNNEncoder
)

class ReactionModelTrainer:
    """反应模型训练器"""
    
    def __init__(self, params):
        """
        参数:
            params: 训练参数字典
        """
        self.params = params
        self._set_seed()
        
    def _set_seed(self):
        """设置随机种子确保可重复性"""
        seed = self.params['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pl.seed_everything(seed)
    
    def _get_short_path(self, path):
        """获取路径的最后一个文件夹名"""
        return os.path.basename(path)
    
    def _setup_logging(self):
        """配置日志和检查点"""
        short_path = self._get_short_path(self.params['data_path'])
        
        model_name = f"reaction_model_{short_path}_ep{self.params['max_epochs']}"
        if self.params.get('use_mpnn', False):
            model_name += "_mpnn"
        else:
            model_name += f"_fp_bits{self.params['fp_bits']}_fp_radius{self.params['fp_radius']}"
            
        base_log_dir = self.params['base_log_dir']
        model_dir = os.path.join(base_log_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        logger = CSVLogger(save_dir=base_log_dir, name=model_name)
        
        checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        
        return logger, checkpoint_callback, model_dir
    
    def _create_feature_configs(self):
        """创建特征配置"""
        use_mpnn = self.params.get('use_mpnn', False)
        
        molecule_encoder_params = {
            'use_mpnn': use_mpnn
        }
        if not use_mpnn:
            molecule_encoder_params.update({
                'radius': self.params['fp_radius'],
                'n_bits': self.params['fp_bits']
            })
        else:
            molecule_encoder_params.update({
                'hidden_size': self.params['mpnn_hidden_size'],
                'depth': self.params['mpnn_depth'],
                'dropout': self.params['dropout']
            })
            
        return {
            'reactant1': FeatureConfig(
                'molecule', 
                'Reactant1_Smiles',
                molecule_encoder_params
            ),
            'reactant2': FeatureConfig(
                'molecule',
                'Reactant2_Smiles', 
                molecule_encoder_params
            ),
            'catalyst': FeatureConfig(
                'molecule',
                'Catalyst',
                molecule_encoder_params
            ),
            'reagent': FeatureConfig(
                'categorical',
                'Reagent',
                {'n_categories': 2}
            ),
            'temperature': FeatureConfig(
                'categorical',
                'Temperature',
                {'n_categories': 2}
            ),
            'order': FeatureConfig(
                'numerical',
                'order',
                {'input_dim': 6}
            )
        }
    
    def train(self, data_path):
        """训练模型
        
        参数:
            data_path: 数据文件路径的根目录
        """
        # 设置日志
        logger, checkpoint_callback, model_dir = self._setup_logging()
        
        # 创建特征配置
        feature_configs = self._create_feature_configs()
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            feature_configs=feature_configs,
            batch_size=self.params['batch_size'],
            seed=self.params['seed']
        )
        
        # 创建编码器
        encoders = {}
        for name, config in feature_configs.items():
            if config.type == 'molecule':
                if config.params['use_mpnn']:
                    encoders[name] = MPNNEncoder(**config.params)
                else:
                    encoders[name] = MorganFingerprintEncoder(**config.params)
            elif config.type == 'categorical':
                encoders[name] = OneHotEncoder(**config.params)
            elif config.type == 'numerical':
                encoders[name] = NumericalEncoder(**config.params)
        
        # 创建模型
        model = FlexibleMLP(
            encoders=encoders,
            mlp_dims=self.params['mlp_dims'],
            dropout=self.params['dropout'],
            learning_rate=self.params['learning_rate'],
            weight_decay=self.params['weight_decay']
        )
        
        # 设置是否绘制训练曲线
        model.plot_training_curves = self.params.get('plot_training_curves', True)
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=self.params['max_epochs'],
            accelerator='cuda',
            devices=1,
            enable_checkpointing=True,
            log_every_n_steps=50,
            default_root_dir=model_dir,
            logger=logger,
            callbacks=[checkpoint_callback]
        )
        
        # 训练和评估
        trainer.fit(model, train_loader, val_loader)
        
        # 在所有数据集上测试
        results = {}
        for name, loader in [
            ("test", test_loader),
            ("val", val_loader),
            ("train", train_loader)
        ]:
            results[name] = trainer.test(model, dataloaders=loader)[0]
            print(f"{name} results:", results[name])
        
        # 保存参数
        yaml_path = os.path.join(logger.log_dir, 'hparams.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(self.params, f, default_flow_style=False)
            
        return model, trainer, results 