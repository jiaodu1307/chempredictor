import os
import logging
import yaml
import numpy as np
import torch
import random

class ChemPredictor:
    """化学反应预测器类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化预测器
        
        Args:
            config_path (str): 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is None:
            raise ValueError("必须提供配置文件路径")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 设置随机数种子
        seed = self.config.get('random_seed', 42)
        self._set_random_seed(seed)
        
        # 设置计算设备
        self.device = self._setup_device()
        
        # 初始化pipeline
        self.pipeline = self._setup_pipeline()
        
    def _set_random_seed(self, seed: int):
        """设置随机数种子以确保结果可重现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.logger.info(f"已设置随机数种子: {seed}")
        
    def _setup_device(self) -> str:
        """设置计算设备"""
        device = self.config['pipeline']['steps']['model_training'].get('device', 'auto')
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，将使用CPU")
            device = 'cpu'
            
        if device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"使用GPU: {gpu_name}")
        else:
            self.logger.info("使用CPU进行计算")
            
        return device
        
    def _setup_pipeline(self):
        """设置处理流水线"""
        from .pipeline import Pipeline
        return Pipeline(self.config['pipeline'], device=self.device) 