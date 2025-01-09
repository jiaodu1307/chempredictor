from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import torch.nn as nn

class BaseTrainer(ABC):
    """训练器基类"""
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """设置训练器"""
        pass
    
    @abstractmethod
    def train(self, data_path: str) -> Dict[str, Any]:
        """训练模型
        
        参数:
            data_path: 数据路径
        返回:
            训练结果字典
        """
        pass
    
    @abstractmethod
    def evaluate(self, data_path: str) -> Dict[str, float]:
        """评估模型
        
        参数:
            data_path: 数据路径
        返回:
            评估指标字典
        """
        pass
    
    @abstractmethod
    def save_model(self, save_path: str):
        """保存模型
        
        参数:
            save_path: 保存路径
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str):
        """加载模型
        
        参数:
            model_path: 模型路径
        """
        pass 