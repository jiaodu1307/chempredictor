from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Any, Dict
from ..utils.exceptions import EncoderError

class BaseEncoder(ABC, nn.Module):
    """所有编码器的基类"""
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self._validate_config()
    
    def _validate_config(self):
        """验证编码器配置"""
        pass  # 由子类实现具体验证逻辑
    
    @abstractmethod
    def forward(self, x: Any) -> torch.Tensor:
        """将输入编码为张量
        
        参数:
            x: 输入数据
        返回:
            编码后的张量
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """编码器输出维度"""
        pass
    
    def get_config(self) -> Dict:
        """获取编码器配置"""
        return self.config 

class MorganFingerprintEncoder(BaseEncoder):
    def _validate_config(self):
        """验证Morgan指纹编码器配置"""
        if self.radius < 0:
            raise EncoderError(f"指纹半径必须大于0: {self.radius}")
        if self.n_bits <= 0:
            raise EncoderError(f"指纹位数必须大于0: {self.n_bits}") 