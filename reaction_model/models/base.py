import torch
import torch.nn as nn
from typing import Dict, Any
from ..encoders.base import BaseEncoder

class BaseModel(nn.Module):
    """模型基类"""
    def __init__(self, encoders: Dict[str, BaseEncoder], **kwargs):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.config = kwargs
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config
    
    def encode_features(self, features: Dict[str, Any]) -> torch.Tensor:
        """编码输入特征"""
        encoded = []
        for name, encoder in self.encoders.items():
            if name in features:
                encoded.append(encoder(features[name]))
        return torch.cat(encoded, dim=-1) 