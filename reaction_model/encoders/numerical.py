from .base import BaseEncoder
import torch
import torch.nn as nn

class NumericalEncoder(BaseEncoder):
    """数值特征编码器"""
    def __init__(self, input_dim: int = 1, normalize: bool = True, **kwargs):
        super().__init__(input_dim=input_dim, normalize=normalize, **kwargs)
        self.input_dim = input_dim
        self.normalize = normalize
        if normalize:
            self.norm = nn.BatchNorm1d(input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tensor(x, dtype=torch.float32)
        if self.normalize:
            x = self.norm(x)
        return x
    
    @property
    def output_dim(self) -> int:
        return self.input_dim 