from .base import BaseEncoder
import torch

class OneHotEncoder(BaseEncoder):
    """一热编码器"""
    def __init__(self, n_categories: int, **kwargs):
        super().__init__(n_categories=n_categories, **kwargs)
        self.n_categories = n_categories
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tensor(x, dtype=torch.long)
        return torch.nn.functional.one_hot(x, num_classes=self.n_categories).float()
    
    @property
    def output_dim(self) -> int:
        return self.n_categories 