from ..base import BaseEncoder
from abc import abstractmethod
from typing import List
import torch

class MoleculeEncoder(BaseEncoder):
    """分子编码器基类"""
    @abstractmethod
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        """编码单个SMILES字符串"""
        pass
    
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """批量编码SMILES列表"""
        encodings = [self.encode_smiles(s) for s in smiles_list]
        return torch.stack(encodings)
    
    @abstractmethod
    def preprocess_smiles(self, smiles: str) -> str:
        """预处理SMILES字符串"""
        pass 