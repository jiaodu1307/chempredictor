from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset
from ..encoders.base import BaseEncoder
from ..utils.exceptions import DataError

class ChemicalDataset(Dataset):
    """化学数据集基类"""
    def __init__(
        self,
        data_dict: Dict[str, List[Any]],
        encoders: Dict[str, BaseEncoder],
        target_column: str,
        transform: Optional[callable] = None
    ):
        self.data = data_dict
        self.encoders = encoders
        self.target_column = target_column
        self.transform = transform
        
        # 验证数据
        self._validate_data()
    
    def _validate_data(self):
        """验证数据完整性和有效性"""
        # 验证目标列是否存在
        if self.target_column not in self.data:
            raise DataError(f"目标列 {self.target_column} 不存在")
            
        # 验证数据长度一致性
        length = len(self.data[self.target_column])
        for key, values in self.data.items():
            if len(values) != length:
                raise DataError(f"特征 {key} 的长度与目标长度不匹配")
        
        # 验证编码器与特征的对应关系
        for name, encoder in self.encoders.items():
            if name not in self.data:
                raise DataError(f"编码器 {name} 对应的特征不存在")
    
    def __len__(self):
        return len(self.data[self.target_column])
    
    def __getitem__(self, idx):
        # 获取特征
        features = {}
        for name, values in self.data.items():
            if name != self.target_column:
                encoder = self.encoders.get(name)
                if encoder is not None:
                    features[name] = encoder(values[idx])
                else:
                    features[name] = values[idx]
        
        # 获取目标
        target = torch.tensor(
            self.data[self.target_column][idx],
            dtype=torch.float32
        )
        
        # 应用转换
        if self.transform is not None:
            features, target = self.transform(features, target)
            
        return features, target 