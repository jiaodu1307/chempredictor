import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from .encoders import BaseEncoder, MorganFingerprintEncoder, OneHotEncoder, NumericalEncoder, MPNNEncoder

class ReactionDataset(Dataset):
    """反应数据集类"""
    def __init__(self, data_dict: Dict[str, List[Any]], targets: List[float]):
        """
        参数:
            data_dict: 特征列表字典 {特征名称: [特征值]}
            targets: 标签产率列表
        """
        self.data = data_dict
        self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        features = {k: v[idx] for k, v in self.data.items()}
        return features, self.targets[idx]


def create_dataloaders(
    df: pd.DataFrame,
    feature_columns: Dict[str, str],
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """从DataFrame创建训练/验证/测试数据加载器
    
    参数:
        df: 输入的DataFrame
        feature_columns: 特征名称与列名称的映射字典
        batch_size: 数据加载器的批量大小
        train_split: 训练数据的比例
        val_split: 验证数据的比例
        seed: 随机种子
    
    返回:
        训练加载器、验证加载器和测试加载器的元组
    """
    # 设置随机种子
    torch.manual_seed(seed)
    
    # 创建特征字典
    data_dict = {
        feature_name: df[col_name].tolist()
        for feature_name, col_name in feature_columns.items()
    }
    
    # 获取目标
    targets = df['Yield'].values
    
    # 计算分割索引
    n_samples = len(df)
    indices = torch.randperm(n_samples)
    
    train_size = int(train_split * n_samples)
    val_size = int(val_split * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 创建数据集
    def create_subset(indices):
        subset_dict = {
            k: [v[i] for i in indices]
            for k, v in data_dict.items()
        }
        subset_targets = [targets[i] for i in indices]
        return ReactionDataset(subset_dict, subset_targets)
    
    train_dataset = create_subset(train_indices)
    val_dataset = create_subset(val_indices)
    test_dataset = create_subset(test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader

class FeatureConfig:
    """反应特征配置类"""
    def __init__(self, feature_type: str, column_name: str, encoder_params: dict = None):
        """
        参数:
            feature_type: 特征类型 ('molecule', 'categorical', 'numerical')
            column_name: DataFrame中的列名
            encoder_params: 编码器参数
            trainable: 是否需要训练编码器
        """
        self.type = feature_type
        self.column = column_name
        self.params = encoder_params or {}
        self.trainable = feature_type == 'molecule' and encoder_params.get('use_mpnn', False)

def create_encoders(feature_configs: Dict[str, FeatureConfig]) -> Dict[str, BaseEncoder]:
    """根据配置创建编码器
    
    参数:
        feature_configs: {特征名称: 特征配置}的字典
    返回:
        编码器字典
    """
    encoders = {}
    for name, config in feature_configs.items():
        if config.type == 'molecule':
            if config.params.get('use_mpnn', False):
                encoders[name] = MPNNEncoder(**config.params)
            else:
                encoders[name] = MorganFingerprintEncoder(**config.params)
        elif config.type == 'categorical':
            encoders[name] = OneHotEncoder(**config.params)
        elif config.type == 'numerical':
            encoders[name] = NumericalEncoder(**config.params)
            
    return encoders
