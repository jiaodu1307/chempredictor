import torch
from typing import Dict, Tuple
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from ..encoders.base import BaseEncoder

from .dataset import ChemicalDataset

def create_dataloaders(
    df: pd.DataFrame,
    encoders: Dict[str, BaseEncoder],
    target_column: str,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试数据加载器"""
    torch.manual_seed(seed)
    
    # 计算分割索引
    n_samples = len(df)
    indices = torch.randperm(n_samples)
    
    train_size = int(train_split * n_samples)
    val_size = int(val_split * n_samples)
    
    # 分割数据
    train_df = df.iloc[indices[:train_size]]
    val_df = df.iloc[indices[train_size:train_size + val_size]]
    test_df = df.iloc[indices[train_size + val_size:]]
    
    # 创建数据集
    def create_dataset(split_df):
        return ChemicalDataset(
            data_dict=split_df.to_dict('list'),
            encoders=encoders,
            target_column=target_column
        )
    
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    test_dataset = create_dataset(test_df)
    
    # 创建数据加载器
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )

def load_split_data(
    data_dir: Path,
    encoders: Dict[str, BaseEncoder],
    target_column: str,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """从已划分的数据目录加载数据"""
    # 读取已划分的数据
    train_df = pd.read_csv(data_dir / 'train_data.csv')
    val_df = pd.read_csv(data_dir / 'val_data.csv')
    test_df = pd.read_csv(data_dir / 'test_data.csv')
    
    # 创建数据集
    def create_dataset(split_df):
        return ChemicalDataset(
            data_dict=split_df.to_dict('list'),
            encoders=encoders,
            target_column=target_column
        )
    
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    test_dataset = create_dataset(test_df)
    
    # 创建数据加载器
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    ) 