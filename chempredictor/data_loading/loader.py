"""
数据加载模块 - 提供从不同格式加载数据的功能
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

class ChemDataset(Dataset):
    """化学数据集类
    
    用于处理化学数据的PyTorch数据集类，支持特征和目标的转换、缓存和批处理。
    """
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        初始化数据集
        
        Args:
            X: 特征数组（已经过预处理和编码的数值数组）
            y: 目标数组
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (特征, 目标值)元组，如果没有目标值，则返回(特征, None)
        """
        X_item = self.X[idx]
        y_item = self.y[idx] if self.y is not None else torch.tensor(float('nan'))
        return X_item, y_item

class DataLoader:
    """
    数据加载器
    
    支持从CSV、Excel、JSON等格式加载数据，并提供批处理功能
    """
    
    @staticmethod
    def _fill_mean(df: pd.DataFrame, col: str) -> pd.Series:
        """使用均值填充缺失值"""
        return df[col].fillna(df[col].mean())
    
    @staticmethod
    def _fill_median(df: pd.DataFrame, col: str) -> pd.Series:
        """使用中位数填充缺失值"""
        return df[col].fillna(df[col].median())
    
    @staticmethod
    def _fill_mode(df: pd.DataFrame, col: str) -> pd.Series:
        """使用众数填充缺失值"""
        return df[col].fillna(df[col].mode()[0])
    
    @staticmethod
    def _drop_na(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """删除包含缺失值的行"""
        return df.dropna(subset=[col])
    
    def __init__(self, file_type: Optional[str] = None, 
                 target_column: Optional[str] = None,
                 feature_columns: Optional[List[str]] = None,
                 missing_value_strategy: str = "mean",
                 batch_size: int = 32,
                 num_workers: int = 0,
                 shuffle: bool = True,
                 pandas_kwargs: Optional[Dict[str, Any]] = None):
        """
        初始化数据加载器
        
        Args:
            file_type: 文件类型，'csv'、'excel'或'json'，如果为None则自动检测
            target_column: 目标列名
            feature_columns: 特征列名列表，如果为None则使用除目标列外的所有列
            missing_value_strategy: 缺失值处理策略，'mean'、'median'、'mode'或'drop'
            batch_size: 批大小
            num_workers: 数据加载的工作进程数
            shuffle: 是否打乱数据
            pandas_kwargs: 传递给pandas读取函数的额外参数
        """
        self.file_type = file_type
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.missing_value_strategy = missing_value_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pandas_kwargs = pandas_kwargs or {}
        
        self.logger = logging.getLogger(__name__)
        
        # 缺失值处理函数映射
        self.missing_value_handlers = {
            "mean": self._fill_mean,
            "median": self._fill_median,
            "mode": self._fill_mode,
            "drop": self._drop_na
        }
        
        if missing_value_strategy not in self.missing_value_handlers:
            raise ValueError(f"未知的缺失值处理策略: {missing_value_strategy}")
    
    def load(self, file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """加载数据"""
        self.logger.info(f"从{file_path}加载数据")
        
        # 加载原始数据
        df = self._load_raw_data(file_path)
        self.logger.info(f"加载了{len(df)}行数据，列: {list(df.columns)}")
        
        # 处理缺失值
        df = self._handle_missing_values(df)
        
        # 分离特征和目标
        X, y = self._split_features_target(df)
        
        # 保存原始特征名称
        self.original_feature_names = list(X.columns)
        
        # 处理特征
        X_processed = self._process_features(X)
        
        # 处理目标值
        if y is not None:
            y_processed = y.astype(np.float32)
        else:
            self.logger.warning(f"未找到目标列 '{self.target_column}'")
            return X_processed, None
        
        return X_processed, y_processed
    
    def _load_raw_data(self, file_path: str) -> pd.DataFrame:
        """加载原始数据"""
        file_type = self._detect_file_type(file_path)
        if file_type == 'csv':
            return pd.read_csv(file_path, **self.pandas_kwargs)
        elif file_type == 'excel':
            return pd.read_excel(file_path, **self.pandas_kwargs)
        elif file_type == 'json':
            return pd.read_json(file_path, **self.pandas_kwargs)
        raise ValueError(f"不支持的文件类型: {file_type}")
    
    def _detect_file_type(self, file_path: str) -> str:
        """检测文件类型"""
        if self.file_type:
            return self.file_type
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return 'csv'
        elif ext in ['.xls', '.xlsx']:
            return 'excel'
        elif ext == '.json':
            return 'json'
        raise ValueError(f"无法自动检测文件类型: {ext}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        for col in df.columns:
            if df[col].isna().any():
                self.logger.info(f"列{col}包含{df[col].isna().sum()}个缺失值")
                try:
                    df[col] = self.missing_value_handlers[self.missing_value_strategy](df, col)
                except Exception as e:
                    self.logger.warning(f"使用{self.missing_value_strategy}处理失败: {e}")
                    if self.missing_value_strategy != "drop":
                        df = df.dropna(subset=[col])
        return df
    
    def _split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """分离特征和目标"""
        if self.target_column and self.target_column in df.columns:
            y = df[self.target_column]
            X = df[self.feature_columns] if self.feature_columns else df.drop(columns=[self.target_column])
        else:
            y = None
            X = df[self.feature_columns] if self.feature_columns else df
        return X, y
    
    def _process_features(self, X: pd.DataFrame) -> np.ndarray:
        """处理特征"""
        # 检测每列的类型
        processed_arrays = []
        for col in X.columns:
            col_data = X[col]
            try:
                # 尝试转换为数值
                processed = pd.to_numeric(col_data).values.astype(np.float32)
                processed_arrays.append(processed.reshape(-1, 1))
            except (ValueError, TypeError):
                # 如果不能转换为数值，进行独热编码
                dummies = pd.get_dummies(col_data, prefix=col).values.astype(np.float32)
                processed_arrays.append(dummies)
        
        # 合并所有处理后的特征
        return np.hstack(processed_arrays)
    
    def save(self, X: Union[pd.DataFrame, torch.Tensor], 
             y: Optional[Union[pd.Series, torch.Tensor]], 
             file_path: str) -> None:
        """
        保存数据到文件
        
        参数:
            X: 特征数据（DataFrame或张量）
            y: 目标序列（Series、张量或None）
            file_path: 保存路径
        """
        self.logger.info(f"保存数据到{file_path}")
        
        # 将张量转换为DataFrame/Series
        if isinstance(X, torch.Tensor):
            X = pd.DataFrame(X.cpu().numpy())
            # 如果有原始的列名，尝试恢复
            if hasattr(self, 'original_feature_names'):
                if len(self.original_feature_names) == X.shape[1]:
                    X.columns = self.original_feature_names
                else:
                    X.columns = [f'feature_{i}' for i in range(X.shape[1])]
            else:
                X.columns = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, torch.Tensor):
            y = pd.Series(y.cpu().numpy())
        
        # 确保数据类型正确
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X必须是pandas DataFrame或PyTorch张量")
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("y必须是pandas Series或PyTorch张量")
        
        # 合并特征和目标
        if y is not None:
            if self.target_column is not None:
                df = X.copy()
                df[self.target_column] = y
            else:
                df = pd.concat([X, pd.DataFrame({'target': y})], axis=1)
        else:
            df = X
        
        # 检测文件类型并保存
        file_ext = os.path.splitext(file_path)[1].lower()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            if file_ext == '.csv':
                df.to_csv(file_path, index=False, **self.pandas_kwargs)
            elif file_ext in ['.xls', '.xlsx']:
                df.to_excel(file_path, index=False, **self.pandas_kwargs)
            elif file_ext == '.json':
                df.to_json(file_path, orient='records', **self.pandas_kwargs)
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
            
            self.logger.info(f"已成功保存{len(df)}行数据到{file_path}")
            self.logger.info(f"数据列: {list(df.columns)}")
            
        except Exception as e:
            self.logger.error(f"保存数据到{file_path}时发生错误: {str(e)}")
            raise 