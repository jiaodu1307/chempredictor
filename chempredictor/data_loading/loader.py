"""
数据加载模块 - 提供从不同格式加载数据的功能
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

class DataLoader:
    """
    数据加载器
    
    支持从CSV、Excel、JSON等格式加载数据
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
                 **kwargs):
        """
        初始化数据加载器
        
        参数:
            file_type: 文件类型，'csv'、'excel'或'json'，如果为None则自动检测
            target_column: 目标列名
            feature_columns: 特征列名列表，如果为None则使用除目标列外的所有列
            missing_value_strategy: 缺失值处理策略，'mean'、'median'、'mode'或'drop'
            **kwargs: 其他参数，将传递给相应的pandas读取函数
        """
        self.file_type = file_type
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.missing_value_strategy = missing_value_strategy
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
        # 缺失值处理函数映射
        self.missing_value_handlers = {
            "mean": self._fill_mean,
            "median": self._fill_median,
            "mode": self._fill_mode,
            "drop": self._drop_na
        }
        
        if missing_value_strategy not in self.missing_value_handlers:
            raise ValueError(f"未知的缺失值处理策略: {missing_value_strategy}，"
                            f"支持的策略: {list(self.missing_value_handlers.keys())}")
    
    def load(self, file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        加载数据
        
        参数:
            file_path: 数据文件路径
            
        返回:
            特征数据框和目标序列（如果有目标列）
        """
        self.logger.info(f"从{file_path}加载数据")
        
        # 检测文件类型
        file_type = self.file_type
        if file_type is None:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                file_type = 'csv'
            elif file_ext in ['.xls', '.xlsx']:
                file_type = 'excel'
            elif file_ext == '.json':
                file_type = 'json'
            else:
                raise ValueError(f"无法自动检测文件类型: {file_ext}，请明确指定file_type")
        
        # 加载数据
        if file_type == 'csv':
            df = pd.read_csv(file_path, **self.kwargs)
        elif file_type == 'excel':
            df = pd.read_excel(file_path, **self.kwargs)
        elif file_type == 'json':
            df = pd.read_json(file_path, **self.kwargs)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
        
        self.logger.info(f"加载了{len(df)}行数据，列: {list(df.columns)}")
        
        # 处理缺失值
        for col in df.columns:
            if df[col].isna().any():
                self.logger.info(f"列{col}包含{df[col].isna().sum()}个缺失值，使用{self.missing_value_strategy}策略处理")
                try:
                    df[col] = self.missing_value_handlers[self.missing_value_strategy](df, col)
                except Exception as e:
                    self.logger.warning(f"无法使用{self.missing_value_strategy}策略处理列{col}的缺失值: {e}")
                    if self.missing_value_strategy != "drop":
                        self.logger.warning(f"尝试使用drop策略")
                        df = df.dropna(subset=[col])
        
        # 分离特征和目标
        if self.target_column is not None and self.target_column in df.columns:
            y = df[self.target_column]
            
            # 如果指定了特征列，则只使用这些列
            if self.feature_columns is not None:
                missing_cols = [col for col in self.feature_columns if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"以下特征列不存在: {missing_cols}")
                X = df[[col for col in self.feature_columns if col in df.columns]]
            else:
                # 否则使用除目标列外的所有列
                X = df.drop(columns=[self.target_column])
        else:
            if self.target_column is not None:
                self.logger.warning(f"目标列{self.target_column}不存在，将返回所有列作为特征")
            
            # 如果指定了特征列，则只使用这些列
            if self.feature_columns is not None:
                missing_cols = [col for col in self.feature_columns if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"以下特征列不存在: {missing_cols}")
                X = df[[col for col in self.feature_columns if col in df.columns]]
            else:
                # 否则使用所有列
                X = df
            
            y = None
        
        self.logger.info(f"特征形状: {X.shape}")
        if y is not None:
            self.logger.info(f"目标形状: {y.shape}")
        
        return X, y
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        检测数据框中各列的数据类型
        
        参数:
            df: 数据框
            
        返回:
            列名到类型的映射，类型包括'numeric'、'categorical'、'text'和'smiles'
        """
        column_types = {}
        
        for col in df.columns:
            # 检查是否为数值类型
            if pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = 'numeric'
                continue
            
            # 检查是否为分类类型（唯一值较少）
            n_unique = df[col].nunique()
            if n_unique < 10 or (n_unique / len(df) < 0.05):
                column_types[col] = 'categorical'
                continue
            
            # 检查是否为SMILES（包含特定字符和模式）
            sample = df[col].dropna().sample(min(10, len(df))).tolist()
            is_smiles = all(
                isinstance(s, str) and 
                any(c in s for c in '()=[]#') and 
                any(c.isupper() for c in s if c.isalpha())
                for s in sample
            )
            
            if is_smiles:
                column_types[col] = 'smiles'
            else:
                # 默认为文本
                column_types[col] = 'text'
        
        return column_types
    
    def save(self, X: pd.DataFrame, y: Optional[pd.Series], file_path: str) -> None:
        """
        保存数据到文件
        
        参数:
            X: 特征数据框
            y: 目标序列
            file_path: 保存路径
        """
        self.logger.info(f"保存数据到{file_path}")
        
        # 合并特征和目标
        if y is not None:
            if self.target_column is not None:
                df = X.copy()
                df[self.target_column] = y
            else:
                df = pd.concat([X, pd.DataFrame({'target': y})], axis=1)
        else:
            df = X
        
        # 检测文件类型
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 保存数据
        if file_ext == '.csv':
            df.to_csv(file_path, index=False)
        elif file_ext in ['.xls', '.xlsx']:
            df.to_excel(file_path, index=False)
        elif file_ext == '.json':
            df.to_json(file_path, orient='records')
        else:
            raise ValueError(f"不支持的文件类型: {file_ext}")
        
        self.logger.info(f"已保存{len(df)}行数据") 