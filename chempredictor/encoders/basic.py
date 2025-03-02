"""
基本特征编码器模块 - 提供常用的特征编码方法
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any
import logging

from chempredictor.encoders.base import BaseEncoder, register_encoder

try:
    from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
    from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.getLogger(__name__).warning("scikit-learn未安装，基本编码器功能将不可用")

@register_encoder("onehot_encoder")
class OneHotEncoder(BaseEncoder):
    """
    One-Hot编码器
    
    将分类特征转换为One-Hot编码
    """
    
    def __init__(self, handle_unknown: str = 'error', sparse: bool = False, **kwargs):
        """
        初始化One-Hot编码器
        
        参数:
            handle_unknown: 处理未知类别的策略，'error'或'ignore'
            sparse: 是否返回稀疏矩阵
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn未安装，无法使用OneHotEncoder")
            
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self.encoder = SklearnOneHotEncoder(
            handle_unknown=handle_unknown,
            sparse=sparse
        )
        self.categories_ = None
        self.output_dim = None
        
    def fit(self, data: Union[pd.Series, np.ndarray, List]) -> "OneHotEncoder":
        """
        拟合编码器
        
        参数:
            data: 输入数据
            
        返回:
            拟合后的编码器实例
        """
        # 确保数据是二维的
        if isinstance(data, pd.Series):
            X = data.values.reshape(-1, 1)
        elif isinstance(data, list):
            X = np.array(data).reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            X = data.reshape(-1, 1)
        else:
            X = data
            
        self.encoder.fit(X)
        self.categories_ = self.encoder.categories_
        self.output_dim = sum(len(cat) for cat in self.categories_)
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        转换数据
        
        参数:
            data: 输入数据
            
        返回:
            编码后的数据
        """
        if not self.is_fitted:
            self.fit(data)
            
        # 确保数据是二维的
        if isinstance(data, pd.Series):
            X = data.values.reshape(-1, 1)
        elif isinstance(data, list):
            X = np.array(data).reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            X = data.reshape(-1, 1)
        else:
            X = data
            
        result = self.encoder.transform(X)
        
        # 如果结果是稀疏矩阵，转换为密集矩阵
        if hasattr(result, 'toarray'):
            result = result.toarray()
            
        return result
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        返回:
            输出特征的维度
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未拟合，无法获取输出维度")
        return self.output_dim
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        返回:
            特征名称列表
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未拟合，无法获取特征名称")
            
        feature_names = []
        for i, categories in enumerate(self.categories_):
            for category in categories:
                if isinstance(category, (int, float, bool)):
                    feature_names.append(f"onehot_{i}_{category}")
                else:
                    feature_names.append(f"onehot_{i}_{str(category)}")
                    
        return feature_names

@register_encoder("label_encoder")
class LabelEncoder(BaseEncoder):
    """
    标签编码器
    
    将分类特征转换为整数编码
    """
    
    def __init__(self, **kwargs):
        """
        初始化标签编码器
        
        参数:
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn未安装，无法使用LabelEncoder")
            
        self.encoder = SklearnLabelEncoder()
        self.classes_ = None
        
    def fit(self, data: Union[pd.Series, np.ndarray, List]) -> "LabelEncoder":
        """
        拟合编码器
        
        参数:
            data: 输入数据
            
        返回:
            拟合后的编码器实例
        """
        # 确保数据是一维的
        if isinstance(data, pd.Series):
            X = data.values
        elif isinstance(data, list):
            X = np.array(data)
        else:
            X = data.ravel() if data.ndim > 1 else data
            
        self.encoder.fit(X)
        self.classes_ = self.encoder.classes_
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        转换数据
        
        参数:
            data: 输入数据
            
        返回:
            编码后的数据
        """
        if not self.is_fitted:
            self.fit(data)
            
        # 确保数据是一维的
        if isinstance(data, pd.Series):
            X = data.values
        elif isinstance(data, list):
            X = np.array(data)
        else:
            X = data.ravel() if data.ndim > 1 else data
            
        return self.encoder.transform(X).reshape(-1, 1)
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        返回:
            输出特征的维度
        """
        return 1
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        返回:
            特征名称列表
        """
        return ["label_encoded"]

@register_encoder("standard_scaler")
class StandardScaler(BaseEncoder):
    """
    标准化编码器
    
    将数值特征标准化为均值为0，方差为1
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, **kwargs):
        """
        初始化标准化编码器
        
        参数:
            with_mean: 是否减去均值
            with_std: 是否除以标准差
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn未安装，无法使用StandardScaler")
            
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = SklearnStandardScaler(with_mean=with_mean, with_std=with_std)
        
    def fit(self, data: Union[pd.Series, np.ndarray, List]) -> "StandardScaler":
        """
        拟合编码器
        
        参数:
            data: 输入数据
            
        返回:
            拟合后的编码器实例
        """
        # 确保数据是二维的
        if isinstance(data, pd.Series):
            X = data.values.reshape(-1, 1)
        elif isinstance(data, list):
            X = np.array(data).reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            X = data.reshape(-1, 1)
        else:
            X = data
            
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        转换数据
        
        参数:
            data: 输入数据
            
        返回:
            编码后的数据
        """
        if not self.is_fitted:
            self.fit(data)
            
        # 确保数据是二维的
        if isinstance(data, pd.Series):
            X = data.values.reshape(-1, 1)
        elif isinstance(data, list):
            X = np.array(data).reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            X = data.reshape(-1, 1)
        else:
            X = data
            
        return self.scaler.transform(X)
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        返回:
            输出特征的维度
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未拟合，无法获取输出维度")
        return 1 if len(self.scaler.scale_) == 1 else len(self.scaler.scale_)
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        返回:
            特征名称列表
        """
        n_features = self.get_output_dim()
        return [f"scaled_{i}" for i in range(n_features)]

@register_encoder("minmax_scaler")
class MinMaxScaler(BaseEncoder):
    """
    最小-最大缩放编码器
    
    将数值特征缩放到指定范围
    """
    
    def __init__(self, feature_range: tuple = (0, 1), **kwargs):
        """
        初始化最小-最大缩放编码器
        
        参数:
            feature_range: 目标范围
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn未安装，无法使用MinMaxScaler")
            
        self.feature_range = feature_range
        self.scaler = SklearnMinMaxScaler(feature_range=feature_range)
        
    def fit(self, data: Union[pd.Series, np.ndarray, List]) -> "MinMaxScaler":
        """
        拟合编码器
        
        参数:
            data: 输入数据
            
        返回:
            拟合后的编码器实例
        """
        # 确保数据是二维的
        if isinstance(data, pd.Series):
            X = data.values.reshape(-1, 1)
        elif isinstance(data, list):
            X = np.array(data).reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            X = data.reshape(-1, 1)
        else:
            X = data
            
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        转换数据
        
        参数:
            data: 输入数据
            
        返回:
            编码后的数据
        """
        if not self.is_fitted:
            self.fit(data)
            
        # 确保数据是二维的
        if isinstance(data, pd.Series):
            X = data.values.reshape(-1, 1)
        elif isinstance(data, list):
            X = np.array(data).reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            X = data.reshape(-1, 1)
        else:
            X = data
            
        return self.scaler.transform(X)
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        返回:
            输出特征的维度
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未拟合，无法获取输出维度")
        return 1 if len(self.scaler.scale_) == 1 else len(self.scaler.scale_)
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        返回:
            特征名称列表
        """
        n_features = self.get_output_dim()
        return [f"minmax_scaled_{i}" for i in range(n_features)] 