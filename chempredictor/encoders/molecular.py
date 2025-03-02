"""
分子编码器模块 - 提供各种分子表示方法
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any
import logging

from chempredictor.encoders.base import BaseEncoder, register_encoder

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import DataStructs
    from rdkit.Chem.AtomPairs import Pairs, Torsions
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.getLogger(__name__).warning("RDKit未安装，分子编码器功能将不可用")

@register_encoder("morgan_fingerprint")
class MorganFingerprintEncoder(BaseEncoder):
    """
    Morgan指纹编码器
    
    使用RDKit的Morgan指纹算法将SMILES字符串转换为分子指纹
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048, chiral: bool = False, **kwargs):
        """
        初始化Morgan指纹编码器
        
        参数:
            radius: 指纹半径
            n_bits: 指纹位数
            chiral: 是否考虑手性
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit未安装，无法使用MorganFingerprintEncoder")
            
        self.radius = radius
        self.n_bits = n_bits
        self.chiral = chiral
        self.output_dim = n_bits
        
    def fit(self, data: Union[pd.Series, List[str]]) -> "MorganFingerprintEncoder":
        """
        拟合编码器（对于Morgan指纹，无需拟合）
        
        参数:
            data: SMILES字符串列表或Series
            
        返回:
            拟合后的编码器实例
        """
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        将SMILES字符串转换为Morgan指纹
        
        参数:
            data: SMILES字符串列表或Series
            
        返回:
            指纹数组，形状为(n_samples, n_bits)
        """
        if not self.is_fitted:
            self.fit(data)
            
        # 确保数据是列表
        if isinstance(data, pd.Series):
            smiles_list = data.tolist()
        else:
            smiles_list = data
            
        # 计算指纹
        fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"无法解析SMILES: {smiles}，将使用零向量")
                fp = np.zeros(self.n_bits)
            else:
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.n_bits, useChirality=self.chiral
                )
                fp = np.zeros(self.n_bits)
                DataStructs.ConvertToNumpyArray(morgan_fp, fp)
            fps.append(fp)
            
        return np.array(fps)
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        返回:
            输出特征的维度
        """
        return self.n_bits
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        返回:
            特征名称列表
        """
        return [f"morgan_fp_{i}" for i in range(self.n_bits)]

@register_encoder("rdkit_fingerprint")
class RDKitFingerprintEncoder(BaseEncoder):
    """
    RDKit指纹编码器
    
    使用RDKit的指纹算法将SMILES字符串转换为分子指纹
    """
    
    def __init__(self, min_path: int = 1, max_path: int = 7, n_bits: int = 2048, **kwargs):
        """
        初始化RDKit指纹编码器
        
        参数:
            min_path: 最小路径长度
            max_path: 最大路径长度
            n_bits: 指纹位数
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit未安装，无法使用RDKitFingerprintEncoder")
            
        self.min_path = min_path
        self.max_path = max_path
        self.n_bits = n_bits
        self.output_dim = n_bits
        
    def fit(self, data: Union[pd.Series, List[str]]) -> "RDKitFingerprintEncoder":
        """
        拟合编码器（对于RDKit指纹，无需拟合）
        
        参数:
            data: SMILES字符串列表或Series
            
        返回:
            拟合后的编码器实例
        """
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        将SMILES字符串转换为RDKit指纹
        
        参数:
            data: SMILES字符串列表或Series
            
        返回:
            指纹数组，形状为(n_samples, n_bits)
        """
        if not self.is_fitted:
            self.fit(data)
            
        # 确保数据是列表
        if isinstance(data, pd.Series):
            smiles_list = data.tolist()
        else:
            smiles_list = data
            
        # 计算指纹
        fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"无法解析SMILES: {smiles}，将使用零向量")
                fp = np.zeros(self.n_bits)
            else:
                rdkit_fp = Chem.RDKFingerprint(
                    mol, minPath=self.min_path, maxPath=self.max_path, fpSize=self.n_bits
                )
                fp = np.zeros(self.n_bits)
                DataStructs.ConvertToNumpyArray(rdkit_fp, fp)
            fps.append(fp)
            
        return np.array(fps)
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        返回:
            输出特征的维度
        """
        return self.n_bits
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        返回:
            特征名称列表
        """
        return [f"rdkit_fp_{i}" for i in range(self.n_bits)]

@register_encoder("maccs_keys")
class MACCSKeysEncoder(BaseEncoder):
    """
    MACCS Keys编码器
    
    使用RDKit的MACCS Keys算法将SMILES字符串转换为分子指纹
    """
    
    def __init__(self, **kwargs):
        """
        初始化MACCS Keys编码器
        
        参数:
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit未安装，无法使用MACCSKeysEncoder")
            
        self.output_dim = 167  # MACCS Keys固定为167位
        
    def fit(self, data: Union[pd.Series, List[str]]) -> "MACCSKeysEncoder":
        """
        拟合编码器（对于MACCS Keys，无需拟合）
        
        参数:
            data: SMILES字符串列表或Series
            
        返回:
            拟合后的编码器实例
        """
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        将SMILES字符串转换为MACCS Keys
        
        参数:
            data: SMILES字符串列表或Series
            
        返回:
            指纹数组，形状为(n_samples, 167)
        """
        if not self.is_fitted:
            self.fit(data)
            
        # 确保数据是列表
        if isinstance(data, pd.Series):
            smiles_list = data.tolist()
        else:
            smiles_list = data
            
        # 计算指纹
        fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"无法解析SMILES: {smiles}，将使用零向量")
                fp = np.zeros(167)
            else:
                maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
                fp = np.zeros(167)
                DataStructs.ConvertToNumpyArray(maccs_fp, fp)
            fps.append(fp)
            
        return np.array(fps)
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        返回:
            输出特征的维度
        """
        return 167
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        返回:
            特征名称列表
        """
        return [f"maccs_key_{i}" for i in range(167)] 