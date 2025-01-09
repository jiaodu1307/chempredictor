import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from abc import ABC, abstractmethod

from chemprop.nn.agg import SumAggregation
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn.message_passing.base import BondMessagePassing
from chemprop.data import MoleculeDatapoint, BatchMolGraph, MoleculeDataset


class BaseEncoder(ABC, nn.Module):
    """所有编码器的基类"""
    @abstractmethod
    def forward(self, x):
        pass

    @property
    @abstractmethod
    def output_dim(self):
        pass


class MorganFingerprintEncoder(BaseEncoder):
    """用于分子的Morgan指纹编码器"""
    def __init__(self, radius=2, n_bits=2048):
        super().__init__()
        self.radius = radius
        self.n_bits = n_bits
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
    
    def _smiles_to_fp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        
        mol = Chem.AddHs(mol)
        fp = self.morgan_gen.GetFingerprint(mol)
        fp_array = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        return fp_array
    
    def forward(self, smiles_list):
        """将SMILES列表转换为指纹张量"""
        fps = [self._smiles_to_fp(s) for s in smiles_list]
        return torch.tensor(fps, dtype=torch.float32)
    
    @property
    def output_dim(self):
        return self.n_bits


class OneHotEncoder(BaseEncoder):
    """用于分类特征的一热编码器"""
    def __init__(self, n_categories):
        super().__init__()
        self.n_categories = n_categories
    
    def forward(self, x):
        """将整数类别转换为一热张量"""
        x = torch.tensor(x, dtype=torch.long)
        return torch.nn.functional.one_hot(x, num_classes=self.n_categories).float()
    
    @property
    def output_dim(self):
        return self.n_categories


class NumericalEncoder(BaseEncoder):
    """用于数值特征的简单编码器，可选归一化"""
    def __init__(self, input_dim=1, normalize=True):
        super().__init__()
        self.input_dim = input_dim
        self.normalize = normalize
        if normalize:
            self.norm = nn.BatchNorm1d(input_dim)
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if self.normalize:
            x = self.norm(x)
        return x
    
    @property
    def output_dim(self):
        return self.input_dim


class MultiModalEncoder(BaseEncoder):
    """组合不同类型输入的多个编码器"""
    def __init__(self, encoders):
        """
        参数:
            encoders: {特征名称: 编码器实例}的字典
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
    
    def forward(self, inputs_dict):
        """
        参数:
            inputs_dict: {特征名称: 特征数据}的字典
        返回:
            连接的编码特征
        """
        encoded = []
        for name, encoder in self.encoders.items():
            if name in inputs_dict:
                encoded.append(encoder(inputs_dict[name]))
        return torch.cat(encoded, dim=-1)
    
    @property
    def output_dim(self):
        return sum(encoder.output_dim for encoder in self.encoders.values())


class MPNNEncoder(BaseEncoder):
    """基于消息传递神经网络的分子编码器"""
    def __init__(self, hidden_size=300, depth=3, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 使用Chemprop的分子图特征提取器
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        
        # 使用Chemprop的消息传递层
        self.encoder = BondMessagePassing(
            d_v=self.featurizer.atom_fdim,  # 原子特征维度
            d_e=self.featurizer.bond_fdim,  # 键特征维度
            d_h=hidden_size,     # 隐藏层维度
            bias=True,           # 使用偏置项
            depth=depth,         # 消息传递迭代次数
            dropout=dropout,     # dropout概率
            activation="relu",   # 激活函数
            undirected=False,    # 有向边
            d_vd=None           # 不使用额外的原子描述符
        )
        
        # 使用Chemprop的聚合层
        self.aggregator = SumAggregation()
    
    def forward(self, smiles_list):
        """
        将SMILES列表转换为分子表示
        
        参数:
            smiles_list: SMILES字符串列表
        返回:
            分子的图表示
        """
        # 创建分子数据点
        datapoints = [MoleculeDatapoint.from_smi(smi) for smi in smiles_list]
        dataset = MoleculeDataset(datapoints)
        
        # 使用特征提取器获取分子图
        batch = self.featurizer(dataset)
        
        # 使用MPNN进行编码,不使用额外的原子描述符
        node_features = self.encoder(batch, V_d=None)
        
        # 聚合得到分子表示
        mol_vecs = self.aggregator(node_features, batch)
        
        return mol_vecs
    
    @property
    def output_dim(self):
        return self.hidden_size
