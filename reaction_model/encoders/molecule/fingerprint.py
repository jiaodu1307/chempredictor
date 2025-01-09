from ..base import BaseEncoder
import torch
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
import numpy as np

class MorganFingerprintEncoder(BaseEncoder):
    """Morgan指纹编码器"""
    def __init__(self, radius: int = 2, n_bits: int = 2048, **kwargs):
        super().__init__(radius=radius, n_bits=n_bits, **kwargs)
        self.radius = radius
        self.n_bits = n_bits
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
    
    def _smiles_to_fp(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        
        mol = Chem.AddHs(mol)
        fp = self.morgan_gen.GetFingerprint(mol)
        fp_array = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        return fp_array
    
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        fps = [self._smiles_to_fp(s) for s in smiles_list]
        return torch.tensor(fps, dtype=torch.float32)
    
    @property
    def output_dim(self) -> int:
        return self.n_bits 