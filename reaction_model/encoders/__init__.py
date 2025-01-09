from .base import BaseEncoder
from .molecule.fingerprint import MorganFingerprintEncoder
from .molecule.mpnn import MPNNEncoder
from .categorical import OneHotEncoder
from .numerical import NumericalEncoder

# 编码器类型映射
ENCODER_REGISTRY = {
    'fingerprint': MorganFingerprintEncoder,
    'mpnn': MPNNEncoder,
    'onehot': OneHotEncoder,
    'numerical': NumericalEncoder
}

def get_encoder(encoder_type: str) -> Type[BaseEncoder]:
    """获取编码器类"""
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"未知的编码器类型: {encoder_type}")
    return ENCODER_REGISTRY[encoder_type]

__all__ = [
    'BaseEncoder',
    'MorganFingerprintEncoder',
    'MPNNEncoder',
    'OneHotEncoder',
    'NumericalEncoder',
    'get_encoder'
] 