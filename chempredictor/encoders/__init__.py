"""
编码器模块 - 提供各种特征编码方法
"""

from chempredictor.encoders.base import (
    BaseEncoder, 
    register_encoder, 
    get_encoder,
    ENCODER_REGISTRY
)

# 导入所有编码器，确保它们被注册
from chempredictor.encoders.molecular import (
    MorganFingerprintEncoder,
    RDKitFingerprintEncoder,
    MACCSKeysEncoder
)

from chempredictor.encoders.basic import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)

__all__ = [
    "BaseEncoder",
    "register_encoder",
    "get_encoder",
    "ENCODER_REGISTRY",
    "MorganFingerprintEncoder",
    "RDKitFingerprintEncoder",
    "MACCSKeysEncoder",
    "OneHotEncoder",
    "LabelEncoder",
    "StandardScaler",
    "MinMaxScaler"
] 