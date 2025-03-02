"""
模型模块 - 提供各种预测模型
"""

from chempredictor.models.base import (
    BaseModel, 
    register_model, 
    get_model,
    MODEL_REGISTRY
)

# 导入所有模型，确保它们被注册
from chempredictor.models.traditional import (
    RandomForestModel,
    XGBoostModel,
    LightGBMModel
)

__all__ = [
    "BaseModel",
    "register_model",
    "get_model",
    "MODEL_REGISTRY",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel"
] 