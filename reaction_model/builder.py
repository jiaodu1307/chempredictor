from typing import Dict, Any, Optional
from .utils.config import ExperimentConfig
from .utils.registry import ENCODER_REGISTRY, MODEL_REGISTRY
from .models.base import BaseModel
from .utils.exceptions import ModelError

class ModelBuilder:
    """模型构建器"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.encoders = {}
        self.model = None
    
    def build_encoders(self) -> Dict[str, Any]:
        """构建特征编码器"""
        try:
            for name, encoder_config in self.config.feature_configs.items():
                if encoder_config.type not in ENCODER_REGISTRY:
                    raise ModelError(f"未知的编码器类型: {encoder_config.type}")
                    
                encoder_cls = ENCODER_REGISTRY.get(encoder_config.type)
                self.encoders[name] = encoder_cls(**encoder_config.params)
                
            return self.encoders
        except Exception as e:
            raise ModelError(f"构建编码器失败: {str(e)}")
    
    def build_model(self) -> BaseModel:
        """构建模型"""
        try:
            if not self.encoders:
                self.build_encoders()
                
            if self.config.model_config.type not in MODEL_REGISTRY:
                raise ModelError(f"未知的模型类型: {self.config.model_config.type}")
                
            model_cls = MODEL_REGISTRY.get(self.config.model_config.type)
            self.model = model_cls(
                encoders=self.encoders,
                **self.config.model_config.__dict__
            )
            return self.model
        except Exception as e:
            raise ModelError(f"构建模型失败: {str(e)}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelBuilder':
        """从YAML配置文件创建构建器"""
        config = ExperimentConfig.from_yaml(yaml_path)
        return cls(config) 