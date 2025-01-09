from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import os
from .exceptions import ConfigError

@dataclass
class EncoderConfig:
    """编码器配置"""
    type: str
    params: Dict = None
    preprocessing: Dict = None

@dataclass
class ModelConfig:
    """模型配置"""
    type: str
    hidden_dims: List[int]
    dropout: float = 0.1
    activation: str = 'relu'

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    early_stopping: bool = True
    patience: int = 10
    
@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    data_path: Path
    feature_configs: Dict[str, EncoderConfig]
    model_config: ModelConfig
    training_config: TrainingConfig
    output_dir: Path
    seed: int = 42 

    def validate(self):
        """验证配置的有效性"""
        # 验证路径
        if not self.data_path.exists():
            raise ConfigError(f"数据路径不存在: {self.data_path}")
        
        # 验证特征配置
        if not self.feature_configs:
            raise ConfigError("未指定特征配置")
        
        # 验证模型配置
        if not self.model_config.hidden_dims:
            raise ConfigError("未指定模型隐藏层维度")
        
        # 验证训练配置
        if self.training_config.batch_size <= 0:
            raise ConfigError(f"批次大小必须大于0: {self.training_config.batch_size}")
        if self.training_config.max_epochs <= 0:
            raise ConfigError(f"最大训练轮数必须大于0: {self.training_config.max_epochs}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """从YAML文件加载配置"""
        if not os.path.exists(yaml_path):
            raise ConfigError(f"配置文件不存在: {yaml_path}")
            
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML文件格式错误: {e}")
        
        try:
            # 转换路径字符串为Path对象
            config_dict['data_path'] = Path(config_dict['data_path'])
            config_dict['output_dir'] = Path(config_dict['output_dir'])
            
            # 创建配置实例
            config = cls(**config_dict)
            # 验证配置
            config.validate()
            return config
        except (KeyError, TypeError) as e:
            raise ConfigError(f"配置格式错误: {e}") 