"""
配置管理模块
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from .exceptions import ConfigError

class Config:
    """配置管理类"""
    
    DEFAULT_CONFIG = {
        'random_seed': 42,
        'device': 'auto',
        'logging': {
            'level': 'INFO',
            'file': 'logs/chempredictor.log'
        },
        'pipeline': {
            'steps': {
                'data_loading': {
                    'batch_size': 32,
                    'num_workers': 4
                },
                'feature_encoding': {
                    'smiles': {
                        'encoder': 'morgan_fingerprint',
                        'params': {
                            'radius': 2,
                            'num_bits': 2048
                        }
                    }
                },
                'model_training': {
                    'type': 'neural_network',
                    'task_type': 'regression',
                    'params': {
                        'learning_rate': 0.001,
                        'num_epochs': 100,
                        'early_stopping_patience': 10
                    }
                }
            }
        },
        'output': {
            'save_model': True,
            'model_path': 'models/',
            'predictions_path': 'predictions/'
        }
    }
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典
            
        Raises:
            ConfigError: 配置加载失败时
        """
        self.config = self._load_config(config_path, config_dict)
        
    def _load_config(
        self,
        config_path: Optional[str],
        config_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """加载配置"""
        if config_path and config_dict:
            raise ConfigError("只能提供config_path或config_dict中的一个")
            
        if config_path:
            return self._load_from_file(config_path)
        elif config_dict:
            return self._merge_with_default(config_dict)
        else:
            return self.DEFAULT_CONFIG.copy()
            
    def _load_from_file(self, config_path: str) -> Dict[str, Any]:
        """从文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            return self._merge_with_default(user_config)
        except Exception as e:
            raise ConfigError(f"无法加载配置文件 {config_path}: {str(e)}")
            
    def _merge_with_default(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """将用户配置与默认配置合并"""
        config = self.DEFAULT_CONFIG.copy()
        self._deep_update(config, user_config)
        return config
        
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """递归更新字典"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
                
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
        
    def __getitem__(self, key: str) -> Any:
        """获取配置值"""
        return self.config[key]
        
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        return key in self.config 