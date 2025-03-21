"""
ChemPredictor核心模块 - 提供主要的预测器类
"""

from __future__ import annotations

# 标准库导入
import os
import json
import logging
from typing import Dict, Any, Union, Optional
from pathlib import Path

# 第三方库导入
import yaml

# 本地模块导入
from .pipeline.builder import build_pipeline
from .utils.logging import setup_logging
from .exceptions import ConfigError, ModelError

class ChemPredictor:
    """
    化学性质预测器主类
    
    提供用于训练模型和预测化学性质的高级接口
    
    Attributes:
        config: 配置字典
        pipeline: 处理流水线实例
        logger: 日志记录器
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None, 
        config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化ChemPredictor
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典，与config_path互斥
            
        Raises:
            ConfigError: 当配置加载失败时
        """
        try:
            self.config = self._load_config(config_path, config_dict)
            self._setup_logging()
            self.logger = logging.getLogger(__name__)
            self.logger.info("Initializing ChemPredictor")
            self.pipeline = self._build_pipeline()
        except Exception as e:
            raise ConfigError(f"Initialization failed: {str(e)}") from e
            
    def _load_config(
        self, 
        config_path: Optional[str], 
        config_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """加载配置"""
        if config_path and config_dict:
            raise ConfigError("Can only provide either config_path or config_dict")
            
        if config_path:
            return self._load_config_from_file(config_path)
        elif config_dict:
            return config_dict
        else:
            return self._load_default_config()
            
    def _load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """从文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load config file {config_path}: {str(e)}") from e
            
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        default_config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        return self._load_config_from_file(str(default_config_path))
        
    def _setup_logging(self) -> None:
        """设置日志系统"""
        if 'logging' in self.config:
            setup_logging(self.config['logging'])
        else:
            setup_logging()
            
    def _build_pipeline(self):
        """构建处理流水线"""
        try:
            pipeline_config = self.config.get('pipeline', {})
            if not isinstance(pipeline_config, dict):
                raise ValueError("Pipeline configuration must be a dictionary")
            
            steps = pipeline_config.get('steps', {})
            if not isinstance(steps, dict):
                raise ValueError("Pipeline.steps configuration must be a dictionary")
            
            # 确保配置格式正确并扁平化
            for step_name, step_config in steps.items():
                if not isinstance(step_config, dict):
                    raise ValueError(f"Configuration for step '{step_name}' must be a dictionary")
                
                # 扁平化数据加载配置
                if step_name == 'data_loading':
                    file_config = step_config.get('file_config', {})
                    loader_config = step_config.get('loader_config', {})
                    if file_config or loader_config:
                        # 合并file_config和loader_config
                        steps[step_name] = {**file_config, **loader_config}
                        self.logger.info(f"Flattened data loading configuration: {steps[step_name]}")
                    else:
                        # 如果没有嵌套结构，保持原样
                        self.logger.info(f"Using original data loading configuration: {step_config}")
            
            return build_pipeline(pipeline_config)
        except Exception as e:
            raise ModelError(f"Failed to build pipeline: {str(e)}") from e
            
    def train(self, data_path: str) -> None:
        """
        使用提供的数据训练模型
        
        Args:
            data_path: 训练数据文件路径
            
        Raises:
            ModelError: 训练过程中出现错误
        """
        try:
            self.logger.info(f"Starting model training with data: {data_path}")
            self.pipeline.fit(data_path)
            self._save_model_if_configured()
        except Exception as e:
            raise ModelError(f"Training failed: {str(e)}") from e
            
    def _save_model_if_configured(self) -> None:
        """如果配置中指定了保存模型，则保存"""
        if self.config.get('output', {}).get('save_model', False):
            model_path = Path(self.config.get('output', {}).get('model_path', 'models/'))
            model_path.mkdir(parents=True, exist_ok=True)
            save_path = model_path / 'model.pkl'
            self.pipeline.save(str(save_path))
            self.logger.info(f"Model saved to: {save_path}")
    
    def predict(self, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        对新数据进行预测
        
        参数:
            data: 数据文件路径或包含特征的字典
            
        返回:
            包含预测结果的字典
        """
        self.logger.info("Starting prediction")
        predictions = self.pipeline.predict(data)
        
        # 如果配置中指定了保存预测结果
        if isinstance(data, str) and self.config.get('output', {}).get('predictions_path'):
            predictions_path = self.config.get('output', {}).get('predictions_path')
            os.makedirs(predictions_path, exist_ok=True)
            
            base_filename = os.path.basename(data).split('.')[0]
            output_file = os.path.join(predictions_path, f"{base_filename}_predictions.json")
            
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2)
            
            self.logger.info(f"Predictions saved to: {output_file}")
        
        return predictions
    
    def evaluate(self, data_path: str) -> Dict[str, Any]:
        """
        评估模型在给定数据上的性能
        
        参数:
            data_path: 评估数据文件路径
            
        返回:
            包含评估指标的字典
        """
        self.logger.info(f"Starting model evaluation with data: {data_path}")
        return self.pipeline.evaluate(data_path)
    
    def save(self, path: str) -> None:
        """
        保存模型到指定路径
        
        参数:
            path: 保存路径
        """
        self.logger.info(f"Saving model to: {path}")
        self.pipeline.save(path)
    
    def load(self, path: str) -> None:
        """
        从指定路径加载模型
        
        参数:
            path: 模型路径
        """
        self.logger.info(f"Loading model from: {path}")
        self.pipeline.load(path) 