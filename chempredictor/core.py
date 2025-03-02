"""
ChemPredictor核心模块 - 提供主要的预测器类
"""

import os
import yaml
import logging
from typing import Dict, Any, Union, Optional

from chempredictor.pipeline.builder import build_pipeline
from chempredictor.utils.logging import setup_logging

class ChemPredictor:
    """
    化学性质预测器主类
    
    提供用于训练模型和预测化学性质的高级接口
    """
    
    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        """
        初始化ChemPredictor
        
        参数:
            config_path: 配置文件路径
            config_dict: 配置字典，与config_path互斥
        """
        if config_path and config_dict:
            raise ValueError("只能提供config_path或config_dict中的一个")
            
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            # 使用默认配置
            default_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'configs', 'default.yaml'
            )
            with open(default_config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        # 设置日志
        if 'logging' in self.config:
            setup_logging(self.config['logging'])
        else:
            setup_logging()
            
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化ChemPredictor")
        
        # 构建pipeline
        self.pipeline = build_pipeline(self.config['pipeline'])
        
    def train(self, data_path: str) -> None:
        """
        使用提供的数据训练模型
        
        参数:
            data_path: 训练数据文件路径
        """
        self.logger.info(f"开始训练模型，使用数据: {data_path}")
        self.pipeline.fit(data_path)
        
        # 如果配置中指定了保存模型
        if self.config.get('output', {}).get('save_model', False):
            model_path = self.config.get('output', {}).get('model_path', 'models/')
            os.makedirs(model_path, exist_ok=True)
            self.pipeline.save(os.path.join(model_path, 'model.pkl'))
            self.logger.info(f"模型已保存到: {model_path}")
    
    def predict(self, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        对新数据进行预测
        
        参数:
            data: 数据文件路径或包含特征的字典
            
        返回:
            包含预测结果的字典
        """
        self.logger.info("开始预测")
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
            
            self.logger.info(f"预测结果已保存到: {output_file}")
        
        return predictions
    
    def evaluate(self, data_path: str) -> Dict[str, Any]:
        """
        评估模型在给定数据上的性能
        
        参数:
            data_path: 评估数据文件路径
            
        返回:
            包含评估指标的字典
        """
        self.logger.info(f"开始评估模型，使用数据: {data_path}")
        return self.pipeline.evaluate(data_path)
    
    def save(self, path: str) -> None:
        """
        保存模型到指定路径
        
        参数:
            path: 保存路径
        """
        self.logger.info(f"保存模型到: {path}")
        self.pipeline.save(path)
    
    def load(self, path: str) -> None:
        """
        从指定路径加载模型
        
        参数:
            path: 模型路径
        """
        self.logger.info(f"从{path}加载模型")
        self.pipeline.load(path) 