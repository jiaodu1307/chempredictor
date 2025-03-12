"""
管道构建模块 - 提供构建管道的功能
"""

import logging
from typing import Dict, Any

from chempredictor.pipeline.pipeline import Pipeline

def build_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    根据配置构建管道
    
    参数:
        config: 管道配置字典
        
    返回:
        构建的管道实例
    """
    logger = logging.getLogger(__name__)
    logger.info("Building pipeline")
    
    # 验证配置
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a dictionary, not {type(config)}")
    
    # 检查必要的配置项
    steps = config.get('steps', {})
    if not steps:
        logger.warning("No steps defined in configuration, using default settings")
    
    # 创建管道
    return Pipeline(config) 