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
    logger.info("构建管道")
    
    # 验证配置
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是字典，而不是{type(config)}")
    
    # 检查必要的配置项
    steps = config.get('steps', {})
    if not steps:
        logger.warning("配置中没有定义步骤，将使用默认设置")
    
    # 创建管道
    return Pipeline(config) 