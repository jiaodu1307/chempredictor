"""
日志工具模块
"""

import os
import logging
from typing import Dict, Any, Optional

def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    设置日志配置
    
    参数:
        config: 日志配置字典，包含level和file等键
    """
    if config is None:
        config = {}
    
    # 获取日志级别
    level_str = config.get('level', 'INFO')
    level = getattr(logging, level_str.upper())
    
    # 基本配置
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 如果指定了日志文件
    log_file = config.get('file')
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        ))
        
        # 添加到根日志器
        logging.getLogger('').addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('rdkit').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.debug("日志系统已初始化") 