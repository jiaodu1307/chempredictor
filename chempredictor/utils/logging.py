"""
日志配置模块
"""

import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'chempredictor.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        'chempredictor': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console']
    }
}

def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None
) -> None:
    """
    设置日志系统
    
    Args:
        config: 日志配置字典，如果为None则使用默认配置
        log_dir: 日志文件目录，如果为None则使用当前目录
    """
    if config is None:
        config = DEFAULT_LOG_CONFIG.copy()
    
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        config['handlers']['file']['filename'] = str(log_path / 'chempredictor.log')
    
    logging.config.dictConfig(config)
    logger = logging.getLogger('chempredictor')
    logger.info('日志系统初始化完成') 