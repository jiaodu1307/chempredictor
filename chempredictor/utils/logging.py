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
        final_config = DEFAULT_LOG_CONFIG.copy()
    else:
        # 确保配置包含必要的字段
        final_config = DEFAULT_LOG_CONFIG.copy()
        
        # 如果传入的配置是简单格式，转换为完整格式
        if 'version' not in config:
            # 处理简单格式的配置
            handlers = []
            if config.get('file'):
                final_config['handlers']['file']['filename'] = config['file']
                handlers.append('file')
            if config.get('level'):
                final_config['loggers']['chempredictor']['level'] = config['level']
                for handler in final_config['handlers'].values():
                    handler['level'] = config['level']
            if handlers:
                final_config['loggers']['chempredictor']['handlers'] = handlers
        else:
            # 如果是完整格式，合并配置
            for section in ['formatters', 'handlers', 'loggers']:
                if section in config:
                    final_config[section].update(config[section])
    
    # 设置日志文件路径
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        final_config['handlers']['file']['filename'] = str(log_path / 'chempredictor.log')
    
    # 应用配置
    logging.config.dictConfig(final_config)
    logger = logging.getLogger('chempredictor')
    logger.info('日志系统初始化完成') 