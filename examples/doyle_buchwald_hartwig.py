#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Doyle-Buchwald-Hartwig反应产率预测示例 - 使用MLP模型
"""

import os
import sys
import logging
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from chempredictor import ChemPredictor
from chempredictor.utils.profiling import profile_section, log_performance, profile_memory
from chempredictor.data_loading import DataLoader
from chempredictor.utils.logging import setup_logging

# 设置日志配置
setup_logging({
    'level': 'INFO',
    'log_file': str(Path(__file__).parent.parent / 'logs' / 'doyle_buchwald_hartwig.log')
})

logger = logging.getLogger(__name__)

def create_config() -> str:
    """
    创建配置
    
    Returns:
        str: 配置文件路径
    """
    # 获取用户主目录
    chempredictor_home = Path.home() / '.chempredictor'
    
    config = {
        # 添加全局随机数种子设置
        "random_seed": 42,
        
        "pipeline": {
            "steps": {
                "data_loading": {
                    "file_type": "csv",
                    "target_column": "Output",
                    "feature_columns": ["Ligand", "Additive", "Base", "Aryl_halide"],
                    "missing_value_strategy": "drop",
                    "batch_size": 32,
                    "num_workers": 0,
                    "shuffle": True,
                    "pandas_kwargs": {
                        "index_col": None,
                        "encoding": "utf-8"
                    }
                },
                "feature_encoding": {
                    "Ligand": {
                        "encoder": "morgan_fingerprint",
                        "params": {
                            "radius": 2,
                            "n_bits": 2048,
                            "chiral": True
                        }
                    },
                    "Additive": {
                        "encoder": "onehot_encoder",
                        "params": {
                            "handle_unknown": "ignore",
                            "sparse_output": False
                        }
                    },
                    "Base": {
                        "encoder": "onehot_encoder",
                        "params": {
                            "handle_unknown": "ignore",
                            "sparse_output": False
                        }
                    },
                    "Aryl_halide": {
                        "encoder": "onehot_encoder",
                        "params": {
                            "handle_unknown": "ignore",
                            "sparse_output": False
                        }
                    }
                },
                "model_training": {
                    "type": "mlp",
                    "task_type": "regression",
                    "params": {
                        "device": "cuda",
                        "hidden_layer_sizes": [512],
                        "activation": "relu",
                        "learning_rate": 0.001,
                        "weight_decay": 0.0001,
                        "batch_size": 32,
                        "max_epochs": 50,
                        "validation_fraction": 0.2,
                        "patience": 10,
                        "random_state": 42,
                        "verbose": True,
                        "optimizer": {
                            "type": "adam",
                            "betas": [0.9, 0.999],
                            "eps": 1e-8
                        },
                        "scheduler": {
                            "type": "reduce_on_plateau",
                            "factor": 0.5,
                            "patience": 5,
                            "min_lr": 1e-6,
                            "threshold": 1e-4
                        }
                    }
                },
                "evaluation": {
                    "metrics": {
                        "regression": ["rmse", "mae", "r2", "mse"]
                    },
                    "feature_importance": False,
                    "shap_analysis": False,
                    "cross_validation": {
                        "enabled": True,
                        "n_splits": 5,
                        "shuffle": True
                    }
                }
            }
        },
        "logging": {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": str(chempredictor_home / "logs" / "doyle_buchwald_hartwig_mlp.log"),
                    "maxBytes": 10485760,
                    "backupCount": 5
                }
            },
            "loggers": {
                "chempredictor": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console"]
            }
        },
        "output": {
            "save_model": True,
            "model_path": str(chempredictor_home / "models" / "doyle_buchwald_hartwig_mlp_model.pkl"),
            "predictions_path": str(chempredictor_home / "results"),
            "report_format": "json",
            "save_checkpoints": True,
            "checkpoint_dir": str(chempredictor_home / "checkpoints")
        },
        "cache": {
            "enabled": True,
            "dir": str(chempredictor_home / "cache"),
            "max_size": 1024
        }
    }
    
    # 创建必要的目录
    for dir_path in [
        chempredictor_home / "models",
        chempredictor_home / "results",
        chempredictor_home / "configs",
        chempredictor_home / "logs",
        chempredictor_home / "cache",
        chempredictor_home / "checkpoints"
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 保存配置到文件
    config_path = chempredictor_home / "configs" / "doyle_buchwald_hartwig_mlp_config.yaml"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return str(config_path)

@log_performance
def format_predictions(predictions: np.ndarray) -> str:
    """
    格式化预测结果
    
    Args:
        predictions: 预测结果数组
        
    Returns:
        str: 格式化后的预测结果字符串
    """
    if isinstance(predictions, (list, np.ndarray)):
        if len(predictions) == 1:
            return f"{predictions[0]:.2f}%"
        else:
            return "\n".join(f"- 样本 {i+1}: {pred:.2f}%" for i, pred in enumerate(predictions))
    else:
        return f"{predictions:.2f}%"

def check_hardware() -> Dict[str, Any]:
    """
    检查硬件环境
    
    Returns:
        Dict[str, Any]: 硬件信息字典
    """
    with profile_section("硬件检测"):
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            device = "cuda" if cuda_available else "cpu"
            info = {
                "device": device,
                "cuda_available": cuda_available
            }
            if cuda_available:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            info = {
                "device": "cpu",
                "cuda_available": False
            }
    return info

@profile_memory
@log_performance
def train_model():
    """训练Doyle-Buchwald-Hartwig反应模型"""
    
    # 数据加载配置
    data_loading_config = {
        'file_type': 'csv',
        'target_column': 'Output',
        'feature_columns': [
            'Ligand', 'Additive', 'Base', 'Aryl_halide'
        ],
        'missing_value_strategy': 'mean',
        'batch_size': 32,
        'num_workers': 2,
        'shuffle': True,
        'pandas_kwargs': {
            'index_col': None,
            'encoding': 'utf-8'
        }
    }
    
    # 初始化数据加载器
    data_loader = DataLoader(**data_loading_config)
    
    # 加载训练数据
    train_data = data_loader.load('data/doyle_buchwald_hartwig_train.csv')
    
    logger.info("开始训练模型...")
    # ... 模型训练代码 ...

def main():
    """主函数"""
    # 检查硬件环境
    hardware_info = check_hardware()
    logger.info("===== 硬件信息 =====")
    logger.info(f"检测到可用设备: {hardware_info['device'].upper()}")
    if hardware_info['cuda_available']:
        logger.info(f"GPU型号: {hardware_info['gpu_name']}")
        logger.info(f"GPU内存: {hardware_info['gpu_memory'] / 1024**3:.2f} GB")
    
    try:
        with profile_section("配置初始化"):
            # 创建配置
            config_path = create_config()
            logger.info(f"已创建配置文件: {config_path}")
            
            # 初始化预测器
            predictor = ChemPredictor(config_path=config_path)
        
        with profile_section("模型训练"):
            # 训练模型
            project_root = Path(__file__).parent.parent
            data_path = project_root / 'data' / 'raw' / 'doyle_buchwald-hartwig_dataset.csv'
            predictor.train(data_path)
        
        with profile_section("模型评估"):
            # 评估模型
            logger.info("===== 模型评估 =====")
            eval_results = predictor.evaluate(data_path)
            
            # 输出评估指标
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
            
            # 特征重要性分析
            if eval_results.get('feature_importance'):
                logger.info("\n特征重要性排名:")
                for feature, importance in eval_results['feature_importance'].items():
                    logger.info(f"- {feature}: {importance:.4f}")
            
            # SHAP值分析
            if eval_results.get('shap_values'):
                logger.info("\nSHAP值分析已完成，结果已保存")
            
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 