import os
import logging
import yaml
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from chempredictor import ChemPredictor
from chempredictor.utils.logging import setup_logging
from chempredictor.utils.profiling import profile_memory, log_performance, profile_section
from chempredictor.data_loading import DataLoader

# 设置日志配置
log_config = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'file': {
            'type': 'file',
            'filename': str(Path(__file__).parent.parent / 'logs' / 'whu_suzuki.log'),
            'mode': 'a'
        },
        'console': {
            'type': 'console'
        }
    }
}

randon_seed = 42 # 随机种子

setup_logging(log_config)
logger = logging.getLogger(__name__)

def create_config() -> Dict[str, Any]:
    """创建配置字典"""
    config = {
        'random_seed': randon_seed,
        'device': 'auto',
        'logging': {
            'level': 'INFO',
            'file': 'logs/whu_suzuki.log'
        },
        'pipeline': {
            'steps': {
                'data_loading': {
                    'file_type': 'csv',
                    'target_column': 'Yield',
                    'feature_columns': [
                        'Reactant1_Smiles',
                        'Reactant2_Smiles',
                        'Product_Smiles',
                        'Catalyst_Smiles',
                        'Reagent_Smiles',
                        'Temperature',
                        'order'
                    ],
                    'batch_size': 32,
                    'num_workers': 4,
                    'shuffle': True,
                    'pandas_kwargs': {
                        'index_col': None,
                        'encoding': 'utf-8'
                    }
                },
                'feature_encoding': {
                    'Reactant1_Smiles': {
                        'encoder': 'morgan_fingerprint',
                        'params': {
                            'radius': 2,
                            'n_bits': 1024
                        }
                    },
                    'Reactant2_Smiles': {
                        'encoder': 'morgan_fingerprint',
                        'params': {
                            'radius': 2,
                            'n_bits': 1024
                        }
                    },
                    'Product_Smiles': {
                        'encoder': 'morgan_fingerprint',
                        'params': {
                            'radius': 2,
                            'n_bits': 1024
                        }
                    },
                    'Catalyst_Smiles': {
                        "encoder": "onehot_encoder",
                        "params": {
                            "handle_unknown": "ignore",
                            "sparse_output": False
                        }
                    },
                    'Reagent_Smiles': {
                        'encoder': 'morgan_fingerprint',
                        'params': {
                            'radius': 2,
                            'n_bits': 1024
                        }
                    },
                    'Temperature': {
                        'encoder': 'standard_scaler',
                        'params': {}
                    },
                    'order': {
                        'encoder': 'onehot_encoder',
                        'params': {
                            'handle_unknown': 'ignore',
                            'sparse_output': False
                        }
                    }
                },
                                'model_training': {
                    'type': 'mlp',
                    'task_type': 'regression',
                    'params': {
                        'device': 'cuda',
                        'hidden_layer_sizes': [512],
                        'activation': 'relu',
                        'learning_rate': 0.001,
                        'weight_decay': 0.0001,
                        'batch_size': 32,
                        'max_epochs': 200,
                        'validation_fraction': 0.1,
                        'patience': 15,
                        'random_state': randon_seed,
                        'verbose': True,
                        'optimizer': {
                            'type': 'adam',
                            'betas': [0.9, 0.999],
                            'eps': 1e-8
                        },
                        'scheduler': {
                            'type': 'reduce_on_plateau',
                            'factor': 0.5,
                            'patience': 7,
                            'min_lr': 1e-6,
                            'threshold': 1e-4
                        }
                    }
                },
                'evaluation': {
                    'metrics': {
                        "regression": ["rmse", "mae", "r2", "mse"]
                    },
                    'feature_importance': False,
                    'shap_analysis': False,
                    'cross_validation': {
                        'enabled': True,
                        'n_splits': 5,
                        'shuffle': True
                    }
                }
            }
        },
        'output': {
            'save_model': True,
            'model_path': 'models/whu_suzuki/',
            'predictions_path': 'predictions/whu_suzuki/'
        }
    }
    
    return config

def save_config(config: Dict[str, Any], path: str = 'configs/whu_suzuki.yaml') -> None:
    """保存配置到YAML文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    return path

@log_performance
def format_predictions(predictions: Dict[str, Any]) -> str:
    """
    格式化预测结果
    
    Args:
        predictions: 预测结果字典，包含评估指标
        
    Returns:
        str: 格式化后的评估指标字符串
    """
    result = []
    
    # 处理评估指标
    if isinstance(predictions, dict):
        result.append("评估指标:")
        for metric_name, value in predictions.items():
            if isinstance(value, dict):
                result.append(f"{metric_name}:")
                for k, v in value.items():
                    result.append(f"  {k}: {v:.4f}")
            else:
                result.append(f"{metric_name}: {value:.4f}")
    
    return "\n".join(result)

def check_hardware() -> Dict[str, Any]:
    """检查硬件配置"""
    import torch
    
    hardware_info = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    }
    
    return hardware_info

@log_performance
def train_model():
    """训练武汉大学铃木偶联反应模型"""
    logger = logging.getLogger(__name__)
    
    # 数据加载配置
    data_loading_config = {
        'file_type': 'csv',
        'target_column': 'Yield',
        'feature_columns': [
            'Reactant1_Smiles',
            'Reactant2_Smiles',
            'Product_Smiles',
            'Catalyst_Smiles',
            'Reagent_Smiles',
            'Temperature',
            'order'
        ],
        'missing_value_strategy': 'mean',
        'batch_size': 32,
        'num_workers': 4,
        'shuffle': True,
        'pandas_kwargs': {
            'index_col': None,
            'encoding': 'utf-8'
        }
    }
    
    # 初始化数据加载器
    data_loader = DataLoader(**data_loading_config)
    
    # 加载训练数据
    project_root = Path(__file__).parent.parent
    train_data = project_root / 'data' / 'raw' / 'whu_suzuki_dataset.csv'
    data = data_loader.load(str(train_data))
    
    logger.info("开始训练模型...")
    return data

def main():
    """主函数"""
    logger = logging.getLogger(__name__)
    
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
            config = create_config()
            config_path = save_config(config)
            logger.info(f"已创建配置文件: {config_path}")
            
            # 初始化预测器
            predictor = ChemPredictor(config_dict=config)
            
            # 创建必要的目录
            project_root = Path(__file__).parent.parent
            for dir_path in ['data/raw', 'models/whu_suzuki', 'logs', 'configs', 'predictions/whu_suzuki']:
                (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        with profile_section("数据加载与预处理"):
            # 加载数据
            train_data = train_model()
            logger.info(f"数据加载完成，形状: {train_data.shape if hasattr(train_data, 'shape') else '未知'}")
        
        with profile_section("模型训练"):
            # 训练模型
            project_root = Path(__file__).parent.parent
            data_path = project_root / 'data' / 'raw' / 'whu_suzuki_dataset.csv'
            logger.info(f"开始训练模型，使用数据: {data_path}")
            predictor.train(str(data_path))
        
        with profile_section("模型评估"):
            # 评估模型
            logger.info("\n===== 模型评估 =====")
            eval_results = predictor.evaluate(str(data_path))
            
            # 输出评估指标
            logger.info("\n回归性能指标:")
            metrics = ['rmse', 'mae', 'r2', 'mse']
            for metric in metrics:
                if metric in eval_results:
                    logger.info(f"{metric.upper()}: {eval_results[metric]:.4f}")
            
            # 交叉验证结果
            if 'cv_results' in eval_results:
                logger.info("\n交叉验证结果:")
                for metric, values in eval_results['cv_results'].items():
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    logger.info(f"{metric}: {mean_value:.4f} (±{std_value:.4f})")
            
            # 特征重要性分析
            if eval_results.get('feature_importance'):
                logger.info("\n特征重要性排名:")
                importances = eval_results['feature_importance']
                if isinstance(importances, dict):
                    # 按重要性排序
                    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                    for feature, importance in sorted_features:
                        logger.info(f"- {feature}: {importance:.4f}")
            
            # SHAP值分析
            if eval_results.get('shap_values'):
                logger.info("\nSHAP值分析已完成")
                shap_values = eval_results['shap_values']
                if isinstance(shap_values, dict):
                    logger.info("特征SHAP值排名:")
                    sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
                    for feature, value in sorted_shap:
                        logger.info(f"- {feature}: {value:.4f}")
        
        with profile_section("模型保存"):
            # 保存模型
            model_path = project_root / 'models' / 'whu_suzuki' / 'model.pkl'
            predictor.save(str(model_path))
            logger.info(f"模型已保存到: {model_path}")
            
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()