#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Doyle-Buchwald-Hartwig反应产率预测示例 - 使用MLP模型
"""

import os
import sys
import logging
import yaml
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chempredictor import ChemPredictor

def create_config():
    """创建配置"""
    # 获取用户主目录
    user_home = os.path.expanduser('~')
    chempredictor_home = os.path.join(user_home, '.chempredictor')
    
    config = {
        # 添加全局随机数种子设置
        "random_seed": 42,
        
        "pipeline": {
            "steps": {
                "data_loading": {
                    "file_type": "csv",
                    "target_column": "Output",
                    "feature_columns": ["Ligand", "Additive", "Base", "Aryl_halide"],
                    "missing_value_strategy": "drop"
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
                        "encoder": "morgan_fingerprint",
                        "params": {
                            "radius": 2,
                            "n_bits": 1024,
                            "chiral": True
                        }
                    },
                    "Base": {
                        "encoder": "morgan_fingerprint",
                        "params": {
                            "radius": 2,
                            "n_bits": 1024,
                            "chiral": True
                        }
                    },
                    "Aryl_halide": {
                        "encoder": "morgan_fingerprint",
                        "params": {
                            "radius": 2,
                            "n_bits": 1024,
                            "chiral": True
                        }
                    }
                },
                "model_training": {
                    "type": "mlp",
                    "task_type": "regression",
                    "device": "cuda",  # 可选值: "auto", "cpu", "cuda"
                    "params": {
                        # PyTorch Lightning MLP参数
                        "hidden_layer_sizes": [256],
                        "activation": "relu",
                        "learning_rate": 0.001,  # 对应LightningMLP中的learning_rate参数
                        "weight_decay": 0.0001,  # 对应LightningMLP中的weight_decay参数
                        "batch_size": 32,
                        "max_epochs": 40,  # 改为max_epochs以匹配PyTorch Lightning
                        "validation_fraction": 0.1,
                        "patience": 10,  # 改为patience以匹配EarlyStopping回调
                        "random_state": 2,
                        "verbose": True,
                        
                        # 优化器和学习率调度器的额外参数
                        "optimizer": {
                            "type": "adam",
                            "betas": [0.9, 0.999],
                            "eps": 1e-8
                        },
                        "scheduler": {
                            "factor": 0.5,
                            "patience": 5,
                            "min_lr": 1e-6,
                            "threshold": 1e-4
                        }
                    }
                },
                "evaluation": {
                    "metrics": {
                        "regression": ["rmse", "mae", "r2"]
                    },
                    "feature_importance": False,
                    "shap_analysis": False
                }
            }
        },
        "logging": {
            "level": "INFO",
            "file": "logs/doyle_buchwald_hartwig_mlp.log"
        },
        "output": {
            "save_model": True,
            "model_path": os.path.join(chempredictor_home, "models", "doyle_buchwald_hartwig_mlp_model.pkl"),
            "predictions_path": os.path.join(chempredictor_home, "results"),
            "report_format": "json"
        }
    }
    
    # 创建必要的目录
    os.makedirs(os.path.join(chempredictor_home, "models"), exist_ok=True)
    os.makedirs(os.path.join(chempredictor_home, "results"), exist_ok=True)
    os.makedirs(os.path.join(chempredictor_home, "configs"), exist_ok=True)
    
    # 保存配置到文件
    config_path = os.path.join(chempredictor_home, "configs", "doyle_buchwald_hartwig_mlp_config.yaml")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config_path

def format_predictions(predictions) -> str:
    """格式化预测结果"""
    if isinstance(predictions, (list, np.ndarray)):
        if len(predictions) == 1:
            return f"{predictions[0]:.2f}%"
        else:
            return "\n".join(f"- 样本 {i+1}: {pred:.2f}%" for i, pred in enumerate(predictions))
    else:
        return f"{predictions:.2f}%"

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 获取用户主目录
    user_home = os.path.expanduser('~')
    chempredictor_home = os.path.join(user_home, 'reaction_model')
    
    # 检测是否可用
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        print(f"\n===== 硬件信息 =====")
        print(f"检测到可用设备: {device.upper()}")
        if cuda_available:
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    except ImportError:
        device = "cpu"
        print("\n===== 硬件信息 =====")
        print("未检测到PyTorch，将使用CPU进行训练")
    
    # 创建配置
    config_path = create_config()
    
    # 根据实际情况修改配置中的device设置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['pipeline']['steps']['model_training']['device'] = device
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"已创建配置文件: {config_path}")
    print(f"训练设备设置为: {device.upper()}")
    
    # 初始化预测器
    predictor = ChemPredictor(config_path=config_path)
    
    # 训练模型
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'raw', 'doyle_buchwald-hartwig_dataset.csv')
    predictor.train(data_path)
    
    # 评估模型
    print("\n===== 模型评估 =====")
    eval_results = predictor.evaluate(data_path)
    print("评估指标:")
    for metric, value in eval_results.items():
        if metric not in ['feature_importance', 'shap_values']:
            print(f"- {metric}: {value:.4f}")
    
    # 显示SHAP值分析结果
    if 'shap_values' in eval_results:
        print("\nSHAP值分析已完成，结果已保存")
    
    # 获取训练历史
    if hasattr(predictor.pipeline.model, 'get_training_history'):
        history = predictor.pipeline.model.get_training_history()
        print("\n===== 训练历史 =====")
        print(f"总迭代次数: {history['n_iter']}")
        print(f"最终损失: {history['loss'][-1]:.4f}")
        if history['val_loss']:
            print(f"最佳验证损失: {history['best_loss']:.4f} (迭代 {history['best_iter']})")
    
    try:
        # 保存模型
        model_path = os.path.join(chempredictor_home, "models", "doyle_buchwald_hartwig_mlp_model.pkl")
        predictor.save(model_path)
        print(f"\n模型已保存到: {model_path}")
        print(f"所有文件都保存在: {chempredictor_home}")
    except Exception as e:
        print(f"\n保存模型时发生错误: {str(e)}")
        print(f"请确保目录 {chempredictor_home} 存在且有写入权限")

if __name__ == "__main__":
    main() 