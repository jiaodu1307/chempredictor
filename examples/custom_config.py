#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自定义配置示例 - 展示如何使用自定义配置
"""

import os
import sys
import logging
import yaml

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chempredictor import ChemPredictor

def create_custom_config():
    """创建自定义配置"""
    config = {
        "pipeline": {
            "steps": {
                "data_loading": {
                    "file_type": "csv",
                    "target_column": "Yield",
                    "feature_columns": ["Reactant", "Solvent", "Temperature"],
                    "missing_value_strategy": "mean"
                },
                "feature_encoding": {
                    "Reactant": {
                        "encoder": "morgan_fingerprint",
                        "params": {
                            "radius": 3,
                            "n_bits": 1024,
                            "chiral": True
                        }
                    },
                    "Solvent": {
                        "encoder": "onehot_encoder",
                        "params": {
                            "handle_unknown": "ignore"
                        }
                    },
                    "Temperature": {
                        "encoder": "minmax_scaler",
                        "params": {
                            "feature_range": (0, 1)
                        }
                    }
                },
                "model_training": {
                    "type": "random_forest",
                    "task_type": "regression",
                    "params": {
                        "n_estimators": 200,
                        "max_depth": 10,
                        "random_state": 42
                    },
                    "cv": {
                        "method": "kfold",
                        "n_splits": 5,
                        "shuffle": True,
                        "random_state": 42
                    }
                },
                "evaluation": {
                    "metrics": {
                        "regression": ["rmse", "r2", "mae"],
                        "classification": ["accuracy", "f1", "roc_auc"]
                    },
                    "feature_importance": True,
                    "shap_analysis": False
                }
            }
        },
        "logging": {
            "level": "INFO",
            "file": "logs/custom_config.log"
        },
        "output": {
            "save_model": True,
            "model_path": "models/custom_model.pkl",
            "predictions_path": "results/",
            "report_format": "json"
        }
    }
    
    # 保存配置到文件
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'custom_config.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config_path

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建自定义配置
    config_path = create_custom_config()
    print(f"已创建自定义配置: {config_path}")
    
    # 使用配置文件初始化预测器
    predictor = ChemPredictor(config_path=config_path)
    
    # 训练模型
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example_reactions.csv')
    predictor.train(data_path)
    
    # 预测新数据
    results = predictor.predict({
        "Reactant": "CC(=O)O",
        "Solvent": "Methanol",
        "Temperature": 90
    })
    
    print("\n预测结果:")
    print(f"预测值: {results['predictions']}")
    
    # 评估模型
    eval_results = predictor.evaluate(data_path)
    print("\n评估结果:")
    for metric, value in eval_results.items():
        if metric != 'feature_importance' and metric != 'shap_values':
            print(f"{metric}: {value}")
    
    # 特征重要性
    if 'feature_importance' in eval_results:
        print("\n特征重要性 (前5个):")
        for i, (feature, importance) in enumerate(eval_results['feature_importance'].items()):
            if i >= 5:
                break
            print(f"{feature}: {importance:.4f}")
    
    print("\n完成!")

if __name__ == "__main__":
    main() 