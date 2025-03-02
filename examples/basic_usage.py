#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基本使用示例 - 展示ChemPredictor的基本功能
"""

import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chempredictor import ChemPredictor

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 使用默认配置初始化预测器
    predictor = ChemPredictor()
    
    # 训练模型
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example_reactions.csv')
    predictor.train(data_path)
    
    # 预测新数据
    # 方法1：使用文件
    results = predictor.predict(data_path)
    print("\n预测结果 (使用文件):")
    print(f"预测值: {results['predictions'][:5]}...")  # 只显示前5个结果
    
    # 方法2：使用字典
    single_prediction = predictor.predict({
        "Reactant": "CCO",
        "Solvent": "Water",
        "Temperature": 100
    })
    print("\n预测结果 (使用字典):")
    print(f"预测值: {single_prediction['predictions']}")
    
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
    
    # 保存模型
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'example_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    predictor.save(model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 加载模型
    new_predictor = ChemPredictor()
    new_predictor.load(model_path)
    
    # 使用加载的模型进行预测
    new_results = new_predictor.predict({
        "Reactant": "C1=CC=CC=C1",
        "Solvent": "Ethanol",
        "Temperature": 80
    })
    print("\n使用加载的模型预测:")
    print(f"预测值: {new_results['predictions']}")

if __name__ == "__main__":
    main() 