#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基本功能测试
"""

import os
import sys
import unittest
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chempredictor import ChemPredictor

class TestChemPredictor(unittest.TestCase):
    """测试ChemPredictor的基本功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example_reactions.csv')
        
        # 确保数据文件存在
        if not os.path.exists(self.data_path):
            self.skipTest("测试数据文件不存在")
    
    def tearDown(self):
        """测试后的清理工作"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """测试初始化"""
        predictor = ChemPredictor()
        self.assertIsNotNone(predictor)
        self.assertIsNotNone(predictor.config)
        self.assertIsNotNone(predictor.pipeline)
    
    def test_train_predict(self):
        """测试训练和预测"""
        predictor = ChemPredictor()
        
        # 训练模型
        predictor.train(self.data_path)
        
        # 预测
        results = predictor.predict({
            "Reactant": "CCO",
            "Solvent": "Water",
            "Temperature": 100
        })
        
        self.assertIn('predictions', results)
        self.assertIsNotNone(results['predictions'])
    
    def test_evaluate(self):
        """测试评估"""
        predictor = ChemPredictor()
        
        # 训练模型
        predictor.train(self.data_path)
        
        # 评估
        metrics = predictor.evaluate(self.data_path)
        
        # 检查是否包含基本指标
        if predictor.config['pipeline']['steps']['model_training']['task_type'] == 'regression':
            self.assertIn('rmse', metrics)
            self.assertIn('r2', metrics)
        else:
            self.assertIn('accuracy', metrics)
    
    def test_save_load(self):
        """测试保存和加载"""
        predictor = ChemPredictor()
        
        # 训练模型
        predictor.train(self.data_path)
        
        # 保存模型
        model_path = os.path.join(self.temp_dir, 'model.pkl')
        predictor.save(model_path)
        
        # 检查文件是否存在
        self.assertTrue(os.path.exists(model_path))
        
        # 加载模型
        new_predictor = ChemPredictor()
        new_predictor.load(model_path)
        
        # 预测
        results = new_predictor.predict({
            "Reactant": "CCO",
            "Solvent": "Water",
            "Temperature": 100
        })
        
        self.assertIn('predictions', results)
        self.assertIsNotNone(results['predictions'])

if __name__ == '__main__':
    unittest.main()
