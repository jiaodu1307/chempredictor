"""
管道模块 - 提供数据处理和模型训练的管道
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Optional, Union, Tuple

from chempredictor.data_loading import DataLoader
from chempredictor.encoders import get_encoder
from chempredictor.models import get_model
from chempredictor.evaluation import Evaluator

class Pipeline:
    """
    数据处理和模型训练管道
    
    管理数据加载、特征编码、模型训练和评估的完整流程
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化管道
        
        参数:
            config: 管道配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化管道")
        
        # 初始化组件
        self._init_components()
        
        # 状态标志
        self.is_fitted = False
        self.feature_names = None
        self.encoded_feature_names = None
        
    def _init_components(self) -> None:
        """
        初始化管道组件
        """
        steps = self.config.get('steps', {})
        
        # 数据加载器
        data_loading_config = steps.get('data_loading', {})
        self.data_loader = DataLoader(**data_loading_config)
        
        # 特征编码器
        self.feature_encoders = {}
        feature_encoding_config = steps.get('feature_encoding', {})
        for feature_name, encoder_config in feature_encoding_config.items():
            encoder_type = encoder_config.get('encoder')
            encoder_params = encoder_config.get('params', {})
            self.feature_encoders[feature_name] = get_encoder(encoder_type, **encoder_params)
        
        # 模型
        model_config = steps.get('model_training', {})
        model_type = model_config.get('type')
        task_type = model_config.get('task_type', 'regression')
        model_params = model_config.get('params', {})
        self.model = get_model(model_type, task_type=task_type, **model_params)
        
        # 评估器
        eval_config = steps.get('evaluation', {})
        metrics = eval_config.get('metrics', {}).get(task_type, None)
        feature_importance = eval_config.get('feature_importance', True)
        shap_analysis = eval_config.get('shap_analysis', False)
        self.evaluator = Evaluator(
            task_type=task_type,
            metrics=metrics,
            feature_importance=feature_importance,
            shap_analysis=shap_analysis
        )
        
    def fit(self, data_path: str) -> "Pipeline":
        """
        拟合管道
        
        参数:
            data_path: 训练数据文件路径
            
        返回:
            拟合后的管道实例
        """
        self.logger.info(f"拟合管道，使用数据: {data_path}")
        
        # 加载数据
        X, y = self.data_loader.load(data_path)
        
        if y is None:
            raise ValueError("无法找到目标列，请检查配置中的target_column设置")
        
        # 保存原始特征名称
        self.feature_names = list(X.columns)
        
        # 编码特征
        X_encoded, encoded_feature_names = self._encode_features(X, fit=True)
        self.encoded_feature_names = encoded_feature_names
        
        # 训练模型
        self.model.fit(X_encoded, y)
        
        # 设置状态
        self.is_fitted = True
        self.train_data_y = y  # 保存训练集目标值，用于计算Q²
        
        return self
    
    def predict(self, data: Union[str, pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用管道进行预测
        
        参数:
            data: 数据文件路径、数据框或特征字典
            
        返回:
            包含预测结果的字典
        """
        if not self.is_fitted:
            raise ValueError("管道尚未拟合，请先调用fit方法")
        
        self.logger.info("使用管道进行预测")
        
        # 加载或处理数据
        if isinstance(data, str):
            # 从文件加载
            X, y = self.data_loader.load(data)
        elif isinstance(data, pd.DataFrame):
            # 直接使用数据框
            X = data
            y = None
        elif isinstance(data, dict):
            # 从字典创建数据框
            X = pd.DataFrame([data])
            y = None
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
        
        # 编码特征
        X_encoded, _ = self._encode_features(X, fit=False)
        
        # 预测
        y_pred = self.model.predict(X_encoded)
        
        # 如果是分类任务，尝试获取概率
        if self.model.task_type == "classification":
            try:
                y_proba = self.model.predict_proba(X_encoded)
                probabilities = y_proba.tolist() if isinstance(y_proba, np.ndarray) else None
            except (AttributeError, NotImplementedError):
                self.logger.warning("模型不支持predict_proba方法，无法获取概率")
                probabilities = None
        else:
            probabilities = None
        
        # 构建结果
        results = {
            "predictions": y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
        }
        
        if probabilities:
            results["probabilities"] = probabilities
        
        # 如果有真实值，计算评估指标
        if y is not None:
            eval_results = self.evaluate(data)
            results["metrics"] = eval_results
        
        return results
    
    def evaluate(self, data: Union[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """
        评估模型性能
        
        参数:
            data: 数据文件路径或(X, y)元组
            
        返回:
            包含评估指标的字典
        """
        if not self.is_fitted:
            raise ValueError("管道尚未拟合，请先调用fit方法")
        
        self.logger.info("评估模型性能")
        
        # 加载或处理数据
        if isinstance(data, str):
            # 从文件加载
            X, y = self.data_loader.load(data)
            if y is None:
                raise ValueError("无法找到目标列，请检查配置中的target_column设置")
            
            # 编码特征
            X_encoded, _ = self._encode_features(X, fit=False)
        elif isinstance(data, tuple) and len(data) == 2:
            # 直接使用X和y
            X_encoded, y = data
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
        
        # 使用评估器评估模型
        return self.evaluator.evaluate(
            model=self.model,
            X=X_encoded,
            y=y,
            feature_names=self.encoded_feature_names,
            y_train=self.train_data_y if hasattr(self, 'train_data_y') else None
        )
    
    def _encode_features(self, X: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        编码特征
        
        参数:
            X: 特征数据框
            fit: 是否拟合编码器
            
        返回:
            编码后的特征矩阵和特征名称列表
        """
        encoded_features = []
        encoded_feature_names = []
        
        for feature_name, encoder in self.feature_encoders.items():
            if feature_name not in X.columns:
                self.logger.warning(f"特征'{feature_name}'不在数据中，将被跳过")
                continue
            
            # 获取特征数据
            feature_data = X[feature_name]
            
            # 编码特征
            if fit:
                encoded = encoder.fit_transform(feature_data)
            else:
                encoded = encoder.transform(feature_data)
            
            # 获取编码后的特征名称
            feature_names = encoder.get_feature_names()
            
            # 添加到结果中
            encoded_features.append(encoded)
            encoded_feature_names.extend([f"{feature_name}_{name}" for name in feature_names])
        
        # 合并所有编码后的特征
        if encoded_features:
            X_encoded = np.hstack(encoded_features)
        else:
            raise ValueError("没有可用的特征编码器")
        
        return X_encoded, encoded_feature_names
    
    def save(self, path: str) -> None:
        """
        保存管道到文件
        
        参数:
            path: 保存路径
        """
        self.logger.info(f"保存管道到: {path}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存管道
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> "Pipeline":
        """
        从文件加载管道
        
        参数:
            path: 文件路径
            
        返回:
            加载的管道实例
        """
        logging.getLogger(__name__).info(f"从{path}加载管道")
        return joblib.load(path) 