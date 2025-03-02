# ChemPredictor

ChemPredictor是一个用于化学性质预测的Python库框架，提供了模块化的架构和丰富的功能，用于构建和评估化学性质预测模型。

## 功能特点

- **模块化设计**：独立的数据加载、特征编码、模型训练和评估模块
- **多种分子表示**：支持Morgan指纹、RDKit指纹、MACCS Keys等多种分子表示方法
- **多种模型支持**：内置随机森林、XGBoost、LightGBM等多种机器学习模型
- **灵活的配置系统**：通过YAML配置文件或字典配置整个预测流程
- **全面的评估指标**：提供RMSE、MAE、R²、Q²、准确率、F1分数等多种评估指标
- **特征重要性分析**：支持基于模型的特征重要性和SHAP值分析

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```python
from chempredictor import ChemPredictor

# 使用默认配置初始化
predictor = ChemPredictor()

# 训练模型
predictor.train("data/example_reactions.csv")

# 预测新数据
results = predictor.predict("data/test_reactions.csv")
print(results)

# 评估模型
metrics = predictor.evaluate("data/test_reactions.csv")
print(metrics)
```

### 使用自定义配置

```python
from chempredictor import ChemPredictor

# 使用自定义配置文件
predictor = ChemPredictor(config_path="configs/my_config.yaml")

# 或者使用配置字典
config = {
    "pipeline": {
        "steps": {
            "data_loading": {
                "file_type": "csv",
                "target_column": "Yield"
            },
            "feature_encoding": {
                "Reactant": {
                    "encoder": "morgan_fingerprint",
                    "params": {"radius": 2, "n_bits": 2048}
                }
            },
            "model_training": {
                "type": "random_forest",
                "task_type": "regression"
            }
        }
    }
}
predictor = ChemPredictor(config_dict=config)

# 训练和预测
predictor.train("data/example_reactions.csv")
results = predictor.predict({"Reactant": "CCO", "Solvent": "Water", "Temperature": 100})
```

## 项目结构

```
chempredictor/
├── __init__.py           # 包初始化
├── core.py               # 核心ChemPredictor类
├── data_loading/         # 数据加载模块
├── encoders/             # 特征编码模块
├── models/               # 预测模型模块
├── evaluation/           # 评估模块
├── pipeline/             # 管道模块
└── utils/                # 工具模块

configs/                  # 配置文件目录
data/                     # 示例数据目录
```

## 配置文件示例

```yaml
pipeline:
  steps:
    - data_loading:
        file_type: csv
        target_column: Yield
        feature_columns: [Reactant, Solvent, Temperature]
        missing_value_strategy: mean
    
    - feature_encoding:
        Reactant: 
          encoder: morgan_fingerprint
          params:
            radius: 2
            n_bits: 2048
            chiral: true
        Solvent:
          encoder: onehot_encoder
        Temperature:
          encoder: standard_scaler
    
    - model_training:
        type: xgboost
        task_type: regression
        params:
          n_estimators: 100
          max_depth: 6
          learning_rate: 0.1
    
    - evaluation:
        metrics: 
          regression: [rmse, r2, mae]
          classification: [accuracy, f1, roc_auc]
        feature_importance: true
        shap_analysis: true
```

## 扩展功能

### 添加新的编码器

```python
from chempredictor.encoders import BaseEncoder, register_encoder

@register_encoder("my_encoder")
class MyEncoder(BaseEncoder):
    def __init__(self, param1=1, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        
    def fit(self, data):
        # 实现拟合逻辑
        self.is_fitted = True
        return self
        
    def transform(self, data):
        # 实现转换逻辑
        return transformed_data
        
    def get_output_dim(self):
        return output_dimension
```

### 添加新的模型

```python
from chempredictor.models import BaseModel, register_model

@register_model("my_model")
class MyModel(BaseModel):
    def __init__(self, task_type="regression", **kwargs):
        super().__init__(task_type=task_type, **kwargs)
        # 初始化模型
        
    def fit(self, X, y):
        # 实现训练逻辑
        self.is_fitted = True
        return self
        
    def predict(self, X):
        # 实现预测逻辑
        return predictions
```

## 许可证

MIT
