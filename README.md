# Reaction Model

一个用于化学反应预测的灵活深度学习框架。支持多种分子表示方法和特征编码器，可用于预测反应产率、选择性等化学性质。

## 特性

- 支持多种分子编码方式（Morgan指纹、MPNN等）
- 灵活的特征处理（分类特征、数值特征）
- 可配置的模型架构
- 基于PyTorch Lightning的训练框架
- 完整的错误处理和验证
- 支持回归和分类任务

## 安装

### 基础安装

```bash
git clone https://github.com/username/reaction-model.git
cd reaction-model
pip install -e ".[dev]"
```

## 快速开始

### 1. 准备数据
将您的反应数据准备为CSV格式，包含以下列：
- SMILES: 反应物的SMILES表示
- 目标值: 如产率、选择性等

```python
from reaction_model.data import ReactionDataset
dataset = ReactionDataset(
csv_path="path/to/data.csv",
smiles_col="SMILES",
target_col="yield"
)
```

### 2. 配置实验

创建YAML配置文件定义模型架构和训练参数：

```yaml
model:
type: "mlp"
hidden_dims: [256, 128, 64]
encoder:
type: "morgan"
radius: 2
num_bits: 2048
training:
batch_size: 32
max_epochs: 100
learning_rate: 0.001
```

### 3. 训练模型

```python
from reaction_model.train import train_model
train_model(config_path="path/to/config.yaml", dataset=dataset)
```

### 4. 预测

```python
smiles = "CC(=O)Oc1ccccc1C(=O)O"
prediction = trainer.predict(smiles)
print(f"预测值: {prediction}")
```

## 高级用法

### 自定义编码器

```python
from reaction_model.encoders import BaseEncoder
class CustomEncoder(BaseEncoder):
def encode(self, smiles):
# 实现自定义编码逻辑
pass
```

