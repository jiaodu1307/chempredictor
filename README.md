# ChemPredictor

ChemPredictor 是一个用于化学性质预测的高性能Python库。它提供了灵活的数据处理流水线、多种分子表示方法和先进的机器学习模型。

## 特性

- 🧪 支持多种分子表示方法（Morgan指纹、SMILES等）
- 🤖 内置多种机器学习模型
- 📊 灵活的数据处理流水线
- 🚀 高性能计算支持（CPU/GPU）
- 💾 智能缓存机制
- 📈 性能监控工具
- 🔍 完整的错误处理
- 📝 详细的日志记录

## 安装

```bash
pip install chempredictor
```

## 快速开始

### 基本用法

```python
from chempredictor import ChemPredictor

# 初始化预测器
predictor = ChemPredictor()

# 训练模型
predictor.train('data/training.csv')

# 进行预测
results = predictor.predict('data/test.csv')
```

### 使用配置文件

```python
from chempredictor import ChemPredictor

# 使用自定义配置
config_path = 'config/my_config.yaml'
predictor = ChemPredictor(config_path=config_path)
```

配置文件示例 (config/my_config.yaml):
```yaml
random_seed: 42
device: 'cuda'  # 或 'cpu'
pipeline:
  steps:
    data_loading:
      batch_size: 32
      num_workers: 4
    feature_encoding:
      smiles:
        encoder: 'morgan_fingerprint'
        params:
          radius: 2
          num_bits: 2048
    model_training:
      type: 'neural_network'
      task_type: 'regression'
      params:
        learning_rate: 0.001
        num_epochs: 100
```

### 性能监控

```python
from chempredictor.utils.profiling import profile_section, log_performance

@log_performance
def process_data():
    with profile_section("数据处理"):
        # 处理逻辑
        pass
```

### 使用缓存

```python
from chempredictor.utils.cache import cache_result, memory_cache

@cache_result()
def expensive_calculation():
    # 耗时计算
    pass

@memory_cache()
def frequent_operation():
    # 频繁操作
    pass
```

## 开发指南

### 环境设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/chempredictor.git
cd chempredictor

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -e ".[dev]"
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行带覆盖率报告的测试
pytest --cov=chempredictor tests/

# 运行特定测试
pytest tests/test_config.py
```

### 代码质量检查

```bash
# 格式化代码
black chempredictor/

# 运行代码检查
flake8 chempredictor/

# 类型检查
mypy chempredictor/
```

### 构建文档

```bash
cd docs
make html
```

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件