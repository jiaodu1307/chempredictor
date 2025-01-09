import pytest
from pathlib import Path
from reaction_model.utils.config import ExperimentConfig
from reaction_model.utils.exceptions import ConfigError

def test_config_validation(sample_config, tmp_path):
    # 创建测试数据文件
    data_file = tmp_path / "test.csv"
    data_file.write_text("dummy data")
    
    # 更新配置中的路径
    sample_config.data_path = data_file
    sample_config.output_dir = tmp_path
    
    # 验证应该通过
    sample_config.validate()

def test_config_validation_fails():
    with pytest.raises(ConfigError):
        ExperimentConfig(
            name='test',
            data_path=Path('nonexistent.csv'),
            feature_configs={},
            model_config=None,
            training_config=None,
            output_dir=Path('outputs')
        ).validate()

def test_config_from_yaml(tmp_path):
    # 创建测试配置文件
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
        name: test
        data_path: data.csv
        feature_configs:
          reactant_a:
            type: fingerprint
            params:
              radius: 2
              n_bits: 1024
        model_config:
          type: mlp
          hidden_dims: [64, 32]
        training_config:
          batch_size: 16
          max_epochs: 10
        output_dir: outputs
    """)
    
    with pytest.raises(ConfigError):
        # 应该失败，因为data_path不存在
        ExperimentConfig.from_yaml(str(config_file)) 