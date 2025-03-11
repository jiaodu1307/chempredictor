"""
配置类单元测试
"""
import pytest
from pathlib import Path
import yaml

from chempredictor.config import Config
from chempredictor.exceptions import ConfigError

def test_config_init_with_dict(sample_config):
    """测试使用字典初始化配置"""
    config = Config(config_dict=sample_config)
    assert config['random_seed'] == 42
    assert config['device'] == 'cpu'
    
def test_config_init_with_file(tmp_path, sample_config):
    """测试使用文件初始化配置"""
    config_file = tmp_path / 'test_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
        
    config = Config(config_path=str(config_file))
    assert config['random_seed'] == 42
    
def test_config_init_with_both_raises():
    """测试同时提供文件和字典时抛出异常"""
    with pytest.raises(ConfigError):
        Config(config_path='test.yaml', config_dict={})
        
def test_config_default_values():
    """测试默认配置值"""
    config = Config()
    assert 'random_seed' in config
    assert 'pipeline' in config
    assert 'logging' in config
    
def test_config_merge_with_default(sample_config):
    """测试配置合并"""
    config = Config(config_dict=sample_config)
    # 检查默认值是否被保留
    assert 'output' in config
    assert config['output']['save_model'] is True
    
def test_config_get_with_default():
    """测试获取配置值时使用默认值"""
    config = Config()
    assert config.get('non_existent', 'default') == 'default'
    
def test_invalid_config_file():
    """测试无效配置文件"""
    with pytest.raises(ConfigError):
        Config(config_path='non_existent.yaml')
        
def test_deep_update():
    """测试深度更新字典"""
    base = {'a': {'b': 1, 'c': 2}, 'd': 3}
    update = {'a': {'b': 10}, 'e': 4}
    config = Config(config_dict=update)
    
    # 检查更新是否正确
    assert config['a']['b'] == 10
    assert config['a']['c'] == 2  # 保留原值
    assert config['e'] == 4  # 新增值 