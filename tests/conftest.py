"""
Pytest配置文件
"""
import os
import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """测试数据目录"""
    return Path(__file__).parent / 'test_data'

@pytest.fixture
def sample_config():
    """样例配置"""
    return {
        'random_seed': 42,
        'device': 'cpu',
        'pipeline': {
            'steps': {
                'data_loading': {
                    'batch_size': 32
                },
                'feature_encoding': {
                    'smiles': {
                        'encoder': 'morgan_fingerprint',
                        'params': {'radius': 2}
                    }
                },
                'model_training': {
                    'type': 'neural_network',
                    'task_type': 'regression'
                }
            }
        }
    }

@pytest.fixture
def sample_data():
    """样例数据"""
    return {
        'smiles': ['CC(=O)O', 'CCO', 'c1ccccc1'],
        'property': [4.76, 5.23, 2.85]
    } 