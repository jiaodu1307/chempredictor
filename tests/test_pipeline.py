"""
Pipeline类单元测试
"""
import pytest
from unittest.mock import Mock, patch

from chempredictor.pipeline import Pipeline
from chempredictor.exceptions import PipelineError

@pytest.fixture
def mock_data_loader():
    """模拟数据加载器"""
    return Mock()

@pytest.fixture
def mock_encoder():
    """模拟特征编码器"""
    return Mock()

@pytest.fixture
def mock_model():
    """模拟模型"""
    return Mock()

def test_pipeline_init(sample_config):
    """测试Pipeline初始化"""
    with patch('chempredictor.pipeline.DataLoader') as mock_loader, \
         patch('chempredictor.pipeline.EncoderFactory') as mock_factory, \
         patch('chempredictor.pipeline.get_model') as mock_get_model:
        
        pipeline = Pipeline(sample_config['pipeline'])
        
        assert pipeline.device == 'cpu'
        assert isinstance(pipeline.steps, dict)
        mock_loader.assert_called_once()
        mock_factory.assert_called_once()
        mock_get_model.assert_called_once()

def test_pipeline_init_missing_config():
    """测试缺少配置时的错误处理"""
    with pytest.raises(PipelineError):
        Pipeline({})

def test_pipeline_setup_steps(sample_config):
    """测试步骤设置"""
    with patch('chempredictor.pipeline.DataLoader') as mock_loader, \
         patch('chempredictor.pipeline.EncoderFactory') as mock_factory, \
         patch('chempredictor.pipeline.get_model') as mock_get_model:
        
        pipeline = Pipeline(sample_config['pipeline'])
        
        assert 'data_loading' in pipeline.steps
        assert 'feature_encoding' in pipeline.steps
        assert hasattr(pipeline, 'model')

@pytest.mark.integration
def test_pipeline_fit(sample_data, tmp_path):
    """测试模型训练流程"""
    config = {
        'steps': {
            'data_loading': {'batch_size': 32},
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
    
    # 创建测试数据文件
    import pandas as pd
    data_file = tmp_path / 'test_data.csv'
    pd.DataFrame(sample_data).to_csv(data_file, index=False)
    
    pipeline = Pipeline(config)
    pipeline.fit(str(data_file))
    
    # 验证模型是否已训练
    assert hasattr(pipeline.model, 'predict')

@pytest.mark.integration
def test_pipeline_predict(sample_data):
    """测试模型预测流程"""
    config = {
        'steps': {
            'data_loading': {'batch_size': 32},
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
    
    pipeline = Pipeline(config)
    predictions = pipeline.predict({'smiles': sample_data['smiles']})
    
    assert isinstance(predictions, dict)
    assert 'predictions' in predictions 