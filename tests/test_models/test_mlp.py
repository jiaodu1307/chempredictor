import pytest
import torch
from reaction_model.models.mlp import MLPModel
from reaction_model.encoders import MorganFingerprintEncoder, NumericalEncoder

@pytest.fixture
def sample_model():
    encoders = {
        'reactant_a': MorganFingerprintEncoder(radius=2, n_bits=1024),
        'temperature': NumericalEncoder(normalize=True)
    }
    return MLPModel(
        encoders=encoders,
        hidden_dims=[64, 32],
        dropout=0.1,
        learning_rate=0.001
    )

def test_model_init(sample_model):
    assert isinstance(sample_model, MLPModel)
    assert len(sample_model.encoders) == 2

def test_model_forward(sample_model, sample_data):
    features = {
        'reactant_a': sample_data['reactant_a'].iloc[:2].tolist(),
        'temperature': torch.tensor([[25.], [30.]], dtype=torch.float32)
    }
    output = sample_model(features)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 1)

def test_model_training_step(sample_model, sample_data):
    features = {
        'reactant_a': sample_data['reactant_a'].iloc[:2].tolist(),
        'temperature': torch.tensor([[25.], [30.]], dtype=torch.float32)
    }
    target = torch.tensor([[0.75], [0.82]], dtype=torch.float32)
    batch = (features, target)
    
    loss = sample_model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # 标量 