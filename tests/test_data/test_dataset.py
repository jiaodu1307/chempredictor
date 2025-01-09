import pytest
import torch
from reaction_model.data.dataset import ChemicalDataset
from reaction_model.utils.exceptions import DataError
from reaction_model.encoders import MorganFingerprintEncoder, NumericalEncoder

def test_dataset_init(sample_data):
    encoders = {
        'reactant_a': MorganFingerprintEncoder(radius=2, n_bits=1024),
        'temperature': NumericalEncoder(normalize=True)
    }
    dataset = ChemicalDataset(
        data_dict=sample_data.to_dict('list'),
        encoders=encoders,
        target_column='Yield'
    )
    assert len(dataset) == len(sample_data)

def test_dataset_validation(sample_data):
    encoders = {
        'invalid_feature': MorganFingerprintEncoder(radius=2, n_bits=1024)
    }
    with pytest.raises(DataError):
        ChemicalDataset(
            data_dict=sample_data.to_dict('list'),
            encoders=encoders,
            target_column='Yield'
        )

def test_dataset_getitem(sample_data):
    encoders = {
        'reactant_a': MorganFingerprintEncoder(radius=2, n_bits=1024),
        'temperature': NumericalEncoder(normalize=True)
    }
    dataset = ChemicalDataset(
        data_dict=sample_data.to_dict('list'),
        encoders=encoders,
        target_column='Yield'
    )
    features, target = dataset[0]
    assert isinstance(features, dict)
    assert isinstance(target, torch.Tensor) 