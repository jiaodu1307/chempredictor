import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from reaction_model.utils.config import (
    ExperimentConfig,
    ModelConfig,
    EncoderConfig,
    TrainingConfig
)

@pytest.fixture
def sample_smiles():
    return [
        "CC(=O)O",  # 乙酸
        "CCO",      # 乙醇
        "C1=CC=CC=C1"  # 苯
    ]

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'reactant_a': ["CC(=O)O", "CCO", "C1=CC=CC=C1"],
        'temperature': [25, 30, 35],
        'Yield': [0.75, 0.82, 0.68]
    })

@pytest.fixture
def sample_config():
    return ExperimentConfig(
        name='test_experiment',
        data_path=Path('tests/data/test.csv'),
        feature_configs={
            'reactant_a': EncoderConfig(
                type='fingerprint',
                params={'radius': 2, 'n_bits': 1024}
            ),
            'temperature': EncoderConfig(
                type='numerical',
                params={'normalize': True}
            )
        },
        model_config=ModelConfig(
            type='mlp',
            hidden_dims=[64, 32]
        ),
        training_config=TrainingConfig(
            batch_size=16,
            max_epochs=10
        ),
        output_dir=Path('tests/outputs')
    ) 