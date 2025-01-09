import pytest
import torch
from reaction_model.encoders.molecule.fingerprint import MorganFingerprintEncoder
from reaction_model.utils.exceptions import EncoderError

def test_fingerprint_encoder_init():
    encoder = MorganFingerprintEncoder(radius=2, n_bits=1024)
    assert encoder.radius == 2
    assert encoder.n_bits == 1024

def test_fingerprint_encoder_invalid_params():
    with pytest.raises(EncoderError):
        MorganFingerprintEncoder(radius=-1, n_bits=1024)
    with pytest.raises(EncoderError):
        MorganFingerprintEncoder(radius=2, n_bits=0)

def test_fingerprint_encoder_forward(sample_smiles):
    encoder = MorganFingerprintEncoder(radius=2, n_bits=1024)
    output = encoder(sample_smiles)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (len(sample_smiles), 1024) 