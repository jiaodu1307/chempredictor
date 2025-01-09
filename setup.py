from setuptools import setup, find_packages

setup(
    name="reaction_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "pytorch-lightning>=1.5.0",
        "rdkit>=2020.09.1",
        "pandas>=1.2.0",
        "pyyaml>=5.4.1",
    ],
    extras_require={
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
        ],
    },
) 