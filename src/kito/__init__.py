"""
Kito: Effortless PyTorch Training

Define your model, Kito handles the rest.
"""

__version__ = "0.1.0"

# Import main classes for top-level access
from kito.engine import Engine
from kito.module import KitoModule

# Import common data classes
from kito.data.datasets import H5Dataset, MemDataset, KitoDataset
from kito.data.datapipeline import GenericDataPipeline
from kito.data.preprocessing import (
    Preprocessing,
    Pipeline,
    Normalize,
    Standardization,
    ToTensor
)

# Import registries
from kito.data.registry import DATASETS, PREPROCESSING

# Define what's available with "from kito import *"
__all__ = [
    # Core
    "Engine",
    "KitoModule",

    # Data
    "H5Dataset",
    "MemDataset",
    "KitoDataset",
    "GenericDataPipeline",

    # Preprocessing
    "Preprocessing",
    "Pipeline",
    "Normalize",
    "Standardization",
    "ToTensor",

    # Registries
    "DATASETS",
    "PREPROCESSING",
]