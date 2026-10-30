"""
Effortless PyTorch training and (scientific) experimentation

Define your model setup, Kito handles the rest.
"""

__version__ = "1.0.0"
__author__ = "Giuseppe Costantino"

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

# logging
from kito.utils.logging_utils import configure_logging, get_logger
# Users can write kito.configure_logging(logging.DEBUG) or pass stream=open('run.log', 'a')

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

    # Logging
    "get_logger",
    "configure_logging"
]