"""
GPU-First ML Library with CUML, Polars, and Arrow.
"""

__version__ = "1.0.0"

from .config import Config, get_config, set_config
from .main import train_pipeline

__all__ = [
    'Config',
    'get_config',
    'set_config',
    'train_pipeline'
]

