"""
GPU-First ML Library with CUML, Polars, and Arrow.
"""

__version__ = "1.0.0"

from .config import Config, get_config, set_config
from .main import train_pipeline
from . import constants
from .utils.message_loader import get_message, get_message_safe
from .utils.gc_utils import (
    AggressiveGC,
    get_gc_manager,
    collect_after_chunk,
    collect_after_operation,
    context_manager_collect,
    memory_efficient_chunked_processing
)

__all__ = [
    'Config',
    'get_config',
    'set_config',
    'train_pipeline',
    'constants',
    'get_message',
    'get_message_safe',
    'AggressiveGC',
    'get_gc_manager',
    'collect_after_chunk',
    'collect_after_operation',
    'context_manager_collect',
    'memory_efficient_chunked_processing'
]

