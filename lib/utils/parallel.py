"""
Parallelization utilities for GPU/CPU operations.
"""
from typing import Callable, List, Any, Optional
from functools import partial
import multiprocessing as mp

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class ParallelExecutor:
    """Parallel execution utilities."""
    
    def __init__(self, config=None):
        """
        Initialize parallel executor.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.n_jobs = self.config.n_jobs if self.config.n_jobs > 0 else mp.cpu_count()
    
    def parallel_map(
        self,
        func: Callable,
        items: List[Any],
        n_jobs: Optional[int] = None,
        backend: str = 'threading'
    ) -> List[Any]:
        """
        Apply function to items in parallel.
        
        Args:
            func: Function to apply
            items: List of items to process
            n_jobs: Number of parallel jobs (uses config default if None)
            backend: Backend type ('threading', 'multiprocessing', 'loky')
            
        Returns:
            List of results
        """
        n_jobs = n_jobs or self.n_jobs
        
        if not JOBLIB_AVAILABLE:
            self.logger.warning("joblib not available, using sequential execution")
            return [func(item) for item in items]
        
        self.logger.debug(f"Executing {len(items)} items in parallel (n_jobs={n_jobs}, backend={backend})")
        
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(func)(item) for item in items
        )
        
        return results
    
    def parallel_folds(
        self,
        fold_func: Callable,
        folds: List[Any],
        n_jobs: Optional[int] = None
    ) -> List[Any]:
        """
        Execute cross-validation folds in parallel.
        
        Args:
            fold_func: Function to execute on each fold
            folds: List of fold data
            n_jobs: Number of parallel jobs
            
        Returns:
            List of fold results
        """
        return self.parallel_map(fold_func, folds, n_jobs=n_jobs, backend='multiprocessing')
    
    def parallel_models(
        self,
        model_func: Callable,
        models: List[Any],
        n_jobs: Optional[int] = None
    ) -> List[Any]:
        """
        Execute model training in parallel.
        
        Args:
            model_func: Function to execute on each model
            models: List of model configurations
            n_jobs: Number of parallel jobs
            
        Returns:
            List of model results
        """
        return self.parallel_map(model_func, models, n_jobs=n_jobs, backend='threading')


class GPUManager:
    """GPU memory and device management."""
    
    def __init__(self, config=None):
        """
        Initialize GPU manager.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.device = None
        self._setup_device()
    
    def _setup_device(self):
        """Setup GPU device."""
        if TORCH_AVAILABLE and torch.cuda.is_available() and self.config.use_gpu:
            self.device = torch.device(f'cuda:{self.config.gpu_id}')
            self.logger.info(f"Using GPU device: {self.device}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU device")
    
    def get_device(self):
        """Get current device."""
        return self.device
    
    def clear_cache(self):
        """Clear GPU cache."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("Cleared GPU cache")
    
    def get_memory_info(self) -> dict:
        """Get GPU memory information."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        return {}


def parallel_feature_extraction(
    extractor_func: Callable,
    texts: List[str],
    n_jobs: Optional[int] = None
) -> List[Any]:
    """
    Extract features in parallel.
    
    Args:
        extractor_func: Feature extraction function
        texts: List of texts to process
        n_jobs: Number of parallel jobs
        
    Returns:
        List of extracted features
    """
    executor = ParallelExecutor()
    return executor.parallel_map(extractor_func, texts, n_jobs=n_jobs)

