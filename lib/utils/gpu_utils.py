"""
GPU utility functions for efficient CUML/cuDF conversions.
"""
from typing import Union, Optional
from scipy.sparse import csr_matrix
import numpy as np

try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

try:
    import cuml
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


def to_gpu_if_needed(
    X: Union[np.ndarray, csr_matrix],
    use_gpu: bool = True,
    sparse_to_dense: bool = False
) -> Union[np.ndarray, csr_matrix, 'cudf.DataFrame']:
    """
    Convert data to GPU format if needed and available.
    
    Args:
        X: Input data (numpy array or sparse matrix)
        use_gpu: Whether to use GPU
        sparse_to_dense: Whether to convert sparse to dense (for CUML compatibility)
        
    Returns:
        GPU data (cuDF DataFrame) or original data
    """
    if not use_gpu or not CUDF_AVAILABLE:
        return X
    
    if isinstance(X, csr_matrix):
        if sparse_to_dense:
            X_dense = X.toarray()
            return cudf.DataFrame(X_dense)
        else:
            # Keep sparse for now (some CUML models support sparse)
            return X
    elif isinstance(X, np.ndarray):
        if not isinstance(X, cudf.DataFrame):
            return cudf.DataFrame(X)
        return X
    else:
        return X


def from_gpu_if_needed(
    result: Union[np.ndarray, 'cudf.DataFrame', 'cupy.ndarray']
) -> np.ndarray:
    """
    Convert GPU result back to numpy if needed.
    
    Args:
        result: GPU result (cuDF DataFrame, cupy array) or numpy array
        
    Returns:
        NumPy array
    """
    if hasattr(result, 'values'):
        # cuDF DataFrame
        return result.values
    elif hasattr(result, 'get'):
        # cupy array
        return result.get()
    else:
        return np.asarray(result)


def check_gpu_availability() -> bool:
    """Check if GPU is available and accessible."""
    if not CUML_AVAILABLE or not CUDF_AVAILABLE:
        return False
    
    # Check if CUDA device is actually accessible
    try:
        import cupy as cp
        cp.cuda.Device(0).use()
        return True
    except Exception:
        # CUDA not available (login node, no GPU, driver mismatch, etc.)
        return False

