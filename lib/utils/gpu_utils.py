"""
GPU utility functions for efficient CUML/cuDF conversions.
"""
from typing import Union, Optional
from scipy.sparse import csr_matrix
from scipy import sparse as sp
import numpy as np

from ..logging.logger import get_logger

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

logger = get_logger(__name__)


def _shape_or_none(X) -> Optional[tuple]:
    if hasattr(X, "shape"):
        try:
            return tuple(X.shape)
        except Exception:
            return None
    return None


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
        logger.debug("GPU convert skip: use_gpu=%s cudf=%s shape=%s type=%s",
                     use_gpu, CUDF_AVAILABLE, _shape_or_none(X), type(X).__name__)
        return X
    
    if isinstance(X, csr_matrix):
        if sparse_to_dense:
            X_dense = X.toarray()
            logger.debug("GPU convert: sparse->dense for cuDF shape=%s", _shape_or_none(X_dense))
            return cudf.DataFrame(X_dense)
        else:
            # Keep sparse for now (some CUML models support sparse)
            logger.debug("GPU convert: keeping sparse shape=%s", _shape_or_none(X))
            return X
    elif isinstance(X, np.ndarray):
        if not isinstance(X, cudf.DataFrame):
            logger.debug("GPU convert: numpy->cuDF shape=%s", _shape_or_none(X))
            return cudf.DataFrame(X)
        logger.debug("GPU convert: already cuDF shape=%s", _shape_or_none(X))
        return X
    else:
        logger.debug("GPU convert: no-op for type=%s", type(X).__name__)
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
    if sp.issparse(result):
        logger.debug("GPU convert back: sparse passthrough shape=%s", _shape_or_none(result))
        return result
    if CUDF_AVAILABLE:
        try:
            import cudf
            if isinstance(result, (cudf.DataFrame, cudf.Series)):
                logger.debug("GPU convert back: cuDF->numpy shape=%s", _shape_or_none(result))
                if hasattr(result, "to_numpy"):
                    return result.to_numpy()
                if hasattr(result, "values_host"):
                    return result.values_host
                return result.to_pandas().values
        except Exception:
            pass
    if hasattr(result, 'get'):
        # cupy array
        logger.debug("GPU convert back: cupy->numpy shape=%s", _shape_or_none(result))
        return result.get()
    if hasattr(result, 'values'):
        # Fallback for dataframe/series
        logger.debug("GPU convert back: values->numpy shape=%s", _shape_or_none(result))
        return np.asarray(result.values)
    logger.debug("GPU convert back: numpy.asarray type=%s shape=%s", type(result).__name__, _shape_or_none(result))
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
