"""
Scaling, imputation, PCA, normalization using CUML for GPU acceleration.
"""
import numpy as np
from scipy import sparse as sp
from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod

try:
    import cuml
    from cuml.preprocessing import StandardScaler as CUMLStandardScaler
    from cuml.preprocessing import MinMaxScaler as CUMLMinMaxScaler
    from cuml.preprocessing import Normalizer as CUMLNormalizer
    from cuml.impute import SimpleImputer as CUMLSimpleImputer
    from cuml.decomposition import PCA as CUMLPCA
    from cuml.decomposition import TruncatedSVD as CUMLTruncatedSVD
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    # Fallback to sklearn if CUML not available
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA, TruncatedSVD

from ..config import get_config
from ..logging.logger import get_logger
from ..utils.gpu_utils import to_gpu_if_needed, from_gpu_if_needed
from ..utils.gc_utils import collect_after_operation

logger = get_logger(__name__)


class BasePreprocessor(ABC):
    """Base class for preprocessors."""
    
    def __init__(self, use_gpu: Optional[bool] = None):
        self.config = get_config()
        self.use_gpu = use_gpu if use_gpu is not None else self.config.use_gpu
        self.logger = get_logger(self.__class__.__name__)
        self._fitted = False
    
    @abstractmethod
    def fit(self, X):
        """Fit the preprocessor."""
        pass
    
    @abstractmethod
    def transform(self, X):
        """Transform the data."""
        pass
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class Scaler(BasePreprocessor):
    """GPU-accelerated standard scaler."""
    
    def __init__(self, use_gpu: Optional[bool] = None, with_mean: bool = True, with_std: bool = True):
        super().__init__(use_gpu)
        self.with_mean = with_mean
        self.with_std = with_std
        
        if self.use_gpu and CUML_AVAILABLE:
            self.scaler = CUMLStandardScaler(
                with_mean=with_mean,
                with_std=with_std
            )
        else:
            self.scaler = StandardScaler(
                with_mean=with_mean,
                with_std=with_std
            )
    
    def fit(self, X):
        """Fit the scaler."""
        self.logger.debug("Fitting StandardScaler")
        self.logger.debug("Scaler input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=False)
        if sp.issparse(X_gpu) and self.with_mean:
            self.logger.warning("Sparse input detected; disabling mean-centering for scaler.")
            self.with_mean = False
            if self.use_gpu and CUML_AVAILABLE:
                self.scaler = CUMLStandardScaler(with_mean=False, with_std=self.with_std)
            else:
                self.scaler = StandardScaler(with_mean=False, with_std=self.with_std)
        self.scaler.fit(X_gpu)
        del X_gpu
        if self.use_gpu and CUML_AVAILABLE:
            collect_after_operation("scaler_fit", aggressive=True)
        self._fitted = True
        return self
    
    def transform(self, X):
        """Transform the data."""
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        self.logger.debug("Transforming with StandardScaler")
        self.logger.debug("Scaler transform input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=False)
        result = self.scaler.transform(X_gpu)
        del X_gpu
        result_cpu = from_gpu_if_needed(result)
        del result
        if self.use_gpu and CUML_AVAILABLE:
            collect_after_operation("scaler_transform", aggressive=True)
        return result_cpu


class MinMaxScaler(BasePreprocessor):
    """GPU-accelerated min-max scaler."""
    
    def __init__(self, use_gpu: Optional[bool] = None, feature_range: Tuple[float, float] = (0, 1)):
        super().__init__(use_gpu)
        self.feature_range = feature_range
        
        if self.use_gpu and CUML_AVAILABLE:
            self.scaler = CUMLMinMaxScaler(feature_range=feature_range)
        else:
            self.scaler = MinMaxScaler(feature_range=feature_range)
    
    def fit(self, X):
        """Fit the scaler."""
        self.logger.debug("Fitting MinMaxScaler")
        self.logger.debug("MinMaxScaler input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=False)
        self.scaler.fit(X_gpu)
        self._fitted = True
        return self
    
    def transform(self, X):
        """Transform the data."""
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        self.logger.debug("Transforming with MinMaxScaler")
        self.logger.debug("MinMaxScaler transform input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=False)
        result = self.scaler.transform(X_gpu)
        return from_gpu_if_needed(result)


class Imputer(BasePreprocessor):
    """GPU-accelerated imputer."""
    
    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        strategy: str = 'mean',
        missing_values: Union[float, int] = np.nan
    ):
        super().__init__(use_gpu)
        self.strategy = strategy
        self.missing_values = missing_values
        
        if self.use_gpu and CUML_AVAILABLE:
            # CUML SimpleImputer
            self.imputer = CUMLSimpleImputer(
                missing_values=missing_values,
                strategy=strategy
            )
        else:
            self.imputer = SimpleImputer(
                missing_values=missing_values,
                strategy=strategy
            )
    
    def fit(self, X):
        """Fit the imputer."""
        self.logger.debug(f"Fitting Imputer with strategy={self.strategy}")
        self.logger.debug("Imputer input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=False)
        self.imputer.fit(X_gpu)
        self._fitted = True
        return self
    
    def transform(self, X):
        """Transform the data."""
        if not self._fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self.logger.debug("Transforming with Imputer")
        self.logger.debug("Imputer transform input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=False)
        result = self.imputer.transform(X_gpu)
        return from_gpu_if_needed(result)


class PCAReducer(BasePreprocessor):
    """GPU-accelerated PCA for dimensionality reduction."""
    
    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        n_components: Union[int, float] = 0.95,
        random_state: Optional[int] = None
    ):
        super().__init__(use_gpu)
        self.n_components = n_components
        self.random_state = random_state or get_config().random_state
        
        if self.use_gpu and CUML_AVAILABLE:
            if isinstance(n_components, float):
                # Use TruncatedSVD for variance-based reduction
                self.reducer = CUMLTruncatedSVD(
                    n_components=int(n_components * 100) if n_components < 1 else int(n_components),
                    random_state=self.random_state
                )
            else:
                self.reducer = CUMLPCA(
                    n_components=n_components,
                    random_state=self.random_state
                )
        else:
            if isinstance(n_components, float):
                self.reducer = TruncatedSVD(
                    n_components=int(n_components * 100) if n_components < 1 else int(n_components),
                    random_state=self.random_state
                )
            else:
                self.reducer = PCA(
                    n_components=n_components,
                    random_state=self.random_state
                )
    
    def fit(self, X):
        """Fit the PCA reducer."""
        self.logger.debug(f"Fitting PCA with n_components={self.n_components}")
        self.logger.debug("PCA input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=False)
        self.reducer.fit(X_gpu)
        del X_gpu
        if self.use_gpu and CUML_AVAILABLE:
            collect_after_operation("pca_fit", aggressive=True)
        self._fitted = True
        return self
    
    def transform(self, X):
        """Transform the data."""
        if not self._fitted:
            raise ValueError("PCA must be fitted before transform")
        
        self.logger.debug("Transforming with PCA")
        self.logger.debug("PCA transform input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=False)
        result = self.reducer.transform(X_gpu)
        del X_gpu
        result_cpu = from_gpu_if_needed(result)
        del result
        if self.use_gpu and CUML_AVAILABLE:
            collect_after_operation("pca_transform", aggressive=True)
        return result_cpu
    
    def get_explained_variance_ratio(self):
        """Get explained variance ratio."""
        if hasattr(self.reducer, 'explained_variance_ratio_'):
            return self.reducer.explained_variance_ratio_
        return None


class DataNormalizer(BasePreprocessor):
    """GPU-accelerated data normalizer."""
    
    def __init__(self, use_gpu: Optional[bool] = None, norm: str = 'l2'):
        super().__init__(use_gpu)
        self.norm = norm
        
        if self.use_gpu and CUML_AVAILABLE:
            self.normalizer = CUMLNormalizer(norm=norm)
        else:
            self.normalizer = Normalizer(norm=norm)
    
    def fit(self, X):
        """Fit the normalizer (no-op for normalizers)."""
        self.logger.debug(f"Fitting Normalizer with norm={self.norm}")
        self.logger.debug("Normalizer input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        # Normalizer doesn't need fit, but we check GPU availability
        self._fitted = True
        return self
    
    def transform(self, X):
        """Transform the data."""
        if not self._fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        self.logger.debug("Transforming with Normalizer")
        self.logger.debug("Normalizer transform input type=%s shape=%s", type(X).__name__, getattr(X, "shape", None))
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=False)
        result = self.normalizer.transform(X_gpu)
        return from_gpu_if_needed(result)


class PreprocessingPipeline:
    """Pipeline for multiple preprocessing steps."""
    
    def __init__(self, steps: list, use_gpu: Optional[bool] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            steps: List of (name, preprocessor) tuples
            use_gpu: Whether to use GPU
        """
        self.steps = steps
        self.use_gpu = use_gpu
        self.logger = get_logger(self.__class__.__name__)
    
    def fit(self, X):
        """Fit all preprocessors in the pipeline."""
        self.logger.info("Fitting preprocessing pipeline")
        X_transformed = X
        for name, preprocessor in self.steps:
            self.logger.debug(f"Fitting {name}")
            self.logger.debug("Pipeline step %s input type=%s shape=%s", name, type(X_transformed).__name__, getattr(X_transformed, "shape", None))
            preprocessor.fit(X_transformed)
            X_transformed = preprocessor.transform(X_transformed)
            self.logger.debug("Pipeline step %s output type=%s shape=%s", name, type(X_transformed).__name__, getattr(X_transformed, "shape", None))
        return self
    
    def transform(self, X):
        """Transform data through the pipeline."""
        self.logger.debug("Transforming through preprocessing pipeline")
        X_transformed = X
        for name, preprocessor in self.steps:
            self.logger.debug("Pipeline step %s input type=%s shape=%s", name, type(X_transformed).__name__, getattr(X_transformed, "shape", None))
            X_transformed = preprocessor.transform(X_transformed)
            self.logger.debug("Pipeline step %s output type=%s shape=%s", name, type(X_transformed).__name__, getattr(X_transformed, "shape", None))
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
