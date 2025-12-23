"""
Preprocessing pipeline that works with config.
"""
from typing import Optional
from scipy.sparse import csr_matrix
import numpy as np

from ..config import get_config
from ..logging.logger import get_logger
from .preprocessor import Scaler, Imputer, PCAReducer, DataNormalizer

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Preprocessing pipeline based on config."""
    
    def __init__(self, config=None):
        """Initialize preprocessing pipeline from config."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        
        # Build pipeline based on config
        self.steps = []
        
        if self.config.impute_missing:
            self.steps.append(('imputer', Imputer(use_gpu=self.config.use_gpu)))
        
        if self.config.scale_features:
            self.steps.append(('scaler', Scaler(use_gpu=self.config.use_gpu)))
        
        if self.config.use_pca:
            self.steps.append(('pca', PCAReducer(
                n_components=self.config.pca_components,
                use_gpu=self.config.use_gpu
            )))
        
        if self.config.normalize:
            self.steps.append(('normalizer', DataNormalizer(use_gpu=self.config.use_gpu)))
        
        self._fitted = False
    
    def fit(self, X):
        """Fit all preprocessors in the pipeline."""
        self.logger.info("Fitting preprocessing pipeline")
        X_transformed = X
        for name, preprocessor in self.steps:
            self.logger.debug(f"Fitting {name}")
            preprocessor.fit(X_transformed)
            X_transformed = preprocessor.transform(X_transformed)
        self._fitted = True
        return self
    
    def transform(self, X):
        """Transform data through the pipeline."""
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        self.logger.debug("Transforming through preprocessing pipeline")
        X_transformed = X
        for name, preprocessor in self.steps:
            X_transformed = preprocessor.transform(X_transformed)
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

