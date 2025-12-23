"""
CUML-based Bayesian models (Naive Bayes) with GPU support.
"""
import numpy as np
from typing import Optional, Dict, Any
from scipy.sparse import csr_matrix
import pickle

try:
    import cuml
    from cuml.naive_bayes import MultinomialNB as CUMLMultinomialNB
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB
    from sklearn.naive_bayes import ComplementNB as SKComplementNB

from .base import BaseModel
from ..config import get_config
from ..logging.logger import get_logger
from ..utils.gpu_utils import to_gpu_if_needed, from_gpu_if_needed

logger = get_logger(__name__)


class BayesianModel(BaseModel):
    """CUML-based Naive Bayes model."""
    
    def __init__(
        self,
        config=None,
        model_type: str = 'multinomial',
        alpha: float = 1.0,
        fit_prior: bool = True,
        **kwargs
    ):
        """
        Initialize Bayesian model.
        
        Args:
            config: Configuration object
            model_type: Type of NB model ('multinomial' or 'complement')
            alpha: Smoothing parameter
            fit_prior: Whether to fit class priors
            **kwargs: Additional arguments
        """
        super().__init__(config, model_name="bayesian")
        
        self.model_type = model_type
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.kwargs = kwargs
        
        # Initialize model
        if self.use_gpu and CUML_AVAILABLE:
            self.model = CUMLMultinomialNB(
                alpha=alpha,
                fit_prior=fit_prior,
                **kwargs
            )
            self.logger.info("Initialized CUML MultinomialNB")
        else:
            if model_type == 'complement':
                self.model = SKComplementNB(
                    alpha=alpha,
                    fit_prior=fit_prior,
                    **kwargs
                )
                self.logger.info("Initialized sklearn ComplementNB")
            else:
                self.model = SKMultinomialNB(
                    alpha=alpha,
                    fit_prior=fit_prior,
                    **kwargs
                )
                self.logger.info("Initialized sklearn MultinomialNB")
    
    def fit(self, X, y, **kwargs):
        """Fit the model."""
        self.logger.info("Fitting Bayesian model")
        
        # Naive Bayes: CUML requires dense, sklearn can use sparse
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=True)
        self.model.fit(X_gpu, y, **kwargs)
        self._fitted = True
        
        self.logger.info("Bayesian model fitted")
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        self._ensure_fitted()
        
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=True)
        predictions = self.model.predict(X_gpu)
        return from_gpu_if_needed(predictions)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        self._ensure_fitted()
        
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=True)
        probabilities = self.model.predict_proba(X_gpu)
        return from_gpu_if_needed(probabilities)
    
    def save(self, path: str):
        """Save model to disk."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving Bayesian model to {path}")
        
        model_state = {
            'model': self.model,
            'model_type': self.model_type,
            'alpha': self.alpha,
            'fit_prior': self.fit_prior,
            'kwargs': self.kwargs,
            'fitted': self._fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
        self.logger.info("Model saved")
    
    @classmethod
    def load(cls, path: str, config=None):
        """Load model from disk."""
        import pickle
        
        logger.info(f"Loading Bayesian model from {path}")
        
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        model = cls(
            config=config,
            model_type=model_state['model_type'],
            alpha=model_state['alpha'],
            fit_prior=model_state['fit_prior'],
            **model_state['kwargs']
        )
        
        model.model = model_state['model']
        model._fitted = model_state['fitted']
        
        logger.info("Model loaded")
        return model

