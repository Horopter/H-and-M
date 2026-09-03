"""
CUML-based Bayesian models (Naive Bayes) with GPU support.
"""
import numpy as np
from typing import Optional, Dict, Any
from scipy.sparse import csr_matrix
from scipy import sparse as sp
import pickle

try:
    import cuml
    from cuml.naive_bayes import MultinomialNB as CUMLMultinomialNB
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

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
        self.cpu_model = None
        self._cpu_fitted = False
        
        # Initialize model
        if self.use_gpu and CUML_AVAILABLE and self.model_type in ('multinomial', 'complement'):
            self.model = CUMLMultinomialNB(
                alpha=alpha,
                fit_prior=fit_prior,
                **kwargs
            )
            self.logger.info("Initialized CUML MultinomialNB")
        else:
            self.model = self._init_cpu_model(self.model_type)
            self.logger.info("Initialized sklearn Naive Bayes")
    
    def fit(self, X, y, **kwargs):
        """Fit the model."""
        self.logger.info("Fitting Bayesian model")
        X_cpu = from_gpu_if_needed(X)
        if self._requires_non_negative() and self._has_negative_values(X_cpu):
            if self.model_type != 'gaussian':
                self.logger.warning(
                    "Negative values detected for %s NB; switching to GaussianNB.",
                    self.model_type
                )
            self.model_type = 'gaussian'
            self.use_gpu = False
            self.model = self._init_cpu_model(self.model_type)
        if self.model_type == 'gaussian' and sp.issparse(X_cpu):
            self.logger.warning(
                "GaussianNB requires dense input; converting sparse matrix to dense for fit."
            )
            X_cpu = X_cpu.toarray()
        try:
            # Naive Bayes: CUML requires dense, sklearn can use sparse (except GaussianNB).
            if self._use_gpu_model():
                X_gpu = to_gpu_if_needed(X, True, sparse_to_dense=True)
                self.model.fit(X_gpu, y, **kwargs)
            else:
                self.model.fit(X_cpu, y, **kwargs)
            # Fit CPU fallback for robust predict/proba
            self.cpu_model = self._init_cpu_model(self.model_type)
            self.cpu_model.fit(X_cpu, y, **kwargs)
            self._cpu_fitted = True
            self._fitted = True
            self.logger.info("Bayesian model fitted")
            return self
        except Exception as e:
            if self.use_gpu and CUML_AVAILABLE:
                self.logger.warning("CUML Bayesian fit failed; falling back to sklearn. error=%s", e)
                self.model_type = 'multinomial' if self.model_type not in ('gaussian', 'complement') else self.model_type
                self.model = self._init_cpu_model(self.model_type)
                self.use_gpu = False
                self.model.fit(X_cpu, y, **kwargs)
                self.cpu_model = self.model
                self._cpu_fitted = True
                self._fitted = True
                self.logger.info("Bayesian model fitted (CPU fallback)")
                return self
            raise
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        self._ensure_fitted()
        X_cpu = from_gpu_if_needed(X)
        if self.model_type == 'gaussian' and sp.issparse(X_cpu):
            self.logger.warning(
                "GaussianNB requires dense input; converting sparse matrix to dense for predict."
            )
            X_cpu = X_cpu.toarray()
        try:
            if self._use_gpu_model():
                X_gpu = to_gpu_if_needed(X, True, sparse_to_dense=True)
                predictions = self.model.predict(X_gpu)
            else:
                predictions = self.model.predict(X_cpu)
            return from_gpu_if_needed(predictions)
        except Exception as e:
            if self.use_gpu and CUML_AVAILABLE and self.cpu_model is not None and self._cpu_fitted:
                self.logger.warning("CUML Bayesian predict failed; using CPU fallback. error=%s", e)
                predictions = self.cpu_model.predict(X_cpu)
                return from_gpu_if_needed(predictions)
            raise
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        self._ensure_fitted()
        X_cpu = from_gpu_if_needed(X)
        if self.model_type == 'gaussian' and sp.issparse(X_cpu):
            self.logger.warning(
                "GaussianNB requires dense input; converting sparse matrix to dense for predict_proba."
            )
            X_cpu = X_cpu.toarray()
        try:
            if self._use_gpu_model():
                X_gpu = to_gpu_if_needed(X, True, sparse_to_dense=True)
                probabilities = self.model.predict_proba(X_gpu)
            else:
                probabilities = self.model.predict_proba(X_cpu)
            return from_gpu_if_needed(probabilities)
        except Exception as e:
            if self.use_gpu and CUML_AVAILABLE and self.cpu_model is not None and self._cpu_fitted:
                self.logger.warning("CUML Bayesian predict_proba failed; using CPU fallback. error=%s", e)
                probabilities = self.cpu_model.predict_proba(X_cpu)
                return from_gpu_if_needed(probabilities)
            raise

    def _init_cpu_model(self, model_type: Optional[str] = None):
        """Initialize sklearn fallback model."""
        from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB
        from sklearn.naive_bayes import ComplementNB as SKComplementNB
        from sklearn.naive_bayes import GaussianNB as SKGaussianNB
        model_type = model_type or self.model_type
        if model_type == 'complement':
            return SKComplementNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                **self.kwargs
            )
        if model_type == 'gaussian':
            var_smoothing = self.kwargs.get('var_smoothing', 1e-9)
            return SKGaussianNB(var_smoothing=var_smoothing)
        return SKMultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior, **self.kwargs)

    def _use_gpu_model(self) -> bool:
        if not (self.use_gpu and CUML_AVAILABLE):
            return False
        return type(self.model).__module__.startswith("cuml")

    def _requires_non_negative(self) -> bool:
        return self.model_type in ('multinomial', 'complement')

    def _has_negative_values(self, X) -> bool:
        if X is None:
            return False
        if sp.issparse(X):
            if X.data is None or X.data.size == 0:
                return False
            return np.nanmin(X.data) < 0
        try:
            arr = np.asarray(X)
        except Exception:
            return False
        if arr.size == 0:
            return False
        return np.nanmin(arr) < 0
    
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
