"""
CUML-based Logistic Regression with GPU support.
"""
import numpy as np
from typing import Optional, Dict, Any
from scipy.sparse import csr_matrix
from scipy import sparse as sp
import pickle

try:
    import cuml
    from cuml.linear_model import LogisticRegression as CUMLLogisticRegression
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from .base import BaseModel
from ..config import get_config
from ..logging.logger import get_logger
from ..utils.gpu_utils import to_gpu_if_needed, from_gpu_if_needed

logger = get_logger(__name__)


class LogisticRegressionModel(BaseModel):
    """CUML-based Logistic Regression model."""
    
    def __init__(
        self,
        config=None,
        C: float = 1.0,
        penalty: str = 'l2',
        l1_ratio: float = 0.5,
        class_weight: Optional[str] = None,
        max_iter: int = 2000,
        tol: float = 1e-4,
        linesearch_max_iter: int = 100,
        **kwargs
    ):
        """
        Initialize Logistic Regression model.
        
        Args:
            config: Configuration object
            C: Regularization strength
            penalty: Penalty type ('l1', 'l2', 'elasticnet')
            l1_ratio: L1 ratio for elasticnet
            class_weight: Class weight ('balanced' or None)
            max_iter: Maximum iterations
            **kwargs: Additional arguments
        """
        super().__init__(config, model_name="logistic_regression")
        
        self.C = C
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.linesearch_max_iter = linesearch_max_iter
        self.solver = None
        self.kwargs = kwargs
        
        # Initialize model
        if self.use_gpu and CUML_AVAILABLE:
            base_kwargs = {
                "C": C,
                "penalty": penalty,
                "l1_ratio": l1_ratio if penalty == 'elasticnet' else None,
                "class_weight": class_weight,
                "max_iter": max_iter,
                "tol": tol
            }
            try:
                import inspect
                sig = inspect.signature(CUMLLogisticRegression)
                if "linesearch_max_iter" in sig.parameters:
                    base_kwargs["linesearch_max_iter"] = linesearch_max_iter
                base_kwargs = {k: v for k, v in base_kwargs.items() if k in sig.parameters}
            except Exception:
                pass
            self.model = CUMLLogisticRegression(**base_kwargs, **kwargs)
            self.logger.info("Initialized CUML LogisticRegression")
        else:
            self.solver = 'saga' if penalty in ['l1', 'elasticnet'] else 'lbfgs'
            self.model = self._init_cpu_model(self.solver)
            self.logger.info("Initialized sklearn LogisticRegression")
    
    def fit(self, X, y, **kwargs):
        """Fit the model."""
        self.logger.info("Fitting LogisticRegression model")

        if not (self.use_gpu and CUML_AVAILABLE) and sp.issparse(X):
            if self.solver in ('lbfgs', 'newton-cg'):
                self.logger.warning(
                    "Sparse input with solver=%s; switching to saga for sklearn LogisticRegression",
                    self.solver
                )
                self.solver = 'saga'
                self.model = self._init_cpu_model(self.solver)

        # Convert to GPU format if needed
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=True)
        try:
            self.model.fit(X_gpu, y, **kwargs)
        except TypeError as e:
            if not (self.use_gpu and CUML_AVAILABLE) and "Sparse data was passed" in str(e):
                self.logger.warning("Sparse input rejected by solver; retrying with dense array. error=%s", e)
                X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)
                self.model.fit(X_dense, y, **kwargs)
            else:
                raise
        self._fitted = True
        
        self.logger.info("LogisticRegression model fitted")
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

    def _init_cpu_model(self, solver: str):
        """Initialize sklearn LogisticRegression model."""
        return SKLogisticRegression(
            C=self.C,
            penalty=self.penalty,
            l1_ratio=self.l1_ratio if self.penalty == 'elasticnet' else None,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=solver,
            **self.kwargs
        )
    
    def save(self, path: str):
        """Save model to disk."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving LogisticRegression model to {path}")
        
        # Save model state
        model_state = {
            'model': self.model,
            'C': self.C,
            'penalty': self.penalty,
            'l1_ratio': self.l1_ratio,
            'class_weight': self.class_weight,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'linesearch_max_iter': self.linesearch_max_iter,
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
        
        logger.info(f"Loading LogisticRegression model from {path}")
        
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        # Recreate model
        model = cls(
            config=config,
            C=model_state['C'],
            penalty=model_state['penalty'],
            l1_ratio=model_state['l1_ratio'],
            class_weight=model_state['class_weight'],
            max_iter=model_state['max_iter'],
            tol=model_state.get('tol', 1e-4),
            linesearch_max_iter=model_state.get('linesearch_max_iter', 100),
            **model_state['kwargs']
        )
        
        model.model = model_state['model']
        model._fitted = model_state['fitted']
        
        logger.info("Model loaded")
        return model
