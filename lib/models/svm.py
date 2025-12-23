"""
CUML-based SVM with GPU support.
"""
import numpy as np
from typing import Optional, Dict, Any
from scipy.sparse import csr_matrix
import pickle

try:
    import cuml
    from cuml.svm import SVC as CUMLSVC
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    from sklearn.svm import SVC as SKSVC

from .base import BaseModel
from ..config import get_config
from ..logging.logger import get_logger
from ..utils.gpu_utils import to_gpu_if_needed, from_gpu_if_needed

logger = get_logger(__name__)


class SVMModel(BaseModel):
    """CUML-based SVM model."""
    
    def __init__(
        self,
        config=None,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        class_weight: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize SVM model.
        
        Args:
            config: Configuration object
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly')
            gamma: Gamma parameter ('scale', 'auto', or float)
            class_weight: Class weight ('balanced' or None)
            **kwargs: Additional arguments
        """
        super().__init__(config, model_name="svm")
        
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        self.kwargs = kwargs
        
        # Initialize model
        if self.use_gpu and CUML_AVAILABLE:
            self.model = CUMLSVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                class_weight=class_weight,
                **kwargs
            )
            self.logger.info("Initialized CUML SVC")
        else:
            self.model = SKSVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                class_weight=class_weight,
                probability=True,  # Enable probability estimates
                **kwargs
            )
            self.logger.info("Initialized sklearn SVC")
    
    def fit(self, X, y, **kwargs):
        """Fit the model."""
        self.logger.info("Fitting SVM model")
        
        X_gpu = to_gpu_if_needed(X, self.use_gpu and CUML_AVAILABLE, sparse_to_dense=True)
        self.model.fit(X_gpu, y, **kwargs)
        self._fitted = True
        
        self.logger.info("SVM model fitted")
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
        
        self.logger.info(f"Saving SVM model to {path}")
        
        model_state = {
            'model': self.model,
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'class_weight': self.class_weight,
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
        
        logger.info(f"Loading SVM model from {path}")
        
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        model = cls(
            config=config,
            C=model_state['C'],
            kernel=model_state['kernel'],
            gamma=model_state['gamma'],
            class_weight=model_state['class_weight'],
            **model_state['kwargs']
        )
        
        model.model = model_state['model']
        model._fitted = model_state['fitted']
        
        logger.info("Model loaded")
        return model

