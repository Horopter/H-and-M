"""
XGBoost model with GPU support.
"""
import numpy as np
from typing import Optional, Dict, Any
from scipy.sparse import csr_matrix
import pickle

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .base import BaseModel
from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost model with GPU support."""
    
    def __init__(
        self,
        config=None,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        tree_method: str = 'gpu_hist',
        **kwargs
    ):
        """
        Initialize XGBoost model.
        
        Args:
            config: Configuration object
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            n_estimators: Number of estimators
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            tree_method: Tree construction method ('gpu_hist' for GPU)
            **kwargs: Additional arguments
        """
        super().__init__(config, model_name="xgboost")
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoostModel")
        
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
        # Use GPU if available and configured
        if self.use_gpu:
            tree_method = tree_method or 'gpu_hist'
        else:
            tree_method = 'hist'
        
        self.tree_method = tree_method
        self.kwargs = kwargs
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            use_label_encoder=False,
            eval_metric='logloss',
            **kwargs
        )
        
        self.logger.info(f"Initialized XGBoost with tree_method={tree_method}")
    
    def fit(self, X, y, **kwargs):
        """Fit the model."""
        self.logger.info("Fitting XGBoost model")
        
        # XGBoost can handle sparse matrices directly
        # Convert to DMatrix for efficiency if needed
        if isinstance(X, csr_matrix):
            # XGBoost works with sparse matrices
            self.model.fit(X, y, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)
        
        self._fitted = True
        
        self.logger.info("XGBoost model fitted")
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        self._ensure_fitted()
        
        predictions = self.model.predict(X)
        return np.array(predictions)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        self._ensure_fitted()
        
        probabilities = self.model.predict_proba(X)
        return np.array(probabilities)
    
    def save(self, path: str):
        """Save model to disk."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving XGBoost model to {path}")
        
        model_state = {
            'model': self.model,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'tree_method': self.tree_method,
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
        
        logger.info(f"Loading XGBoost model from {path}")
        
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        model = cls(
            config=config,
            max_depth=model_state['max_depth'],
            learning_rate=model_state['learning_rate'],
            n_estimators=model_state['n_estimators'],
            subsample=model_state['subsample'],
            colsample_bytree=model_state['colsample_bytree'],
            tree_method=model_state['tree_method'],
            **model_state['kwargs']
        )
        
        model.model = model_state['model']
        model._fitted = model_state['fitted']
        
        logger.info("Model loaded")
        return model

