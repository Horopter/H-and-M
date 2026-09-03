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
    _GPU_TREE_METHOD_SUPPORTED: Optional[bool] = None
    _GPU_TREE_METHOD_LOGGED: bool = False
    
    def __init__(
        self,
        config=None,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        tree_method: Optional[str] = None,
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
        
        if tree_method is None:
            tree_method = getattr(self.config, "xgb_tree_method", "gpu_hist")

        # Use GPU if available and configured
        if self.use_gpu:
            tree_method = tree_method or 'gpu_hist'
        else:
            tree_method = 'hist'

        tree_method = self._resolve_tree_method(tree_method)
        
        self.tree_method = tree_method
        self.kwargs = kwargs
        self.model = self._build_model(tree_method)
        self.logger.info(f"Initialized XGBoost with tree_method={tree_method}")
    
    def fit(self, X, y, **kwargs):
        """Fit the model."""
        self.logger.info("Fitting XGBoost model")
        
        # XGBoost can handle sparse matrices directly
        # Convert to DMatrix for efficiency if needed
        try:
            if isinstance(X, csr_matrix):
                # XGBoost works with sparse matrices
                self.model.fit(X, y, **kwargs)
            else:
                self.model.fit(X, y, **kwargs)
        except Exception as e:
            if self.tree_method == 'gpu_hist' and "gpu_hist" in str(e):
                if not self.__class__._GPU_TREE_METHOD_LOGGED:
                    self.logger.warning("XGBoost GPU tree_method failed; retrying with 'hist'. error=%s", e)
                    self.__class__._GPU_TREE_METHOD_LOGGED = True
                self.__class__._GPU_TREE_METHOD_SUPPORTED = False
                self.tree_method = 'hist'
                self.use_gpu = False
                self.model = self._build_model(self.tree_method)
                if isinstance(X, csr_matrix):
                    self.model.fit(X, y, **kwargs)
                else:
                    self.model.fit(X, y, **kwargs)
            else:
                raise
        
        self._fitted = True
        
        self.logger.info("XGBoost model fitted")
        return self

    def _build_model(self, tree_method: str):
        """Build XGBoost classifier with current hyperparameters."""
        filtered_kwargs = dict(self.kwargs)
        filtered_kwargs.pop('use_label_encoder', None)
        return xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method=tree_method,
            eval_metric='logloss',
            **filtered_kwargs
        )

    @classmethod
    def _resolve_tree_method(cls, tree_method: str) -> str:
        if tree_method != 'gpu_hist':
            return tree_method
        support = cls._detect_gpu_support()
        if support is False:
            if not cls._GPU_TREE_METHOD_LOGGED:
                logger.warning("XGBoost build lacks CUDA support; forcing tree_method='hist'")
                cls._GPU_TREE_METHOD_LOGGED = True
            return 'hist'
        return tree_method

    @classmethod
    def _detect_gpu_support(cls) -> Optional[bool]:
        if cls._GPU_TREE_METHOD_SUPPORTED is not None:
            return cls._GPU_TREE_METHOD_SUPPORTED
        support = None
        try:
            if hasattr(xgb, "build_info"):
                info = xgb.build_info()
                if isinstance(info, dict):
                    use_cuda = info.get("USE_CUDA")
                    if use_cuda is None and "build_info" in info and isinstance(info["build_info"], dict):
                        use_cuda = info["build_info"].get("USE_CUDA")
                    if use_cuda is not None:
                        if isinstance(use_cuda, str):
                            support = use_cuda.strip().lower() in ("1", "true", "on", "yes")
                        else:
                            support = bool(use_cuda)
        except Exception:
            support = None
        if support is None:
            try:
                if hasattr(xgb.core, "_has_cuda_support"):
                    support = bool(xgb.core._has_cuda_support())
            except Exception:
                support = None
        cls._GPU_TREE_METHOD_SUPPORTED = support
        return support
    
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
