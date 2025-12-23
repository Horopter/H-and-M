"""
Base model interface with checkpoint support.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np
from pathlib import Path

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Base class for all models with checkpoint support."""
    
    def __init__(self, config=None, model_name: str = "model"):
        """
        Initialize base model.
        
        Args:
            config: Configuration object
            model_name: Name of the model
        """
        self.config = config or get_config()
        self.model_name = model_name
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self._fitted = False
        self.use_gpu = self.config.use_gpu
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Fit the model.
        
        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional arguments
        """
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted probabilities
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str, config=None):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            config: Configuration object
            
        Returns:
            Loaded model instance
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def set_params(self, **params):
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        if self.model is not None and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
    
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._fitted
    
    def _ensure_fitted(self):
        """Ensure model is fitted before prediction."""
        if not self._fitted:
            raise ValueError(f"{self.model_name} must be fitted before prediction")

