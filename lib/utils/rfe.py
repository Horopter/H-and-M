"""
Recursive Feature Elimination (RFE) for feature selection.
"""
import numpy as np
from typing import Dict, Any, Optional, List
from scipy.sparse import csr_matrix

try:
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..config import get_config
from ..logging.logger import get_logger
from ..utils.gc_utils import collect_after_chunk, collect_after_operation

logger = get_logger(__name__)


class RecursiveFeatureElimination:
    """Recursive Feature Elimination for feature selection."""
    
    def __init__(self, config=None):
        """Initialize RFE."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.selector = None
        self.n_features_selected = None
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("sklearn not available, RFE disabled")
    
    def fit_transform(
        self,
        X: csr_matrix,
        y: np.ndarray,
        n_features_to_select: Optional[int] = None,
        step: float = 0.1,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform RFE and return selected features.
        
        Args:
            X: Feature matrix
            y: Target values
            n_features_to_select: Number of features to select (None = auto)
            step: Fraction of features to remove at each step
            cv: Number of CV folds for RFECV
            
        Returns:
            Dictionary with RFE results
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("RFE skipped: sklearn not available")
            return {'n_features_selected': X.shape[1], 'feature_ranking': None}
        
        self.logger.info(f"Performing RFE on {X.shape[1]} features")
        
        # Convert sparse to dense if needed (RFE requires dense)
        # Process in chunks to save memory
        from ..constants import DEFAULT_CHUNK_SIZE
        chunk_size = getattr(self.config, 'chunk_size', DEFAULT_CHUNK_SIZE) if hasattr(self, 'config') else DEFAULT_CHUNK_SIZE
        if isinstance(X, csr_matrix):
            if X.shape[0] > chunk_size:
                chunks = []
                for i in range(0, X.shape[0], chunk_size):
                    chunk = X[i:i+chunk_size].toarray()
                    chunks.append(chunk)
                    del chunk
                    collect_after_chunk(i // chunk_size, aggressive=True)
                X_dense = np.vstack(chunks)
                del chunks
                collect_after_chunk(None, aggressive=True)
            else:
                X_dense = X.toarray()
        else:
            X_dense = X
        
        # Use Logistic Regression as base estimator
        estimator = LogisticRegression(
            random_state=self.config.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        
        # Use RFECV for automatic feature selection
        if n_features_to_select is None:
            self.logger.info("Using RFECV for automatic feature selection")
            self.selector = RFECV(
                estimator,
                step=int(max(1, X.shape[1] * step)),
                cv=cv,
                scoring='f1',
                n_jobs=-1
            )
        else:
            self.logger.info(f"Using RFE to select {n_features_to_select} features")
            self.selector = RFE(
                estimator,
                n_features_to_select=n_features_to_select,
                step=int(max(1, X.shape[1] * step))
            )
        
        # Fit selector
        self.selector.fit(X_dense, y)
        del X_dense
        collect_after_operation("rfe_selector_fit", aggressive=True)
        
        self.n_features_selected = self.selector.n_features_
        
        results = {
            'n_features_selected': self.n_features_selected,
            'n_features_original': X.shape[1],
            'feature_ranking': self.selector.ranking_ if hasattr(self.selector, 'ranking_') else None,
            'support': self.selector.support_ if hasattr(self.selector, 'support_') else None
        }
        
        if hasattr(self.selector, 'grid_scores_'):
            results['cv_scores'] = self.selector.grid_scores_.tolist()
        
        self.logger.info(f"RFE selected {self.n_features_selected} features from {X.shape[1]}")
        
        return results
    
    def transform(self, X: csr_matrix) -> csr_matrix:
        """
        Apply feature selection to new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Selected features
        """
        if self.selector is None:
            self.logger.warning("RFE not fitted, returning original features")
            return X
        
        # Convert sparse to dense if needed (chunked for memory)
        from ..constants import DEFAULT_CHUNK_SIZE
        chunk_size = getattr(self.config, 'chunk_size', DEFAULT_CHUNK_SIZE) if hasattr(self, 'config') else DEFAULT_CHUNK_SIZE
        if isinstance(X, csr_matrix):
            if X.shape[0] > chunk_size:
                chunks = []
                for i in range(0, X.shape[0], chunk_size):
                    chunk = self.selector.transform(X[i:i+chunk_size].toarray())
                    chunks.append(chunk)
                    del chunk
                    collect_after_chunk(i // chunk_size, aggressive=True)
                X_selected = np.vstack(chunks)
                del chunks
                collect_after_chunk(None, aggressive=True)
            else:
                X_selected = self.selector.transform(X.toarray())
        else:
            X_selected = self.selector.transform(X)
        
        # Convert back to sparse
        return csr_matrix(X_selected)

