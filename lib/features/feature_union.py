"""
Combine all feature types into a unified feature matrix.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from typing import Dict, List, Optional, Union

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class FeatureUnion:
    """Union of multiple feature extractors."""
    
    def __init__(self, config=None):
        """
        Initialize feature union.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.feature_names: List[str] = []
    
    def combine_features(
        self,
        nlp_features: Optional[csr_matrix] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        encodings: Optional[Dict[str, Union[np.ndarray, csr_matrix]]] = None,
        stylistic_features: Optional[np.ndarray] = None
    ) -> csr_matrix:
        """
        Combine all feature types into a single feature matrix.
        
        Args:
            nlp_features: Sparse matrix of NLP features (TF-IDF, etc.)
            embeddings: Dictionary of embedding arrays
            encodings: Dictionary of encoded features
            stylistic_features: Dense array of stylistic features
            
        Returns:
            Combined feature matrix (sparse)
        """
        self.logger.info("Combining all features")
        feature_blocks = []
        feature_names = []
        
        # NLP features (sparse)
        if nlp_features is not None:
            if nlp_features.shape[0] == 0 or nlp_features.shape[1] == 0:
                self.logger.warning("NLP features are empty, skipping")
            else:
                feature_blocks.append(nlp_features)
                feature_names.append(f"nlp_features_{nlp_features.shape[1]}")
                self.logger.debug(f"Added NLP features: {nlp_features.shape}")
        
        # Stylistic features (dense -> sparse)
        if stylistic_features is not None:
            if len(stylistic_features.shape) < 2 or stylistic_features.shape[0] == 0 or stylistic_features.shape[1] == 0:
                self.logger.warning("Stylistic features are empty, skipping")
            else:
                stylistic_sparse = csr_matrix(stylistic_features)
                feature_blocks.append(stylistic_sparse)
                feature_names.append(f"stylistic_features_{stylistic_features.shape[1]}")
                self.logger.debug(f"Added stylistic features: {stylistic_features.shape}")
        
        # Embeddings (dense -> sparse)
        if embeddings:
            for name, emb_array in embeddings.items():
                if emb_array is None or len(emb_array.shape) < 2 or emb_array.shape[0] == 0 or emb_array.shape[1] == 0:
                    self.logger.warning(f"Embedding {name} is empty, skipping")
                    continue
                emb_sparse = csr_matrix(emb_array)
                feature_blocks.append(emb_sparse)
                feature_names.append(f"{name}_embeddings_{emb_array.shape[1]}")
                self.logger.debug(f"Added {name} embeddings: {emb_array.shape}")
        
        # Encodings (sparse or dense -> sparse)
        if encodings:
            for name, enc_array in encodings.items():
                if isinstance(enc_array, csr_matrix):
                    feature_blocks.append(enc_array)
                else:
                    # Dense array -> sparse
                    enc_sparse = csr_matrix(enc_array.reshape(-1, 1) if enc_array.ndim == 1 else enc_array)
                    feature_blocks.append(enc_sparse)
                feature_names.append(f"{name}_encoding")
                self.logger.debug(f"Added {name} encoding")
        
        if not feature_blocks:
            raise ValueError("No features provided to combine")
        
        # Combine all features horizontally
        combined = hstack(feature_blocks)
        self.feature_names = feature_names
        
        self.logger.info(f"Combined features: shape {combined.shape}")
        return combined
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        return self.feature_names

