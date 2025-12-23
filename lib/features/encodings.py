"""
Encoding strategies with leakage prevention.
"""
import numpy as np
from typing import Dict, List, Optional, Union
from collections import Counter
from sklearn.model_selection import KFold

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class LabelEncoder:
    """Label encoding for categorical features."""
    
    def __init__(self):
        """Initialize label encoder."""
        self.label_to_index: Dict[str, int] = {}
        self.index_to_label: Dict[int, str] = {}
        self.logger = get_logger(self.__class__.__name__)
        self._fitted = False
    
    def fit(self, values: List[str]):
        """
        Fit the label encoder.
        
        Args:
            values: List of categorical values
        """
        self.logger.debug("Fitting LabelEncoder")
        unique_values = sorted(set(values))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_values)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        self._fitted = True
        return self
    
    def transform(self, values: List[str]) -> np.ndarray:
        """
        Transform values to encoded integers.
        
        Args:
            values: List of categorical values
            
        Returns:
            Array of encoded integers
        """
        if not self._fitted:
            raise ValueError("LabelEncoder must be fitted before transform")
        
        self.logger.debug("Transforming with LabelEncoder")
        encoded = []
        for val in values:
            if val in self.label_to_index:
                encoded.append(self.label_to_index[val])
            else:
                # Unknown value - assign to a default (could use -1 or most common)
                encoded.append(0)
        return np.array(encoded, dtype=np.int32)
    
    def fit_transform(self, values: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(values).transform(values)
    
    def inverse_transform(self, encoded: np.ndarray) -> List[str]:
        """
        Inverse transform encoded values back to labels.
        
        Args:
            encoded: Array of encoded integers
            
        Returns:
            List of original labels
        """
        return [self.index_to_label.get(int(idx), '') for idx in encoded]


class TargetEncoder:
    """Target encoding with cross-validation to prevent leakage."""
    
    def __init__(self, n_splits: int = 5, random_state: Optional[int] = None):
        """
        Initialize target encoder.
        
        Args:
            n_splits: Number of CV folds for encoding
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean: float = 0.0
        self.category_means: Dict[str, float] = {}
        self.logger = get_logger(self.__class__.__name__)
        self._fitted = False
    
    def fit(self, categories: List[str], targets: np.ndarray):
        """
        Fit the target encoder using CV to prevent leakage.
        
        Args:
            categories: List of categorical values
            targets: Target values
        """
        self.logger.debug("Fitting TargetEncoder with CV")
        
        categories = np.array(categories)
        targets = np.array(targets)
        
        # Calculate global mean
        self.global_mean = np.mean(targets)
        
        # Use cross-validation to calculate category means
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        category_sums = {}
        category_counts = {}
        
        for train_idx, val_idx in kf.split(categories):
            train_cats = categories[train_idx]
            train_targs = targets[train_idx]
            
            # Calculate means for each category in training fold
            for cat in np.unique(train_cats):
                cat_mask = train_cats == cat
                cat_mean = np.mean(train_targs[cat_mask])
                
                if cat not in category_sums:
                    category_sums[cat] = 0.0
                    category_counts[cat] = 0
                
                category_sums[cat] += cat_mean
                category_counts[cat] += 1
        
        # Average across folds
        self.category_means = {
            cat: category_sums[cat] / category_counts[cat]
            for cat in category_sums
        }
        
        self._fitted = True
        return self
    
    def transform(self, categories: List[str], smoothing: float = 1.0) -> np.ndarray:
        """
        Transform categories to target-encoded values.
        
        Args:
            categories: List of categorical values
            smoothing: Smoothing parameter (higher = more global mean influence)
            
        Returns:
            Array of encoded values
        """
        if not self._fitted:
            raise ValueError("TargetEncoder must be fitted before transform")
        
        self.logger.debug("Transforming with TargetEncoder")
        encoded = []
        
        for cat in categories:
            if cat in self.category_means:
                # Smooth with global mean
                cat_mean = self.category_means[cat]
                n_cat = 1  # Simplified - could track actual counts
                encoded_val = (cat_mean * n_cat + self.global_mean * smoothing) / (n_cat + smoothing)
                encoded.append(encoded_val)
            else:
                # Unknown category - use global mean
                encoded.append(self.global_mean)
        
        return np.array(encoded, dtype=np.float32)
    
    def fit_transform(
        self,
        categories: List[str],
        targets: np.ndarray,
        smoothing: float = 1.0
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(categories, targets).transform(categories, smoothing)


class FrequencyEncoder:
    """Frequency encoding for categorical features."""
    
    def __init__(self):
        """Initialize frequency encoder."""
        self.frequencies: Dict[str, float] = {}
        self.logger = get_logger(self.__class__.__name__)
        self._fitted = False
    
    def fit(self, values: List[str]):
        """
        Fit the frequency encoder.
        
        Args:
            values: List of categorical values
        """
        self.logger.debug("Fitting FrequencyEncoder")
        counter = Counter(values)
        total = len(values)
        self.frequencies = {val: count / total for val, count in counter.items()}
        self._fitted = True
        return self
    
    def transform(self, values: List[str]) -> np.ndarray:
        """
        Transform values to frequencies.
        
        Args:
            values: List of categorical values
            
        Returns:
            Array of frequency values
        """
        if not self._fitted:
            raise ValueError("FrequencyEncoder must be fitted before transform")
        
        self.logger.debug("Transforming with FrequencyEncoder")
        encoded = [self.frequencies.get(val, 0.0) for val in values]
        return np.array(encoded, dtype=np.float32)
    
    def fit_transform(self, values: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(values).transform(values)


class OneHotEncoder:
    """One-hot encoding (sparse format)."""
    
    def __init__(self):
        """Initialize one-hot encoder."""
        self.categories: List[str] = []
        self.category_to_index: Dict[str, int] = {}
        self.logger = get_logger(self.__class__.__name__)
        self._fitted = False
    
    def fit(self, values: List[str]):
        """
        Fit the one-hot encoder.
        
        Args:
            values: List of categorical values
        """
        self.logger.debug("Fitting OneHotEncoder")
        self.categories = sorted(set(values))
        self.category_to_index = {cat: idx for idx, cat in enumerate(self.categories)}
        self._fitted = True
        return self
    
    def transform(self, values: List[str], sparse: bool = True):
        """
        Transform values to one-hot encoding.
        
        Args:
            values: List of categorical values
            sparse: Whether to return sparse matrix
            
        Returns:
            One-hot encoded array or sparse matrix
        """
        if not self._fitted:
            raise ValueError("OneHotEncoder must be fitted before transform")
        
        self.logger.debug("Transforming with OneHotEncoder")
        
        n_samples = len(values)
        n_categories = len(self.categories)
        
        if sparse:
            from scipy.sparse import csr_matrix
            row_indices = []
            col_indices = []
            
            for i, val in enumerate(values):
                if val in self.category_to_index:
                    row_indices.append(i)
                    col_indices.append(self.category_to_index[val])
            
            data = np.ones(len(row_indices), dtype=np.float32)
            return csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_categories))
        else:
            encoded = np.zeros((n_samples, n_categories), dtype=np.float32)
            for i, val in enumerate(values):
                if val in self.category_to_index:
                    encoded[i, self.category_to_index[val]] = 1.0
            return encoded
    
    def fit_transform(self, values: List[str], sparse: bool = True):
        """Fit and transform in one step."""
        return self.fit(values).transform(values, sparse)


class EncodingPipeline:
    """Pipeline for multiple encoding strategies."""
    
    def __init__(self):
        """Initialize encoding pipeline."""
        self.encoders: Dict[str, Union[LabelEncoder, TargetEncoder, FrequencyEncoder, OneHotEncoder]] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def add_encoder(self, name: str, encoder: Union[LabelEncoder, TargetEncoder, FrequencyEncoder, OneHotEncoder]):
        """
        Add an encoder to the pipeline.
        
        Args:
            name: Name of the encoder
            encoder: Encoder instance
        """
        self.encoders[name] = encoder
        self.logger.debug(f"Added encoder: {name}")
    
    def fit(self, data: Dict[str, List], targets: Optional[np.ndarray] = None):
        """
        Fit all encoders in the pipeline.
        
        Args:
            data: Dictionary of column name to values
            targets: Optional target values for target encoding
        """
        self.logger.info("Fitting encoding pipeline")
        
        for name, encoder in self.encoders.items():
            if name not in data:
                self.logger.warning(f"Column {name} not found in data, skipping encoder")
                continue
            
            if isinstance(encoder, TargetEncoder):
                if targets is None:
                    self.logger.warning(f"TargetEncoder {name} requires targets, skipping")
                    continue
                encoder.fit(data[name], targets)
            else:
                encoder.fit(data[name])
        
        return self
    
    def transform(self, data: Dict[str, List]) -> Dict[str, Union[np.ndarray, object]]:
        """
        Transform data using all encoders.
        
        Args:
            data: Dictionary of column name to values
            
        Returns:
            Dictionary of encoded values
        """
        self.logger.debug("Transforming with encoding pipeline")
        encoded = {}
        
        for name, encoder in self.encoders.items():
            if name not in data:
                continue
            
            try:
                if isinstance(encoder, TargetEncoder):
                    encoded[name] = encoder.transform(data[name])
                elif isinstance(encoder, OneHotEncoder):
                    encoded[name] = encoder.transform(data[name], sparse=True)
                else:
                    encoded[name] = encoder.transform(data[name])
            except Exception as e:
                self.logger.warning(f"Encoding failed for {name}: {e}")
        
        return encoded

