"""
Data splitting utilities: 20% train, 20% val, 60% test with random sampling.
"""
import polars as pl
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """Data splitting with 20% train, 20% val, 60% test."""
    
    def __init__(self, config=None):
        """Initialize data splitter."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
    
    def split_20_20_60(
        self,
        df: pl.DataFrame,
        label_column: Optional[str] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Split data into 20% train, 20% val, 60% test with random sampling.
        
        Args:
            df: Polars DataFrame
            label_column: Label column for stratified split
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        random_state = random_state or self.config.random_state
        label_col = label_column or self.config.label_column
        
        self.logger.info("Splitting data: 20% train, 20% val, 60% test (random sampling)")
        
        if label_col not in df.columns:
            self.logger.warning("No label column found, using random split")
            # Random split without stratification
            indices = np.arange(len(df))
            np.random.seed(random_state)
            np.random.shuffle(indices)
            
            n_train = int(len(df) * 0.2)
            n_val = int(len(df) * 0.2)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        else:
            # Stratified split to maintain label distribution
            labels = df[label_col].to_numpy()
            indices = np.arange(len(df))
            
            # First split: 40% (train+val) vs 60% (test)
            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=0.6,
                stratify=labels,
                random_state=random_state
            )
            
            # Second split: 20% train vs 20% val from the 40%
            train_val_labels = labels[train_val_indices]
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=0.5,  # 50% of 40% = 20% of total
                stratify=train_val_labels,
                random_state=random_state
            )
        
        train_df = df[train_indices]
        val_df = df[val_indices]
        test_df = df[test_indices]
        
        self.logger.info(
            f"Split complete - Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
            f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
            f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)"
        )
        
        return train_df, val_df, test_df
    
    def sample_for_cv_fold(
        self,
        df: pl.DataFrame,
        original_size: int,
        sample_percent: float = 0.1,
        label_column: Optional[str] = None,
        random_state: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Sample data for a CV fold: up to 10% of original data per fold.
        
        Args:
            df: DataFrame to sample from
            original_size: Original dataset size (before any splits)
            sample_percent: Percentage of original data to sample (default 0.1 = 10%)
            label_column: Label column for stratified sampling
            random_state: Random seed
            
        Returns:
            Sampled DataFrame
        """
        random_state = random_state or self.config.random_state
        label_col = label_column or self.config.label_column
        
        # Calculate sample size: 10% of original data
        n_sample = int(original_size * sample_percent)
        
        # Don't sample more than available
        n_sample = min(n_sample, len(df))
        
        if n_sample == len(df):
            self.logger.debug(f"Sample size ({n_sample}) equals available data, returning all")
            return df
        
        self.logger.debug(f"Sampling {n_sample} samples ({sample_percent*100}% of original {original_size})")
        
        if label_col in df.columns:
            # Stratified sampling
            labels = df[label_col].to_numpy()
            indices = np.arange(len(df))
            
            # Use train_test_split to get stratified sample
            _, sampled_indices = train_test_split(
                indices,
                train_size=n_sample / len(df),
                stratify=labels,
                random_state=random_state
            )
        else:
            # Random sampling
            indices = np.arange(len(df))
            np.random.seed(random_state)
            sampled_indices = np.random.choice(indices, size=n_sample, replace=False)
        
        return df[sampled_indices]


class TemporalSplitter:
    """Temporal-aware data splitting to prevent temporal leakage."""
    
    def __init__(self, config=None):
        """
        Initialize temporal splitter.
        
        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
    
    def temporal_split(
        self,
        df: pl.DataFrame,
        train_ratio: float = 0.8,
        time_column: Optional[str] = None,
        label_column: Optional[str] = None
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data temporally to prevent leakage.
        
        Args:
            df: Polars DataFrame
            train_ratio: Ratio of data for training
            time_column: Column with temporal information (if None, uses index order)
            label_column: Label column for stratified split
            
        Returns:
            Tuple of (train_df, test_df)
        """
        self.logger.info("Performing temporal split")
        
        # If time column provided, sort by it
        if time_column and time_column in df.columns:
            df = df.sort(time_column)
            self.logger.debug(f"Sorted by time column: {time_column}")
        else:
            # Use index order (assumes data is already in temporal order)
            self.logger.debug("Using index order for temporal split")
        
        # Calculate split point
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        
        # Split temporally
        train_df = df.head(n_train)
        test_df = df.tail(n_total - n_train)
        
        self.logger.info(
            f"Temporal split - Train: {len(train_df)}, Test: {len(test_df)}"
        )
        
        return train_df, test_df
    
    def stratified_temporal_split(
        self,
        df: pl.DataFrame,
        train_ratio: float = 0.8,
        time_column: Optional[str] = None,
        label_column: Optional[str] = None
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Stratified temporal split to maintain label distribution.
        
        Args:
            df: Polars DataFrame
            train_ratio: Ratio of data for training
            time_column: Column with temporal information
            label_column: Label column name
            
        Returns:
            Tuple of (train_df, test_df)
        """
        label_col = label_column or self.config.label_column
        
        if label_col not in df.columns:
            self.logger.warning("No label column found, using regular temporal split")
            return self.temporal_split(df, train_ratio, time_column, label_column)
        
        self.logger.info("Performing stratified temporal split")
        
        # Sort by time if column provided
        if time_column and time_column in df.columns:
            df = df.sort(time_column)
        
        # Get label values
        labels = df[label_col].to_numpy()
        
        # Create stratified split maintaining temporal order
        # Group by label and split each group temporally
        train_indices = []
        test_indices = []
        
        for label in np.unique(labels):
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]
            
            n_label = len(label_indices)
            n_train_label = int(n_label * train_ratio)
            
            train_indices.extend(label_indices[:n_train_label])
            test_indices.extend(label_indices[n_train_label:])
        
        # Sort indices to maintain temporal order
        train_indices = sorted(train_indices)
        test_indices = sorted(test_indices)
        
        train_df = df[train_indices]
        test_df = df[test_indices]
        
        self.logger.info(
            f"Stratified temporal split - Train: {len(train_df)}, Test: {len(test_df)}"
        )
        
        return train_df, test_df
    
    def create_cv_folds_temporal(
        self,
        df: pl.DataFrame,
        n_splits: int = 5,
        time_column: Optional[str] = None,
        label_column: Optional[str] = None,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ) -> list:
        """
        Create temporal-aware CV folds.
        
        Args:
            df: Polars DataFrame
            n_splits: Number of CV folds
            time_column: Column with temporal information
            label_column: Label column for stratification
            shuffle: Whether to shuffle (not recommended for temporal)
            random_state: Random seed
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        random_state = random_state or self.config.random_state
        label_col = label_column or self.config.label_column
        
        # Sort by time if column provided
        if time_column and time_column in df.columns:
            df = df.sort(time_column)
        
        if label_col in df.columns:
            labels = df[label_col].to_numpy()
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            folds = list(skf.split(np.arange(len(df)), labels))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            folds = list(kf.split(np.arange(len(df))))
        
        return folds
