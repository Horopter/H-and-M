"""
Data leakage and temporal leakage detection utilities.
"""
import numpy as np
from typing import List, Set, Tuple, Optional, Dict, Any
import polars as pl

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class LeakageDetector:
    """Detect data leakage and temporal leakage."""
    
    def __init__(self, config=None):
        """
        Initialize leakage detector.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
    
    def check_text_duplicates(
        self,
        train_texts: List[str],
        val_texts: List[str],
        test_texts: Optional[List[str]] = None
    ) -> Dict[str, Set[str]]:
        """
        Check for exact text duplicates between datasets.
        
        Args:
            train_texts: Training texts
            val_texts: Validation texts
            test_texts: Test texts (optional)
            
        Returns:
            Dictionary of overlap information
        """
        self.logger.info("Checking for text duplicates")
        
        train_set = set(str(t).strip() for t in train_texts)
        val_set = set(str(t).strip() for t in val_texts)
        test_set = set(str(t).strip() for t in test_texts) if test_texts else set()
        
        overlaps = {
            'train_val_overlap': train_set & val_set,
            'train_test_overlap': train_set & test_set if test_set else set(),
            'val_test_overlap': val_set & test_set if test_set else set()
        }
        
        # Log findings
        for key, overlap_set in overlaps.items():
            if overlap_set:
                self.logger.warning(f"{key}: {len(overlap_set)} duplicate texts found")
            else:
                self.logger.debug(f"{key}: No duplicates found")
        
        return overlaps
    
    def check_temporal_leakage(
        self,
        train_indices: List[int],
        val_indices: List[int],
        time_column: Optional[np.ndarray] = None
    ) -> bool:
        """
        Check for temporal leakage in train/val split.
        
        Args:
            train_indices: Training indices
            val_indices: Validation indices
            time_column: Optional time column values
            
        Returns:
            True if temporal leakage detected
        """
        self.logger.info("Checking for temporal leakage")
        
        if time_column is not None:
            train_times = time_column[train_indices]
            val_times = time_column[val_indices]
            
            # Check if any validation time is before training time
            max_train_time = np.max(train_times)
            min_val_time = np.min(val_times)
            
            if min_val_time < max_train_time:
                self.logger.warning(
                    f"Temporal leakage detected: "
                    f"min_val_time={min_val_time} < max_train_time={max_train_time}"
                )
                return True
            else:
                self.logger.debug("No temporal leakage detected")
                return False
        else:
            # Check index-based temporal order
            max_train_idx = max(train_indices)
            min_val_idx = min(val_indices)
            
            if min_val_idx <= max_train_idx:
                self.logger.warning(
                    f"Potential temporal leakage: "
                    f"min_val_idx={min_val_idx} <= max_train_idx={max_train_idx}"
                )
                return True
            else:
                self.logger.debug("No temporal leakage detected (index-based)")
                return False
    
    def check_cv_temporal_leakage(
        self,
        folds: List[Tuple[List[int], List[int]]],
        time_column: Optional[np.ndarray] = None
    ) -> List[bool]:
        """
        Check for temporal leakage in CV folds.
        
        Args:
            folds: List of (train_indices, val_indices) tuples
            time_column: Optional time column values
            
        Returns:
            List of leakage flags for each fold
        """
        self.logger.info("Checking CV folds for temporal leakage")
        
        leakage_flags = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            has_leakage = self.check_temporal_leakage(train_idx, val_idx, time_column)
            leakage_flags.append(has_leakage)
            
            if has_leakage:
                self.logger.warning(f"Temporal leakage detected in fold {fold_idx}")
        
        return leakage_flags
    
    def check_feature_leakage(
        self,
        feature_names: List[str],
        target_name: str = 'label'
    ) -> List[str]:
        """
        Check for feature names that might indicate leakage.
        
        Args:
            feature_names: List of feature names
            target_name: Name of target variable
            
        Returns:
            List of suspicious feature names
        """
        self.logger.info("Checking for feature leakage")
        
        suspicious = []
        target_lower = target_name.lower()
        
        for feat_name in feature_names:
            feat_lower = feat_name.lower()
            
            # Check for target name in feature name
            if target_lower in feat_lower and feat_lower != target_lower:
                suspicious.append(feat_name)
                self.logger.warning(f"Suspicious feature name: {feat_name}")
            
            # Check for common leakage indicators
            leakage_indicators = ['target', 'label', 'y_', 'outcome', 'result']
            if any(indicator in feat_lower for indicator in leakage_indicators):
                if feat_lower != target_lower:
                    suspicious.append(feat_name)
                    self.logger.warning(f"Potential leakage feature: {feat_name}")
        
        if not suspicious:
            self.logger.debug("No obvious feature leakage detected")
        
        return suspicious
    
    def validate_target_encoding(
        self,
        train_categories: List[str],
        train_targets: np.ndarray,
        val_categories: List[str],
        val_targets: np.ndarray
    ) -> bool:
        """
        Validate that target encoding doesn't leak information.
        
        Args:
            train_categories: Training categories
            train_targets: Training targets
            val_categories: Validation categories
            val_targets: Validation targets
            
        Returns:
            True if validation passes
        """
        self.logger.info("Validating target encoding")
        
        # Check if validation categories are in training
        train_cat_set = set(train_categories)
        val_cat_set = set(val_categories)
        
        unknown_cats = val_cat_set - train_cat_set
        if unknown_cats:
            self.logger.warning(
                f"Found {len(unknown_cats)} unknown categories in validation set. "
                "Target encoding should handle these properly."
            )
        
        # Check for perfect correlation (potential leakage)
        from collections import defaultdict
        cat_to_target = defaultdict(list)
        
        for cat, target in zip(train_categories, train_targets):
            cat_to_target[cat].append(target)
        
        # Check if any category has only one class (perfect separation)
        perfect_separation = []
        for cat, targets in cat_to_target.items():
            if len(set(targets)) == 1:
                perfect_separation.append(cat)
        
        if perfect_separation:
            self.logger.warning(
                f"Found {len(perfect_separation)} categories with perfect class separation. "
                "This might indicate leakage."
            )
            return False
        
        self.logger.debug("Target encoding validation passed")
        return True
    
    def comprehensive_check(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: Optional[pl.DataFrame] = None,
        text_column: Optional[str] = None,
        time_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive leakage check.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame (optional)
            text_column: Name of text column
            time_column: Name of time column (optional)
            
        Returns:
            Dictionary of check results
        """
        self.logger.info("Performing comprehensive leakage check")
        
        text_col = text_column or self.config.text_column
        
        results = {}
        
        # Text duplicates
        train_texts = train_df[text_col].to_list()
        val_texts = val_df[text_col].to_list()
        test_texts = test_df[text_col].to_list() if test_df is not None else None
        
        results['text_duplicates'] = self.check_text_duplicates(
            train_texts, val_texts, test_texts
        )
        
        # Temporal leakage
        if time_column and time_column in train_df.columns:
            train_times = train_df[time_column].to_numpy()
            val_times = val_df[time_column].to_numpy()
            
            # Simple check: all val times should be >= max train time
            max_train_time = np.max(train_times)
            min_val_time = np.min(val_times)
            
            results['temporal_leakage'] = min_val_time < max_train_time
            
            if results['temporal_leakage']:
                self.logger.warning("Temporal leakage detected!")
        else:
            results['temporal_leakage'] = False
            self.logger.debug("No time column provided, skipping temporal check")
        
        # Feature leakage (if feature names available)
        # This would need to be called separately with feature names
        
        return results

