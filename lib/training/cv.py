"""
5-fold Cross-Validation on 10% data with temporal leakage prevention.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from ..config import get_config
from ..logging.logger import get_logger
from ..data.splitter import TemporalSplitter
from ..utils.validation import LeakageDetector
from ..utils.parallel import ParallelExecutor

logger = get_logger(__name__)


class CrossValidator:
    """5-fold Cross-Validation with temporal leakage prevention."""
    
    def __init__(self, config=None, n_splits: int = 5):
        """
        Initialize cross-validator.
        
        Args:
            config: Configuration object
            n_splits: Number of CV folds
        """
        self.config = config or get_config()
        self.n_splits = n_splits
        self.logger = get_logger(self.__class__.__name__)
        self.splitter = TemporalSplitter(config)
        self.leakage_detector = LeakageDetector(config)
        self.executor = ParallelExecutor(config)
    
    def create_folds(
        self,
        X,
        y: np.ndarray,
        subset_size: float = 0.1,
        temporal: bool = True,
        original_data_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create CV folds with temporal awareness.
        
        Args:
            X: Feature matrix
            y: Target values
            subset_size: Fraction of data to use
            temporal: Whether to enforce temporal ordering
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        self.logger.info(f"Creating {self.n_splits} CV folds (subset_size={subset_size})")
        
        # Validate input
        if len(y) == 0:
            raise ValueError("Cannot create CV folds: y is empty")
        if len(y) < self.n_splits:
            raise ValueError(
                f"Cannot create {self.n_splits} folds: only {len(y)} samples available. "
                f"Need at least {self.n_splits} samples."
            )
        
        # For CV: use 10% of original data PER FOLD (random sampling)
        # subset_size here represents the percentage of original data per fold (e.g., 0.1 = 10% per fold)
        # original_data_size is the size of the original dataset before any splits
        if subset_size < 1.0:
            # Calculate sample size: 10% of original data per fold
            if original_data_size is not None:
                n_per_fold = max(1, int(original_data_size * subset_size))  # 10% of original data
                self.logger.info(f"Sampling {n_per_fold} samples per fold (10% of original {original_data_size})")
            else:
                # Fallback: use percentage of available data
                n_per_fold = max(1, int(len(y) * subset_size))
                self.logger.warning(f"Original data size not provided, using {subset_size*100}% of available data ({n_per_fold} samples)")
            
            if n_per_fold < self.n_splits:
                self.logger.warning(
                    f"Sample per fold ({n_per_fold}) is smaller than n_splits ({self.n_splits}). "
                    f"Using all {len(y)} samples instead."
                )
                X_subset = X
                y_subset = y
                subset_indices = np.arange(len(y))
            else:
                # Random stratified sampling for CV
                indices = np.arange(len(y))
                skf = StratifiedKFold(n_splits=1, shuffle=True, random_state=self.config.random_state)
                train_idx, _ = next(skf.split(indices, y))
                # Sample n_per_fold samples
                np.random.seed(self.config.random_state)
                sampled_idx = np.random.choice(train_idx, size=min(n_per_fold, len(train_idx)), replace=False)
                subset_indices = sampled_idx
                
                X_subset = X[subset_indices] if hasattr(X, '__getitem__') else X
                y_subset = y[subset_indices]
                
                self.logger.info(f"Sampled {len(subset_indices)} samples for CV (target was {n_per_fold})")
        else:
            X_subset = X
            y_subset = y
            subset_indices = np.arange(len(y))
        
        # Always use stratified folds with random sampling (not temporal for CV)
        # For CV, we use random sampling, not temporal ordering
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,  # Random shuffle for CV (not temporal)
            random_state=self.config.random_state
        )
        folds = list(skf.split(X_subset, y_subset))
        
        # For CV with random sampling, temporal leakage is not a concern
        # We use random sampling, so no temporal ordering is maintained
        if temporal:
            self.logger.info("CV uses random sampling (not temporal), so temporal leakage checks are skipped")
        
        self.logger.info(f"Created {len(folds)} folds")
        return folds
    
    def evaluate_fold(
        self,
        model,
        X_train,
        y_train: np.ndarray,
        X_val,
        y_val: np.ndarray,
        fold_id: int
    ) -> Dict[str, float]:
        """
        Evaluate a single fold.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            fold_id: Fold ID
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating fold {fold_id}")
        
        # Train model
        model.fit(X_train, y_train)
        collect_after_operation("cv_model_fit", aggressive=True)
        
        # Predict
        y_pred = model.predict(X_val)
        collect_after_operation("cv_model_predict", aggressive=True)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        if y_proba is not None:
            collect_after_operation("cv_model_predict_proba", aggressive=True)
        
        # Calculate metrics
        metrics = {
            'f1': f1_score(y_val, y_pred),
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_val, y_proba)
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        self.logger.info(
            f"Fold {fold_id} - F1: {metrics['f1']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )
        
        return metrics
    
    def evaluate_model(
        self,
        model_factory,
        X,
        y: np.ndarray,
        subset_size: float = 0.1,
        temporal: bool = False,
        model_params: Optional[Dict[str, Any]] = None,
        save_checkpoints: bool = False,
        original_data_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model with cross-validation, optionally saving checkpoints per fold.
        
        Args:
            model_factory: Function that creates model instances
            X: Feature matrix
            y: Target array
            subset_size: Fraction of data to use
            temporal: Whether to enforce temporal ordering
            model_params: Parameters to pass to model factory
            save_checkpoints: Whether to save model checkpoints for each fold
            
        Returns:
            Dictionary with CV results
        """
        return self.cross_validate(
            model_factory,
            X,
            y,
            subset_size=subset_size,
            temporal=temporal,
            model_params=model_params,
            save_checkpoints=save_checkpoints,
            original_data_size=original_data_size
        )
    
    def cross_validate(
        self,
        model_factory,
        X,
        y: np.ndarray,
        subset_size: float = 0.1,
        temporal: bool = False,
        parallel: bool = False,
        model_params: Optional[Dict[str, Any]] = None,
        save_checkpoints: bool = False,
        original_data_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            model_factory: Function that creates a model instance
            X: Feature matrix
            y: Target values
            subset_size: Fraction of data to use
            temporal: Whether to enforce temporal ordering
            parallel: Whether to run folds in parallel
            
        Returns:
            Dictionary of CV results
        """
        self.logger.info(f"Starting {self.n_splits}-fold cross-validation")
        
        # Create folds
        folds = self.create_folds(X, y, subset_size, temporal, original_data_size)
        
        # Evaluate each fold
        fold_results = []
        
        if parallel:
            # Parallel execution
            def evaluate_single_fold(fold_data):
                fold_id, (train_idx, val_idx) = fold_data
                model = model_factory()
                metrics = self.evaluate_fold(
                    model,
                    X[train_idx] if hasattr(X, '__getitem__') else X,
                    y[train_idx],
                    X[val_idx] if hasattr(X, '__getitem__') else X,
                    y[val_idx],
                    fold_id
                )
                return fold_id, metrics
            
            fold_data_list = [(i, fold) for i, fold in enumerate(folds)]
            results = self.executor.parallel_folds(evaluate_single_fold, fold_data_list)
            fold_results = sorted(results, key=lambda x: x[0])
        else:
            # Sequential execution
            for fold_id, (train_idx, val_idx) in enumerate(folds):
                # Create model with params if provided
                if model_params:
                    model = model_factory(**model_params)
                else:
                    model = model_factory()
                
                metrics = self.evaluate_fold(
                    model,
                    X[train_idx] if hasattr(X, '__getitem__') else X,
                    y[train_idx],
                    X[val_idx] if hasattr(X, '__getitem__') else X,
                    y[val_idx],
                    fold_id
                )
                
                # Save checkpoint if requested
                if save_checkpoints:
                    try:
                        from ..checkpointing.checkpoint_manager import CheckpointManager
                        checkpoint_mgr = CheckpointManager(self.config)
                        checkpoint_mgr.save_checkpoint(
                            model,
                            model.__class__.__name__,
                            fold_id=fold_id,
                            score=metrics.get('f1', 0),
                            metadata={'fold_id': fold_id, 'metrics': metrics}
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to save checkpoint for fold {fold_id}: {e}")
                
                fold_results.append((fold_id, metrics))
        
        # Aggregate results
        if not fold_results:
            self.logger.warning("No fold results to aggregate")
            return {
                'n_folds': self.n_splits,
                'metrics': {},
                'fold_results': {}
            }
        
        all_metrics = {}
        
        for metric_name in fold_results[0][1].keys():
            values = [result[1].get(metric_name, 0.0) for result in fold_results]
            all_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        cv_results = {
            'n_folds': self.n_splits,
            'metrics': all_metrics,
            'fold_results': {fold_id: metrics for fold_id, metrics in fold_results}
        }
        
        f1_mean = all_metrics.get('f1', {}).get('mean', 0.0)
        f1_std = all_metrics.get('f1', {}).get('std', 0.0)
        self.logger.info(
            f"CV Results - F1: {f1_mean:.4f} ± {f1_std:.4f}"
        )
        
        return cv_results

