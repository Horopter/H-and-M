"""
GridSearch hyperparameter tuning on 10% data.
"""
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from itertools import product
from scipy.sparse import csr_matrix
import hashlib
import json
from pathlib import Path

from ..checkpointing.checkpoint_manager import CheckpointManager

from ..config import get_config
from ..logging.logger import get_logger
from ..training.cv import CrossValidator
from ..utils.parallel import ParallelExecutor

logger = get_logger(__name__)


class GridSearch:
    """Grid search for hyperparameter tuning."""
    
    def __init__(self, config=None):
        """
        Initialize grid search.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.cv = CrossValidator(config)
        self.executor = ParallelExecutor(config)
    
    def generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from grid.
        
        Args:
            param_grid: Dictionary of parameter name to list of values
            
        Returns:
            List of parameter dictionaries
        """
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def evaluate_params(
        self,
        model_factory: Callable,
        param_dict: Dict[str, Any],
        X,
        y: np.ndarray,
        subset_size: float = 0.1,
        cv_folds: int = 5,
        original_data_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single parameter combination.
        
        Args:
            model_factory: Function that creates model with parameters
            param_dict: Parameter dictionary
            X: Feature matrix
            y: Target values
            subset_size: Fraction of data to use
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with parameters and CV results
        """
        self.logger.debug(f"Evaluating parameters: {param_dict}")
        
        param_payload = json.dumps(param_dict, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_payload.encode('utf-8')).hexdigest()[:8]
        checkpoint_tag = f"grid_{param_hash}"
        checkpoint_metadata = {
            'params': param_dict,
            'param_hash': param_hash
        }
        
        # Create model with parameters
        model = model_factory(**param_dict)
        if cv_folds and self.cv.n_splits != cv_folds:
            self.cv.n_splits = cv_folds

        cached_results = self._load_cached_cv_results(
            model.__class__.__name__,
            checkpoint_tag,
            self.cv.n_splits
        )
        if cached_results is not None:
            mean_f1 = cached_results.get('metrics', {}).get('f1', {}).get('mean', 0.0)
            self.logger.info(
                "Loaded cached CV results for %s (%s) - Mean F1: %.4f",
                model.__class__.__name__,
                checkpoint_tag,
                mean_f1
            )
            return {
                'params': param_dict,
                'mean_f1': mean_f1,
                'cv_results': cached_results,
                'cached': True
            }
        
        # Perform CV with random sampling (not temporal)
        cv_results = self.cv.cross_validate(
            lambda: model_factory(**param_dict),
            X,
            y,
            subset_size=subset_size,
            temporal=False,  # Random sampling for CV
            parallel=False,
            original_data_size=original_data_size,
            save_checkpoints=True,
            checkpoint_tag=checkpoint_tag,
            checkpoint_metadata=checkpoint_metadata,
            checkpoint_data=True
        )
        
        # Extract mean F1 score
        mean_f1 = cv_results.get('metrics', {}).get('f1', {}).get('mean', 0.0)
        
        result = {
            'params': param_dict,
            'mean_f1': mean_f1,
            'cv_results': cv_results
        }
        
        self.logger.info(f"Parameters {param_dict} - Mean F1: {mean_f1:.4f}")
        
        return result

    def _load_cached_cv_results(
        self,
        model_type: str,
        checkpoint_tag: str,
        n_splits: int
    ) -> Optional[Dict[str, Any]]:
        checkpoint_mgr = CheckpointManager(self.config)
        base_dir = Path(checkpoint_mgr.checkpoint_dir) / model_type / checkpoint_tag
        fold_results = {}

        for fold_id in range(n_splits):
            meta_path = base_dir / f"fold_{fold_id}" / "metadata.json"
            if not meta_path.exists():
                return None
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                metrics = meta.get('metrics', {})
                if not metrics:
                    return None
                fold_results[fold_id] = metrics
            except Exception:
                return None

        metric_names = set()
        for metrics in fold_results.values():
            metric_names.update(metrics.keys())

        all_metrics = {}
        for metric_name in metric_names:
            values = [fold_results[fid].get(metric_name, 0.0) for fid in sorted(fold_results)]
            all_metrics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }

        return {
            'n_folds': n_splits,
            'metrics': all_metrics,
            'fold_results': fold_results,
            'cached': True
        }
    
    def search(
        self,
        model_factory: Callable,
        param_grid: Dict[str, List],
        X,
        y: np.ndarray,
        subset_size: float = 0.1,
        cv_folds: int = 5,
        scoring: str = 'f1',
        parallel: bool = True,
        original_data_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform grid search.
        
        Args:
            model_factory: Function that creates model with parameters
            param_grid: Dictionary of parameter name to list of values
            X: Feature matrix
            y: Target values
            subset_size: Fraction of data to use
            cv_folds: Number of CV folds
            scoring: Scoring metric
            parallel: Whether to run in parallel
            
        Returns:
            Dictionary with best parameters and all results
        """
        self.logger.info("Starting grid search")
        
        # Generate parameter combinations
        param_combinations = self.generate_param_combinations(param_grid)
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Evaluate all combinations
        if parallel:
            def evaluate_combination(params):
                return self.evaluate_params(
                    model_factory,
                    params,
                    X,
                    y,
                    subset_size,
                    cv_folds,
                    original_data_size
                )
            
            results = self.executor.parallel_map(evaluate_combination, param_combinations)
        else:
            results = [
                self.evaluate_params(model_factory, params, X, y, subset_size, cv_folds, original_data_size)
                for params in param_combinations
            ]
        
        # Find best parameters
        best_result = max(results, key=lambda r: r['mean_f1'])
        best_params = best_result['params']
        best_score = best_result['mean_f1']
        
        grid_search_results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'n_combinations': len(param_combinations)
        }
        
        self.logger.info(
            f"Grid search complete - Best F1: {best_score:.4f}, "
            f"Best params: {best_params}"
        )
        
        return grid_search_results
    
    def get_best_model(
        self,
        model_factory: Callable,
        grid_search_results: Dict[str, Any]
    ):
        """
        Get best model with best parameters.
        
        Args:
            model_factory: Function that creates model with parameters
            grid_search_results: Results from grid search
            
        Returns:
            Model instance with best parameters
        """
        best_params = grid_search_results['best_params']
        model = model_factory(**best_params)
        
        self.logger.info(f"Created model with best parameters: {best_params}")
        return model
