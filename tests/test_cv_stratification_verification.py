"""
Comprehensive test to verify CV is ALWAYS stratified.
"""
import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.training.cv import CrossValidator
from lib.config import Config


class TestCVStratificationVerification(unittest.TestCase):
    """Comprehensive verification that CV is always stratified."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config({
            'cv_folds': 5,
            'random_state': 42,
            'subset_size': 1.0
        })
        self.cv = CrossValidator(self.config, n_splits=5)
    
    def test_always_stratified(self):
        """Test that CV is ALWAYS stratified regardless of temporal setting."""
        n_samples = 1000
        n_features = 100
        
        # Create imbalanced dataset (70/30 split)
        y = np.concatenate([
            np.zeros(700, dtype=int),
            np.ones(300, dtype=int)
        ])
        np.random.seed(42)
        np.random.shuffle(y)
        
        X = np.random.randn(n_samples, n_features)
        
        # Test with temporal=False
        folds_no_temp = self.cv.create_folds(X, y, subset_size=1.0, temporal=False)
        
        # Test with temporal=True
        folds_with_temp = self.cv.create_folds(X, y, subset_size=1.0, temporal=True)
        
        # Both should be stratified
        for folds, name in [(folds_no_temp, "non-temporal"), (folds_with_temp, "temporal")]:
            with self.subTest(fold_type=name):
                for fold_id, (train_idx, val_idx) in enumerate(folds):
                    y_train = y[train_idx]
                    y_val = y[val_idx]
                    
                    # Calculate class distributions
                    train_dist_0 = np.sum(y_train == 0) / len(y_train)
                    train_dist_1 = np.sum(y_train == 1) / len(y_train)
                    val_dist_0 = np.sum(y_val == 0) / len(y_val)
                    val_dist_1 = np.sum(y_val == 1) / len(y_val)
                    
                    # Overall distribution
                    overall_dist_0 = np.sum(y == 0) / len(y)
                    overall_dist_1 = np.sum(y == 1) / len(y)
                    
                    # Train and val distributions should be close to overall (within 5%)
                    self.assertAlmostEqual(
                        train_dist_0, overall_dist_0, delta=0.05,
                        msg=f"{name} Fold {fold_id}: Train class 0 distribution {train_dist_0:.3f} "
                            f"not close to overall {overall_dist_0:.3f}"
                    )
                    self.assertAlmostEqual(
                        val_dist_0, overall_dist_0, delta=0.05,
                        msg=f"{name} Fold {fold_id}: Val class 0 distribution {val_dist_0:.3f} "
                            f"not close to overall {overall_dist_0:.3f}"
                    )
                    
                    # Train and val should be similar to each other (within 3%)
                    self.assertAlmostEqual(
                        train_dist_0, val_dist_0, delta=0.03,
                        msg=f"{name} Fold {fold_id}: Train {train_dist_0:.3f} and Val {val_dist_0:.3f} "
                            f"distributions differ too much"
                    )
    
    def test_stratified_kfold_used(self):
        """Verify StratifiedKFold is actually used."""
        import inspect
        from lib.training.cv import CrossValidator
        
        # Check source code
        source = inspect.getsource(CrossValidator.create_folds)
        
        self.assertIn('StratifiedKFold', source)
        self.assertIn('skf.split', source)
        
        # Verify it's imported
        from lib.training import cv
        import importlib
        importlib.reload(cv)
        
        self.assertTrue(hasattr(cv, 'StratifiedKFold'))


if __name__ == '__main__':
    unittest.main()

