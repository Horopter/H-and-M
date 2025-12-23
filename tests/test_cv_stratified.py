"""
Tests to verify stratified 5-fold CV implementation.
"""
import unittest
import numpy as np
from scipy.sparse import csr_matrix

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.training.cv import CrossValidator
from lib.config import Config


class TestStratifiedCV(unittest.TestCase):
    """Test that CV is properly stratified."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config({
            'cv_folds': 5,
            'random_state': 42,
            'subset_size': 1.0  # Use full data for testing
        })
        self.cv = CrossValidator(self.config, n_splits=5)
    
    def test_stratified_folds(self):
        """Test that folds maintain class distribution."""
        # Create imbalanced dataset
        n_samples = 1000
        n_features = 100
        
        # 70% class 0, 30% class 1
        y = np.concatenate([
            np.zeros(700, dtype=int),
            np.ones(300, dtype=int)
        ])
        np.random.seed(42)
        np.random.shuffle(y)
        
        X = np.random.randn(n_samples, n_features)
        
        # Create folds
        folds = self.cv.create_folds(X, y, subset_size=1.0, temporal=False)
        
        self.assertEqual(len(folds), 5)
        
        # Check each fold maintains stratification
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            # Check class distribution in train and val
            train_dist = np.bincount(y_train) / len(y_train)
            val_dist = np.bincount(y_val) / len(y_val)
            
            # Should be approximately equal (within 5%)
            self.assertAlmostEqual(
                train_dist[0], val_dist[0], delta=0.05,
                msg=f"Fold {fold_id}: Class distribution not maintained"
            )
            
            # Log for debugging
            print(f"Fold {fold_id}: Train dist={train_dist}, Val dist={val_dist}")
    
    def test_cv_metrics(self):
        """Test CV evaluation."""
        n_samples = 100
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Simple model factory
        def model_factory():
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=42, max_iter=100)
        
        # Run CV
        results = self.cv.cross_validate(
            model_factory,
            X,
            y,
            subset_size=1.0,
            temporal=False,
            parallel=False
        )
        
        # Check results structure
        self.assertIn('n_folds', results)
        self.assertIn('metrics', results)
        self.assertIn('fold_results', results)
        
        self.assertEqual(results['n_folds'], 5)
        self.assertEqual(len(results['fold_results']), 5)
        
        # Check metrics exist
        self.assertIn('f1', results['metrics'])
        self.assertIn('accuracy', results['metrics'])
        
        # Check metric structure
        f1_metrics = results['metrics']['f1']
        self.assertIn('mean', f1_metrics)
        self.assertIn('std', f1_metrics)
        self.assertIn('values', f1_metrics)
        self.assertEqual(len(f1_metrics['values']), 5)


if __name__ == '__main__':
    unittest.main()

