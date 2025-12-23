"""
Tests for all model implementations.
"""
import unittest
import numpy as np
from scipy.sparse import csr_matrix

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.models.logistic_regression import LogisticRegressionModel
from lib.models.svm import SVMModel
from lib.models.bayesian import BayesianModel
from lib.models.xgboost import XGBoostModel
from lib.config import Config


class TestModels(unittest.TestCase):
    """Test all model implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config({'use_gpu': False})  # CPU for testing
        self.n_samples = 200
        self.n_features = 50
        
        # Create synthetic data
        np.random.seed(42)
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # Split
        split_idx = int(0.8 * self.n_samples)
        self.X_train = self.X[:split_idx]
        self.y_train = self.y[:split_idx]
        self.X_val = self.X[split_idx:]
        self.y_val = self.y[split_idx:]
    
    def test_logistic_regression(self):
        """Test Logistic Regression model."""
        model = LogisticRegressionModel(self.config, C=1.0)
        
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_val)
        probabilities = model.predict_proba(self.X_val)
        
        self.assertEqual(len(predictions), len(self.y_val))
        self.assertEqual(probabilities.shape, (len(self.y_val), 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6))
    
    def test_svm(self):
        """Test SVM model."""
        model = SVMModel(self.config, C=1.0, kernel='linear')
        
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_val)
        probabilities = model.predict_proba(self.X_val)
        
        self.assertEqual(len(predictions), len(self.y_val))
        self.assertEqual(probabilities.shape, (len(self.y_val), 2))
    
    def test_bayesian(self):
        """Test Bayesian model."""
        model = BayesianModel(self.config, alpha=1.0)
        
        # Bayesian models need non-negative features
        X_train_pos = np.abs(self.X_train) + 1
        X_val_pos = np.abs(self.X_val) + 1
        
        model.fit(X_train_pos, self.y_train)
        predictions = model.predict(X_val_pos)
        probabilities = model.predict_proba(X_val_pos)
        
        self.assertEqual(len(predictions), len(self.y_val))
        self.assertEqual(probabilities.shape, (len(self.y_val), 2))
    
    def test_xgboost(self):
        """Test XGBoost model."""
        try:
            model = XGBoostModel(self.config, n_estimators=10, max_depth=3)
            
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_val)
            probabilities = model.predict_proba(self.X_val)
            
            self.assertEqual(len(predictions), len(self.y_val))
            self.assertEqual(probabilities.shape, (len(self.y_val), 2))
        except ImportError:
            self.skipTest("XGBoost not available")
    
    def test_model_save_load(self):
        """Test model save/load functionality."""
        import tempfile
        import os
        
        model = LogisticRegressionModel(self.config, C=1.0)
        model.fit(self.X_train, self.y_train)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            save_path = f.name
        
        try:
            model.save(save_path)
            self.assertTrue(os.path.exists(save_path))
            
            # Load
            loaded_model = LogisticRegressionModel.load(save_path, self.config)
            self.assertTrue(loaded_model.is_fitted())
            
            # Compare predictions
            orig_pred = model.predict(self.X_val)
            loaded_pred = loaded_model.predict(self.X_val)
            
            np.testing.assert_array_equal(orig_pred, loaded_pred)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


if __name__ == '__main__':
    unittest.main()

