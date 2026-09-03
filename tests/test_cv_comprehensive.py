"""
Comprehensive tests for ALL functions in cv.py.
"""
import unittest
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.training.cv import CrossValidator
from lib.config import get_config


class TestCrossValidator(unittest.TestCase):
    """Test CrossValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        self.config.random_state = 42
    
    def test_init(self):
        """Test CrossValidator.__init__."""
        cv = CrossValidator(self.config, n_splits=5)
        self.assertEqual(cv.n_splits, 5)
        self.assertIsNotNone(cv.config)
        self.assertIsNotNone(cv.logger)
    
    def test_create_folds(self):
        """Test CrossValidator.create_folds method."""
        cv = CrossValidator(self.config, n_splits=3)
        
        X = csr_matrix(np.random.rand(100, 50))
        y = np.array([0, 1] * 50)
        
        folds = cv.create_folds(X, y, subset_size=0.5, temporal=False)
        
        self.assertIsInstance(folds, list)
        self.assertGreater(len(folds), 0)
    
    @patch('lib.training.cv.collect_after_operation')
    def test_evaluate_fold(self, mock_collect):
        """Test CrossValidator.evaluate_fold method."""
        cv = CrossValidator(self.config)
        
        # Mock model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
        
        X_train = csr_matrix(np.random.rand(50, 20))
        X_val = csr_matrix(np.random.rand(20, 20))
        y_train = np.array([0, 1] * 25)
        y_val = np.array([0, 1] * 10)
        
        model_factory = lambda **kwargs: mock_model
        
        result = cv.evaluate_fold(
            model_factory,
            X_train, y_train,
            X_val, y_val,
            fold_id=0,
            model_params={}
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('metrics', result)
        # Verify GC function was called
        self.assertTrue(mock_collect.called)
    
    def test_evaluate_model(self):
        """Test CrossValidator.evaluate_model method."""
        cv = CrossValidator(self.config, n_splits=3)
        
        # Mock model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
        
        X = csr_matrix(np.random.rand(100, 50))
        y = np.array([0, 1] * 50)
        
        model_factory = lambda **kwargs: mock_model
        
        result = cv.evaluate_model(
            model_factory,
            X, y,
            subset_size=0.3,
            temporal=False
        )
        
        self.assertIsInstance(result, dict)
    
    def test_cross_validate(self):
        """Test CrossValidator.cross_validate method."""
        cv = CrossValidator(self.config, n_splits=3)
        
        # Mock model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
        
        X = csr_matrix(np.random.rand(100, 50))
        y = np.array([0, 1] * 50)
        
        model_factory = lambda **kwargs: mock_model
        
        result = cv.cross_validate(
            model_factory,
            X, y,
            subset_size=0.3,
            temporal=False
        )
        
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()

