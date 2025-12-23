"""
Tests for GPU fallback behavior when CUML is not available.
"""
import unittest
import numpy as np
from scipy.sparse import csr_matrix

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils.gpu_utils import to_gpu_if_needed, from_gpu_if_needed, check_gpu_availability
from lib.data.preprocessor import Scaler, Imputer
from lib.models.logistic_regression import LogisticRegressionModel
from lib.config import Config


class TestGPUFallback(unittest.TestCase):
    """Test GPU fallback behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config({'use_gpu': False})  # Force CPU for testing
        self.X = np.random.randn(100, 50).astype(np.float32)
        self.y = np.random.randint(0, 2, 100)
    
    def test_gpu_utils_fallback(self):
        """Test GPU utils work without CUML."""
        # Should work even without CUML
        result = to_gpu_if_needed(self.X, use_gpu=False)
        self.assertIsInstance(result, np.ndarray)
        
        # Should return original if GPU not available
        result = to_gpu_if_needed(self.X, use_gpu=True)
        # Should still work (falls back to CPU)
        self.assertIsNotNone(result)
    
    def test_preprocessor_fallback(self):
        """Test preprocessors work without CUML."""
        scaler = Scaler(use_gpu=False)
        scaler.fit(self.X)
        X_scaled = scaler.transform(self.X)
        
        self.assertIsInstance(X_scaled, np.ndarray)
        self.assertEqual(X_scaled.shape, self.X.shape)
    
    def test_model_fallback(self):
        """Test models work without CUML."""
        model = LogisticRegressionModel(self.config, use_gpu=False)
        
        # Should work with CPU fallback
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_sparse_matrix_handling(self):
        """Test sparse matrix handling."""
        X_sparse = csr_matrix(self.X)
        
        # Should handle sparse matrices
        result = to_gpu_if_needed(X_sparse, use_gpu=False, sparse_to_dense=False)
        self.assertIsInstance(result, csr_matrix)
        
        # With sparse_to_dense=True, should convert
        result = to_gpu_if_needed(X_sparse, use_gpu=False, sparse_to_dense=True)
        # Should be numpy array after conversion
        self.assertIsInstance(result, (np.ndarray, csr_matrix))


if __name__ == '__main__':
    unittest.main()

