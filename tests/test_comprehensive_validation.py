"""
Comprehensive validation tests for production readiness.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import constants
from lib.config import Config
from lib.utils.message_loader import get_message, get_message_safe


class TestComprehensiveValidation(unittest.TestCase):
    """Comprehensive validation tests."""
    
    def test_chunk_size_is_1000(self):
        """Verify chunk size is 1000."""
        self.assertEqual(constants.DEFAULT_CHUNK_SIZE, 1000)
        config = Config()
        self.assertEqual(config.chunk_size, 1000)
    
    def test_all_constants_defined(self):
        """Test all constants are defined and not None."""
        self.assertIsNotNone(constants.DEFAULT_DATA_PATH)
        self.assertIsNotNone(constants.DEFAULT_CHUNK_SIZE)
        self.assertIsNotNone(constants.DEFAULT_EMBEDDING_BATCH_SIZE)
        self.assertIsNotNone(constants.DEFAULT_GRADIENT_ACCUMULATION_STEPS)
        self.assertIsNotNone(constants.DEFAULT_MODELS)
        self.assertIsNotNone(constants.DEFAULT_HYPERPARAMETER_GRIDS)
    
    def test_message_loader_robustness(self):
        """Test message loader handles edge cases."""
        # Test missing category
        msg1 = get_message('nonexistent', 'key')
        self.assertIsInstance(msg1, str)
        
        # Test missing key
        msg2 = get_message('stage', 'nonexistent_key')
        self.assertIsInstance(msg2, str)
        
        # Test with formatting
        msg3 = get_message_safe('data', 'loaded_rows', default='Loaded {rows} rows', rows=100)
        self.assertIn('100', msg3)
    
    def test_config_defaults_match_constants(self):
        """Test config defaults match constants."""
        config = Config()
        
        self.assertEqual(config.chunk_size, constants.DEFAULT_CHUNK_SIZE)
        self.assertEqual(config.embedding_batch_size, constants.DEFAULT_EMBEDDING_BATCH_SIZE)
        self.assertEqual(config.gradient_accumulation_steps, constants.DEFAULT_GRADIENT_ACCUMULATION_STEPS)
        self.assertEqual(config.models, constants.DEFAULT_MODELS)
        self.assertEqual(config.hyperparameter_grids, constants.DEFAULT_HYPERPARAMETER_GRIDS)
    
    def test_hyperparameter_grids_structure(self):
        """Test hyperparameter grids have correct structure."""
        grids = constants.DEFAULT_HYPERPARAMETER_GRIDS
        
        for model_type in constants.DEFAULT_MODELS:
            self.assertIn(model_type, grids)
            self.assertIsInstance(grids[model_type], dict)
            self.assertGreater(len(grids[model_type]), 0)
    
    def test_env_variable_names(self):
        """Test environment variable names are strings."""
        self.assertIsInstance(constants.ENV_CHUNK_SIZE, str)
        self.assertIsInstance(constants.ENV_EMBEDDING_BATCH_SIZE, str)
        self.assertIsInstance(constants.ENV_GRADIENT_ACCUMULATION_STEPS, str)
    
    def test_model_constants(self):
        """Test model name constants."""
        self.assertEqual(constants.MODEL_LOGREG, 'logreg')
        self.assertEqual(constants.MODEL_SVM, 'svm')
        self.assertEqual(constants.MODEL_BAYESIAN, 'bayesian')
        self.assertEqual(constants.MODEL_XGBOOST, 'xgboost')
        self.assertEqual(constants.MODEL_NEURAL_NET, 'neural_net')
        
        # All model constants should be in DEFAULT_MODELS
        self.assertIn(constants.MODEL_LOGREG, constants.DEFAULT_MODELS)
        self.assertIn(constants.MODEL_SVM, constants.DEFAULT_MODELS)
        self.assertIn(constants.MODEL_BAYESIAN, constants.DEFAULT_MODELS)
        self.assertIn(constants.MODEL_XGBOOST, constants.DEFAULT_MODELS)
        self.assertIn(constants.MODEL_NEURAL_NET, constants.DEFAULT_MODELS)


if __name__ == '__main__':
    unittest.main()

