"""
Unit tests for constants module.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import constants


class TestConstants(unittest.TestCase):
    """Test constants module."""
    
    def test_path_constants(self):
        """Test path constants are defined."""
        self.assertIsNotNone(constants.DEFAULT_DATA_PATH)
        self.assertIsNotNone(constants.DEFAULT_CHECKPOINT_DIR)
        self.assertIsNotNone(constants.DEFAULT_LOG_DIR)
        self.assertIsNotNone(constants.DEFAULT_MODEL_DIR)
        self.assertIsNotNone(constants.DEFAULT_OUTPUT_DIR)
    
    def test_file_constants(self):
        """Test file name constants."""
        self.assertEqual(constants.DEFAULT_TRAIN_FILE, 'train.csv')
        self.assertEqual(constants.DEFAULT_VAL_FILE, 'val.csv')
        self.assertEqual(constants.DEFAULT_TEST_FILE, 'test.csv')
    
    def test_column_constants(self):
        """Test column name constants."""
        self.assertEqual(constants.DEFAULT_TEXT_COLUMN, 'text')
        self.assertEqual(constants.DEFAULT_LABEL_COLUMN, 'label')
        self.assertEqual(constants.DEFAULT_ID_COLUMN, 'id')
    
    def test_chunk_size(self):
        """Test chunk size is 3000."""
        self.assertEqual(constants.DEFAULT_CHUNK_SIZE, 3000)
    
    def test_model_constants(self):
        """Test model name constants."""
        self.assertIn(constants.MODEL_LOGREG, constants.DEFAULT_MODELS)
        self.assertIn(constants.MODEL_SVM, constants.DEFAULT_MODELS)
        self.assertIn(constants.MODEL_BAYESIAN, constants.DEFAULT_MODELS)
        self.assertIn(constants.MODEL_XGBOOST, constants.DEFAULT_MODELS)
        self.assertIn(constants.MODEL_NEURAL_NET, constants.DEFAULT_MODELS)
    
    def test_hyperparameter_grids(self):
        """Test hyperparameter grids are defined."""
        self.assertIn(constants.MODEL_LOGREG, constants.DEFAULT_HYPERPARAMETER_GRIDS)
        self.assertIn(constants.MODEL_SVM, constants.DEFAULT_HYPERPARAMETER_GRIDS)
        self.assertIn(constants.MODEL_BAYESIAN, constants.DEFAULT_HYPERPARAMETER_GRIDS)
        self.assertIn(constants.MODEL_XGBOOST, constants.DEFAULT_HYPERPARAMETER_GRIDS)
        self.assertIn(constants.MODEL_NEURAL_NET, constants.DEFAULT_HYPERPARAMETER_GRIDS)
    
    def test_embedding_constants(self):
        """Test embedding constants."""
        self.assertEqual(constants.DEFAULT_WORD2VEC_DIM, 300)
        self.assertIsNotNone(constants.DEFAULT_SENTENCE_TRANSFORMER_MODEL)
        self.assertIsNotNone(constants.DEFAULT_BERT_MODEL)
        self.assertIn(constants.DEFAULT_EMBEDDING_AGGREGATION, [
            constants.EMBEDDING_AGGREGATION_MEAN,
            constants.EMBEDDING_AGGREGATION_MAX,
            constants.EMBEDDING_AGGREGATION_WEIGHTED
        ])
    
    def test_env_variables(self):
        """Test environment variable names are defined."""
        self.assertIsNotNone(constants.ENV_DATA_PATH)
        self.assertIsNotNone(constants.ENV_USE_GPU)
        self.assertIsNotNone(constants.ENV_CHUNK_SIZE)


if __name__ == '__main__':
    unittest.main()

