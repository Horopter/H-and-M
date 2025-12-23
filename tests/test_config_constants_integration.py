"""
Integration tests for config using constants.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import Config
from lib import constants


class TestConfigConstantsIntegration(unittest.TestCase):
    """Test config uses constants correctly."""
    
    def test_config_uses_constants(self):
        """Test config defaults match constants."""
        config = Config()
        
        # Paths
        self.assertEqual(config.data_path, constants.DEFAULT_DATA_PATH)
        self.assertEqual(config.checkpoint_dir, constants.DEFAULT_CHECKPOINT_DIR)
        
        # Files
        self.assertEqual(config.train_file, constants.DEFAULT_TRAIN_FILE)
        self.assertEqual(config.val_file, constants.DEFAULT_VAL_FILE)
        self.assertEqual(config.test_file, constants.DEFAULT_TEST_FILE)
        
        # Columns
        self.assertEqual(config.text_column, constants.DEFAULT_TEXT_COLUMN)
        self.assertEqual(config.label_column, constants.DEFAULT_LABEL_COLUMN)
        self.assertEqual(config.id_column, constants.DEFAULT_ID_COLUMN)
        
        # Training
        self.assertEqual(config.cv_folds, constants.DEFAULT_CV_FOLDS)
        self.assertEqual(config.subset_size, constants.DEFAULT_SUBSET_SIZE)
        self.assertEqual(config.random_state, constants.DEFAULT_RANDOM_STATE)
        self.assertEqual(config.stratified, constants.DEFAULT_STRATIFIED)
        
        # Models
        self.assertEqual(config.models, constants.DEFAULT_MODELS)
        
        # Chunked processing
        self.assertEqual(config.chunk_size, constants.DEFAULT_CHUNK_SIZE)
        self.assertEqual(config.embedding_batch_size, constants.DEFAULT_EMBEDDING_BATCH_SIZE)
        self.assertEqual(config.gradient_accumulation_steps, constants.DEFAULT_GRADIENT_ACCUMULATION_STEPS)
        
        # Hyperparameter grids
        self.assertEqual(config.hyperparameter_grids, constants.DEFAULT_HYPERPARAMETER_GRIDS)
    
    def test_config_hyperparameter_grids(self):
        """Test hyperparameter grids are correctly loaded."""
        config = Config()
        
        self.assertIn(constants.MODEL_LOGREG, config.hyperparameter_grids)
        self.assertIn(constants.MODEL_SVM, config.hyperparameter_grids)
        self.assertIn(constants.MODEL_BAYESIAN, config.hyperparameter_grids)
        self.assertIn(constants.MODEL_XGBOOST, config.hyperparameter_grids)
        self.assertIn(constants.MODEL_NEURAL_NET, config.hyperparameter_grids)
        
        # Check logreg grid
        logreg_grid = config.hyperparameter_grids[constants.MODEL_LOGREG]
        self.assertIn('C', logreg_grid)
        self.assertIn('penalty', logreg_grid)
        self.assertIn('l1_ratio', logreg_grid)


if __name__ == '__main__':
    unittest.main()

