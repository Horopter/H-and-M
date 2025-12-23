"""
Integration tests for the full pipeline.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import polars as pl
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import Config
from lib.data.loader import DataLoader
from lib.training.trainer import Trainer
from lib.training.cv import CrossValidator


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config({
            'data_path': self.temp_dir,
            'checkpoint_dir': str(Path(self.temp_dir) / 'checkpoints'),
            'log_dir': str(Path(self.temp_dir) / 'logs'),
            'models': ['logreg'],  # Test with one model
            'use_gpu': False,  # CPU for testing
            'cv_folds': 3,  # Smaller for faster tests
            'subset_size': 0.5,  # Smaller subset
            'text_column': 'text',
            'label_column': 'label',
            'id_column': 'id'
        })
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self):
        """Create test CSV files."""
        n_train = 100
        n_val = 20
        n_test = 20
        
        # Train
        train_data = {
            'id': list(range(n_train)),
            'text': [f'This is sample text number {i}' for i in range(n_train)],
            'label': np.random.randint(0, 2, n_train).tolist()
        }
        pl.DataFrame(train_data).write_csv(Path(self.temp_dir) / 'train.csv')
        
        # Val
        val_data = {
            'id': list(range(n_train, n_train + n_val)),
            'text': [f'This is validation text number {i}' for i in range(n_val)],
            'label': np.random.randint(0, 2, n_val).tolist()
        }
        pl.DataFrame(val_data).write_csv(Path(self.temp_dir) / 'val.csv')
        
        # Test
        test_data = {
            'id': list(range(n_train + n_val, n_train + n_val + n_test)),
            'text': [f'This is test text number {i}' for i in range(n_test)]
        }
        pl.DataFrame(test_data).write_csv(Path(self.temp_dir) / 'test.csv')
    
    def test_full_pipeline(self):
        """Test full training pipeline."""
        trainer = Trainer(self.config)
        
        # Run pipeline
        results = trainer.run_full_pipeline(experiment_name="test_experiment")
        
        # Check results structure
        self.assertIn('results', results)
        self.assertIn('leakage_check', results)
        self.assertIn('features_shape', results)
        
        # Check model results
        if 'logreg' in results['results']:
            model_result = results['results']['logreg']
            if 'error' not in model_result:
                self.assertIn('metrics', model_result)
                self.assertIn('f1', model_result['metrics'])
    
    def test_cv_stratification(self):
        """Test CV maintains stratification."""
        loader = DataLoader(self.config)
        train_df, _, _ = loader.load_train_val_test()
        
        y = train_df[self.config.label_column].to_numpy()
        X = np.random.randn(len(y), 50)  # Dummy features
        
        cv = CrossValidator(self.config, n_splits=3)
        folds = cv.create_folds(X, y, subset_size=1.0, temporal=False)
        
        # Check stratification
        for train_idx, val_idx in folds:
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            train_dist = np.bincount(y_train) / len(y_train)
            val_dist = np.bincount(y_val) / len(y_val)
            
            # Should be similar (within 10% for small dataset)
            self.assertAlmostEqual(
                train_dist[0], val_dist[0], delta=0.10,
                msg="CV not properly stratified"
            )


if __name__ == '__main__':
    unittest.main()

