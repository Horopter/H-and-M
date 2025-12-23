"""
Tests for data loading functionality.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import polars as pl
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.data.loader import DataLoader
from lib.config import Config


class TestDataLoader(unittest.TestCase):
    """Test DataLoader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config({
            'data_path': self.temp_dir,
            'text_column': 'text',
            'label_column': 'label',
            'id_column': 'id'
        })
        self.loader = DataLoader(self.config)
        
        # Create test CSV files
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self):
        """Create test CSV files."""
        # Train data
        train_data = {
            'id': [1, 2, 3],
            'text': ['This is text one', 'This is text two', 'This is text three'],
            'label': [0, 1, 0]
        }
        train_df = pl.DataFrame(train_data)
        train_df.write_csv(Path(self.temp_dir) / 'train.csv')
        
        # Val data
        val_data = {
            'id': [4, 5],
            'text': ['This is text four', 'This is text five'],
            'label': [1, 0]
        }
        val_df = pl.DataFrame(val_data)
        val_df.write_csv(Path(self.temp_dir) / 'val.csv')
        
        # Test data
        test_data = {
            'id': [6, 7],
            'text': ['This is text six', 'This is text seven']
        }
        test_df = pl.DataFrame(test_data)
        test_df.write_csv(Path(self.temp_dir) / 'test.csv')
    
    def test_load_csv(self):
        """Test CSV loading."""
        df = self.loader.load_csv('train.csv')
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn('text', df.columns)
        self.assertIn('label', df.columns)
    
    def test_load_train_val_test(self):
        """Test loading all datasets."""
        train_df, val_df, test_df = self.loader.load_train_val_test()
        
        self.assertIsInstance(train_df, pl.DataFrame)
        self.assertIsInstance(val_df, pl.DataFrame)
        self.assertIsInstance(test_df, pl.DataFrame)
        
        self.assertEqual(len(train_df), 3)
        self.assertEqual(len(val_df), 2)
        self.assertEqual(len(test_df), 2)
        
        self.assertIn('label', train_df.columns)
        self.assertIn('label', val_df.columns)
        self.assertNotIn('label', test_df.columns)
    
    def test_save_load_arrow(self):
        """Test Arrow format save/load."""
        df = self.loader.load_csv('train.csv')
        
        arrow_path = Path(self.temp_dir) / 'test.arrow'
        self.loader.save_arrow(df, str(arrow_path))
        self.assertTrue(arrow_path.exists())
        
        loaded_df = self.loader.load_arrow(str(arrow_path))
        self.assertEqual(len(loaded_df), len(df))
        self.assertEqual(list(loaded_df.columns), list(df.columns))
    
    def test_validate_schema(self):
        """Test schema validation."""
        df = self.loader.load_csv('train.csv')
        
        # Should pass with correct columns
        self.assertTrue(
            self.loader.validate_schema(df, ['text', 'label'], 'train')
        )
        
        # Should fail with missing columns
        with self.assertRaises(ValueError):
            self.loader.validate_schema(df, ['text', 'label', 'missing'], 'train')
    
    def test_check_missing_values(self):
        """Test missing value detection."""
        df = self.loader.load_csv('train.csv')
        missing = self.loader.check_missing_values(df)
        self.assertIsInstance(missing, dict)


if __name__ == '__main__':
    unittest.main()

