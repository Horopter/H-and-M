"""
Unit tests for chunked CSV loading.
"""
import unittest
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from lib.config import Config
from lib.data.loader import DataLoader


class TestDataLoaderChunked(unittest.TestCase):
    """Test chunked CSV loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config({
            'data_path': str(self.temp_dir),
            'chunk_size': 3,  # Small chunk size for testing
            'text_column': 'text',
            'label_column': 'label',
            'id_column': 'id'
        })
        self.loader = DataLoader(self.config)
        
        # Create test CSV with more rows than chunk size
        self._create_test_data()
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self):
        """Create test CSV file."""
        data = {
            'id': list(range(1, 11)),
            'text': [f'Text {i}' for i in range(1, 11)],
            'label': [i % 2 for i in range(1, 11)]
        }
        df = pl.DataFrame(data)
        df.write_csv(self.temp_dir / 'train.csv')
    
    def test_chunked_loading(self):
        """Test chunked CSV loading."""
        df = self.loader.load_csv('train.csv', chunked=True)
        
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 10)
        self.assertIn('text', df.columns)
        self.assertIn('label', df.columns)
        self.assertIn('id', df.columns)
    
    def test_chunked_loading_disabled(self):
        """Test non-chunked loading."""
        df = self.loader.load_csv('train.csv', chunked=False)
        
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 10)
    
    def test_chunked_loading_empty_file(self):
        """Test chunked loading with empty file."""
        empty_file = self.temp_dir / 'empty.csv'
        empty_file.write_text('id,text,label\n')
        
        # Empty file might return empty DataFrame or raise error
        try:
            df = self.loader.load_csv('empty.csv', chunked=True)
            self.assertEqual(len(df), 0)
        except (ValueError, TypeError):
            # Either is acceptable
            pass
    
    def test_chunked_loading_missing_column(self):
        """Test chunked loading with missing column."""
        data = {
            'id': [1, 2, 3],
            'wrong_col': ['a', 'b', 'c']
        }
        df = pl.DataFrame(data)
        df.write_csv(self.temp_dir / 'wrong.csv')
        
        with self.assertRaises(ValueError):
            self.loader.load_csv('wrong.csv', chunked=True)


if __name__ == '__main__':
    unittest.main()

