"""
Tests for data leakage and validation.
"""
import unittest
import numpy as np
import polars as pl

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils.validation import LeakageDetector
from lib.config import Config


class TestValidation(unittest.TestCase):
    """Test validation and leakage detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.detector = LeakageDetector(self.config)
    
    def test_text_duplicate_detection(self):
        """Test text duplicate detection."""
        train_texts = ['text one', 'text two', 'text three']
        val_texts = ['text four', 'text one', 'text five']  # 'text one' is duplicate
        test_texts = ['text six', 'text two']  # 'text two' is duplicate
        
        overlaps = self.detector.check_text_duplicates(train_texts, val_texts, test_texts)
        
        self.assertIn('train_val_overlap', overlaps)
        self.assertIn('train_test_overlap', overlaps)
        self.assertEqual(len(overlaps['train_val_overlap']), 1)
        self.assertEqual(len(overlaps['train_test_overlap']), 1)
    
    def test_temporal_leakage_detection(self):
        """Test temporal leakage detection."""
        # Create indices where val comes before train (leakage)
        train_indices = [5, 6, 7, 8, 9]
        val_indices = [1, 2, 3, 4]  # Before train - leakage!
        
        time_column = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        has_leakage = self.detector.check_temporal_leakage(
            train_indices, val_indices, time_column
        )
        
        self.assertTrue(has_leakage)
        
        # No leakage case
        train_indices = [1, 2, 3, 4, 5]
        val_indices = [6, 7, 8, 9]
        
        has_leakage = self.detector.check_temporal_leakage(
            train_indices, val_indices, time_column
        )
        
        self.assertFalse(has_leakage)
    
    def test_feature_leakage_detection(self):
        """Test feature name leakage detection."""
        feature_names = [
            'text_length',
            'word_count',
            'label_mean',  # Suspicious!
            'target_encoded',  # Suspicious!
            'f1_score'  # Suspicious!
        ]
        
        suspicious = self.detector.check_feature_leakage(feature_names, 'label')
        
        self.assertGreater(len(suspicious), 0)
        self.assertIn('label_mean', suspicious)
        self.assertIn('target_encoded', suspicious)


if __name__ == '__main__':
    unittest.main()

