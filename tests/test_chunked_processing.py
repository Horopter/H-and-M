"""
Unit tests for chunked processing functionality.
"""
import unittest
import sys
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import Config
from lib.constants import DEFAULT_CHUNK_SIZE


class TestChunkedProcessing(unittest.TestCase):
    """Test chunked processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config({'chunk_size': DEFAULT_CHUNK_SIZE})
        self.assertEqual(self.config.chunk_size, 1000)
    
    def test_chunk_size_config(self):
        """Test chunk size is correctly set."""
        self.assertEqual(self.config.chunk_size, 1000)
        self.assertEqual(self.config.embedding_batch_size, 32)
        self.assertEqual(self.config.gradient_accumulation_steps, 4)
    
    def test_chunk_size_env_override(self):
        """Test chunk size can be overridden by env var."""
        import os
        original = os.environ.get('CHUNK_SIZE')
        try:
            os.environ['CHUNK_SIZE'] = '5000'
            config = Config()
            self.assertEqual(config.chunk_size, 5000)
        finally:
            if original:
                os.environ['CHUNK_SIZE'] = original
            elif 'CHUNK_SIZE' in os.environ:
                del os.environ['CHUNK_SIZE']
    
    def test_sparse_matrix_chunking(self):
        """Test sparse matrix can be processed in chunks."""
        # Create large sparse matrix
        n_rows = 10000
        n_cols = 1000
        X = csr_matrix(np.random.rand(n_rows, n_cols))
        
        # Process in chunks
        chunk_size = self.config.chunk_size
        chunks = []
        for i in range(0, X.shape[0], chunk_size):
            chunk = X[i:i+chunk_size].toarray()
            chunks.append(chunk)
        
        X_dense = np.vstack(chunks)
        self.assertEqual(X_dense.shape, (n_rows, n_cols))
    
    def test_chunk_size_smaller_than_data(self):
        """Test chunking when chunk size is smaller than data."""
        data_size = 5000
        chunk_size = self.config.chunk_size  # Use actual chunk size (1000)
        
        chunks = []
        for i in range(0, data_size, chunk_size):
            end = min(i + chunk_size, data_size)
            chunks.append(list(range(i, end)))
        
        # With chunk_size=1000 and data_size=5000, we get 5 chunks
        self.assertEqual(len(chunks), 5)
        self.assertEqual(len(chunks[0]), 1000)
        self.assertEqual(len(chunks[-1]), 1000)  # Last chunk


if __name__ == '__main__':
    unittest.main()

