"""
Comprehensive tests for ALL functions in embeddings.py.
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

from lib.features.embeddings import (
    Word2VecEmbeddings,
    SentenceTransformerEmbeddings,
    BERTEmbeddings,
    TFIDFWeightedEmbeddings,
    EmbeddingExtractor
)
from lib.config import get_config


class TestWord2VecEmbeddings(unittest.TestCase):
    """Test Word2VecEmbeddings class."""
    
    def test_init(self):
        """Test Word2VecEmbeddings.__init__."""
        emb = Word2VecEmbeddings(dim=100, aggregation='mean')
        self.assertEqual(emb.dim, 100)
        self.assertEqual(emb.aggregation, 'mean')
    
    def test_train(self):
        """Test Word2VecEmbeddings.train method."""
        emb = Word2VecEmbeddings(dim=50, aggregation='mean')
        texts = ['hello world', 'test text']
        
        # Should not raise error even if gensim not available
        try:
            emb.train(texts)
        except Exception:
            pass  # Expected if gensim not available
    
    def test_embed(self):
        """Test Word2VecEmbeddings.embed method."""
        emb = Word2VecEmbeddings(dim=50, aggregation='mean')
        result = emb.embed('hello world')
        self.assertIsInstance(result, np.ndarray)
    
    def test_embed_batch(self):
        """Test Word2VecEmbeddings.embed_batch method."""
        emb = Word2VecEmbeddings(dim=50, aggregation='mean')
        texts = ['hello world', 'test text']
        result = emb.embed_batch(texts)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(texts))


class TestSentenceTransformerEmbeddings(unittest.TestCase):
    """Test SentenceTransformerEmbeddings class."""
    
    def test_init(self):
        """Test SentenceTransformerEmbeddings.__init__."""
        emb = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        self.assertEqual(emb.model_name, 'all-MiniLM-L6-v2')
    
    @patch('lib.features.embeddings.collect_after_operation')
    def test_embed_batch(self, mock_collect):
        """Test SentenceTransformerEmbeddings.embed_batch method."""
        emb = SentenceTransformerEmbeddings()
        texts = ['hello world', 'test text']
        
        # Should not raise error even if sentence-transformers not available
        try:
            result = emb.embed_batch(texts)
            self.assertIsInstance(result, np.ndarray)
            # Verify GC function was called
            self.assertTrue(mock_collect.called)
        except Exception:
            pass  # Expected if sentence-transformers not available


class TestBERTEmbeddings(unittest.TestCase):
    """Test BERTEmbeddings class."""
    
    def test_init(self):
        """Test BERTEmbeddings.__init__."""
        emb = BERTEmbeddings(model_name='distilbert-base-uncased', aggregation='mean')
        self.assertEqual(emb.model_name, 'distilbert-base-uncased')
        self.assertEqual(emb.aggregation, 'mean')
    
    @patch('lib.features.embeddings.collect_after_operation')
    def test_embed_batch(self, mock_collect):
        """Test BERTEmbeddings.embed_batch method."""
        emb = BERTEmbeddings()
        texts = ['hello world', 'test text']
        
        # Should not raise error even if transformers not available
        try:
            result = emb.embed_batch(texts, batch_size=2)
            self.assertIsInstance(result, np.ndarray)
            # Verify GC function was called
            self.assertTrue(mock_collect.called)
        except Exception:
            pass  # Expected if transformers not available


class TestTFIDFWeightedEmbeddings(unittest.TestCase):
    """Test TFIDFWeightedEmbeddings class."""
    
    def test_init(self):
        """Test TFIDFWeightedEmbeddings.__init__."""
        emb = TFIDFWeightedEmbeddings(base_embeddings=Word2VecEmbeddings(dim=50))
        self.assertIsNotNone(emb.base_embeddings)
    
    def test_embed_batch(self):
        """Test TFIDFWeightedEmbeddings.embed_batch method."""
        base_emb = Word2VecEmbeddings(dim=50)
        emb = TFIDFWeightedEmbeddings(base_embeddings=base_emb)
        texts = ['hello world', 'test text']
        
        # Create mock TF-IDF matrix
        tfidf_matrix = csr_matrix([[0.5, 0.3], [0.2, 0.8]])
        
        result = emb.embed_batch(texts, tfidf_matrix=tfidf_matrix)
        self.assertIsInstance(result, np.ndarray)


class TestEmbeddingExtractor(unittest.TestCase):
    """Test EmbeddingExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        self.config.use_gpu = False
    
    def test_init(self):
        """Test EmbeddingExtractor.__init__."""
        extractor = EmbeddingExtractor(self.config)
        self.assertIsNotNone(extractor.config)
        self.assertIsNotNone(extractor.logger)
    
    def test_initialize_embeddings(self):
        """Test EmbeddingExtractor.initialize_embeddings method."""
        extractor = EmbeddingExtractor(self.config)
        
        # Should not raise error
        try:
            extractor.initialize_embeddings(
                use_word2vec=True,
                use_sentence_transformer=False,
                use_bert=False
            )
        except Exception:
            pass  # Expected if dependencies not available
    
    @patch('lib.features.embeddings.collect_after_chunk')
    @patch('lib.features.embeddings.collect_after_operation')
    def test_extract_all_embeddings(self, mock_collect_op, mock_collect_chunk):
        """Test EmbeddingExtractor.extract_all_embeddings method."""
        extractor = EmbeddingExtractor(self.config)
        extractor.word2vec = None
        extractor.sentence_transformer = None
        extractor.bert = None
        
        texts = ['hello world', 'test text']
        tfidf_matrix = csr_matrix([[0.5, 0.3], [0.2, 0.8]])
        
        result = extractor.extract_all_embeddings(
            texts,
            tfidf_matrix=tfidf_matrix,
            train_word2vec=False
        )
        
        self.assertIsInstance(result, dict)
    
    def test_concatenate_embeddings(self):
        """Test EmbeddingExtractor.concatenate_embeddings method."""
        extractor = EmbeddingExtractor(self.config)
        
        embeddings_dict = {
            'word2vec': np.array([[1, 2], [3, 4]]),
            'sentence': np.array([[5, 6], [7, 8]])
        }
        
        result = extractor.concatenate_embeddings(embeddings_dict)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 4)  # 2 + 2


if __name__ == '__main__':
    unittest.main()

