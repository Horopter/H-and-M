"""
All embedding types: Word2Vec, FastText, Sentence Transformers, BERT, TF-IDF weighted.
"""
import numpy as np
from typing import List, Optional, Union, Dict, Any
from scipy.sparse import csr_matrix

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class Word2VecEmbeddings:
    """Word2Vec/FastText embeddings."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        dim: int = 300,
        aggregation: str = 'mean'
    ):
        """
        Initialize Word2Vec embeddings.
        
        Args:
            model_path: Path to pre-trained model (None to train)
            dim: Embedding dimension
            aggregation: How to aggregate word embeddings ('mean', 'max', 'weighted')
        """
        self.model_path = model_path
        self.dim = dim
        self.aggregation = aggregation
        self.model = None
        self.logger = get_logger(self.__class__.__name__)
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if path provided."""
        if self.model_path:
            try:
                import gensim
                self.model = gensim.models.Word2Vec.load(self.model_path)
                self.logger.info(f"Loaded Word2Vec model from {self.model_path}")
            except ImportError:
                self.logger.warning("gensim not available, cannot load pre-trained model")
            except Exception as e:
                self.logger.warning(f"Could not load model from {self.model_path}: {e}")
    
    def train(self, texts: List[str], **kwargs):
        """
        Train Word2Vec model on texts.
        
        Args:
            texts: List of text strings
            **kwargs: Additional arguments for Word2Vec
        """
        try:
            import gensim
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("gensim is required for Word2Vec embeddings")
        
        self.logger.info("Training Word2Vec model")
        
        # Tokenize texts
        tokenized = [text.lower().split() for text in texts]
        
        # Train model
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.dim,
            window=5,
            min_count=2,
            workers=4,
            **kwargs
        )
        
        self.logger.info("Word2Vec model trained")
    
    def _aggregate(self, embeddings: np.ndarray) -> np.ndarray:
        """Aggregate word embeddings."""
        if self.aggregation == 'mean':
            return np.mean(embeddings, axis=0)
        elif self.aggregation == 'max':
            return np.max(embeddings, axis=0)
        elif self.aggregation == 'weighted':
            # Simple weighted average (could use TF-IDF weights)
            weights = np.ones(len(embeddings))
            weights = weights / weights.sum()
            return np.average(embeddings, axis=0, weights=weights)
        else:
            return np.mean(embeddings, axis=0)
    
    def embed(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        words = text.lower().split()
        word_embeddings = []
        
        for word in words:
            if word in self.model.wv:
                word_embeddings.append(self.model.wv[word])
        
        if not word_embeddings:
            # Return zero vector if no words found
            return np.zeros(self.dim, dtype=np.float32)
        
        embeddings_array = np.array(word_embeddings)
        return self._aggregate(embeddings_array).astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings
        """
        self.logger.debug(f"Embedding {len(texts)} texts with Word2Vec")
        embeddings = [self.embed(text) for text in texts]
        return np.array(embeddings, dtype=np.float32)


class SentenceTransformerEmbeddings:
    """Sentence Transformer embeddings."""
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None
    ):
        """
        Initialize Sentence Transformer embeddings.
        
        Args:
            model_name: Name of the model
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.logger = get_logger(self.__class__.__name__)
        self._load_model()
    
    def _load_model(self):
        """Load Sentence Transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
        except ImportError:
            self.logger.warning("sentence-transformers not available")
        except Exception as e:
            self.logger.warning(f"Could not load SentenceTransformer: {e}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        if self.model is None:
            raise ValueError("SentenceTransformer model not loaded")
        
        self.logger.debug(f"Embedding {len(texts)} texts with SentenceTransformer")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


class BERTEmbeddings:
    """BERT-style contextual embeddings."""
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        device: Optional[str] = None,
        aggregation: str = 'mean'
    ):
        """
        Initialize BERT embeddings.
        
        Args:
            model_name: Name of the BERT model
            device: Device to use ('cuda' or 'cpu')
            aggregation: How to aggregate token embeddings ('mean', 'max', 'cls')
        """
        self.model_name = model_name
        self.device = device
        self.aggregation = aggregation
        self.model = None
        self.tokenizer = None
        self.logger = get_logger(self.__class__.__name__)
        self._load_model()
    
    def _load_model(self):
        """Load BERT model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            device = self.device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            
            self.device = device
            self.logger.info(f"Loaded BERT model: {self.model_name} on {device}")
        except ImportError:
            self.logger.warning("transformers not available")
        except Exception as e:
            self.logger.warning(f"Could not load BERT model: {e}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("BERT model not loaded")
        
        import torch
        
        self.logger.debug(f"Embedding {len(texts)} texts with BERT")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                hidden_states = outputs.last_hidden_state
                
                # Aggregate
                if self.aggregation == 'cls':
                    embeddings = hidden_states[:, 0, :].cpu().numpy()
                elif self.aggregation == 'mean':
                    # Mean pooling (excluding padding tokens)
                    attention_mask = encoded['attention_mask'].unsqueeze(-1)
                    embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
                    embeddings = embeddings.cpu().numpy()
                elif self.aggregation == 'max':
                    embeddings = hidden_states.max(1)[0].cpu().numpy()
                else:
                    embeddings = hidden_states.mean(1).cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings).astype(np.float32)


class TFIDFWeightedEmbeddings:
    """TF-IDF weighted embeddings."""
    
    def __init__(
        self,
        base_embeddings: Union[Word2VecEmbeddings, SentenceTransformerEmbeddings, BERTEmbeddings],
        tfidf_matrix: csr_matrix
    ):
        """
        Initialize TF-IDF weighted embeddings.
        
        Args:
            base_embeddings: Base embedding model
            tfidf_matrix: TF-IDF sparse matrix
        """
        self.base_embeddings = base_embeddings
        self.tfidf_matrix = tfidf_matrix
        self.logger = get_logger(self.__class__.__name__)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get TF-IDF weighted embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of weighted embeddings
        """
        self.logger.debug("Computing TF-IDF weighted embeddings")
        
        # Get base embeddings
        base_emb = self.base_embeddings.embed_batch(texts)
        
        # Get TF-IDF weights (assuming same order as texts)
        tfidf_weights = self.tfidf_matrix.toarray()
        
        # Weight embeddings by TF-IDF
        # This is a simplified version - in practice, you'd weight word embeddings
        # For sentence-level embeddings, we can use document-level TF-IDF as a weight
        weighted_emb = base_emb * tfidf_weights.mean(axis=1, keepdims=True)
        
        return weighted_emb.astype(np.float32)


class EmbeddingExtractor:
    """Comprehensive embedding extractor."""
    
    def __init__(self, config=None):
        """
        Initialize embedding extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize embedding models
        self.word2vec = None
        self.sentence_transformer = None
        self.bert = None
        
        # Determine device
        try:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu'
        except ImportError:
            self.device = 'cpu'
    
    def initialize_embeddings(self, use_word2vec: bool = True, use_sentence_transformer: bool = True, use_bert: bool = True):
        """Initialize embedding models."""
        if use_word2vec:
            try:
                self.word2vec = Word2VecEmbeddings(
                    dim=self.config.word2vec_dim,
                    aggregation=self.config.embedding_aggregation
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize Word2Vec: {e}")
        
        if use_sentence_transformer:
            try:
                self.sentence_transformer = SentenceTransformerEmbeddings(
                    model_name=self.config.sentence_transformer_model,
                    device=self.device
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize SentenceTransformer: {e}")
        
        if use_bert:
            try:
                self.bert = BERTEmbeddings(
                    model_name=self.config.bert_model,
                    device=self.device,
                    aggregation=self.config.embedding_aggregation
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize BERT: {e}")
    
    def extract_all_embeddings(self, texts: List[str], tfidf_matrix: Optional[csr_matrix] = None) -> Dict[str, np.ndarray]:
        """
        Extract all available embeddings.
        
        Args:
            texts: List of text strings
            tfidf_matrix: Optional TF-IDF matrix for weighted embeddings
            
        Returns:
            Dictionary of embedding type to embedding array
        """
        self.logger.info("Extracting all embeddings")
        embeddings = {}
        
        if self.word2vec:
            try:
                embeddings['word2vec'] = self.word2vec.embed_batch(texts)
            except Exception as e:
                self.logger.warning(f"Word2Vec embedding failed: {e}")
        
        if self.sentence_transformer:
            try:
                embeddings['sentence_transformer'] = self.sentence_transformer.embed_batch(texts)
            except Exception as e:
                self.logger.warning(f"SentenceTransformer embedding failed: {e}")
        
        if self.bert:
            try:
                embeddings['bert'] = self.bert.embed_batch(texts)
            except Exception as e:
                self.logger.warning(f"BERT embedding failed: {e}")
        
        # TF-IDF weighted embeddings
        if tfidf_matrix is not None and self.sentence_transformer:
            try:
                weighted = TFIDFWeightedEmbeddings(self.sentence_transformer, tfidf_matrix)
                embeddings['tfidf_weighted'] = weighted.embed_batch(texts)
            except Exception as e:
                self.logger.warning(f"TF-IDF weighted embedding failed: {e}")
        
        self.logger.info(f"Extracted {len(embeddings)} embedding types")
        return embeddings
    
    def concatenate_embeddings(self, embeddings_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate multiple embedding types.
        
        Args:
            embeddings_dict: Dictionary of embedding arrays
            
        Returns:
            Concatenated embedding array
        """
        if not embeddings_dict:
            raise ValueError("No embeddings to concatenate")
        
        arrays = list(embeddings_dict.values())
        concatenated = np.hstack(arrays)
        self.logger.debug(f"Concatenated embeddings: shape {concatenated.shape}")
        return concatenated

