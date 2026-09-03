"""
All embedding types: Word2Vec, FastText, Sentence Transformers, BERT, TF-IDF weighted.
"""
import numpy as np
from typing import List, Optional, Union, Dict, Any
from scipy.sparse import csr_matrix
from pathlib import Path
import polars as pl

from ..config import get_config
from ..logging.logger import get_logger
from ..utils.gc_utils import collect_after_chunk, collect_after_operation

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
    
    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
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
        
        if batch_size is None:
            batch_size = 32
        
        self.logger.debug(f"Embedding {len(texts)} texts with SentenceTransformer (batch_size={batch_size})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        result = embeddings.astype(np.float32)
        del embeddings
        
        # GC after GPU-intensive embedding
        if self.device == 'cuda':
            collect_after_operation("sentence_transformer_embed", aggressive=True)
        
        return result


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
                
                # Free GPU tensors immediately
                del outputs, hidden_states, encoded
            
            all_embeddings.append(embeddings)
            del embeddings
            
            # GC after each batch (HYPER-AGGRESSIVE for GPU)
            if self.device == 'cuda':
                collect_after_operation("bert_embed_batch", aggressive=True)
        
        result = np.vstack(all_embeddings).astype(np.float32)
        del all_embeddings
        collect_after_operation("bert_embed_complete", aggressive=True)
        return result


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
        self.config = get_config()
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
        
        # Get TF-IDF weights (assuming same order as texts)
        # Process in chunks to save memory and avoid dense conversions
        from ..constants import DEFAULT_CHUNK_SIZE
        chunk_size = getattr(self.config, "tfidf_chunk_size", max(50, self.config.chunk_size // 4))
        if chunk_size <= 0:
            chunk_size = DEFAULT_CHUNK_SIZE
        chunk_size = min(chunk_size, len(texts)) if len(texts) > 0 else chunk_size
        self.logger.info("TF-IDF weighted chunk_size=%s rows=%s", chunk_size, len(texts))
        
        n_rows = min(self.tfidf_matrix.shape[0], len(texts))
        if n_rows != len(texts):
            self.logger.warning(
                "TF-IDF matrix rows mismatch: rows=%s expected=%s; using min rows",
                self.tfidf_matrix.shape[0], len(texts)
            )
        
        total_chunks = (n_rows + chunk_size - 1) // chunk_size if chunk_size else 1
        weighted_emb = None
        for i in range(0, n_rows, chunk_size):
            chunk_num = (i // chunk_size) + 1
            if chunk_num % 25 == 0 or chunk_num == 1 or chunk_num == total_chunks:
                self.logger.info(
                    "TF-IDF weighted: Processing chunk %s/%s (%s/%s rows)",
                    chunk_num, total_chunks, min(i + chunk_size, n_rows), n_rows
                )
            chunk_len = min(chunk_size, n_rows - i)
            chunk_texts = texts[i:i+chunk_len]
            base_chunk = self.base_embeddings.embed_batch(chunk_texts)
            tfidf_chunk = self.tfidf_matrix[i:i+chunk_len]
            row_means = np.asarray(tfidf_chunk.mean(axis=1), dtype=np.float32)
            if row_means.ndim == 1:
                row_means = row_means.reshape(-1, 1)
            weighted_chunk = (base_chunk * row_means).astype(np.float32)
            if weighted_emb is None:
                weighted_emb = np.empty((n_rows, weighted_chunk.shape[1]), dtype=np.float32)
            weighted_emb[i:i+chunk_len] = weighted_chunk
            del chunk_texts, base_chunk, tfidf_chunk, row_means, weighted_chunk
            collect_after_chunk(i // chunk_size, aggressive=True)
            collect_after_operation("tfidf_weighted_chunk", aggressive=True)
        if weighted_emb is None:
            weighted_emb = np.empty((0, 0), dtype=np.float32)
        
        collect_after_operation("tfidf_weighted_complete", aggressive=True)
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
        self.logger.info(
            "Initializing embeddings: word2vec=%s sentence_transformer=%s bert=%s device=%s",
            use_word2vec, use_sentence_transformer, use_bert, self.device
        )
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
                if getattr(self.sentence_transformer, "model", None) is None:
                    self.logger.warning("SentenceTransformer model not loaded; disabling embeddings")
                    self.sentence_transformer = None
            except Exception as e:
                self.logger.warning(f"Could not initialize SentenceTransformer: {e}")
                self.sentence_transformer = None
        
        if use_bert:
            try:
                self.bert = BERTEmbeddings(
                    model_name=self.config.bert_model,
                    device=self.device,
                    aggregation=self.config.embedding_aggregation
                )
                if getattr(self.bert, "model", None) is None:
                    self.logger.warning("BERT model not loaded; disabling embeddings")
                    self.bert = None
            except Exception as e:
                self.logger.warning(f"Could not initialize BERT: {e}")
                self.bert = None

    def _validate_embedding_shapes(self, embeddings: Dict[str, np.ndarray], n_texts: int) -> Dict[str, Dict[str, Optional[int]]]:
        """Return any embedding types with row counts that don't match expected size."""
        invalid = {}
        for name, emb_array in embeddings.items():
            if emb_array is None:
                invalid[name] = {"observed": None, "expected": n_texts}
                continue
            if not hasattr(emb_array, "shape") or len(emb_array.shape) == 0:
                invalid[name] = {"observed": None, "expected": n_texts}
                continue
            if emb_array.shape[0] != n_texts:
                invalid[name] = {"observed": int(emb_array.shape[0]), "expected": n_texts}
        return invalid

    def _normalize_partial_keys(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize partial_* keys to base embedding names, preferring full keys when present."""
        normalized: Dict[str, np.ndarray] = {}
        source: Dict[str, str] = {}
        for key, arr in embeddings.items():
            base = key[len("partial_"):] if key.startswith("partial_") else key
            is_partial = key.startswith("partial_")
            if base not in normalized:
                normalized[base] = arr
                source[base] = "partial" if is_partial else "full"
            else:
                if source[base] == "partial" and not is_partial:
                    normalized[base] = arr
                    source[base] = "full"
        if set(normalized.keys()) != set(embeddings.keys()):
            self.logger.info(
                "Normalized embedding keys: before=%s after=%s",
                list(embeddings.keys()), list(normalized.keys())
            )
        return normalized

    def _log_embedding_summary(self, embeddings: Dict[str, np.ndarray], n_texts: int, context: str) -> None:
        """Log per-embedding shape details for troubleshooting."""
        if not embeddings:
            self.logger.info("Embeddings summary (%s): none", context)
            return
        for name, emb_array in embeddings.items():
            shape = getattr(emb_array, "shape", None)
            rows = shape[0] if shape and len(shape) > 0 else None
            cols = shape[1] if shape and len(shape) > 1 else None
            match = rows == n_texts if rows is not None else False
            self.logger.info(
                "Embeddings summary (%s): %s shape=%s rows=%s cols=%s expected_rows=%s match=%s",
                context, name, shape, rows, cols, n_texts, match
            )

    def _chunk_stage(self, checkpoint_stage: str, checkpoint_name: str, emb_type: str) -> str:
        return f"{checkpoint_stage}/chunks/{checkpoint_name}/{emb_type}"

    def _load_numpy_parquet(self, path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        df = pl.read_parquet(str(path))
        data_columns = [c for c in df.columns if not c.startswith("_meta_")]
        if not data_columns:
            return None
        if len(data_columns) == 1:
            return df[data_columns[0]].to_numpy()
        return df.select(data_columns).to_numpy()

    def _existing_chunk_files(self, chunk_dir: Path) -> Dict[int, Path]:
        existing: Dict[int, Path] = {}
        if not chunk_dir.exists():
            return existing
        for path in chunk_dir.glob("chunk_*.parquet"):
            parts = path.stem.split("_")
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[-1])
            except ValueError:
                continue
            existing[idx] = path
        return existing
    
    def extract_all_embeddings(self, texts: List[str], tfidf_matrix: Optional[csr_matrix] = None, train_word2vec: bool = False, chunk_size: Optional[int] = None, checkpoint_name: Optional[str] = None, checkpoint_stage: Optional[str] = None, arrow_storage: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """
        Extract all available embeddings.
        
        Args:
            texts: List of text strings
            tfidf_matrix: Optional TF-IDF matrix for weighted embeddings
            train_word2vec: If True and Word2Vec model is not trained, train it on texts
            checkpoint_name: Optional checkpoint name for saving/loading embeddings
            checkpoint_stage: Optional stage name for checkpoint directory
            arrow_storage: Optional ArrowStorage instance for checkpointing
            
        Returns:
            Dictionary of embedding type to embedding array
        """
        self.logger.info("Extracting all embeddings")
        n_texts = len(texts)
        self.logger.info(
            "Embedding inputs: n_texts=%s checkpoint=%s stage=%s",
            n_texts, checkpoint_name, checkpoint_stage
        )
        attempt = 0
        while attempt < 2:
            embeddings = {}
            use_checkpoint = attempt == 0
            self.logger.debug("Embedding extraction attempt=%s use_checkpoint=%s", attempt + 1, use_checkpoint)
            
            # Check for existing embeddings if checkpointing is enabled
            if use_checkpoint and checkpoint_name and arrow_storage:
                existing_embeddings = arrow_storage.load_embeddings(checkpoint_name, checkpoint_stage)
                if existing_embeddings:
                    has_partial = any(k.startswith("partial_") for k in existing_embeddings.keys())
                    existing_embeddings = self._normalize_partial_keys(existing_embeddings)
                    self._log_embedding_summary(existing_embeddings, n_texts, "checkpoint_load")
                    invalid = self._validate_embedding_shapes(existing_embeddings, n_texts)
                    if not invalid and not has_partial:
                        self.logger.info(f"Found existing embeddings checkpoint: {checkpoint_name}, loading...")
                        return existing_embeddings
                    if not invalid and has_partial:
                        self.logger.info(
                            "Partial embeddings found; will attempt to complete missing types. types=%s",
                            list(existing_embeddings.keys())
                        )
                        embeddings.update(existing_embeddings)
                    if invalid:
                        self.logger.warning(
                            "Found embeddings checkpoint but row counts do not match input size; "
                            f"recomputing. Mismatches: {invalid}"
                        )
        
            if chunk_size is None:
                from ..constants import DEFAULT_CHUNK_SIZE
                chunk_size = getattr(self.config, 'chunk_size', DEFAULT_CHUNK_SIZE) if hasattr(self, 'config') else DEFAULT_CHUNK_SIZE
            batch_size = getattr(self.config, 'embedding_batch_size', 32) if hasattr(self, 'config') else 32
            self.logger.info("Embedding chunk_size=%s batch_size=%s", chunk_size, batch_size)
            
            if self.word2vec and "word2vec" not in embeddings:
                # Check if Word2Vec model is trained/loaded
                if self.word2vec.model is None:
                    if train_word2vec:
                        try:
                            self.logger.info("Training Word2Vec model on provided texts...")
                            self.word2vec.train(texts)
                        except Exception as e:
                            self.logger.warning(f"Could not train Word2Vec model: {e}")
                            self.word2vec = None  # Disable Word2Vec if training fails
                    else:
                        self.logger.debug("Word2Vec model not trained and train_word2vec=False, skipping")
                        self.word2vec = None  # Disable Word2Vec if not trained
                
                # Try to extract embeddings if model is available
                if self.word2vec and self.word2vec.model is not None:
                    try:
                        if len(texts) > chunk_size:
                            chunk_embeddings = []
                            total_chunks = (len(texts) + chunk_size - 1) // chunk_size
                            self.logger.info(f"Extracting Word2Vec embeddings for {len(texts)} texts in {total_chunks} chunks...")
                            use_chunk_cache = bool(checkpoint_name and checkpoint_stage and arrow_storage)
                            chunk_stage = self._chunk_stage(checkpoint_stage, checkpoint_name, "word2vec") if use_chunk_cache else None
                            chunk_dir = (arrow_storage.storage_dir / chunk_stage) if use_chunk_cache else None
                            existing_chunks = self._existing_chunk_files(chunk_dir) if use_chunk_cache else {}
                            if use_chunk_cache:
                                self.logger.info(
                                    "Word2Vec chunk resume: found %s/%s chunks in %s",
                                    len(existing_chunks), total_chunks, chunk_dir
                                )
                            for i in range(0, len(texts), chunk_size):
                                chunk_num = (i // chunk_size) + 1
                                if chunk_num % 10 == 0 or chunk_num == 1 or chunk_num == total_chunks:
                                    self.logger.info(f"Word2Vec: Processing chunk {chunk_num}/{total_chunks} ({min(i+chunk_size, len(texts))}/{len(texts)} texts)")
                                chunk_len = min(chunk_size, len(texts) - i)
                                chunk_emb = None
                                if use_chunk_cache and chunk_num in existing_chunks:
                                    cached = self._load_numpy_parquet(existing_chunks[chunk_num])
                                    if cached is not None and cached.shape[0] == chunk_len:
                                        chunk_emb = cached
                                    else:
                                        self.logger.warning(
                                            "Word2Vec chunk cache mismatch for chunk %s; recomputing",
                                            chunk_num
                                        )
                                if chunk_emb is None:
                                    chunk_texts = texts[i:i+chunk_size]
                                    chunk_emb = self.word2vec.embed_batch(chunk_texts)
                                    del chunk_texts
                                    if use_chunk_cache and arrow_storage:
                                        arrow_storage.save_numpy_array(
                                            chunk_emb,
                                            f"chunk_{chunk_num:05d}",
                                            chunk_stage
                                        )
                                chunk_embeddings.append(chunk_emb)
                                del chunk_emb
                                collect_after_chunk(i // chunk_size, aggressive=True)
                            embeddings['word2vec'] = np.vstack(chunk_embeddings)
                            del chunk_embeddings
                            collect_after_chunk(None, aggressive=True)
                            self.logger.info("Word2Vec embedding extraction complete")
                        else:
                            self.logger.info(f"Extracting Word2Vec embeddings for {len(texts)} texts...")
                            embeddings['word2vec'] = self.word2vec.embed_batch(texts)
                            self.logger.info("Word2Vec embedding extraction complete")
                        
                        # Checkpoint Word2Vec embeddings
                        if checkpoint_name and arrow_storage:
                            arrow_storage.save_embeddings({'word2vec': embeddings['word2vec']}, f"{checkpoint_name}_partial", checkpoint_stage)
                            self.logger.info("Checkpointed Word2Vec embeddings")
                        self.logger.info(
                            "Word2Vec embeddings ready: shape=%s expected_rows=%s",
                            getattr(embeddings.get('word2vec'), "shape", None), n_texts
                        )
                    except Exception as e:
                        self.logger.warning(f"Word2Vec embedding failed: {e}")
            
            if self.sentence_transformer and "sentence_transformer" not in embeddings:
                try:
                    if len(texts) > chunk_size:
                        chunk_embeddings = []
                        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
                        self.logger.info(f"Extracting SentenceTransformer embeddings for {len(texts)} texts in {total_chunks} chunks...")
                        use_chunk_cache = bool(checkpoint_name and checkpoint_stage and arrow_storage)
                        chunk_stage = self._chunk_stage(checkpoint_stage, checkpoint_name, "sentence_transformer") if use_chunk_cache else None
                        chunk_dir = (arrow_storage.storage_dir / chunk_stage) if use_chunk_cache else None
                        existing_chunks = self._existing_chunk_files(chunk_dir) if use_chunk_cache else {}
                        if use_chunk_cache:
                            self.logger.info(
                                "SentenceTransformer chunk resume: found %s/%s chunks in %s",
                                len(existing_chunks), total_chunks, chunk_dir
                            )
                        for i in range(0, len(texts), chunk_size):
                            chunk_num = (i // chunk_size) + 1
                            if chunk_num % 10 == 0 or chunk_num == 1 or chunk_num == total_chunks:
                                self.logger.info(f"SentenceTransformer: Processing chunk {chunk_num}/{total_chunks} ({min(i+chunk_size, len(texts))}/{len(texts)} texts)")
                            chunk_len = min(chunk_size, len(texts) - i)
                            chunk_emb = None
                            if use_chunk_cache and chunk_num in existing_chunks:
                                cached = self._load_numpy_parquet(existing_chunks[chunk_num])
                                if cached is not None and cached.shape[0] == chunk_len:
                                    chunk_emb = cached
                                else:
                                    self.logger.warning(
                                        "SentenceTransformer chunk cache mismatch for chunk %s; recomputing",
                                        chunk_num
                                    )
                            if chunk_emb is None:
                                chunk_texts = texts[i:i+chunk_size]
                                chunk_emb = self.sentence_transformer.embed_batch(chunk_texts, batch_size=batch_size)
                                del chunk_texts
                                if use_chunk_cache and arrow_storage:
                                    arrow_storage.save_numpy_array(
                                        chunk_emb,
                                        f"chunk_{chunk_num:05d}",
                                        chunk_stage
                                    )
                            chunk_embeddings.append(chunk_emb)
                            del chunk_emb
                            collect_after_chunk(i // chunk_size, aggressive=True)
                        embeddings['sentence_transformer'] = np.vstack(chunk_embeddings)
                        del chunk_embeddings
                        collect_after_chunk(None, aggressive=True)
                        self.logger.info("SentenceTransformer embedding extraction complete")
                    else:
                        self.logger.info(f"Extracting SentenceTransformer embeddings for {len(texts)} texts...")
                        embeddings['sentence_transformer'] = self.sentence_transformer.embed_batch(texts, batch_size=batch_size)
                        self.logger.info("SentenceTransformer embedding extraction complete")
                    
                    # Checkpoint SentenceTransformer embeddings
                    if checkpoint_name and arrow_storage:
                        arrow_storage.save_embeddings({'sentence_transformer': embeddings['sentence_transformer']}, f"{checkpoint_name}_partial", checkpoint_stage)
                        self.logger.info("Checkpointed SentenceTransformer embeddings")
                    self.logger.info(
                        "SentenceTransformer embeddings ready: shape=%s expected_rows=%s",
                        getattr(embeddings.get('sentence_transformer'), "shape", None), n_texts
                    )
                except Exception as e:
                    self.logger.warning(f"SentenceTransformer embedding failed: {e}")
            
            if self.bert and "bert" not in embeddings:
                try:
                    if len(texts) > chunk_size:
                        chunk_embeddings = []
                        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
                        self.logger.info(f"Extracting BERT embeddings for {len(texts)} texts in {total_chunks} chunks (this may take a while)...")
                        use_chunk_cache = bool(checkpoint_name and checkpoint_stage and arrow_storage)
                        chunk_stage = self._chunk_stage(checkpoint_stage, checkpoint_name, "bert") if use_chunk_cache else None
                        chunk_dir = (arrow_storage.storage_dir / chunk_stage) if use_chunk_cache else None
                        existing_chunks = self._existing_chunk_files(chunk_dir) if use_chunk_cache else {}
                        if use_chunk_cache:
                            self.logger.info(
                                "BERT chunk resume: found %s/%s chunks in %s",
                                len(existing_chunks), total_chunks, chunk_dir
                            )
                        for i in range(0, len(texts), chunk_size):
                            chunk_num = (i // chunk_size) + 1
                            if chunk_num % 5 == 0 or chunk_num == 1 or chunk_num == total_chunks:
                                self.logger.info(f"BERT: Processing chunk {chunk_num}/{total_chunks} ({min(i+chunk_size, len(texts))}/{len(texts)} texts)")
                            chunk_len = min(chunk_size, len(texts) - i)
                            chunk_emb = None
                            if use_chunk_cache and chunk_num in existing_chunks:
                                cached = self._load_numpy_parquet(existing_chunks[chunk_num])
                                if cached is not None and cached.shape[0] == chunk_len:
                                    chunk_emb = cached
                                else:
                                    self.logger.warning(
                                        "BERT chunk cache mismatch for chunk %s; recomputing",
                                        chunk_num
                                    )
                            if chunk_emb is None:
                                chunk_texts = texts[i:i+chunk_size]
                                chunk_emb = self.bert.embed_batch(chunk_texts, batch_size=batch_size)
                                del chunk_texts
                                if use_chunk_cache and arrow_storage:
                                    arrow_storage.save_numpy_array(
                                        chunk_emb,
                                        f"chunk_{chunk_num:05d}",
                                        chunk_stage
                                    )
                            chunk_embeddings.append(chunk_emb)
                            del chunk_emb
                            collect_after_chunk(i // chunk_size, aggressive=True)
                        embeddings['bert'] = np.vstack(chunk_embeddings)
                        del chunk_embeddings
                        collect_after_chunk(None, aggressive=True)
                        self.logger.info("BERT embedding extraction complete")
                    else:
                        self.logger.info(f"Extracting BERT embeddings for {len(texts)} texts...")
                        embeddings['bert'] = self.bert.embed_batch(texts, batch_size=batch_size)
                        self.logger.info("BERT embedding extraction complete")
                    
                    # Checkpoint BERT embeddings
                    if checkpoint_name and arrow_storage:
                        arrow_storage.save_embeddings({'bert': embeddings['bert']}, f"{checkpoint_name}_partial", checkpoint_stage)
                        self.logger.info("Checkpointed BERT embeddings")
                    self.logger.info(
                        "BERT embeddings ready: shape=%s expected_rows=%s",
                        getattr(embeddings.get('bert'), "shape", None), n_texts
                    )
                except Exception as e:
                    self.logger.warning(f"BERT embedding failed: {e}")
            
            # TF-IDF weighted embeddings disabled (not beneficial vs transformer embeddings)
            if tfidf_matrix is not None:
                self.logger.info("TF-IDF weighted embeddings disabled; skipping")
            
            # Final checkpoint of all embeddings
            if embeddings:
                embeddings = self._normalize_partial_keys(embeddings)
                invalid = self._validate_embedding_shapes(embeddings, n_texts)
                if invalid:
                    if attempt == 0:
                        self.logger.warning(
                            "Embedding extraction incomplete; retrying full chunked extraction. "
                            f"Mismatches: {invalid}"
                        )
                        attempt += 1
                        continue
                    raise RuntimeError(
                        "Embedding extraction produced mismatched shapes after retry. "
                        f"Mismatches: {invalid}"
                    )
                self.logger.info(
                    "Embedding consistency check passed: expected_rows=%s types=%s",
                    n_texts, list(embeddings.keys())
                )
                self._log_embedding_summary(embeddings, n_texts, "extraction_complete")
                if checkpoint_name and arrow_storage:
                    arrow_storage.save_embeddings(embeddings, checkpoint_name, checkpoint_stage)
                    self.logger.info(f"Checkpointed all {len(embeddings)} embedding types")
            else:
                self.logger.warning("No embeddings produced in this attempt")
            
            self.logger.info(f"Extracted {len(embeddings)} embedding types")
            return embeddings
        
        return {}
    
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
