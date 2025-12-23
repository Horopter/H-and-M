"""
Extensive NLP feature engineering: TF-IDF, n-grams, stylistic features.
"""
import re
import math
import string
import gzip
import io
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.sparse import csr_matrix, hstack

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class StylisticFeatures:
    """Extract stylistic features from text."""
    
    def __init__(self):
        """Initialize stylistic feature extractor."""
        self.stopwords = set([
            "the", "and", "is", "in", "to", "of", "that", "it", "for", "on",
            "you", "with", "as", "this", "are", "be", "or", "by", "an", "from",
            "at", "was", "have", "not", "but", "we", "they", "which", "one",
            "all", "can", "has", "there", "their", "more", "will", "if", "about",
            "so", "what", "a", "an", "the"
        ])
        self.punct_set = set(string.punctuation)
        self.logger = get_logger(self.__class__.__name__)
    
    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        """Safe division."""
        return float(a) / float(b) if b else 0.0
    
    def _char_entropy(self, s: str) -> float:
        """Calculate character entropy."""
        if not s:
            return 0.0
        counts = {}
        for ch in s:
            counts[ch] = counts.get(ch, 0) + 1
        n = len(s)
        ent = 0.0
        for c in counts.values():
            p = c / n
            ent -= p * math.log(p + 1e-12, 2)
        return ent
    
    def _gzip_ratio(self, s: str) -> float:
        """Calculate gzip compression ratio."""
        if not s:
            return 0.0
        raw = s.encode("utf-8", "ignore")
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode="w") as f:
            f.write(raw)
        compressed_size = len(out.getvalue())
        return compressed_size / max(1, len(raw))
    
    def extract(self, text: str) -> List[float]:
        """
        Extract stylistic features from text.
        
        Args:
            text: Input text string
            
        Returns:
            List of feature values
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        text = text.strip()
        n_chars = len(text)
        words = re.findall(r"\b\w+\b", text.lower())
        n_words = len(words)
        unique = set(words)
        n_unique = len(unique)
        
        # Basic counts
        avg_word_len = np.mean([len(w) for w in words]) if n_words > 0 else 0.0
        long_word_ratio = self._safe_div(sum(1 for w in words if len(w) >= 7), n_words)
        ttr = self._safe_div(n_unique, n_words)  # Type-token ratio
        
        # Hapax legomena
        hapax_ratio = 0.0
        if n_words > 0:
            freq = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            hapax_ratio = self._safe_div(sum(1 for c in freq.values() if c == 1), n_words)
        
        # Character counts
        punct_cnt = sum(1 for ch in text if ch in self.punct_set)
        digit_cnt = sum(1 for ch in text if ch.isdigit())
        upper_cnt = sum(1 for ch in text if ch.isupper())
        space_cnt = sum(1 for ch in text if ch.isspace())
        stop_cnt = sum(1 for w in words if w in self.stopwords)
        
        # Ratios
        punct_ratio = self._safe_div(punct_cnt, n_chars)
        digit_ratio = self._safe_div(digit_cnt, n_chars)
        upper_ratio = self._safe_div(upper_cnt, n_chars)
        space_ratio = self._safe_div(space_cnt, n_chars)
        stop_ratio = self._safe_div(stop_cnt, n_words)
        
        # Entropy
        char_entropy = self._char_entropy(text)
        
        # Sentence statistics
        sentences = [seg.strip() for seg in re.split(r"[.!?]+", text) if seg.strip()]
        sent_lens = [len(seg.split()) for seg in sentences] if sentences else []
        mean_sent_len = np.mean(sent_lens) if sent_lens else 0.0
        var_sent_len = np.var(sent_lens) if sent_lens else 0.0
        short_sent_ratio = self._safe_div(
            sum(1 for l in sent_lens if l <= 7),
            len(sent_lens)
        ) if sent_lens else 0.0
        
        # Repetition features
        def rep_ratio(seq, n):
            if len(seq) < n:
                return 0.0, 0.0
            grams = [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
            total = len(grams)
            uniq = len(set(grams))
            repeats = total - uniq
            return self._safe_div(repeats, total), self._safe_div(uniq, total)
        
        rep1, uniq1 = rep_ratio(words, 1)
        rep2, uniq2 = rep_ratio(words, 2)
        rep3, uniq3 = rep_ratio(words, 3)
        
        # Run lengths
        max_tok_run = 1
        cur = 1
        for i in range(1, n_words):
            if words[i] == words[i-1]:
                cur += 1
            else:
                max_tok_run = max(max_tok_run, cur)
                cur = 1
        if n_words > 0:
            max_tok_run = max(max_tok_run, cur)
        
        max_punct_run = 0
        cur = 0
        for ch in text:
            if ch in self.punct_set:
                cur += 1
            else:
                max_punct_run = max(max_punct_run, cur)
                cur = 0
        max_punct_run = max(max_punct_run, cur)
        
        # Capitalization
        cap_all_ratio = self._safe_div(
            sum(1 for w in re.findall(r"\b\w+\b", text) if w.isupper()),
            n_words
        )
        
        # Compression ratio
        gzip_ratio = self._gzip_ratio(text)
        
        features = [
            n_chars,
            n_words,
            avg_word_len,
            long_word_ratio,
            ttr,
            hapax_ratio,
            punct_ratio,
            digit_ratio,
            upper_ratio,
            space_ratio,
            stop_ratio,
            char_entropy,
            rep1, rep2, rep3,
            uniq1, uniq2, uniq3,
            max_tok_run,
            max_punct_run,
            mean_sent_len,
            var_sent_len,
            short_sent_ratio,
            cap_all_ratio,
            gzip_ratio
        ]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return [
            "n_chars", "n_words", "avg_word_len", "long_word_ratio",
            "type_token_ratio", "hapax_ratio", "punct_ratio", "digit_ratio",
            "upper_ratio", "space_ratio", "stop_ratio", "char_entropy",
            "rep1_ratio", "rep2_ratio", "rep3_ratio",
            "uniq1_ratio", "uniq2_ratio", "uniq3_ratio",
            "max_token_run", "max_punct_run", "mean_sent_len",
            "var_sent_len", "short_sent_ratio", "cap_all_ratio", "gzip_ratio"
        ]


class TFIDFFeatures:
    """TF-IDF feature extraction."""
    
    def __init__(
        self,
        ngram_range: tuple = (1, 3),
        max_features: int = 50000,
        analyzer: str = 'word',
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        """
        Initialize TF-IDF feature extractor.
        
        Args:
            ngram_range: Range of n-grams to extract
            max_features: Maximum number of features
            analyzer: 'word' or 'char'
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            sublinear_tf: Use sublinear TF scaling
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for TF-IDF features")
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            analyzer=analyzer,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            stop_words='english',
            lowercase=True,
            dtype=np.float32
        )
        self.logger = get_logger(self.__class__.__name__)
        self._fitted = False
    
    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer."""
        self.logger.debug("Fitting TF-IDF vectorizer")
        self.vectorizer.fit(texts)
        self._fitted = True
        return self
    
    def transform(self, texts: List[str]) -> csr_matrix:
        """Transform texts to TF-IDF features."""
        if not self._fitted:
            raise ValueError("TF-IDF vectorizer must be fitted before transform")
        
        self.logger.debug("Transforming texts with TF-IDF")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)


class NLPFeatureExtractor:
    """Comprehensive NLP feature extractor."""
    
    def __init__(self, config=None):
        """
        Initialize NLP feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize feature extractors
        self.stylistic = StylisticFeatures()
        
        # Word TF-IDF
        self.word_tfidf = TFIDFFeatures(
            ngram_range=(1, 2),
            max_features=40000,
            analyzer='word',
            min_df=3,
            max_df=0.90
        )
        
        # Character TF-IDF
        self.char_tfidf = TFIDFFeatures(
            ngram_range=(3, 5),
            max_features=30000,
            analyzer='char',
            min_df=3,
            max_df=1.0
        )
    
    def extract_stylistic_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract stylistic features.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of stylistic features
        """
        self.logger.debug("Extracting stylistic features")
        features = [self.stylistic.extract(text) for text in texts]
        return np.array(features, dtype=np.float32)
    
    def extract_tfidf_features(
        self,
        texts: List[str],
        fit: bool = True
    ) -> csr_matrix:
        """
        Extract TF-IDF features.
        
        Args:
            texts: List of text strings
            fit: Whether to fit the vectorizers
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        self.logger.debug("Extracting TF-IDF features")
        
        if fit:
            word_features = self.word_tfidf.fit_transform(texts)
            char_features = self.char_tfidf.fit_transform(texts)
        else:
            word_features = self.word_tfidf.transform(texts)
            char_features = self.char_tfidf.transform(texts)
        
        # Combine word and char features
        combined = hstack([word_features, char_features])
        return combined
    
    def extract_all_features(
        self,
        texts: List[str],
        fit: bool = True
    ) -> csr_matrix:
        """
        Extract all NLP features.
        
        Args:
            texts: List of text strings
            fit: Whether to fit the extractors
            
        Returns:
            Sparse matrix of all features
        """
        self.logger.info("Extracting all NLP features")
        
        # Stylistic features (dense)
        stylistic_features = self.extract_stylistic_features(texts)
        stylistic_sparse = csr_matrix(stylistic_features)
        
        # TF-IDF features (sparse)
        tfidf_features = self.extract_tfidf_features(texts, fit=fit)
        
        # Combine all features
        all_features = hstack([stylistic_sparse, tfidf_features])
        
        self.logger.info(f"Extracted {all_features.shape[1]} total features")
        return all_features
    
    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to features (assumes already fitted).
        
        Args:
            texts: List of text strings
            
        Returns:
            Sparse matrix of features
        """
        return self.extract_all_features(texts, fit=False)

