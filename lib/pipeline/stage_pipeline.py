"""
Stage-based pipeline with caching and proper workflow.
Stages: Exploratory -> Statistical -> Feature Engineering -> RFE -> Training (30% train/val) -> Full Training (100% train/val)
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.sparse import csr_matrix, vstack
from scipy import sparse as sp
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
import json
import shutil
from datetime import datetime

from ..config import get_config
from ..logging.logger import get_logger
from ..data.loader import DataLoader
from ..features.nlp_features import NLPFeatureExtractor
from ..features.embeddings import EmbeddingExtractor
from ..features.feature_union import FeatureUnion
from ..data.preprocessing_pipeline import PreprocessingPipeline
from ..training.cv import CrossValidator
from ..training.grid_search import GridSearch
from ..checkpointing.checkpoint_manager import CheckpointManager
from ..checkpointing.state_manager import StateManager
from ..training.mlflow_tracker import MLFlowTracker
from ..utils.duckdb_reporter import DuckDBReporter
from ..utils.stats_analysis import StatisticalAnalyzer
from ..utils.exploratory import ExploratoryAnalyzer
from ..utils.rfe import RecursiveFeatureElimination
from ..utils.visualization import Visualizer
from ..utils.arrow_storage import ArrowStorage
from ..utils.submission import SubmissionGenerator
from ..utils.gc_utils import collect_after_chunk, collect_after_operation
from ..utils.metrics_utils import select_positive_proba

logger = get_logger(__name__)


class StagePipeline:
    """Stage-based pipeline with caching and proper workflow."""
    
    def __init__(self, config=None):
        """Initialize stage pipeline."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        
        # Set consistent random seed
        np.random.seed(self.config.random_state)
        import random
        random.seed(self.config.random_state)
        
        # Initialize components
        self.data_loader = DataLoader(config)
        # Note: splitter, leakage_detector, encoding_pipeline not used - using provided train/val/test files
        self.checkpoint_manager = CheckpointManager(config)
        self.state_manager = StateManager(config)
        self.cv = CrossValidator(config)
        self.grid_search = GridSearch(config)
        
        # Feature extractors (will be fitted once and cached)
        self.nlp_extractor = NLPFeatureExtractor(config)
        self.embedding_extractor = EmbeddingExtractor(config)
        self.feature_union = FeatureUnion(config)
        
        # Analysis tools
        self.exploratory_analyzer = ExploratoryAnalyzer(config)
        self.stats_analyzer = StatisticalAnalyzer(config)
        self.rfe = RecursiveFeatureElimination(config)
        self.visualizer = Visualizer(config)
        self.arrow_storage = ArrowStorage(config)
        self.submission_generator = SubmissionGenerator(config)
        
        # Cache for features (avoid recomputation)
        self.feature_cache: Dict[str, Any] = {}
        self.preprocessing_cache: Dict[str, Any] = {}
        
        # Results storage
        self.stage_results: Dict[str, Any] = {}
        
        # Cleanup existing checkpoints if DELETE_EXISTING is True
        if self.config.delete_existing:
            self._cleanup_existing_checkpoints()

    def _transform_nlp_in_chunks(self, texts: List[str], split_name: str, expected_cols: Optional[int] = None) -> csr_matrix:
        """Transform texts to NLP features with chunk-level checkpoint resume."""
        chunk_size = self.config.chunk_size
        if len(texts) <= chunk_size:
            return self.nlp_extractor.transform(texts)
        
        stage = f"stage_3/chunks/{split_name}_nlp"
        chunk_dir = self.arrow_storage.storage_dir / stage
        existing_chunks: Dict[int, Path] = {}
        if chunk_dir.exists():
            for data_path in chunk_dir.glob("chunk_*.parquet"):
                if data_path.name.endswith("_metadata.parquet"):
                    continue
                meta_path = data_path.with_name(f"{data_path.stem}_metadata.parquet")
                if not meta_path.exists():
                    continue
                parts = data_path.stem.split("_")
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[-1])
                except ValueError:
                    continue
                existing_chunks[idx] = data_path
        
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        if existing_chunks:
            self.logger.info(
                "NLP %s chunk resume: found %s/%s chunks in %s",
                split_name, len(existing_chunks), total_chunks, chunk_dir
            )
        
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk_num = (i // chunk_size) + 1
            chunk_len = min(chunk_size, len(texts) - i)
            chunk_name = f"chunk_{chunk_num:05d}"
            chunk_mat = None
            if chunk_num in existing_chunks:
                chunk_mat = self.arrow_storage.load_sparse_matrix(chunk_name, stage)
                if chunk_mat is None or chunk_mat.shape[0] != chunk_len or (
                    expected_cols is not None and chunk_mat.shape[1] != expected_cols
                ):
                    self.logger.warning(
                        "NLP %s chunk cache mismatch for chunk %s; recomputing",
                        split_name, chunk_num
                    )
                    chunk_mat = None
            if chunk_mat is None:
                chunk_texts = texts[i:i+chunk_size]
                chunk_mat = self.nlp_extractor.transform(chunk_texts)
                del chunk_texts
                self.arrow_storage.save_sparse_matrix(chunk_mat, chunk_name, stage)
            chunks.append(chunk_mat)
            collect_after_chunk(i // chunk_size, aggressive=True)
        
        combined = vstack(chunks)
        del chunks
        collect_after_operation(f"vstack_{split_name}_nlp_chunks", aggressive=True)
        return combined

    def _load_sparse_metadata(self, name: str, stage: str) -> Optional[Tuple[int, int]]:
        """Load sparse matrix metadata without loading full matrix."""
        meta_path = self.arrow_storage.storage_dir / stage / f"{name}_metadata.parquet"
        data_path = self.arrow_storage.storage_dir / stage / f"{name}.parquet"
        if not meta_path.exists():
            return None
        if not data_path.exists():
            return None
        try:
            df = pl.read_parquet(str(meta_path))
        except Exception as e:
            self.logger.warning("Could not read sparse metadata for %s: %s", meta_path, e)
            return None
        if df.is_empty() or "shape_0" not in df.columns or "shape_1" not in df.columns:
            return None
        return int(df["shape_0"][0]), int(df["shape_1"][0])

    def _embedding_checkpoint_rows_ok(self, checkpoint_name: str, expected_rows: int, stage: str = "stage_3") -> bool:
        """Check embedding row counts via parquet metadata (avoid loading full arrays)."""
        stage_dir = self.arrow_storage.storage_dir / stage
        if not stage_dir.exists():
            return False
        paths = sorted(stage_dir.glob(f"{checkpoint_name}_*.parquet"))
        if not paths:
            return False
        for path in paths:
            try:
                pf = pq.ParquetFile(path)
                rows = pf.metadata.num_rows
            except Exception as e:
                self.logger.warning("Could not read embedding metadata for %s: %s", path.name, e)
                return False
            if rows != expected_rows:
                self.logger.warning(
                    "Embedding checkpoint row mismatch for %s: rows=%s expected=%s",
                    path.name, rows, expected_rows
                )
                return False
        return True

    def _check_deferred_component_metadata(self, train_rows: int, val_rows: int, test_rows: int) -> bool:
        """Check checkpoint metadata before deferred union to avoid premature loading."""
        ok = True
        train_meta = self._load_sparse_metadata("train_nlp_checkpoint", "stage_3")
        if train_meta is None:
            self.logger.warning("Missing train NLP checkpoint metadata for deferred union")
            ok = False
        else:
            if train_meta[0] != train_rows:
                self.logger.warning(
                    "Train NLP rows mismatch in metadata: rows=%s expected=%s",
                    train_meta[0], train_rows
                )
                ok = False
        val_meta = self._load_sparse_metadata("val_nlp_checkpoint", "stage_3")
        if val_meta is None:
            self.logger.warning("Missing val NLP checkpoint metadata for deferred union")
            ok = False
        else:
            if val_meta[0] != val_rows:
                self.logger.warning(
                    "Val NLP rows mismatch in metadata: rows=%s expected=%s",
                    val_meta[0], val_rows
                )
                ok = False
            if train_meta and val_meta[1] != train_meta[1]:
                self.logger.warning(
                    "Val NLP cols mismatch in metadata: cols=%s expected=%s",
                    val_meta[1], train_meta[1]
                )
                ok = False
        test_meta = self._load_sparse_metadata("test_nlp_checkpoint", "stage_3")
        if test_rows is not None and test_meta is not None:
            if test_meta[0] != test_rows:
                self.logger.warning(
                    "Test NLP rows mismatch in metadata: rows=%s expected=%s",
                    test_meta[0], test_rows
                )
                ok = False
            if train_meta and test_meta[1] != train_meta[1]:
                self.logger.warning(
                    "Test NLP cols mismatch in metadata: cols=%s expected=%s",
                    test_meta[1], train_meta[1]
                )
                ok = False
        if self.config.use_embeddings:
            if self.arrow_storage.embeddings_exist("embeddings_train", "stage_3"):
                if not self._embedding_checkpoint_rows_ok("embeddings_train", train_rows):
                    self.logger.warning("Train embeddings checkpoint metadata invalid")
                    ok = False
            else:
                self.logger.warning("Train embeddings missing; will recompute stage 3")
                ok = False
            if self.arrow_storage.embeddings_exist("embeddings_val", "stage_3"):
                if not self._embedding_checkpoint_rows_ok("embeddings_val", val_rows):
                    self.logger.warning("Val embeddings checkpoint metadata invalid")
                    ok = False
            else:
                self.logger.warning("Val embeddings missing; will recompute stage 3")
                ok = False
            if test_rows is not None:
                if self.arrow_storage.embeddings_exist("embeddings_test", "stage_3"):
                    if not self._embedding_checkpoint_rows_ok("embeddings_test", test_rows):
                        self.logger.warning("Test embeddings checkpoint metadata invalid")
                        ok = False
                else:
                    self.logger.warning("Test embeddings missing; will recompute stage 3")
                    ok = False
        return ok

    def _align_embedding_keys(
        self,
        embeddings_train: Dict[str, np.ndarray],
        embeddings_val: Dict[str, np.ndarray],
        embeddings_test: Dict[str, np.ndarray],
        has_test: bool
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Align embedding types across splits to a common key set."""
        train_keys = set(embeddings_train.keys())
        val_keys = set(embeddings_val.keys())
        test_keys = set(embeddings_test.keys()) if has_test else train_keys
        common_keys = train_keys & val_keys & test_keys
        if common_keys != train_keys or common_keys != val_keys or common_keys != test_keys:
            self.logger.warning(
                "Embedding key mismatch across splits; aligning to common set. "
                "train_only=%s val_only=%s test_only=%s common=%s",
                sorted(list(train_keys - common_keys)),
                sorted(list(val_keys - common_keys)),
                sorted(list(test_keys - common_keys)),
                sorted(list(common_keys))
            )
            embeddings_train = {k: v for k, v in embeddings_train.items() if k in common_keys}
            embeddings_val = {k: v for k, v in embeddings_val.items() if k in common_keys}
            embeddings_test = {k: v for k, v in embeddings_test.items() if k in common_keys} if has_test else embeddings_test
        # Drop embedding types with mismatched column counts across splits
        mismatched_dims = []
        for key in sorted(common_keys):
            train_dim = embeddings_train.get(key).shape[1] if embeddings_train.get(key) is not None else None
            val_dim = embeddings_val.get(key).shape[1] if embeddings_val.get(key) is not None else None
            test_dim = embeddings_test.get(key).shape[1] if has_test and embeddings_test.get(key) is not None else train_dim
            if train_dim is None or val_dim is None:
                continue
            if train_dim != val_dim or (has_test and test_dim is not None and train_dim != test_dim):
                mismatched_dims.append((key, train_dim, val_dim, test_dim))
        if mismatched_dims:
            self.logger.warning(
                "Embedding dim mismatch across splits; dropping keys: %s",
                ", ".join([f"{k}(train={t},val={v},test={te})" for k, t, v, te in mismatched_dims])
            )
            drop_keys = {k for k, _, _, _ in mismatched_dims}
            embeddings_train = {k: v for k, v in embeddings_train.items() if k not in drop_keys}
            embeddings_val = {k: v for k, v in embeddings_val.items() if k not in drop_keys}
            embeddings_test = {k: v for k, v in embeddings_test.items() if k not in drop_keys} if has_test else embeddings_test
        return embeddings_train, embeddings_val, embeddings_test

    def _normalize_embeddings(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize partial_* embedding keys to base names."""
        if not embeddings:
            return embeddings
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

    def _load_embeddings_from_arrow(self, has_test: bool) -> Dict[str, Dict[str, np.ndarray]]:
        """Load embeddings dicts from Arrow storage."""
        embeddings_train = self.arrow_storage.load_embeddings("embeddings_train", "stage_3") or {}
        embeddings_val = self.arrow_storage.load_embeddings("embeddings_val", "stage_3") or {}
        embeddings_test = {}
        if has_test and self.arrow_storage.embeddings_exist("embeddings_test", "stage_3"):
            embeddings_test = self.arrow_storage.load_embeddings("embeddings_test", "stage_3") or {}
        embeddings_train = self._normalize_embeddings(embeddings_train)
        embeddings_val = self._normalize_embeddings(embeddings_val)
        embeddings_test = self._normalize_embeddings(embeddings_test)
        if not embeddings_train and self.config.use_embeddings:
            self.logger.warning("No train embeddings found in Arrow storage")
        return {
            "train": embeddings_train,
            "val": embeddings_val,
            "test": embeddings_test
        }

    def _load_components_from_arrow(self) -> Dict[str, Any]:
        """Load NLP/embedding components from Arrow storage for deferred union."""
        train_nlp = self.arrow_storage.load_sparse_matrix("train_nlp_checkpoint", "stage_3")
        val_nlp = self.arrow_storage.load_sparse_matrix("val_nlp_checkpoint", "stage_3")
        test_nlp = None
        test_path = self.arrow_storage.storage_dir / "stage_3" / "test_nlp_checkpoint.parquet"
        if test_path.exists():
            test_nlp = self.arrow_storage.load_sparse_matrix("test_nlp_checkpoint", "stage_3")
        if train_nlp is not None:
            self.logger.info("Loaded train NLP checkpoint for deferred union: shape=%s", train_nlp.shape)
        if val_nlp is not None:
            self.logger.info("Loaded val NLP checkpoint for deferred union: shape=%s", val_nlp.shape)
        if test_nlp is not None:
            self.logger.info("Loaded test NLP checkpoint for deferred union: shape=%s", test_nlp.shape)
        embeddings = {}
        if self.config.use_embeddings:
            embeddings = self._load_embeddings_from_arrow(has_test=test_nlp is not None)
        return {
            "nlp": {"train": train_nlp, "val": val_nlp, "test": test_nlp},
            "embeddings": embeddings,
            "encodings": {}
        }

    def _stage_5_artifacts_complete(
        self,
        train_rows: int,
        val_rows: int,
        test_rows: Optional[int]
    ) -> bool:
        """Validate stage 5 artifacts before skipping RFE."""
        train_meta = self._load_sparse_metadata("train_features_selected", "stage_5")
        val_meta = self._load_sparse_metadata("val_features_selected", "stage_5")
        if train_meta is None or val_meta is None:
            self.logger.warning("Stage 5 artifacts missing: train/val metadata not found")
            return False
        if train_meta[0] != train_rows or val_meta[0] != val_rows:
            self.logger.warning(
                "Stage 5 row mismatch: train_rows=%s expected=%s val_rows=%s expected=%s",
                train_meta[0], train_rows, val_meta[0], val_rows
            )
            return False
        if train_meta[1] != val_meta[1]:
            self.logger.warning(
                "Stage 5 column mismatch: train_cols=%s val_cols=%s",
                train_meta[1], val_meta[1]
            )
            return False
        if test_rows is not None:
            test_meta = self._load_sparse_metadata("test_features_selected", "stage_5")
            if test_meta is None:
                self.logger.warning("Stage 5 artifacts missing: test metadata not found")
                return False
            if test_meta[0] != test_rows:
                self.logger.warning(
                    "Stage 5 test row mismatch: rows=%s expected=%s",
                    test_meta[0], test_rows
                )
                return False
            if test_meta[1] != train_meta[1]:
                self.logger.warning(
                    "Stage 5 test column mismatch: cols=%s expected=%s",
                    test_meta[1], train_meta[1]
                )
                return False
        return True

    def _assemble_features_from_components(self, components: Dict[str, Any]) -> Dict[str, csr_matrix]:
        """Assemble unified features from component dicts."""
        nlp = components.get("nlp", {})
        embeddings = components.get("embeddings", {})
        encodings = components.get("encodings", {})
        has_test = nlp.get("test") is not None
        embeddings_train = embeddings.get("train") or {}
        embeddings_val = embeddings.get("val") or {}
        embeddings_test = embeddings.get("test") or {}
        if embeddings_train or embeddings_val or embeddings_test:
            embeddings_train, embeddings_val, embeddings_test = self._align_embedding_keys(
                embeddings_train, embeddings_val, embeddings_test, has_test
            )
        self.logger.info("Assembling feature matrix from components (nlp + embeddings + encodings)")
        train_features = self.feature_union.combine_features(
            nlp_features=nlp.get("train"),
            embeddings=embeddings_train,
            encodings=encodings.get("train", {}) if isinstance(encodings, dict) else encodings
        )
        val_features = self.feature_union.combine_features(
            nlp_features=nlp.get("val"),
            embeddings=embeddings_val,
            encodings=encodings.get("val", {}) if isinstance(encodings, dict) else encodings
        )
        test_features = None
        if has_test:
            test_features = self.feature_union.combine_features(
                nlp_features=nlp.get("test"),
                embeddings=embeddings_test,
                encodings=encodings.get("test", {}) if isinstance(encodings, dict) else encodings
            )
        self.logger.info(
            "Feature matrix shape meaning: (n_samples, n_features). "
            "Train rows=%s cols=%s; Val rows=%s cols=%s",
            train_features.shape[0], train_features.shape[1],
            val_features.shape[0], val_features.shape[1]
        )
        return {"train": train_features, "val": val_features, "test": test_features}
    
    def stage_1_exploratory(self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """Stage 1: Exploratory Data Analysis."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 1: EXPLORATORY DATA ANALYSIS")
        self.logger.info("=" * 80)
        
        results = self.exploratory_analyzer.analyze(train_df, val_df, test_df)
        
        # Save exploratory results
        output_dir = Path(self.config.output_dir) / "exploratory"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "eda_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.stage_results['exploratory'] = results
        
        # Save stage checkpoint
        self._save_stage_checkpoint('stage_1', results)
        
        self.logger.info("Stage 1 complete: Exploratory analysis")
        return results
    
    def stage_2_statistical_analysis(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> Dict[str, Any]:
        """Stage 2: Statistical analyses (ANOVA, Tukey HSD, F statistic, Cramer's V, etc.)."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 2: STATISTICAL ANALYSIS")
        self.logger.info("=" * 80)
        
        results = self.stats_analyzer.comprehensive_statistical_analysis(
            train_df, val_df, save_dir=str(Path(self.config.output_dir) / "statistical")
        )
        
        self.stage_results['statistical'] = results
        
        # Save stage checkpoint
        self._save_stage_checkpoint('stage_2', results)
        
        self.logger.info("Stage 2 complete: Statistical analysis")
        return results
    
    def stage_3_feature_engineering(self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """Stage 3: Feature engineering (done once, cached)."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 3: FEATURE ENGINEERING")
        self.logger.info("=" * 80)
        self.logger.info(
            "Input shapes: train=%s val=%s test=%s",
            getattr(train_df, "shape", None),
            getattr(val_df, "shape", None),
            getattr(test_df, "shape", None) if test_df is not None else None
        )
        
        # Check cache
        cache_key = f"features_{hash(str(train_df.shape) + str(val_df.shape))}"
        if cache_key in self.feature_cache:
            self.logger.info("Using cached features")
            return self.feature_cache[cache_key]
        
        # Get texts (process in chunks to save memory)
        chunk_size = self.config.chunk_size
        train_texts = train_df[self.config.text_column].to_list()
        val_texts = val_df[self.config.text_column].to_list()
        test_texts = test_df[self.config.text_column].to_list() if test_df is not None else None
        self.logger.info(
            "Text counts: train=%s val=%s test=%s chunk_size=%s",
            len(train_texts), len(val_texts), len(test_texts) if test_texts is not None else None, chunk_size
        )
        
        # NLP features (fit once on train, transform on all with chunking)
        self.logger.info("Extracting NLP features...")
        # Check for existing NLP features
        nlp_checkpoint_path = self.arrow_storage.storage_dir / "stage_3" / "train_nlp_checkpoint.parquet"
        if nlp_checkpoint_path.exists():
            self.logger.info("Found existing NLP features checkpoint, loading...")
            try:
                train_nlp = self.arrow_storage.load_sparse_matrix("train_nlp_checkpoint", "stage_3")
                if train_nlp is None:
                    raise ValueError("Train NLP checkpoint is empty")
                if train_nlp.shape[0] != len(train_texts):
                    raise ValueError(
                        f"Train NLP rows mismatch: rows={train_nlp.shape[0]} expected={len(train_texts)}"
                    )
                self.logger.info("Loaded NLP features from checkpoint")
                # IMPORTANT: Still need to fit vectorizers on training data for transform operations
                # This ensures vectorizers are ready to transform val/test texts
                # We only fit the vectorizers, not recompute features (we already have them from checkpoint)
                self.logger.info("Fitting NLP vectorizers on training data (required for transform operations)...")
                self.nlp_extractor.word_tfidf.fit(train_texts)
                self.nlp_extractor.char_tfidf.fit(train_texts)
                self.logger.info("NLP vectorizers fitted and ready for transform")
            except Exception as e:
                self.logger.warning(f"Could not load NLP checkpoint, recomputing: {e}")
                train_nlp = self.nlp_extractor.extract_all_features(train_texts, fit=True)
                self.arrow_storage.save_sparse_matrix(train_nlp, "train_nlp_checkpoint", "stage_3")
        else:
            train_nlp = self.nlp_extractor.extract_all_features(train_texts, fit=True)
            # Checkpoint NLP features
            self.arrow_storage.save_sparse_matrix(train_nlp, "train_nlp_checkpoint", "stage_3")
            self.logger.info("Checkpointed NLP features")
        
        # Checkpoint after NLP extraction
        self._save_stage_checkpoint('stage_3_nlp', {'status': 'complete', 'shape': list(train_nlp.shape)})
        self.logger.info("Train NLP shape=%s nnz=%s", train_nlp.shape, train_nlp.nnz)
        
        # Check for existing val NLP features
        val_nlp_checkpoint_path = self.arrow_storage.storage_dir / "stage_3" / "val_nlp_checkpoint.parquet"
        if val_nlp_checkpoint_path.exists():
            self.logger.info("Found existing val NLP features checkpoint, loading...")
            try:
                val_nlp = self.arrow_storage.load_sparse_matrix("val_nlp_checkpoint", "stage_3")
                if val_nlp is None:
                    raise ValueError("Val NLP checkpoint is empty")
                if val_nlp.shape[0] != len(val_texts):
                    raise ValueError(
                        f"Val NLP rows mismatch: rows={val_nlp.shape[0]} expected={len(val_texts)}"
                    )
                if val_nlp.shape[1] != train_nlp.shape[1]:
                    raise ValueError(
                        f"Val NLP cols mismatch: cols={val_nlp.shape[1]} expected={train_nlp.shape[1]}"
                    )
                self.logger.info("Loaded val NLP features from checkpoint")
            except Exception as e:
                self.logger.warning(f"Could not load val NLP checkpoint, recomputing: {e}")
                val_nlp = self._transform_nlp_in_chunks(val_texts, "val", train_nlp.shape[1])
                self.arrow_storage.save_sparse_matrix(val_nlp, "val_nlp_checkpoint", "stage_3")
                self.logger.info("Checkpointed val NLP features")
        else:
            val_nlp = self._transform_nlp_in_chunks(val_texts, "val", train_nlp.shape[1])
            self.arrow_storage.save_sparse_matrix(val_nlp, "val_nlp_checkpoint", "stage_3")
            self.logger.info("Checkpointed val NLP features")
        
        if test_texts:
            # Check for existing test NLP features
            test_nlp_checkpoint_path = self.arrow_storage.storage_dir / "stage_3" / "test_nlp_checkpoint.parquet"
            if test_nlp_checkpoint_path.exists():
                self.logger.info("Found existing test NLP features checkpoint, loading...")
                try:
                    test_nlp = self.arrow_storage.load_sparse_matrix("test_nlp_checkpoint", "stage_3")
                    if test_nlp is None:
                        raise ValueError("Test NLP checkpoint is empty")
                    if test_nlp.shape[0] != len(test_texts):
                        raise ValueError(
                            f"Test NLP rows mismatch: rows={test_nlp.shape[0]} expected={len(test_texts)}"
                        )
                    if test_nlp.shape[1] != train_nlp.shape[1]:
                        raise ValueError(
                            f"Test NLP cols mismatch: cols={test_nlp.shape[1]} expected={train_nlp.shape[1]}"
                        )
                    self.logger.info("Loaded test NLP features from checkpoint")
                except Exception as e:
                    self.logger.warning(f"Could not load test NLP checkpoint, recomputing: {e}")
                    test_nlp = self._transform_nlp_in_chunks(test_texts, "test", train_nlp.shape[1])
                    self.arrow_storage.save_sparse_matrix(test_nlp, "test_nlp_checkpoint", "stage_3")
                    self.logger.info("Checkpointed test NLP features")
            else:
                test_nlp = self._transform_nlp_in_chunks(test_texts, "test", train_nlp.shape[1])
                self.arrow_storage.save_sparse_matrix(test_nlp, "test_nlp_checkpoint", "stage_3")
                self.logger.info("Checkpointed test NLP features")
        else:
            test_nlp = None
        if val_nlp is not None:
            self.logger.info("Val NLP shape=%s nnz=%s", val_nlp.shape, val_nlp.nnz)
        if test_nlp is not None:
            self.logger.info("Test NLP shape=%s nnz=%s", test_nlp.shape, test_nlp.nnz)
        
        # Embeddings (fit once on train, transform on all)
        embeddings_train = {}
        embeddings_val = {}
        embeddings_test = {}
        embeddings_from_arrow = False
        defer_union = self.config.defer_embedding_union
        
        if self.config.use_embeddings:
            self.logger.info("Extracting embeddings...")
            embeddings_cached = False
            if defer_union:
                embeddings_cached = (
                    self._embedding_checkpoint_rows_ok("embeddings_train", len(train_texts))
                    and self._embedding_checkpoint_rows_ok("embeddings_val", len(val_texts))
                    and (not test_texts or self._embedding_checkpoint_rows_ok("embeddings_test", len(test_texts)))
                )
            if defer_union and embeddings_cached:
                embeddings_from_arrow = True
                self.logger.info("Embeddings checkpoints already present; skipping load due to deferred union")
            else:
                self.embedding_extractor.initialize_embeddings()
                # Train Word2Vec on training data if needed, then extract all embeddings
                # Check for existing embeddings and resume if available
                embeddings_train = self.embedding_extractor.extract_all_embeddings(
                    train_texts, train_nlp, train_word2vec=True,
                    checkpoint_name="embeddings_train", checkpoint_stage="stage_3",
                    arrow_storage=self.arrow_storage
                )
                self.logger.info("Train embeddings complete, checkpointing...")
                self._save_stage_checkpoint('stage_3_embeddings_train', {'status': 'complete', 'n_embeddings': len(embeddings_train)})
                self.logger.info("Train embeddings keys=%s", list(embeddings_train.keys()))
                for name, emb_array in embeddings_train.items():
                    self.logger.info("Train embedding %s shape=%s", name, getattr(emb_array, "shape", None))
                
                embeddings_val = self.embedding_extractor.extract_all_embeddings(
                    val_texts, train_word2vec=False,
                    checkpoint_name="embeddings_val", checkpoint_stage="stage_3",
                    arrow_storage=self.arrow_storage
                )
                self.logger.info("Val embeddings complete, checkpointing...")
                self._save_stage_checkpoint('stage_3_embeddings_val', {'status': 'complete', 'n_embeddings': len(embeddings_val)})
                self.logger.info("Val embeddings keys=%s", list(embeddings_val.keys()))
                for name, emb_array in embeddings_val.items():
                    self.logger.info("Val embedding %s shape=%s", name, getattr(emb_array, "shape", None))
                
                if test_texts:
                    embeddings_test = self.embedding_extractor.extract_all_embeddings(
                        test_texts, train_word2vec=False,
                        checkpoint_name="embeddings_test", checkpoint_stage="stage_3",
                        arrow_storage=self.arrow_storage
                    )
                    self.logger.info("Test embeddings complete, checkpointing...")
                    self._save_stage_checkpoint('stage_3_embeddings_test', {'status': 'complete', 'n_embeddings': len(embeddings_test)})
                    self.logger.info("Test embeddings keys=%s", list(embeddings_test.keys()))
                    for name, emb_array in embeddings_test.items():
                        self.logger.info("Test embedding %s shape=%s", name, getattr(emb_array, "shape", None))
        
        if self.config.use_embeddings and not defer_union:
            embeddings_train, embeddings_val, embeddings_test = self._align_embedding_keys(
                embeddings_train, embeddings_val, embeddings_test, test_texts is not None
            )
        if self.config.use_embeddings and not embeddings_from_arrow:
            missing = []
            if not embeddings_train:
                missing.append("train")
            if not embeddings_val:
                missing.append("val")
            if test_texts and not embeddings_test:
                missing.append("test")
            if missing:
                raise RuntimeError(
                    "Embeddings missing for splits: "
                    f"{', '.join(missing)}. "
                    "Verify dependencies are installed and clear stale checkpoints before retrying."
                )
        
        # Encodings not used for text-based features (no categorical columns)
        if defer_union:
            components = {
                "nlp": {"train": train_nlp, "val": val_nlp, "test": test_nlp},
                "embeddings": {"train": embeddings_train, "val": embeddings_val, "test": embeddings_test},
                "encodings": {},
                "load_embeddings": embeddings_from_arrow
            }
            component_shapes = {
                "nlp": {
                    "train": list(train_nlp.shape),
                    "val": list(val_nlp.shape),
                    "test": list(test_nlp.shape) if test_nlp is not None else None
                }
            }
            if embeddings_from_arrow:
                component_shapes["embeddings"] = "from_arrow"
            else:
                component_shapes["embeddings"] = {
                    "train": {k: list(v.shape) for k, v in embeddings_train.items()},
                    "val": {k: list(v.shape) for k, v in embeddings_val.items()},
                    "test": {k: list(v.shape) for k, v in embeddings_test.items()} if test_texts else None
                }
            stage_3_results = {
                "deferred_union": True,
                "component_shapes": component_shapes
            }
            self._save_stage_checkpoint('stage_3', stage_3_results)
            features = {"components": components}
            self.feature_cache[cache_key] = features
            self.logger.info("Feature union deferred to stage 4; returning components only")
            return features
        
        # Combine all features (done once)
        train_features = self.feature_union.combine_features(
            nlp_features=train_nlp,
            embeddings=embeddings_train,
            encodings={}  # Empty - no categorical encodings needed
        )
        
        val_features = self.feature_union.combine_features(
            nlp_features=val_nlp,
            embeddings=embeddings_val,
            encodings={}  # Empty - no categorical encodings needed
        )
        
        test_features = self.feature_union.combine_features(
            nlp_features=test_nlp,
            embeddings=embeddings_test,
            encodings={}  # Empty - no categorical encodings needed
        ) if test_nlp is not None else None
        self.logger.info("Combined feature shapes: train=%s val=%s test=%s",
                         train_features.shape, val_features.shape,
                         test_features.shape if test_features is not None else None)
        
        features = {
            'train': train_features,
            'val': val_features,
            'test': test_features
        }
        
        # Cache features
        self.feature_cache[cache_key] = features
        
        # Save features to Arrow format
        self.logger.info("Saving engineered features to Arrow format...")
        self.arrow_storage.save_sparse_matrix(train_features, "train_features", "stage_3")
        self.arrow_storage.save_sparse_matrix(val_features, "val_features", "stage_3")
        if test_features is not None:
            self.arrow_storage.save_sparse_matrix(test_features, "test_features", "stage_3")
        
        # Save feature statistics
        feature_stats = {
            'train_shape_0': train_features.shape[0],
            'train_shape_1': train_features.shape[1],
            'train_nnz': train_features.nnz,
            'val_shape_0': val_features.shape[0],
            'val_shape_1': val_features.shape[1],
            'val_nnz': val_features.nnz
        }
        self.arrow_storage.save_metrics(feature_stats, "feature_statistics", "stage_3")
        
        # Save stage checkpoint
        stage_3_results = {
            'feature_shapes': {
                'train': list(train_features.shape),
                'val': list(val_features.shape),
                'test': list(test_features.shape) if test_features is not None else None
            },
            'feature_statistics': feature_stats
        }
        self._save_stage_checkpoint('stage_3', stage_3_results)
        
        self.logger.info(f"Feature engineering complete. Train: {train_features.shape}, Val: {val_features.shape}")
        self.logger.info("Stage 3 complete: Feature engineering (cached)")
        
        return features
    
    def stage_4_preprocessing(self, features: Dict[str, Any]) -> Dict[str, csr_matrix]:
        """Stage 4: Preprocessing (scaling, imputation, PCA, normalization) - done once, cached."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 4: PREPROCESSING")
        self.logger.info("=" * 80)
        components = features.get("components") if isinstance(features, dict) else None
        if components is not None:
            self.logger.info("Deferred union detected; assembling features in stage 4")
            load_from_arrow = components.get("load_from_arrow") if isinstance(components, dict) else True
            if load_from_arrow:
                components = self._load_components_from_arrow()
            else:
                if components.get("load_embeddings") and self.config.use_embeddings:
                    has_test = components.get("nlp", {}).get("test") is not None
                    components["embeddings"] = self._load_embeddings_from_arrow(has_test=has_test)
            features = self._assemble_features_from_components(components)
        self.logger.info(
            "Preprocessing input shapes: train=%s val=%s test=%s",
            getattr(features.get('train'), "shape", None),
            getattr(features.get('val'), "shape", None),
            getattr(features.get('test'), "shape", None)
        )
        
        # Check cache
        cache_key = f"preprocessing_{hash(str(features['train'].shape))}"
        if cache_key in self.preprocessing_cache:
            self.logger.info("Using cached preprocessed features")
            return self.preprocessing_cache[cache_key]
        
        # Create preprocessing pipeline
        preprocessor = PreprocessingPipeline(self.config)
        
        # Fit on train, transform on all
        processed_train = preprocessor.fit_transform(features['train'])
        collect_after_operation("preprocessing_fit_transform", aggressive=True)
        processed_val = preprocessor.transform(features['val'])
        collect_after_operation("preprocessing_transform_val", aggressive=True)
        processed_test = preprocessor.transform(features['test']) if features.get('test') is not None else None
        if processed_test is not None:
            collect_after_operation("preprocessing_transform_test", aggressive=True)
        
        processed = {
            'train': processed_train,
            'val': processed_val,
            'test': processed_test
        }
        self.logger.info(
            "Preprocessing output shapes: train=%s val=%s test=%s",
            getattr(processed_train, "shape", None),
            getattr(processed_val, "shape", None),
            getattr(processed_test, "shape", None)
        )
        
        # Cache preprocessed features
        self.preprocessing_cache[cache_key] = processed
        
        # Save preprocessed features to Arrow
        self.logger.info("Saving preprocessed features to Arrow format...")
        self.arrow_storage.save_sparse_matrix(processed_train, "train_preprocessed", "stage_4")
        self.arrow_storage.save_sparse_matrix(processed_val, "val_preprocessed", "stage_4")
        if processed_test is not None:
            self.arrow_storage.save_sparse_matrix(processed_test, "test_preprocessed", "stage_4")
        
        # Save stage checkpoint
        stage_4_results = {
            'processed_shapes': {
                'train': list(processed_train.shape),
                'val': list(processed_val.shape),
                'test': list(processed_test.shape) if processed_test is not None else None
            }
        }
        self._save_stage_checkpoint('stage_4', stage_4_results)
        
        self.logger.info(f"Preprocessing complete. Train: {processed_train.shape}, Val: {processed_val.shape}")
        self.logger.info("Stage 4 complete: Preprocessing (cached)")
        
        return processed
    
    def stage_5_rfe(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray,
        X_test: Optional[csr_matrix] = None
    ) -> Dict[str, Any]:
        """Stage 5: Recursive Feature Elimination."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 5: RECURSIVE FEATURE ELIMINATION (RFE)")
        self.logger.info("=" * 80)
        
        # Use subset for RFE (30% of data)
        subset_size = 0.3
        n_subset = max(1, int(len(y_train) * subset_size))
        
        if n_subset < len(y_train):
            indices = np.arange(len(y_train))
            subset_indices = None
            try:
                from sklearn.model_selection import train_test_split
                subset_indices, _ = train_test_split(
                    indices,
                    train_size=n_subset,
                    stratify=y_train,
                    random_state=self.config.random_state
                )
            except Exception as e:
                self.logger.warning("Stratified subset selection failed (%s); using random subset", e)
                rng = np.random.default_rng(self.config.random_state)
                subset_indices = rng.choice(indices, size=n_subset, replace=False)
            # Index sparse matrix with subset indices
            X_rfe = X_train[subset_indices]
            y_rfe = y_train[subset_indices]
        else:
            X_rfe = X_train
            y_rfe = y_train
        
        # Perform RFE
        rfe_results = self.rfe.fit_transform(
            X_rfe, y_rfe, 
            n_features_to_select=None,  # Select optimal number
            step=0.1  # Remove 10% of features at each step
        )
        collect_after_operation("rfe_fit_transform", aggressive=True)
        
        # Apply feature selection to full datasets
        processed_test = X_test
        if processed_test is None:
            # Get processed test features from cache if available
            for _, cached_features in self.preprocessing_cache.items():
                if isinstance(cached_features, dict) and 'test' in cached_features:
                    processed_test = cached_features['test']
                    break
        if processed_test is None:
            # Try to load preprocessed test from Arrow storage
            test_proc_path = self.arrow_storage.storage_dir / "stage_4" / "test_preprocessed.parquet"
            if test_proc_path.exists():
                processed_test = self.arrow_storage.load_sparse_matrix("test_preprocessed", "stage_4")
                if processed_test is not None:
                    self.logger.info("Loaded test preprocessed features for RFE: shape=%s", processed_test.shape)
        
        selected_features = {
            'train': self.rfe.transform(X_train),
            'val': self.rfe.transform(X_val),
            'test': self.rfe.transform(processed_test) if processed_test is not None else None
        }
        
        self.stage_results['rfe'] = rfe_results
        
        # Generate RFE visualization
        plots_dir = Path(self.config.output_dir) / "plots" / "stage_5_rfe"
        plots_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer.plot_rfe_results(
            rfe_results,
            save_path=str(plots_dir / "rfe_results.png")
        )
        
        # Save RFE results to Arrow
        self.arrow_storage.save_metrics(rfe_results, "rfe_results", "stage_5")
        
        # Save selected features
        self.arrow_storage.save_sparse_matrix(selected_features['train'], "train_features_selected", "stage_5")
        self.arrow_storage.save_sparse_matrix(selected_features['val'], "val_features_selected", "stage_5")
        if selected_features['test'] is not None:
            self.arrow_storage.save_sparse_matrix(selected_features['test'], "test_features_selected", "stage_5")
        
        self.logger.info(f"RFE complete. Selected {rfe_results.get('n_features_selected', 'unknown')} features")
        
        # Save stage checkpoint
        self._save_stage_checkpoint('stage_5', rfe_results)
        
        self.logger.info("Stage 5 complete: RFE")
        
        return selected_features
    
    def stage_6_training_30pct(self, X_train: csr_matrix, y_train: np.ndarray, X_val: csr_matrix, y_val: np.ndarray, experiment_name: str = "default") -> Dict[str, Any]:
        """Stage 6: Training on 30% of train/val data with stratified 5-fold CV and hyperparameter tuning."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 6: TRAINING ON 30% OF TRAIN/VAL DATA (CV + HYPERPARAMETER TUNING)")
        self.logger.info("=" * 80)
        
        # Sample 30% of train and 30% of val data for CV and hyperparameter tuning
        train_sample_size = int(len(y_train) * 0.3)
        val_sample_size = int(len(y_val) * 0.3)
        
        self.logger.info(f"Sampling {train_sample_size} from {len(y_train)} train samples (30%)")
        self.logger.info(f"Sampling {val_sample_size} from {len(y_val)} val samples (30%)")
        self.logger.info("CV folds will use 10% of original data per fold (random sampling)")
        
        # Random stratified sampling of 30% from train and val
        from sklearn.model_selection import train_test_split
        
        # Sample 30% of train data (get indices first, then index sparse matrix)
        train_indices = np.arange(len(y_train))
        train_sample_indices, _ = train_test_split(
            train_indices,
            train_size=0.3,
            stratify=y_train,
            random_state=self.config.random_state
        )
        # Index sparse matrix with sample indices
        X_train_30pct = X_train[train_sample_indices]
        y_train_30pct = y_train[train_sample_indices]
        
        # Sample 30% of val data
        val_indices = np.arange(len(y_val))
        val_sample_indices, _ = train_test_split(
            val_indices,
            train_size=0.3,
            stratify=y_val,
            random_state=self.config.random_state
        )
        # Index sparse matrix with sample indices
        X_val_30pct = X_val[val_sample_indices]
        y_val_30pct = y_val[val_sample_indices]
        
        # For CV: use 10% of original data per fold
        cv_sample_per_fold = 0.1  # 10% of original data per fold
        
        all_results = {}
        
        for model_type in self.config.models:
            try:
                self.logger.info(f"Training {model_type} on 30% subset...")
                
                # Create model factory
                model_factory = self._get_model_factory(model_type)
                
                # Grid search on 30% train data, using 10% of original data per fold
                param_grid = self.config.hyperparameter_grids.get(model_type, {})
                if param_grid:
                    # Grid search will use CV internally, which will sample 10% per fold
                    grid_results = self.grid_search.search(
                        model_factory,
                        param_grid,
                        X_train_30pct,
                        y_train_30pct,
                        subset_size=cv_sample_per_fold,  # 10% of original data per fold
                        cv_folds=self.config.cv_folds,
                        scoring='f1',
                        original_data_size=self.original_data_size
                    )
                    best_params = grid_results.get('best_params', {})
                else:
                    best_params = {}
                    grid_results = {}
                
                # Cross-validation on 30% data: use 10% of original data per fold (random sampling)
                cv_results = self.cv.evaluate_model(
                    model_factory,
                    X_train_30pct,
                    y_train_30pct,
                    subset_size=cv_sample_per_fold,  # 10% of original data per fold
                    temporal=False,  # Random sampling, not temporal
                    model_params=best_params,
                    save_checkpoints=False,  # Don't save during 30% training
                    original_data_size=self.original_data_size
                )
                
                all_results[model_type] = {
                    'cv_results': cv_results,
                    'grid_search': grid_results,
                    'best_params': best_params
                }
                
                self.logger.info(f"{model_type} CV F1: {cv_results.get('metrics', {}).get('f1', {}).get('mean', 0):.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {e}", exc_info=True)
                all_results[model_type] = {'error': str(e)}
            progress_summary, best_model_type, best_f1 = self._build_stage_6_progress(all_results)
            stage_6_partial = {
                'status': 'in_progress',
                'completed_models': list(all_results.keys()),
                'model_summary': progress_summary,
                'best_model': best_model_type,
                'best_f1': best_f1 if best_model_type else None,
                'updated_at': datetime.utcnow().isoformat() + "Z"
            }
            self._save_stage_checkpoint('stage_6_partial', stage_6_partial)
        
        # Statistical analysis of CV results
        if self.stats_analyzer:
            self.logger.info("Performing statistical analysis on CV results...")
            plots_dir = Path(self.config.output_dir) / "plots" / "stage_6_training"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            stats_results = self.stats_analyzer.comprehensive_analysis(
                all_results,
                save_dir=str(plots_dir)
            )
            all_results['_statistical_analysis'] = stats_results
        
        # Generate comprehensive visualizations
        self.logger.info("Generating training visualizations...")
        plots_dir = Path(self.config.output_dir) / "plots" / "stage_6_training"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # CV metrics plot
        self.visualizer.plot_cv_metrics(
            all_results,
            save_path=str(plots_dir / "cv_metrics.png")
        )
        
        # Model comparison
        model_metrics = {}
        for model_type, result in all_results.items():
            if 'error' not in result and 'cv_results' in result:
                metrics = result['cv_results'].get('metrics', {})
                model_metrics[model_type] = {
                    k: v.get('mean', 0) for k, v in metrics.items() if isinstance(v, dict)
                }
        
        if model_metrics:
            self.visualizer.plot_model_comparison(
                model_metrics,
                save_path=str(plots_dir / "model_comparison.png")
            )
        
        # Save CV results to Arrow
        self.arrow_storage.save_cv_results(all_results, "training_30pct", "stage_6")
        
        # Save metrics
        for model_type, result in all_results.items():
            if 'error' not in result:
                self.arrow_storage.save_metrics(result, f"{model_type}_30pct", "stage_6")
        
        # Find best model
        best_model_type = None
        best_f1 = -1
        
        for model_type, result in all_results.items():
            if 'error' not in result and 'cv_results' in result:
                cv_metrics = result['cv_results'].get('metrics', {})
                f1_mean = cv_metrics.get('f1', {}).get('mean', 0)
                if f1_mean > best_f1:
                    best_f1 = f1_mean
                    best_model_type = model_type
        
        self.stage_results['training_30pct'] = all_results
        self.stage_results['best_model'] = best_model_type
        self.stage_results['best_f1'] = best_f1
        
        # Save stage checkpoint
        stage_6_checkpoint = {
            'all_results': all_results,
            'best_model': best_model_type,
            'best_f1': best_f1
        }
        self._save_stage_checkpoint('stage_6', stage_6_checkpoint)
        
        if best_model_type:
            self.logger.info(f"Stage 6 complete: Best model is {best_model_type} with F1={best_f1:.4f}")
        else:
            self.logger.warning("Stage 6 complete: No best model found (all models may have failed)")
        
        return all_results

    def _build_stage_6_progress(self, all_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str], float]:
        """Summarize stage 6 progress and best model so far."""
        summary = {}
        best_model_type = None
        best_f1 = -1.0
        for model_type, result in all_results.items():
            if 'error' in result:
                summary[model_type] = {'status': 'error', 'error': result.get('error')}
                continue
            cv_metrics = result.get('cv_results', {}).get('metrics', {})
            f1_mean = cv_metrics.get('f1', {}).get('mean')
            summary[model_type] = {'status': 'completed', 'f1_mean': f1_mean}
            if f1_mean is not None and f1_mean > best_f1:
                best_f1 = f1_mean
                best_model_type = model_type
        return summary, best_model_type, best_f1
    
    def stage_7_full_training(self, X_train: csr_matrix, y_train: np.ndarray, X_val: csr_matrix, y_val: np.ndarray, best_model_type: str, best_params: Dict[str, Any], experiment_name: str = "default") -> Dict[str, Any]:
        """Stage 7: Train best model on ALL train and ALL val data."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 7: TRAINING BEST MODEL ON ALL TRAIN/VAL DATA")
        self.logger.info("=" * 80)
        self.logger.info(f"Best model: {best_model_type} with params: {best_params}")
        self.logger.info(f"Using ALL train data: {len(y_train)} samples")
        self.logger.info(f"Using ALL val data: {len(y_val)} samples")
        
        # Create best model with best params
        model_factory = self._get_model_factory(best_model_type)
        model = model_factory(**best_params)
        
        # Train on ALL train data with 5-fold CV (for model selection and saving per fold)
        # Each fold uses 10% of original data (random sampling)
        original_size = getattr(self, 'original_data_size', len(y_train) * 5)
        cv_sample_per_fold = 0.1  # 10% of original data per fold
        
        cv_results = self.cv.evaluate_model(
            model_factory,
            X_train,
            y_train,
            subset_size=cv_sample_per_fold,  # 10% of original data per fold
            temporal=False,  # Random sampling
            model_params=best_params,
            save_checkpoints=True,  # Save models for each fold
            original_data_size=self.original_data_size
        )
        
        # Also train final model on all training data
        final_model = model_factory(**best_params)
        final_model.fit(X_train, y_train)
        collect_after_operation("model_fit", aggressive=True)
        
        # Final evaluation on validation set
        model = final_model
        y_pred = model.predict(X_val)
        y_proba = None
        supports_proba = getattr(model, "probability", True)
        if hasattr(model, 'predict_proba') and supports_proba:
            try:
                y_proba = model.predict_proba(X_val)
                y_proba = select_positive_proba(model, y_proba, logger=self.logger)
            except Exception as e:
                self.logger.warning("predict_proba failed; skipping ROC-AUC. error=%s", e)
        
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
        
        final_metrics = {
            'f1': f1_score(y_val, y_pred),
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba) if y_proba is not None else 0.0
        }
        best_threshold = None
        threshold_f1 = None
        if y_proba is not None:
            best_threshold, threshold_f1 = self._tune_threshold(y_val, y_proba)
            self.logger.info(
                "Best threshold on validation set: %.3f (F1=%.4f)",
                best_threshold,
                threshold_f1
            )
            self.stage_results['best_threshold'] = best_threshold
        
        # Generate comprehensive visualizations for final model
        self.logger.info("Generating final model visualizations...")
        plots_dir = Path(self.config.output_dir) / "plots" / "stage_7_final"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        self.visualizer.plot_confusion_matrix(
            y_val, y_pred, best_model_type,
            save_path=str(plots_dir / "confusion_matrix.png")
        )
        
        # ROC curve
        if y_proba is not None:
            self.visualizer.plot_roc_curve(
                y_val, y_proba, best_model_type,
                save_path=str(plots_dir / "roc_curve.png")
            )
            
            # Precision-Recall curve
            self.visualizer.plot_precision_recall_curve(
                y_val, y_proba, best_model_type,
                save_path=str(plots_dir / "precision_recall_curve.png")
            )
        
        # Save predictions and probabilities to Arrow
        pred_df = pl.DataFrame({
            'y_true': y_val,
            'y_pred': y_pred,
            'y_proba': y_proba if y_proba is not None else [0.0] * len(y_val)
        })
        self.arrow_storage.save_dataframe(pred_df, "final_predictions", "stage_7")
        
        # Save final metrics
        self.arrow_storage.save_metrics(final_metrics, "final_metrics", "stage_7")
        self.arrow_storage.save_metrics(cv_results, "final_cv_results", "stage_7")
        
        # Save final model
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model,
            best_model_type,
            fold_id=None,
            score=final_metrics['f1'],
            metadata={'stage': 'final', 'params': best_params, 'metrics': final_metrics}
        )
        
        results = {
            'model_type': best_model_type,
            'params': best_params,
            'cv_results': cv_results,
            'final_metrics': final_metrics,
            'checkpoint_path': str(checkpoint_path),
            'best_threshold': best_threshold,
            'threshold_f1': threshold_f1
        }
        
        self.stage_results['full_training'] = results
        
        # Save stage checkpoint
        self._save_stage_checkpoint('stage_7', results)
        
        self.logger.info(f"Stage 7 complete: Final model F1={final_metrics['f1']:.4f}")
        
        return results

    def _tune_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Tune classification threshold to maximize F1 on validation data."""
        from sklearn.metrics import f1_score
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 91)
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in thresholds:
            preds = (y_proba >= threshold).astype(int)
            score = f1_score(y_true, preds)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        return best_threshold, best_f1
    
    def _get_model_factory(self, model_type: str):
        """Get model factory function."""
        from ..models.logistic_regression import LogisticRegressionModel
        from ..models.svm import SVMModel
        from ..models.bayesian import BayesianModel
        from ..models.xgboost import XGBoostModel
        from ..models.neural_network import NeuralNetworkModel
        
        factories = {
            'logreg': lambda **kwargs: LogisticRegressionModel(self.config, **kwargs),
            'svm': lambda **kwargs: SVMModel(self.config, **kwargs),
            'bayesian': lambda **kwargs: BayesianModel(self.config, **kwargs),
            'xgboost': lambda **kwargs: XGBoostModel(self.config, **kwargs),
            'neural_net': lambda **kwargs: NeuralNetworkModel(self.config, **kwargs)
        }
        
        if model_type not in factories:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return factories[model_type]
    
    def _cleanup_existing_checkpoints(self):
        """Delete existing checkpoints and stage results if DELETE_EXISTING is True."""
        self.logger.info("DELETE_EXISTING is True - cleaning up existing checkpoints and stage results")
        
        # Clean up checkpoint directory
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if checkpoint_dir.exists():
            try:
                shutil.rmtree(checkpoint_dir)
                self.logger.info(f"Deleted checkpoint directory: {checkpoint_dir}")
            except Exception as e:
                self.logger.warning(f"Could not delete checkpoint directory {checkpoint_dir}: {e}")
        
        # Clean up Arrow storage (stage results)
        arrow_storage_dir = Path(self.config.output_dir) / "arrow_data"
        if arrow_storage_dir.exists():
            try:
                shutil.rmtree(arrow_storage_dir)
                self.logger.info(f"Deleted Arrow storage directory: {arrow_storage_dir}")
            except Exception as e:
                self.logger.warning(f"Could not delete Arrow storage directory {arrow_storage_dir}: {e}")
        
        # Clean up stage checkpoint JSON files
        checkpoint_json_dir = Path(self.config.output_dir) / "checkpoints"
        if checkpoint_json_dir.exists():
            try:
                for json_file in checkpoint_json_dir.glob("*_results.json"):
                    json_file.unlink()
                    self.logger.debug(f"Deleted checkpoint JSON: {json_file}")
            except Exception as e:
                self.logger.warning(f"Could not delete checkpoint JSON files: {e}")
        
        # Recreate directories
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        arrow_storage_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_json_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Cleanup complete - ready for fresh run")
    
    def _save_stage_checkpoint(self, stage_name: str, stage_results: Dict[str, Any]):
        """Save stage results as checkpoint."""
        try:
            # Save stage results to Arrow storage
            self.arrow_storage.save_metrics(
                stage_results,
                f"{stage_name}_checkpoint",
                stage_name
            )
            
            # Also save to JSON for human readability
            checkpoint_json_path = Path(self.config.output_dir) / "checkpoints" / f"{stage_name}_results.json"
            checkpoint_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_json_path, 'w') as f:
                json.dump(stage_results, f, indent=2, default=str)
            
            self.logger.info(f"Saved checkpoint for {stage_name}")
        except Exception as e:
            self.logger.warning(f"Could not save checkpoint for {stage_name}: {e}")
    
    def run_all_stages(self, experiment_name: str = "default", mlflow_tracker: Optional[Any] = None, duckdb_reporter: Optional[Any] = None, resume_from_stage: Optional[int] = None) -> Dict[str, Any]:
        """Run all pipeline stages in sequence.
        
        Args:
            experiment_name: Name of the experiment
            mlflow_tracker: Optional MLFlow tracker
            duckdb_reporter: Optional DuckDB reporter
            resume_from_stage: Resume from this stage number (1-7). If None, checks for existing checkpoints.
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING STAGE-BASED PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Random seed: {self.config.random_state} (consistent across all stages)")
        
        # Load train/val/test files (they already exist, no splitting needed)
        train_df, val_df, test_df = self.data_loader.load_train_val_test()
        self.logger.info("Loaded train/val/test files")
        
        # Store original sizes for CV sampling (10% of original data per fold)
        self.original_train_size = len(train_df)
        self.original_val_size = len(val_df)
        self.original_data_size = len(train_df) + len(val_df) + len(test_df)
        self.logger.info(f"Original data sizes - Train: {self.original_train_size}, Val: {self.original_val_size}, Test: {len(test_df)}, Total: {self.original_data_size}")
        
        # Determine starting stage
        start_stage = 1
        if resume_from_stage is not None:
            start_stage = resume_from_stage
            self.logger.info(f"Resuming from stage {start_stage} (user specified)")
        else:
            # Auto-detect last completed stage
            for stage_num in range(7, 0, -1):
                checkpoint_path = Path(self.config.output_dir) / "checkpoints" / f"stage_{stage_num}_results.json"
                if checkpoint_path.exists():
                    if stage_num == 5:
                        if not self._stage_5_artifacts_complete(
                            len(train_df),
                            len(val_df),
                            len(test_df) if test_df is not None else None
                        ):
                            start_stage = 5
                            self.logger.warning(
                                "Stage 5 checkpoint found but artifacts are incomplete; resuming from stage 5"
                            )
                            break
                    start_stage = stage_num + 1
                    self.logger.info(f"Found checkpoint for stage {stage_num}, resuming from stage {start_stage}")
                    break
        
        # Stage 1: Exploratory
        exploratory_results = {}
        if start_stage <= 1:
            exploratory_results = self.stage_1_exploratory(train_df, val_df, test_df)
        else:
            self.logger.info(f"Skipping stage 1 (already completed)")
            checkpoint_path = Path(self.config.output_dir) / "checkpoints" / "stage_1_results.json"
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    exploratory_results = json.load(f)
        
        # Stage 2: Statistical analysis
        statistical_results = {}
        if start_stage <= 2:
            statistical_results = self.stage_2_statistical_analysis(train_df, val_df)
            # Save statistical results to Arrow
            if statistical_results:
                self.arrow_storage.save_metrics(statistical_results, "statistical_analysis", "stage_2")
        else:
            self.logger.info(f"Skipping stage 2 (already completed)")
            checkpoint_path = Path(self.config.output_dir) / "checkpoints" / "stage_2_results.json"
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    statistical_results = json.load(f)
        
        # Stage 3: Feature engineering (cached)
        features = {}
        if start_stage <= 3:
            features = self.stage_3_feature_engineering(train_df, val_df, test_df)
        else:
            self.logger.info(f"Skipping stage 3 (already completed), loading features from checkpoint...")
            if self.config.defer_embedding_union:
                if self._check_deferred_component_metadata(len(train_df), len(val_df), len(test_df)):
                    self.logger.info("Stage 3 components look valid; deferring union to stage 4")
                    features = {"components": {"load_from_arrow": True}}
                else:
                    self.logger.warning("Checkpointed stage 3 components are inconsistent; recomputing stage 3...")
                    features = self.stage_3_feature_engineering(train_df, val_df, test_df)
            else:
                # Try to load features from checkpoint
                train_features_path = self.arrow_storage.storage_dir / "stage_3" / "train_features.parquet"
                val_features_path = self.arrow_storage.storage_dir / "stage_3" / "val_features.parquet"
                if train_features_path.exists() and val_features_path.exists():
                    train_features = self.arrow_storage.load_sparse_matrix("train_features", "stage_3")
                    val_features = self.arrow_storage.load_sparse_matrix("val_features", "stage_3")
                    test_features = None
                    test_features_path = self.arrow_storage.storage_dir / "stage_3" / "test_features.parquet"
                    if test_features_path.exists():
                        test_features = self.arrow_storage.load_sparse_matrix("test_features", "stage_3")
                    features = {
                        'train': train_features,
                        'val': val_features,
                        'test': test_features
                    }
                    self.logger.info(f"Loaded features from checkpoint - Train: {train_features.shape}, Val: {val_features.shape}")
                    self.logger.info(
                        "Feature matrix shape meaning: (n_samples, n_features). "
                        "Train rows=%s cols=%s; Val rows=%s cols=%s",
                        train_features.shape[0], train_features.shape[1],
                        val_features.shape[0], val_features.shape[1]
                    )
                    mismatch = False
                    if train_features.shape[1] != val_features.shape[1]:
                        mismatch = True
                        self.logger.warning(
                            "Train/Val feature column mismatch after checkpoint load: "
                            "train_cols=%s val_cols=%s (check embedding sets and vectorizers)",
                            train_features.shape[1], val_features.shape[1]
                        )
                    if train_features.shape[0] != len(train_df):
                        mismatch = True
                        self.logger.warning(
                            "Train feature row mismatch after checkpoint load: rows=%s expected=%s",
                            train_features.shape[0], len(train_df)
                        )
                    if val_features.shape[0] != len(val_df):
                        mismatch = True
                        self.logger.warning(
                            "Val feature row mismatch after checkpoint load: rows=%s expected=%s",
                            val_features.shape[0], len(val_df)
                        )
                    if mismatch:
                        self.logger.warning("Checkpointed stage 3 features are inconsistent; recomputing stage 3...")
                        features = self.stage_3_feature_engineering(train_df, val_df, test_df)
                else:
                    self.logger.warning("Could not load features from checkpoint, recomputing stage 3...")
                    features = self.stage_3_feature_engineering(train_df, val_df, test_df)
        
        # Stage 4: Preprocessing (cached)
        processed_features = {}
        if start_stage <= 4:
            processed_features = self.stage_4_preprocessing(features)
        else:
            self.logger.info(f"Skipping stage 4 (already completed), loading processed features from checkpoint...")
            train_proc_path = self.arrow_storage.storage_dir / "stage_4" / "train_preprocessed.parquet"
            val_proc_path = self.arrow_storage.storage_dir / "stage_4" / "val_preprocessed.parquet"
            if train_proc_path.exists() and val_proc_path.exists():
                processed_features['train'] = self.arrow_storage.load_sparse_matrix("train_preprocessed", "stage_4")
                processed_features['val'] = self.arrow_storage.load_sparse_matrix("val_preprocessed", "stage_4")
                test_proc_path = self.arrow_storage.storage_dir / "stage_4" / "test_preprocessed.parquet"
                processed_features['test'] = None
                if test_proc_path.exists():
                    processed_features['test'] = self.arrow_storage.load_sparse_matrix("test_preprocessed", "stage_4")
                self.logger.info(f"Loaded processed features from checkpoint")
                self.logger.info(
                    "Preprocessed shape meaning: (n_samples, n_features). "
                    "Train rows=%s cols=%s; Val rows=%s cols=%s",
                    processed_features['train'].shape[0], processed_features['train'].shape[1],
                    processed_features['val'].shape[0], processed_features['val'].shape[1]
                )
                mismatch = False
                if processed_features['train'].shape[1] != processed_features['val'].shape[1]:
                    mismatch = True
                    self.logger.warning(
                        "Train/Val preprocessed column mismatch after checkpoint load: "
                        "train_cols=%s val_cols=%s",
                        processed_features['train'].shape[1], processed_features['val'].shape[1]
                    )
                if processed_features['train'].shape[0] != len(train_df):
                    mismatch = True
                    self.logger.warning(
                        "Train preprocessed row mismatch after checkpoint load: rows=%s expected=%s",
                        processed_features['train'].shape[0], len(train_df)
                    )
                if processed_features['val'].shape[0] != len(val_df):
                    mismatch = True
                    self.logger.warning(
                        "Val preprocessed row mismatch after checkpoint load: rows=%s expected=%s",
                        processed_features['val'].shape[0], len(val_df)
                    )
                if mismatch:
                    self.logger.warning("Checkpointed stage 4 features are inconsistent; recomputing stage 4...")
                    processed_features = self.stage_4_preprocessing(features)
            else:
                self.logger.warning("Could not load processed features from checkpoint, recomputing stage 4...")
                processed_features = self.stage_4_preprocessing(features)

        if test_df is not None and processed_features.get('test') is None:
            test_proc_path = self.arrow_storage.storage_dir / "stage_4" / "test_preprocessed.parquet"
            if test_proc_path.exists():
                processed_features['test'] = self.arrow_storage.load_sparse_matrix("test_preprocessed", "stage_4")
                if processed_features['test'] is None:
                    self.logger.warning(
                        "Test preprocessed metadata missing; recomputing stage 4 to include test features"
                    )
                    processed_features = self.stage_4_preprocessing(features)
            else:
                self.logger.warning(
                    "Test preprocessed features missing; recomputing stage 4 to include test features"
                )
                processed_features = self.stage_4_preprocessing(features)
        
        # Get targets
        y_train = train_df[self.config.label_column].to_numpy()
        y_val = val_df[self.config.label_column].to_numpy()
        
        # Stage 5: RFE (optional, can be skipped)
        if self.config.use_rfe:
            if start_stage <= 5:
                rfe_features = self.stage_5_rfe(
                    processed_features['train'],
                    y_train,
                    processed_features['val'],
                    y_val,
                    processed_features.get('test')
                )
                # Update processed_features with RFE-selected features
                processed_features = rfe_features
            else:
                self.logger.info(f"Skipping stage 5 (already completed), loading RFE features from checkpoint...")
                base_features = processed_features.copy()
                train_rfe_path = self.arrow_storage.storage_dir / "stage_5" / "train_features_selected.parquet"
                val_rfe_path = self.arrow_storage.storage_dir / "stage_5" / "val_features_selected.parquet"
                if train_rfe_path.exists() and val_rfe_path.exists():
                    processed_features['train'] = self.arrow_storage.load_sparse_matrix("train_features_selected", "stage_5")
                    processed_features['val'] = self.arrow_storage.load_sparse_matrix("val_features_selected", "stage_5")
                    test_rfe_path = self.arrow_storage.storage_dir / "stage_5" / "test_features_selected.parquet"
                    if test_rfe_path.exists():
                        processed_features['test'] = self.arrow_storage.load_sparse_matrix("test_features_selected", "stage_5")
                    self.logger.info(f"Loaded RFE features from checkpoint")
                    self.logger.info(
                        "RFE shape meaning: (n_samples, n_features). "
                        "Train rows=%s cols=%s; Val rows=%s cols=%s",
                        processed_features['train'].shape[0], processed_features['train'].shape[1],
                        processed_features['val'].shape[0], processed_features['val'].shape[1]
                    )
                    mismatch = False
                    if processed_features['train'].shape[1] != processed_features['val'].shape[1]:
                        mismatch = True
                        self.logger.warning(
                            "Train/Val RFE column mismatch after checkpoint load: "
                            "train_cols=%s val_cols=%s",
                            processed_features['train'].shape[1], processed_features['val'].shape[1]
                        )
                    if processed_features['train'].shape[0] != len(train_df):
                        mismatch = True
                        self.logger.warning(
                            "Train RFE row mismatch after checkpoint load: rows=%s expected=%s",
                            processed_features['train'].shape[0], len(train_df)
                        )
                    if processed_features['val'].shape[0] != len(val_df):
                        mismatch = True
                        self.logger.warning(
                            "Val RFE row mismatch after checkpoint load: rows=%s expected=%s",
                            processed_features['val'].shape[0], len(val_df)
                        )
                    if test_df is not None:
                        if processed_features.get('test') is None:
                            mismatch = True
                            self.logger.warning(
                                "Test RFE features missing after checkpoint load: expected_rows=%s",
                                len(test_df)
                            )
                        elif processed_features['test'].shape[0] != len(test_df):
                            mismatch = True
                            self.logger.warning(
                                "Test RFE row mismatch after checkpoint load: rows=%s expected=%s",
                                processed_features['test'].shape[0], len(test_df)
                            )
                    if mismatch:
                        self.logger.warning("Checkpointed stage 5 features are inconsistent; recomputing stage 5...")
                        rfe_features = self.stage_5_rfe(
                            base_features['train'],
                            y_train,
                            base_features['val'],
                            y_val,
                            base_features.get('test')
                        )
                        processed_features = rfe_features
                else:
                    self.logger.warning("Could not load RFE features from checkpoint, recomputing stage 5...")
                    rfe_features = self.stage_5_rfe(
                        processed_features['train'],
                        y_train,
                        processed_features['val'],
                        y_val,
                        processed_features.get('test')
                    )
                    processed_features = rfe_features
        
        # Stage 6: Training on 30% of train/val with CV and hyperparameter tuning
        training_30pct_results = {}
        if start_stage <= 6:
            training_30pct_results = self.stage_6_training_30pct(
                processed_features['train'],
                y_train,
                processed_features['val'],
                y_val,
                experiment_name
            )
        else:
            self.logger.info(f"Skipping stage 6 (already completed), loading results from checkpoint...")
            checkpoint_path = Path(self.config.output_dir) / "checkpoints" / "stage_6_results.json"
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    stage_6_data = json.load(f)
                    training_30pct_results = stage_6_data.get('all_results', {})
                    self.stage_results['best_model'] = stage_6_data.get('best_model')
                    self.stage_results['best_f1'] = stage_6_data.get('best_f1')
                    self.logger.info(f"Loaded stage 6 results from checkpoint")
            else:
                self.logger.warning("Could not load stage 6 results, recomputing...")
                training_30pct_results = self.stage_6_training_30pct(
                    processed_features['train'],
                    y_train,
                    processed_features['val'],
                    y_val,
                    experiment_name
                )
        
        # Stage 7: Train best model on ALL train/val data
        best_model_type = self.stage_results.get('best_model')
        best_params = training_30pct_results.get(best_model_type, {}).get('best_params', {})
        
        full_training_results = {}
        if start_stage <= 7:
            if best_model_type:
                full_training_results = self.stage_7_full_training(
                    processed_features['train'],
                    y_train,
                    processed_features['val'],
                    y_val,
                    best_model_type,
                    best_params,
                    experiment_name
                )
            else:
                self.logger.error("No best model found, skipping full training")
        else:
            self.logger.info(f"Skipping stage 7 (already completed)")
            checkpoint_path = Path(self.config.output_dir) / "checkpoints" / "stage_7_results.json"
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    full_training_results = json.load(f)
                    if full_training_results.get('best_threshold') is not None:
                        self.stage_results['best_threshold'] = full_training_results.get('best_threshold')
                    self.logger.info(f"Loaded stage 7 results from checkpoint")
        
        # Generate submission file if test data is available
        submission_path = None
        if test_df is not None and processed_features.get('test') is not None and best_model_type:
            self.logger.info("Generating submission file...")
            try:
                # Get test IDs
                test_ids = test_df[self.config.id_column].to_numpy() if self.config.id_column in test_df.columns else None
                
                best_threshold = self.stage_results.get('best_threshold')
                if best_threshold is None:
                    self.logger.info("Tuning threshold on validation set for best model...")
                    threshold_model = self._get_model_factory(best_model_type)(**best_params)
                    threshold_model.fit(processed_features['train'], y_train)
                    supports_proba = getattr(threshold_model, "probability", True)
                    if hasattr(threshold_model, 'predict_proba') and supports_proba:
                        try:
                            val_proba = threshold_model.predict_proba(processed_features['val'])
                            val_proba = select_positive_proba(threshold_model, val_proba, logger=self.logger)
                            best_threshold, threshold_f1 = self._tune_threshold(y_val, val_proba)
                            self.logger.info(
                                "Best threshold on validation set: %.3f (F1=%.4f)",
                                best_threshold,
                                threshold_f1
                            )
                            self.stage_results['best_threshold'] = best_threshold
                        except Exception as e:
                            self.logger.warning("Threshold tuning failed; using default threshold. error=%s", e)
                    else:
                        self.logger.warning("Best model does not support predict_proba; using default threshold.")
                if best_threshold is not None:
                    self.config.submission_threshold = best_threshold
                
                # Train best model on full train+val before test predictions
                X_full = processed_features['train']
                y_full = y_train
                if processed_features.get('val') is not None:
                    if sp.issparse(X_full) or sp.issparse(processed_features['val']):
                        X_full = vstack([X_full, processed_features['val']])
                    else:
                        X_full = np.vstack([X_full, processed_features['val']])
                    y_full = np.concatenate([y_train, y_val])
                
                final_model = self._get_model_factory(best_model_type)(**best_params)
                final_model.fit(X_full, y_full)
                collect_after_operation("final_model_fit_full", aggressive=True)
                supports_proba = getattr(final_model, "probability", True)
                if hasattr(final_model, 'predict_proba') and supports_proba:
                    try:
                        proba = final_model.predict_proba(processed_features['test'])
                        collect_after_operation("final_model_predict_proba", aggressive=True)
                        test_predictions = select_positive_proba(final_model, proba, logger=self.logger)
                    except Exception as e:
                        self.logger.warning("predict_proba failed; using class predictions. error=%s", e)
                        test_predictions = final_model.predict(processed_features['test']).astype(float)
                else:
                    # Fallback to class predictions (0 or 1)
                    test_predictions = final_model.predict(processed_features['test']).astype(float)

                # Generate submission CSV (ONLY CSV file)
                submission_path = self.submission_generator.generate_submission(
                    test_predictions,
                    test_ids
                )
                
                # Save test predictions to Arrow (not CSV)
                test_pred_df = pl.DataFrame({
                    'id': test_ids if test_ids is not None else np.arange(len(test_predictions)),
                    'prediction': test_predictions
                })
                self.arrow_storage.save_dataframe(test_pred_df, "test_predictions", "submission")
                
            except Exception as e:
                self.logger.warning(f"Could not generate submission file: {e}")
        
        # Compile final results
        final_results = {
            'exploratory': exploratory_results,
            'statistical': statistical_results,
            'feature_engineering': {
                'train_shape': processed_features['train'].shape,
                'val_shape': processed_features['val'].shape
            },
            'training_30pct': training_30pct_results,
            'full_training': full_training_results,
            'best_model': best_model_type,
            'best_f1': self.stage_results.get('best_f1'),
            'submission_path': str(submission_path) if submission_path else None
        }
        
        # Log to MLFlow and DuckDB
        if mlflow_tracker:
            mlflow_tracker.log_metrics(final_results.get('full_training', {}).get('final_metrics', {}))
            mlflow_tracker.log_param('best_model', best_model_type or 'none')
        
        if duckdb_reporter:
            duckdb_reporter.log_cv_results(
                experiment_name,
                best_model_type or 'unknown',
                training_30pct_results.get(best_model_type, {}).get('cv_results', {})
            )
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 80)
        
        return final_results
