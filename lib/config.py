"""
Configuration management for GPU-first ML library.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from .constants import (
    DEFAULT_DATA_PATH, DEFAULT_CHECKPOINT_DIR, DEFAULT_LOG_DIR, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_FILE, DEFAULT_VAL_FILE, DEFAULT_TEST_FILE,
    DEFAULT_TEXT_COLUMN, DEFAULT_LABEL_COLUMN, DEFAULT_ID_COLUMN,
    DEFAULT_USE_GPU, DEFAULT_GPU_ID, DEFAULT_NUM_GPUS,
    DEFAULT_CV_FOLDS, DEFAULT_SUBSET_SIZE, DEFAULT_RANDOM_STATE, DEFAULT_STRATIFIED, DEFAULT_USE_RFE, DEFAULT_N_JOBS,
    DEFAULT_MODELS, MODEL_SVM,
    DEFAULT_SKIP_SVM, DEFAULT_SKIP_NEURAL_NET,
    DEFAULT_USE_NLP_FEATURES, DEFAULT_USE_EMBEDDINGS, DEFAULT_USE_ENCODINGS,
    DEFAULT_DEFER_EMBEDDING_UNION,
    DEFAULT_WORD2VEC_DIM, DEFAULT_SENTENCE_TRANSFORMER_MODEL, DEFAULT_BERT_MODEL, DEFAULT_EMBEDDING_AGGREGATION,
    DEFAULT_SCALE_FEATURES, DEFAULT_IMPUTE_MISSING, DEFAULT_USE_PCA, DEFAULT_PCA_COMPONENTS, DEFAULT_NORMALIZE,
    DEFAULT_NN_HIDDEN_LAYERS, DEFAULT_NN_DROPOUT, DEFAULT_NN_LEARNING_RATE, DEFAULT_NN_BATCH_SIZE,
    DEFAULT_NN_EPOCHS, DEFAULT_NN_EARLY_STOPPING_PATIENCE,
    DEFAULT_XGB_TREE_METHOD, DEFAULT_XGB_MAX_DEPTH, DEFAULT_XGB_LEARNING_RATE, DEFAULT_XGB_N_ESTIMATORS,
    DEFAULT_PARALLEL_BACKEND,
    DEFAULT_SAVE_CHECKPOINTS, DEFAULT_CHECKPOINT_FREQUENCY, DEFAULT_KEEP_BEST_N,
    DEFAULT_SUBMISSION_USE_PROBA, DEFAULT_SUBMISSION_THRESHOLD,
    DEFAULT_CHUNK_SIZE, DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_TFIDF_CHUNK_SIZE,
    DEFAULT_LOG_LEVEL, DEFAULT_LOG_TO_FILE, DEFAULT_LOG_TO_CONSOLE,
    DEFAULT_ARROW_FORMAT, DEFAULT_COMPRESSION,
    DEFAULT_HYPERPARAMETER_GRIDS,
    ENV_DATA_PATH, ENV_CHECKPOINT_DIR, ENV_LOG_DIR, ENV_MODEL_DIR, ENV_OUTPUT_DIR,
    ENV_USE_GPU, ENV_GPU_ID, ENV_NUM_GPUS, ENV_USE_RFE, ENV_N_JOBS, ENV_DELETE_EXISTING, ENV_DEFER_EMBEDDING_UNION,
    ENV_XGB_TREE_METHOD, ENV_SUBMISSION_USE_PROBA, ENV_SUBMISSION_THRESHOLD,
    ENV_SKIP_SVM, ENV_SKIP_NEURAL_NET,
    ENV_CHUNK_SIZE, ENV_EMBEDDING_BATCH_SIZE, ENV_GRADIENT_ACCUMULATION_STEPS, ENV_TFIDF_CHUNK_SIZE
)


class Config:
    """Centralized configuration for the ML pipeline."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration from dict or defaults."""
        # Always load defaults first to ensure all attributes exist
        self._load_defaults()
        # Then override with provided dict if any
        if config_dict:
            self._load_from_dict(config_dict)

    def _load_defaults(self):
        """Load default configuration."""
        # Paths
        self.data_path = os.getenv(ENV_DATA_PATH, DEFAULT_DATA_PATH)
        self.checkpoint_dir = os.getenv(
            ENV_CHECKPOINT_DIR, DEFAULT_CHECKPOINT_DIR)
        self.log_dir = os.getenv(ENV_LOG_DIR, DEFAULT_LOG_DIR)
        self.model_dir = os.getenv(ENV_MODEL_DIR, DEFAULT_MODEL_DIR)
        self.output_dir = os.getenv(ENV_OUTPUT_DIR, DEFAULT_OUTPUT_DIR)

        # GPU settings
        self.use_gpu = os.getenv(ENV_USE_GPU, str(
            DEFAULT_USE_GPU)).lower() == 'true'
        self.gpu_id = int(os.getenv(ENV_GPU_ID, str(DEFAULT_GPU_ID)))
        self.num_gpus = int(os.getenv(ENV_NUM_GPUS, str(DEFAULT_NUM_GPUS)))

        # Data settings
        self.train_file = DEFAULT_TRAIN_FILE
        self.val_file = DEFAULT_VAL_FILE
        self.test_file = DEFAULT_TEST_FILE
        self.text_column = DEFAULT_TEXT_COLUMN
        self.label_column = DEFAULT_LABEL_COLUMN
        self.id_column = DEFAULT_ID_COLUMN

        # Training settings
        self.cv_folds = DEFAULT_CV_FOLDS
        self.subset_size = DEFAULT_SUBSET_SIZE
        self.random_state = DEFAULT_RANDOM_STATE
        self.stratified = DEFAULT_STRATIFIED
        self.use_rfe = os.getenv(ENV_USE_RFE, str(
            DEFAULT_USE_RFE)).lower() == 'true'
        self.n_jobs = int(os.getenv(ENV_N_JOBS, str(DEFAULT_N_JOBS)))
        self.skip_svm = os.getenv(ENV_SKIP_SVM, str(
            DEFAULT_SKIP_SVM)).lower() == 'true'
        self.skip_neural_net = os.getenv(ENV_SKIP_NEURAL_NET, str(
            DEFAULT_SKIP_NEURAL_NET)).lower() == 'true'

        # Models to train
        self.models = DEFAULT_MODELS.copy()
        self._apply_model_filters()

        # Feature engineering
        self.use_nlp_features = DEFAULT_USE_NLP_FEATURES
        self.use_embeddings = DEFAULT_USE_EMBEDDINGS
        self.use_encodings = DEFAULT_USE_ENCODINGS
        self.defer_embedding_union = os.getenv(
            ENV_DEFER_EMBEDDING_UNION,
            str(DEFAULT_DEFER_EMBEDDING_UNION)
        ).lower() == 'true'

        # Embedding settings
        self.word2vec_dim = DEFAULT_WORD2VEC_DIM
        self.sentence_transformer_model = DEFAULT_SENTENCE_TRANSFORMER_MODEL
        self.bert_model = DEFAULT_BERT_MODEL
        self.embedding_aggregation = DEFAULT_EMBEDDING_AGGREGATION

        # Preprocessing
        self.scale_features = DEFAULT_SCALE_FEATURES
        self.impute_missing = DEFAULT_IMPUTE_MISSING
        self.use_pca = DEFAULT_USE_PCA
        self.pca_components = DEFAULT_PCA_COMPONENTS
        self.normalize = DEFAULT_NORMALIZE

        # Hyperparameter grids
        self.hyperparameter_grids = DEFAULT_HYPERPARAMETER_GRIDS.copy()

        # Neural network settings
        self.nn_hidden_layers = DEFAULT_NN_HIDDEN_LAYERS.copy()
        self.nn_dropout = DEFAULT_NN_DROPOUT
        self.nn_learning_rate = DEFAULT_NN_LEARNING_RATE
        self.nn_batch_size = DEFAULT_NN_BATCH_SIZE
        self.nn_epochs = DEFAULT_NN_EPOCHS
        self.nn_early_stopping_patience = DEFAULT_NN_EARLY_STOPPING_PATIENCE

        # XGBoost settings
        self.xgb_tree_method = os.getenv(
            ENV_XGB_TREE_METHOD, DEFAULT_XGB_TREE_METHOD)
        self.xgb_max_depth = DEFAULT_XGB_MAX_DEPTH
        self.xgb_learning_rate = DEFAULT_XGB_LEARNING_RATE
        self.xgb_n_estimators = DEFAULT_XGB_N_ESTIMATORS

        # Parallelization
        self.n_jobs = DEFAULT_N_JOBS
        self.parallel_backend = DEFAULT_PARALLEL_BACKEND

        # Checkpointing
        self.save_checkpoints = DEFAULT_SAVE_CHECKPOINTS
        self.checkpoint_frequency = DEFAULT_CHECKPOINT_FREQUENCY
        self.keep_best_n = DEFAULT_KEEP_BEST_N
        self.delete_existing = os.getenv(
            ENV_DELETE_EXISTING, 'false').lower() == 'true'

        # Submission settings
        self.submission_use_proba = os.getenv(
            ENV_SUBMISSION_USE_PROBA, str(DEFAULT_SUBMISSION_USE_PROBA)
        ).lower() == 'true'
        self.submission_threshold = float(
            os.getenv(ENV_SUBMISSION_THRESHOLD, str(
                DEFAULT_SUBMISSION_THRESHOLD))
        )

        # Chunked processing (memory optimization)
        self.chunk_size = int(
            os.getenv(ENV_CHUNK_SIZE, str(DEFAULT_CHUNK_SIZE)))
        self.embedding_batch_size = int(
            os.getenv(ENV_EMBEDDING_BATCH_SIZE, str(DEFAULT_EMBEDDING_BATCH_SIZE)))
        self.gradient_accumulation_steps = int(os.getenv(
            ENV_GRADIENT_ACCUMULATION_STEPS, str(DEFAULT_GRADIENT_ACCUMULATION_STEPS)))
        self.tfidf_chunk_size = int(
            os.getenv(ENV_TFIDF_CHUNK_SIZE, str(DEFAULT_TFIDF_CHUNK_SIZE)))

        # Logging
        self.log_level = DEFAULT_LOG_LEVEL
        self.log_to_file = DEFAULT_LOG_TO_FILE
        self.log_to_console = DEFAULT_LOG_TO_CONSOLE

        # Arrow/Parquet settings
        self.arrow_format = DEFAULT_ARROW_FORMAT
        self.compression = DEFAULT_COMPRESSION

    def _load_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Store unknown keys for flexibility
                setattr(self, key, value)
        self._apply_model_filters()

    def _apply_model_filters(self):
        """Apply model selection filters based on config flags."""
        if not hasattr(self, 'models') or self.models is None:
            self.models = []
        if getattr(self, 'skip_svm', False):
            self.models = [m for m in self.models if m != MODEL_SVM]
        if getattr(self, 'skip_neural_net', False):
            self.models = [m for m in self.models if m != 'neural_net']

    def _get_default_hyperparameter_grids(self) -> Dict[str, Dict[str, List]]:
        """Get default hyperparameter grids for each model."""
        return DEFAULT_HYPERPARAMETER_GRIDS.copy()

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            getattr(self, 'data_path', 'data/'),
            getattr(self, 'checkpoint_dir', 'checkpoints/'),
            getattr(self, 'log_dir', 'logs/'),
            getattr(self, 'model_dir', 'models/'),
            getattr(self, 'output_dir', 'outputs/')
        ]
        for dir_path in dirs:
            if dir_path:  # Skip None/empty
                Path(dir_path).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
        _config.ensure_directories()
    return _config


def set_config(config: Config):
    """Set global configuration instance."""
    global _config
    _config = config
    _config.ensure_directories()
