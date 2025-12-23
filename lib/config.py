"""
Configuration management for GPU-first ML library.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


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
        self.data_path = os.getenv('DATA_PATH', 'data/')
        self.checkpoint_dir = os.getenv('CHECKPOINT_DIR', 'checkpoints/')
        self.log_dir = os.getenv('LOG_DIR', 'logs/')
        self.model_dir = os.getenv('MODEL_DIR', 'models/')
        self.output_dir = os.getenv('OUTPUT_DIR', 'outputs/')
        
        # GPU settings
        self.use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
        self.gpu_id = int(os.getenv('GPU_ID', '0'))
        self.num_gpus = int(os.getenv('NUM_GPUS', '1'))
        
        # Data settings
        self.train_file = 'train.csv'
        self.val_file = 'val.csv'
        self.test_file = 'test.csv'
        self.text_column = 'text'
        self.label_column = 'label'
        self.id_column = 'id'
        
        # Training settings
        self.cv_folds = 5
        self.subset_size = 0.2  # 20% for CV and grid search
        self.random_state = 42  # Consistent seed across all stages
        self.stratified = True
        self.use_rfe = os.getenv('USE_RFE', 'true').lower() == 'true'
        self.n_jobs = int(os.getenv('N_JOBS', '-1'))
        
        # Models to train
        self.models = ['logreg', 'svm', 'bayesian', 'xgboost', 'neural_net']
        
        # Feature engineering
        self.use_nlp_features = True
        self.use_embeddings = True
        self.use_encodings = True
        
        # Embedding settings
        self.word2vec_dim = 300
        self.sentence_transformer_model = 'all-MiniLM-L6-v2'
        self.bert_model = 'distilbert-base-uncased'
        self.embedding_aggregation = 'mean'  # mean, max, weighted
        
        # Preprocessing
        self.scale_features = True
        self.impute_missing = True
        self.use_pca = True
        self.pca_components = 0.95  # Keep 95% variance or int for fixed components
        self.normalize = True
        
        # Hyperparameter grids
        self.hyperparameter_grids = self._get_default_hyperparameter_grids()
        
        # Neural network settings
        self.nn_hidden_layers = [512, 256]
        self.nn_dropout = 0.3
        self.nn_learning_rate = 0.001
        self.nn_batch_size = 32
        self.nn_epochs = 50
        self.nn_early_stopping_patience = 5
        
        # XGBoost settings
        self.xgb_tree_method = 'gpu_hist'
        self.xgb_max_depth = 6
        self.xgb_learning_rate = 0.1
        self.xgb_n_estimators = 100
        
        # Parallelization
        self.n_jobs = -1
        self.parallel_backend = 'threading'
        
        # Checkpointing
        self.save_checkpoints = True
        self.checkpoint_frequency = 1  # Save every N epochs
        self.keep_best_n = 3  # Keep best N checkpoints
        self.delete_existing = os.getenv('DELETE_EXISTING', 'false').lower() == 'true'  # Delete existing checkpoints before starting
        
        # Logging
        self.log_level = 'INFO'  # DEBUG, INFO, WARN, ERROR
        self.log_to_file = True
        self.log_to_console = True
        
        # Arrow/Parquet settings
        self.arrow_format = True
        self.compression = 'snappy'
        
    def _load_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Store unknown keys for flexibility
                setattr(self, key, value)
    
    def _get_default_hyperparameter_grids(self) -> Dict[str, Dict[str, List]]:
        """Get default hyperparameter grids for each model."""
        return {
            'logreg': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'l1_ratio': [0.1, 0.5, 0.9] if 'elasticnet' in ['l1', 'l2', 'elasticnet'] else [0.5],
                'class_weight': [None, 'balanced']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'class_weight': [None, 'balanced']
            },
            'bayesian': {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'fit_prior': [True, False]
            },
            'xgboost': {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [50, 100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'neural_net': {
                'hidden_layers': [[256], [512], [512, 256]],
                'dropout': [0.2, 0.3, 0.4],
                'learning_rate': [0.0001, 0.001, 0.01],
                'batch_size': [16, 32, 64]
            }
        }
    
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

