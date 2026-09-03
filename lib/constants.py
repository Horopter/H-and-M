"""
Centralized constants for the ML pipeline.
All configuration defaults and constants should be defined here.
"""
import os
from typing import List, Dict, Any

# ============================================================================
# PATH CONSTANTS
# ============================================================================
DEFAULT_DATA_PATH = 'data/'
DEFAULT_CHECKPOINT_DIR = 'checkpoints/'
DEFAULT_LOG_DIR = 'logs/'
DEFAULT_MODEL_DIR = 'models/'
DEFAULT_OUTPUT_DIR = 'outputs/'

# ============================================================================
# FILE NAMES
# ============================================================================
DEFAULT_TRAIN_FILE = 'train.csv'
DEFAULT_VAL_FILE = 'val.csv'
DEFAULT_TEST_FILE = 'test.csv'

# ============================================================================
# COLUMN NAMES
# ============================================================================
DEFAULT_TEXT_COLUMN = 'text'
DEFAULT_LABEL_COLUMN = 'label'
DEFAULT_ID_COLUMN = 'id'

# ============================================================================
# GPU SETTINGS
# ============================================================================
DEFAULT_USE_GPU = True
DEFAULT_GPU_ID = 0
DEFAULT_NUM_GPUS = 1

# ============================================================================
# TRAINING SETTINGS
# ============================================================================
DEFAULT_CV_FOLDS = 5
DEFAULT_SUBSET_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_STRATIFIED = True
DEFAULT_USE_RFE = True
DEFAULT_N_JOBS = -1
DEFAULT_SKIP_SVM = False
DEFAULT_SKIP_NEURAL_NET = False

# ============================================================================
# MODEL NAMES
# ============================================================================
MODEL_LOGREG = 'logreg'
MODEL_SVM = 'svm'
MODEL_BAYESIAN = 'bayesian'
MODEL_XGBOOST = 'xgboost'
MODEL_NEURAL_NET = 'neural_net'

DEFAULT_MODELS: List[str] = [
    MODEL_LOGREG,
    MODEL_SVM,
    MODEL_BAYESIAN,
    MODEL_XGBOOST,
    MODEL_NEURAL_NET
]

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
DEFAULT_USE_NLP_FEATURES = True
DEFAULT_USE_EMBEDDINGS = True
DEFAULT_USE_ENCODINGS = True
DEFAULT_DEFER_EMBEDDING_UNION = True

# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================
DEFAULT_WORD2VEC_DIM = 300
DEFAULT_SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_BERT_MODEL = 'distilbert-base-uncased'
DEFAULT_EMBEDDING_AGGREGATION = 'mean'  # mean, max, weighted

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================
DEFAULT_SCALE_FEATURES = True
DEFAULT_IMPUTE_MISSING = True
DEFAULT_USE_PCA = True
DEFAULT_PCA_COMPONENTS = 0.9  # Keep 90% variance or int for fixed components
DEFAULT_NORMALIZE = True

# ============================================================================
# NEURAL NETWORK SETTINGS
# ============================================================================
DEFAULT_NN_HIDDEN_LAYERS: List[int] = [512, 256]
DEFAULT_NN_DROPOUT = 0.3
DEFAULT_NN_LEARNING_RATE = 0.001
DEFAULT_NN_BATCH_SIZE = 32
DEFAULT_NN_EPOCHS = 50
DEFAULT_NN_EARLY_STOPPING_PATIENCE = 5

# ============================================================================
# XGBOOST SETTINGS
# ============================================================================
DEFAULT_XGB_TREE_METHOD = 'gpu_hist'
DEFAULT_XGB_MAX_DEPTH = 6
DEFAULT_XGB_LEARNING_RATE = 0.1
DEFAULT_XGB_N_ESTIMATORS = 100

# ============================================================================
# PARALLELIZATION
# ============================================================================
DEFAULT_PARALLEL_BACKEND = 'threading'

# ============================================================================
# CHECKPOINTING
# ============================================================================
DEFAULT_SAVE_CHECKPOINTS = True
DEFAULT_CHECKPOINT_FREQUENCY = 1
DEFAULT_KEEP_BEST_N = 3

# ============================================================================
# SUBMISSION SETTINGS
# ============================================================================
DEFAULT_SUBMISSION_USE_PROBA = True
DEFAULT_SUBMISSION_THRESHOLD = 0.5

# ============================================================================
# CHUNKED PROCESSING (MEMORY OPTIMIZATION)
# ============================================================================
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_TFIDF_CHUNK_SIZE = 250
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4

# ============================================================================
# LOGGING
# ============================================================================
DEFAULT_LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARN, ERROR
DEFAULT_LOG_TO_FILE = True
DEFAULT_LOG_TO_CONSOLE = True

# ============================================================================
# ARROW/PARQUET SETTINGS
# ============================================================================
DEFAULT_ARROW_FORMAT = True
DEFAULT_COMPRESSION = 'snappy'

# ============================================================================
# HYPERPARAMETER GRIDS
# ============================================================================
HYPERPARAMETER_GRID_LOGREG: Dict[str, List[Any]] = {
    'C': [0.01, 0.1, 1.0],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': [0.1, 0.5, 0.9],
    'class_weight': [None, 'balanced']
}

HYPERPARAMETER_GRID_SVM: Dict[str, List[Any]] = {
    'C': [0.01, 0.1, 1.0],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'class_weight': [None, 'balanced']
}

HYPERPARAMETER_GRID_BAYESIAN: Dict[str, List[Any]] = {
    'alpha': [0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

HYPERPARAMETER_GRID_XGBOOST: Dict[str, List[Any]] = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

HYPERPARAMETER_GRID_NEURAL_NET: Dict[str, List[Any]] = {
    'hidden_layers': [[256], [512], [512, 256]],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64]
}

DEFAULT_HYPERPARAMETER_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    MODEL_LOGREG: HYPERPARAMETER_GRID_LOGREG,
    MODEL_SVM: HYPERPARAMETER_GRID_SVM,
    MODEL_BAYESIAN: HYPERPARAMETER_GRID_BAYESIAN,
    MODEL_XGBOOST: HYPERPARAMETER_GRID_XGBOOST,
    MODEL_NEURAL_NET: HYPERPARAMETER_GRID_NEURAL_NET
}

# ============================================================================
# ENVIRONMENT VARIABLE NAMES
# ============================================================================
ENV_DATA_PATH = 'DATA_PATH'
ENV_CHECKPOINT_DIR = 'CHECKPOINT_DIR'
ENV_LOG_DIR = 'LOG_DIR'
ENV_MODEL_DIR = 'MODEL_DIR'
ENV_OUTPUT_DIR = 'OUTPUT_DIR'
ENV_USE_GPU = 'USE_GPU'
ENV_GPU_ID = 'GPU_ID'
ENV_NUM_GPUS = 'NUM_GPUS'
ENV_USE_RFE = 'USE_RFE'
ENV_N_JOBS = 'N_JOBS'
ENV_DELETE_EXISTING = 'DELETE_EXISTING'
ENV_DEFER_EMBEDDING_UNION = 'DEFER_EMBEDDING_UNION'
ENV_SKIP_SVM = 'SKIP_SVM'
ENV_SKIP_NEURAL_NET = 'SKIP_NEURAL_NET'
ENV_XGB_TREE_METHOD = 'XGB_TREE_METHOD'
ENV_SUBMISSION_USE_PROBA = 'SUBMISSION_USE_PROBA'
ENV_SUBMISSION_THRESHOLD = 'SUBMISSION_THRESHOLD'
ENV_CHUNK_SIZE = 'CHUNK_SIZE'
ENV_TFIDF_CHUNK_SIZE = 'TFIDF_CHUNK_SIZE'
ENV_EMBEDDING_BATCH_SIZE = 'EMBEDDING_BATCH_SIZE'
ENV_GRADIENT_ACCUMULATION_STEPS = 'GRADIENT_ACCUMULATION_STEPS'

# ============================================================================
# OTHER CONSTANTS
# ============================================================================
SEPARATOR_LINE = "=" * 80
EMBEDDING_AGGREGATION_MEAN = 'mean'
EMBEDDING_AGGREGATION_MAX = 'max'
EMBEDDING_AGGREGATION_WEIGHTED = 'weighted'
