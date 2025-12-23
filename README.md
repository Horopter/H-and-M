# H&M: Human vs Machine Generated Text Classification

A comprehensive, production-ready machine learning pipeline for detecting machine-generated text using GPU-accelerated libraries (CUML, Polars, PyTorch) with extensive feature engineering, stratified cross-validation, and statistical analysis.

## Problem Statement

### Description

Large Language Models like ChatGPT, Claude, and Gemini have made it very easy to generate and refine text. For better or worse, machine-generated text is everywhere—including in places where we would prefer humans to have written it. In this Kaggle task, we provide examples of human and machine generated text and ask you to train a classifier to predict whether some new texts are machine generated or not.

### Dataset

The dataset is a collection of dataframes with three columns: "id", "text", and "label". The label is 1 if the text is machine generated. The training and validation dataframes will have the "label" column but the test data will not—you'll need to predict this.

We're providing you with a lot of data but you do not need to use all of it. As a practitioner, sometimes you'll have more data that you need, so we encourage you to consider how much data you want to train on and how you'll use it.

### Task

The task will be to predict the value for the "label" column for the test data file. You will need to create a new dataframe as a .csv file with columns "id" and "label" for the test data. The IDs must match the instances in the test data. You'll upload this dataframe to Kaggle.

### Evaluation

This competition uses a public leaderboard and a private leaderboard. 50% of your submitted predictions determine your public leaderboard score, while the remaining 50% determine your private leaderboard score. You can see your public leaderboard scores and get an idea of how you're doing. Your final grade will be determined by your private leaderboard score.

The scoring will be based on the binary F1 of your prediction (where the positive class is machine generated).

### Submission File

For each ID in the test set, you must predict a label for the label variable. The file should contain a header and have the following format:

```
id,label
2,0
5,0
6,0
etc.
```

## Architecture

### Design Principles

- **GPU-First**: Prioritizes GPU acceleration using CUML, cuDF, PyTorch (CUDA), and XGBoost (GPU)
- **No Pandas**: Uses Polars for all DataFrame operations
- **Arrow Format**: All data stored in Apache Arrow/Parquet format for zero-copy operations
- **Checkpoint-Based**: Full model checkpointing with weights saved per fold, plus stage results checkpointed after each stage
- **Leakage Prevention**: Temporal and data leakage detection built-in
- **Stratified CV**: Guaranteed class distribution across folds
- **Comprehensive Logging**: Structured, timestamped logs with DEBUG/WARN/ERROR levels

### Pipeline Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Loading (Polars)                    │
│              train.csv, val.csv, test.csv → Arrow           │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Feature Engineering Pipeline                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  NLP Features│  │  Embeddings  │  │  Encodings   │      │
│  │  - TF-IDF    │  │  - Word2Vec  │  │  - Label     │      │
│  │  - N-grams   │  │  - FastText   │  │  - Target    │      │
│  │  - Stylistic │  │  - BERT       │  │  - Frequency │      │
│  │  - POS tags  │  │  - Sentence   │  │  - One-Hot   │      │
│  │              │  │    Transformers│  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                        │                                    │
│                        ▼                                    │
│              Feature Union (Sparse Matrix)                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Preprocessing (CUML)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Scaling  │→ │Imputation│→ │   PCA    │→ │Normalize │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│         Stratified 5-Fold CV (10% subset)                   │
│         + GridSearch Hyperparameter Tuning                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                    Model Training                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │LogReg     │  │   SVM    │  │ Bayesian │  │ XGBoost  │   │
│  │(CUML)     │  │  (CUML)  │  │  (CUML)  │  │  (GPU)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐                                               │
│  │Neural Net│                                               │
│  │(PyTorch) │                                               │
│  └──────────┘                                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Experiment Tracking & Reporting                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  MLFlow  │  │  DuckDB  │  │  Stats   │                  │
│  │ Tracking │  │ Reporting│  │ Analysis │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
│  - Metrics    │  - SQL Queries│  - ANOVA                    │
│  - Parameters │  - Reports    │  - Tukey HSD                │
│  - Artifacts  │  - Aggregates │  - UMAP                     │
│  - Plots      │              │  - Visualizations           │
└─────────────────────────────────────────────────────────────┘
```

## Folder Structure

```
Kaggle_1/
├── data/                          # Data files
│   ├── train.csv                  # Training data
│   ├── val.csv                    # Validation data
│   └── test.csv                   # Test data (no labels)
│
├── src/                           # Source code
│   └── run_training_pipeline.py   # Main driver script
│
├── lib/                           # Core library
│   ├── config.py                  # Configuration management
│   ├── main.py                    # Pipeline entry point
│   │
│   ├── data/                      # Data handling
│   │   ├── loader.py              # Polars + Arrow I/O
│   │   ├── preprocessor.py        # CUML preprocessing
│   │   └── splitter.py            # Temporal splitting
│   │
│   ├── features/                  # Feature engineering
│   │   ├── nlp_features.py        # NLP features (TF-IDF, n-grams)
│   │   ├── embeddings.py          # Embeddings (Word2Vec, BERT, etc.)
│   │   ├── encodings.py           # Encodings (Label, Target, etc.)
│   │   └── feature_union.py       # Feature combination
│   │
│   ├── models/                    # Model implementations
│   │   ├── base.py                # Base model interface
│   │   ├── logistic_regression.py # CUML Logistic Regression
│   │   ├── svm.py                 # CUML SVM
│   │   ├── bayesian.py            # CUML Naive Bayes
│   │   ├── xgboost.py             # XGBoost (GPU)
│   │   └── neural_network.py      # PyTorch MLP
│   │
│   ├── training/                  # Training orchestration
│   │   ├── cv.py                  # Stratified 5-fold CV
│   │   ├── grid_search.py         # Hyperparameter tuning
│   │   ├── trainer.py             # Main trainer
│   │   └── mlflow_tracker.py      # MLFlow integration
│   │
│   ├── checkpointing/             # Checkpoint management
│   │   ├── checkpoint_manager.py  # Model checkpointing
│   │   └── state_manager.py       # Training state
│   │
│   ├── logging/                   # Logging
│   │   └── logger.py              # Structured logger
│   │
│   └── utils/                     # Utilities
│       ├── gpu_utils.py           # GPU conversion utilities
│       ├── validation.py          # Leakage detection
│       ├── parallel.py            # Parallelization
│       ├── arrow_utils.py         # Arrow helpers
│       ├── duckdb_reporter.py     # DuckDB reporting
│       └── stats_analysis.py      # Statistical analyses
│
├── notebooks/                     # Jupyter notebooks
│   └── *.ipynb                    # All notebook files
│
├── submissions/                   # Submission files
│   └── submission*.csv            # Kaggle submission CSVs
│
├── checkpoints/                   # Model checkpoints (created at runtime)
│   └── {model_type}/
│       └── fold_{fold_id}/
│           └── model.arrow        # Arrow-format checkpoints
│
├── logs/                          # Log files (created at runtime)
│   ├── *.out                      # Standard output logs
│   └── *.err                      # Error logs
│
├── outputs/                       # Outputs (created at runtime)
│   ├── reports/                   # DuckDB reports
│   ├── plots/                     # PNG plots (training curves, UMAP, etc.)
│   └── results.duckdb             # DuckDB database
│
├── mlruns/                        # MLFlow runs (created at runtime)
│   └── {experiment_name}/
│       └── {run_id}/
│           ├── metrics/           # Metrics
│           ├── params/            # Parameters
│           └── artifacts/         # Artifacts (plots, models)
│
├── scripts/                       # Utility scripts
│   ├── install_dependencies.sh    # Install core packages (login node)
│   ├── install_all.sh             # Complete installation script
│   ├── validate_pipeline.py       # Pre-flight validation
│   ├── quick_check.sh             # Quick environment check
│   └── slurm_jobs/
│       ├── slurm_ml_training.sh   # SLURM batch script for training
│       └── slurm_install_large_packages.sh  # SLURM job for large packages
│
├── tests/                         # Test suite
│   ├── test_data_loader.py        # Data loading tests
│   ├── test_cv_stratified.py     # CV stratification tests
│   ├── test_gpu_fallback.py      # GPU fallback tests
│   ├── test_models.py            # Model tests
│   ├── test_validation.py        # Leakage detection tests
│   ├── test_integration.py       # Integration tests
│   └── run_tests.py              # Test runner
│
├── requirements.txt                # Main requirements (references split files)
├── requirements-core.txt           # Core dependencies (small packages)
├── requirements-ml.txt             # ML packages (medium size)
├── requirements-stats.txt          # Statistical packages
├── requirements-viz.txt             # Visualization packages
├── requirements-nlp.txt             # NLP packages (large - install in compute job)
├── requirements-torch.txt           # PyTorch (very large - install in compute job)
└── README.md                      # This file
```

## Key Features

### 1. GPU-First Architecture
- **CUML**: GPU-accelerated ML algorithms (Logistic Regression, SVM, Naive Bayes)
- **Polars**: High-performance DataFrame library (replaces Pandas)
- **PyTorch**: Neural networks with CUDA support
- **XGBoost**: GPU tree method (`gpu_hist`)
- **Automatic Fallback**: Gracefully falls back to CPU if GPU unavailable

### 2. Extensive Feature Engineering
- **NLP Features**: TF-IDF, n-grams (1-3), stylistic features (readability, sentiment, POS tags)
- **Embeddings**: Word2Vec, FastText, Sentence Transformers, BERT, TF-IDF-weighted embeddings
- **Encodings**: Label, Target, Frequency, One-Hot (with leakage prevention)

### 3. Robust Training Pipeline
- **Stratified 5-Fold CV**: Guaranteed class distribution across folds
- **Temporal Awareness**: Prevents temporal leakage while maintaining stratification
- **GridSearch**: Hyperparameter tuning on 10% subset
- **Subset Training**: Uses 10% of data for CV and grid search (configurable)

### 4. Preprocessing Pipeline
- **Scaling**: StandardScaler or MinMaxScaler (CUML)
- **Imputation**: Missing value imputation (CUML)
- **PCA**: Dimensionality reduction (CUML)
- **Normalization**: L2 normalization

### 5. Model Support
- **Logistic Regression** (CUML)
- **SVM** (CUML)
- **Naive Bayes** (CUML)
- **XGBoost** (GPU)
- **Neural Network** (PyTorch MLP)

### 6. Experiment Tracking
- **MLFlow**: Metrics, parameters, artifacts, training curves
- **DuckDB**: Structured results storage, SQL queries, aggregated reports
- **Statistical Analysis**: ANOVA, Tukey HSD, UMAP visualizations

### 7. Checkpointing
- **Per-Fold Checkpoints**: Models saved for each CV fold
- **Arrow Format**: Efficient serialization
- **Metadata**: Training state, metrics, hyperparameters

### 8. Data Leakage Prevention
- **Temporal Leakage Detection**: Ensures no future data in training
- **Data Leakage Detection**: Checks for duplicate IDs, feature leakage
- **Stratified Splits**: Maintains class distribution

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- SLURM (for cluster execution)

### Setup

1. **Clone the repository** (if applicable)

2. **Install dependencies**:

**On Login Node (for core packages):**
```bash
bash scripts/install_dependencies.sh
```

**For Large Packages (torch, transformers) - Use SLURM Job:**
```bash
sbatch scripts/slurm_jobs/slurm_install_large_packages.sh
```

**Or use the complete installation script:**
```bash
bash scripts/install_all.sh
```

This will automatically:
- Install core packages on the login node
- Submit a SLURM job for large packages (torch, transformers)

**Note**: Large packages (torch, transformers) will be killed on login nodes due to memory limits. Always use the SLURM job for these.

**Note**: CUML/RAPIDS installation can be challenging. The pipeline gracefully handles missing CUML and falls back to CPU. If you want GPU acceleration:
```bash
# Option 1: Conda (recommended for CUML)
conda install -c rapidsai -c conda-forge -c nvidia cuml cudf

# Option 2: pip (may have compatibility issues)
pip install cuml-cu11  # or cuml-cu12 depending on CUDA version
```

3. **Verify installation**:
```bash
./scripts/quick_check.sh
python scripts/validate_pipeline.py
```

## Usage

### Local Execution

```bash
# Run full pipeline
python src/run_training_pipeline.py \
    --data-path data/ \
    --checkpoint-dir checkpoints/ \
    --log-dir logs/ \
    --output-dir outputs/ \
    --models logreg svm bayesian xgboost neural_net \
    --cv-folds 5 \
    --subset-size 0.1 \
    --use-gpu \
    --run-stats \
    --experiment-name "baseline_experiment"
```

### SLURM Execution

```bash
# Submit job
sbatch scripts/slurm_jobs/slurm_ml_training.sh

# With custom parameters
ML_MODELS="logreg svm" \
ML_CV_FOLDS=5 \
ML_USE_GPU=true \
ML_EXPERIMENT_NAME="custom_experiment" \
sbatch scripts/slurm_jobs/slurm_ml_training.sh
```

### Environment Variables

The SLURM script supports the following environment variables:

- `ML_MODELS`: Space-separated list of models (default: all)
- `ML_CV_FOLDS`: Number of CV folds (default: 5)
- `ML_SUBSET_SIZE`: Fraction for CV/grid search (default: 0.1)
- `ML_USE_GPU`: Enable GPU (default: true)
- `ML_RUN_STATS`: Run statistical analyses (default: true)
- `ML_EXPERIMENT_NAME`: Experiment name (default: auto-generated)

## Output Structure

### Checkpoints
```
checkpoints/
├── logistic_regression/
│   ├── fold_0/
│   │   └── model.arrow
│   ├── fold_1/
│   │   └── model.arrow
│   └── ...
├── svm/
│   └── ...
└── ...
```

### Reports
```
outputs/
├── reports/
│   └── experiment_report_{timestamp}.txt
├── plots/
│   ├── training_curves.png
│   ├── umap_visualization.png
│   ├── metric_comparison.png
│   └── ...
└── results.duckdb
```

### MLFlow
```
mlruns/
└── {experiment_name}/
    └── {run_id}/
        ├── metrics/
        ├── params/
        └── artifacts/
```

## Testing

Run the comprehensive test suite:

```bash
python tests/run_tests.py
```

Tests cover:
- Data loading and Arrow I/O
- Stratified CV verification
- GPU fallback behavior
- Model implementations
- Leakage detection
- Integration tests

## Configuration

Configuration is managed through `lib/config.py` and can be overridden via:
1. Environment variables
2. Command-line arguments
3. Configuration dictionary

Key settings:
- `cv_folds`: Number of CV folds (default: 5)
- `subset_size`: Fraction for CV/grid search (default: 0.1)
- `use_gpu`: Enable GPU (default: true)
- `models`: List of models to train
- `use_nlp_features`: Enable NLP features (default: true)
- `use_embeddings`: Enable embeddings (default: true)
- `delete_existing`: Delete existing checkpoints before starting (default: false, set via `DELETE_EXISTING` env var)

## Troubleshooting

### Installation Issues

#### Memory Issues on Login Nodes

If `pip install` gets killed (exit code 137 or "Killed" message), it's due to memory limits on login nodes. Use the provided installation scripts:

1. **Core packages** (safe on login node):
   ```bash
   bash scripts/install_dependencies.sh
   ```

2. **Large packages** (must use compute node):
   ```bash
   sbatch scripts/slurm_jobs/slurm_install_large_packages.sh
   ```

3. **Complete installation** (automated):
   ```bash
   bash scripts/install_all.sh
   ```

#### CUML Installation Issues
- The pipeline automatically falls back to CPU if CUML is unavailable
- Check GPU availability: `python -c "from lib.utils.gpu_utils import check_gpu_availability; print(check_gpu_availability())"`
- For CUML installation, prefer conda over pip

#### Training Memory Issues
- Reduce `subset_size` for CV/grid search
- Use fewer models: `--models logreg svm`
- Disable embeddings: Set `use_embeddings=False` in config

### SLURM Issues
- Check job logs: `logs/slurm-{job_id}.out` and `logs/slurm-{job_id}.err`
- Verify GPU allocation: `squeue -u $USER`
- Check environment: `scripts/validate_pipeline.py`

## Contributing

This is a production-ready codebase following best practices:
- Modular architecture
- Comprehensive error handling
- Extensive logging
- Full test coverage
- GPU/CPU fallback
- Leakage prevention

## License

[Specify license if applicable]

## Acknowledgments

Built for Kaggle competition on machine-generated text detection.

