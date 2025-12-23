#!/bin/bash
#
# SLURM Batch Script for GPU-First ML Training Pipeline
#
# Trains multiple models (logreg, SVM, Bayesian, XGBoost, Neural Network)
# with stratified 5-fold CV, hyperparameter tuning, and comprehensive reporting.
#
# Usage:
#   sbatch scripts/slurm_jobs/slurm_ml_training.sh

#SBATCH --job-name=kaggle_pipeline_1
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/ml_training-%j.out
#SBATCH --error=logs/ml_training-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL
#SBATCH --export=ALL

set -euo pipefail
set -o errtrace
umask 077

# ============================================================================
# Environment Setup
# ============================================================================

unset MallocStackLogging || true
unset MallocStackLoggingNoCompact || true
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02
module load cuda/12.1 || true

mkdir -p logs .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"
export WORK_DIR="${SLURM_TMPDIR:-$PWD}"
export ORIG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export VENV_DIR="$ORIG_DIR/venv"

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    echo "$@" >&1
    echo "$@" >&2
    sync 2>/dev/null || true
}

# ============================================================================
# Virtual Environment Setup
# ============================================================================

log "Activating virtual environment: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    log "✗ ERROR: Virtual environment not found: $VENV_DIR"
    log "  Create with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV_DISABLE_PROMPT=1

# ============================================================================
# Environment Variables
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ============================================================================
# System Information
# ============================================================================

log "=========================================="
log "GPU-FIRST ML TRAINING PIPELINE"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $(pwd)"
log "Python:      $(which python3 2>/dev/null || which python 2>/dev/null || echo 'not found')"
log "Python version: $(python3 --version 2>&1 || python --version 2>&1 || echo 'unknown')"
log "=========================================="

# ============================================================================
# Verify Prerequisites
# ============================================================================

log "Verifying prerequisites..."

# Check critical Python packages
PREREQ_PACKAGES=("polars" "numpy" "scikit-learn" "scipy" "joblib" "pyarrow")

MISSING_PACKAGES=()

for pkg in "${PREREQ_PACKAGES[@]}"; do
    case "$pkg" in
        "scikit-learn")
            if ! python3 -c "import sklearn" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg (sklearn) found"
            fi
            ;;
        *)
            if ! python3 -c "import $pkg" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg found"
            fi
            ;;
    esac
done

# Optional packages (warn if missing, but DO NOT fail)
# NOTE: CUML/cudf are notoriously difficult to install - graceful fallback to CPU
OPTIONAL_PACKAGES=("cuml" "cudf" "xgboost" "torch" "mlflow" "duckdb" "umap-learn" "statsmodels")

for pkg in "${OPTIONAL_PACKAGES[@]}"; do
    case "$pkg" in
        "umap-learn")
            if ! python3 -c "import umap" 2>/dev/null; then
                log "⚠ WARNING: $pkg not found (UMAP visualization will be disabled)"
            else
                log "✓ $pkg found"
            fi
            ;;
        "cuml"|"cudf")
            # CUML/cudf are optional - CPU fallback available
            if ! python3 -c "import ${pkg}" 2>/dev/null; then
                log "⚠ INFO: $pkg not found (will use CPU fallback - this is OK)"
            else
                log "✓ $pkg found (GPU acceleration enabled)"
            fi
            ;;
        *)
            # Use timeout to prevent hanging, and catch core dumps
            if timeout 10 python3 -c "import ${pkg//-/_}" 2>/dev/null; then
                log "✓ $pkg found"
            else
                EXIT_CODE=$?
                if [ $EXIT_CODE -eq 124 ]; then
                    log "⚠ WARNING: $pkg import timed out (may be slow to load)"
                elif [ $EXIT_CODE -eq 134 ] || [ $EXIT_CODE -eq 139 ]; then
                    log "⚠ WARNING: $pkg import crashed (core dump) - package may be corrupted"
                    log "  Try: pip uninstall $pkg && pip install $pkg"
                else
                    log "⚠ WARNING: $pkg not found (some features may be disabled)"
                fi
            fi
            ;;
    esac
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    log "✗ ERROR: Missing required packages: ${MISSING_PACKAGES[*]}"
    log "  Install with: pip install -r requirements.txt"
    exit 1
fi

# Verify data files exist
DATA_DIR="${ML_DATA_PATH:-data}"
TRAIN_FILE="$ORIG_DIR/$DATA_DIR/train.csv"
VAL_FILE="$ORIG_DIR/$DATA_DIR/val.csv"
TEST_FILE="$ORIG_DIR/$DATA_DIR/test.csv"

if [ ! -f "$TRAIN_FILE" ]; then
    log "✗ ERROR: Training data not found: $TRAIN_FILE"
    exit 1
fi
log "✓ Training data found: $TRAIN_FILE"

if [ ! -f "$VAL_FILE" ]; then
    log "✗ ERROR: Validation data not found: $VAL_FILE"
    exit 1
fi
log "✓ Validation data found: $VAL_FILE"

if [ ! -f "$TEST_FILE" ]; then
    log "⚠ WARNING: Test data not found: $TEST_FILE (optional)"
fi

log "✅ All prerequisites verified"

# ============================================================================
# Pre-flight Validation
# ============================================================================

# Define variables needed for validation (must be before validation section)
PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null || echo "python3")
LOG_FILE="$ORIG_DIR/logs/ml_training_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

log "=========================================="
log "Running Pre-flight Validation"
log "=========================================="

VALIDATION_SCRIPT="$ORIG_DIR/scripts/validate_pipeline.py"
DIAGNOSTIC_SCRIPT="$ORIG_DIR/scripts/diagnose_crash.py"
MINIMAL_TEST="$ORIG_DIR/scripts/minimal_test.py"

# First, run minimal test to see if Python can import basic packages
if [ -f "$MINIMAL_TEST" ]; then
    log "Running minimal import test..."
    if timeout 60 "$PYTHON_CMD" -u "$MINIMAL_TEST" 2>&1 | tee -a "$LOG_FILE"; then
        log "✓ Minimal test passed"
    else
        MINIMAL_EXIT=${PIPESTATUS[0]}
        log "✗ ERROR: Minimal test failed (exit code: $MINIMAL_EXIT)"
        log "  Python environment has serious issues - cannot proceed"
        log "  Check logs: $LOG_FILE"
        log "  Try: pip install --force-reinstall --no-cache-dir numpy scipy polars"
        exit $MINIMAL_EXIT
    fi
fi

# If minimal test passes, try full validation (but make it optional)
if [ -f "$VALIDATION_SCRIPT" ]; then
    log "Running full validation script..."
    
    # Run with timeout and better error handling
    if timeout 300 "$PYTHON_CMD" -u "$VALIDATION_SCRIPT" 2>&1 | tee -a "$LOG_FILE"; then
        VALIDATION_EXIT_CODE=${PIPESTATUS[0]}
        if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
            log "✓ Validation passed"
        else
            log "⚠ WARNING: Validation failed (exit code: $VALIDATION_EXIT_CODE)"
            log "  Continuing anyway (validation is non-blocking)"
            log "  Check logs for details: $LOG_FILE"
        fi
    else
        VALIDATION_EXIT_CODE=${PIPESTATUS[0]}
        if [ $VALIDATION_EXIT_CODE -eq 134 ] || [ $VALIDATION_EXIT_CODE -eq 139 ]; then
            log "⚠ WARNING: Validation script crashed (core dump)"
            log "  This indicates a package import issue, but continuing anyway"
            log "  The pipeline will attempt to run and may work despite this"
            log "  Check logs: $LOG_FILE"
        else
            log "⚠ WARNING: Validation script failed (exit code: $VALIDATION_EXIT_CODE)"
            log "  Continuing anyway - validation is non-blocking"
        fi
    fi
else
    log "⚠ WARNING: Validation script not found: $VALIDATION_SCRIPT"
    log "  Skipping pre-flight validation"
fi

# ============================================================================
# Pipeline Execution
# ============================================================================

log "=========================================="
log "Starting ML Training Pipeline"
log "=========================================="

# Configuration
MODELS="${ML_MODELS:-logreg svm bayesian xgboost neural_net}"
CV_FOLDS="${ML_CV_FOLDS:-5}"
SUBSET_SIZE="${ML_SUBSET_SIZE:-0.1}"
USE_GPU="${ML_USE_GPU:-true}"
EXPERIMENT_NAME="${ML_EXPERIMENT_NAME:-}"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-}"
DUCKDB_PATH="${ML_DUCKDB_PATH:-results.duckdb}"
RUN_STATS="${ML_RUN_STATS:-true}"

log "Models: $MODELS"
log "CV Folds: $CV_FOLDS (stratified)"
log "Subset Size: $SUBSET_SIZE"
log "Use GPU: $USE_GPU"
log "Run Stats: $RUN_STATS"

PIPELINE_START=$(date +%s)
# PYTHON_CMD and LOG_FILE already defined above (before validation section)

cd "$ORIG_DIR" || exit 1

# Build command
GPU_FLAG=""
if [ "$USE_GPU" = "true" ] || [ "$USE_GPU" = "1" ] || [ "$USE_GPU" = "yes" ]; then
    GPU_FLAG="--use-gpu"
fi

STATS_FLAG=""
if [ "$RUN_STATS" = "true" ] || [ "$RUN_STATS" = "1" ] || [ "$RUN_STATS" = "yes" ]; then
    STATS_FLAG="--run-stats"
fi

MLFLOW_FLAG=""
if [ -n "$MLFLOW_URI" ]; then
    MLFLOW_FLAG="--mlflow-tracking-uri $MLFLOW_URI"
fi

log "Running training pipeline..."
log "Log file: $LOG_FILE"

if "$PYTHON_CMD" -u "$ORIG_DIR/src/run_training_pipeline.py" \
    --data-path "$DATA_DIR" \
    --checkpoint-dir checkpoints \
    --log-dir logs \
    --output-dir outputs \
    --models $MODELS \
    --cv-folds "$CV_FOLDS" \
    --subset-size "$SUBSET_SIZE" \
    --duckdb-path "$DUCKDB_PATH" \
    $GPU_FLAG \
    $STATS_FLAG \
    ${EXPERIMENT_NAME:+--experiment-name "$EXPERIMENT_NAME"} \
    ${MLFLOW_FLAG:+$MLFLOW_FLAG} \
    2>&1 | tee "$LOG_FILE"; then
    
    PIPELINE_END=$(date +%s)
    PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
    log "✓ Training pipeline completed successfully in ${PIPELINE_DURATION}s ($((${PIPELINE_DURATION} / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/outputs"
    log "DuckDB database: $ORIG_DIR/$DUCKDB_PATH"
else
    PIPELINE_END=$(date +%s)
    PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
    log "✗ ERROR: Training pipeline failed after ${PIPELINE_DURATION}s"
    log "Check log file: $LOG_FILE"
    
    # Check for specific error types
    if grep -q "CUDA out of memory\|OutOfMemoryError\|OOM" "$LOG_FILE" 2>/dev/null; then
        log "✗ ERROR: CUDA Out of Memory detected"
        log "  Try reducing batch size or model complexity"
    fi
    exit 1
fi

log ""
log "============================================================"
log "ML TRAINING PIPELINE EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${PIPELINE_DURATION}s ($((${PIPELINE_DURATION} / 60)) minutes)"
log "Models trained: $MODELS"
log "CV Folds: $CV_FOLDS (stratified)"
log "Output directory: $ORIG_DIR/outputs"
log "DuckDB database: $ORIG_DIR/$DUCKDB_PATH"
log "Log file: $LOG_FILE"
log "============================================================"

