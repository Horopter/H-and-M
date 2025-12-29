#!/bin/bash
#SBATCH --job-name=hvsm-notebook
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=14:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load python3.11-anaconda/2024.02
module load cuda/12.1 || true

mkdir -p logs runs weights .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"

# Allocator hygiene
export MALLOC_TRIM_THRESHOLD_=134217728
export MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb=128,garbage_collection_threshold:0.8"
export PYTHONUNBUFFERED=1

# Caches to local disk
export WORK_DIR="${SLURM_TMPDIR:-$PWD}"
export HF_HOME="$WORK_DIR/.cache/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

ln -snf "$WORK_DIR/runs" runs || true
ln -snf "$WORK_DIR/weights" weights || true
mkdir -p "$WORK_DIR/runs" "$WORK_DIR/weights"

export TOKENIZERS_PARALLELISM=false

# BLAS
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_SERVICE_FORCE_INTEL=1

# Kernel name
KNAME="hvsm-${SLURM_JOB_ID:-$$}"
python -m ipykernel install --user --name "${KNAME}" \
  --display-name "Python 3.11 (${KNAME})"
trap 'jupyter kernelspec remove -y "${KNAME}" 2>/dev/null || true' EXIT

# Dependencies
python - <<'PY'
import os, sys, subprocess
pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.check_call(pip + [
  "papermill==2.6.0", "ipykernel==6.29.5", "jupyterlab==4.2.5",
  "spacy==3.7.5", "en-core-web-sm@https://github.com/explosion/"
  "spacy-models/releases/download/en_core_web_sm-3.7.1/"
  "en_core_web_sm-3.7.1-py3-none-any.whl",
  "transformers==4.44.2", "datasets>=2.20,<4",
  "torchtext==0.18.0", "xgboost==2.1.1",
  "ftfy==6.2.3", "emoji==2.14.0", "textstat==0.7.4",
  "nltk==3.9.1", "scipy==1.11.4", "seaborn==0.13.2",
  "pandas>=2.2.2,<3", "wordfreq==3.1.1",
  "tqdm==4.66.4", "joblib==1.4.2", "scikit-learn==1.5.2"
])

def has(cmd):
    from shutil import which
    return which(cmd) is not None

cuda_tag = None
if has("nvidia-smi"):
    if os.environ.get("CUDA_HOME", "").endswith("/12.1"):
        cuda_tag = "cu121"
    elif os.environ.get("CUDA_HOME", "").endswith("/11.8"):
        cuda_tag = "cu118"

if cuda_tag:
    idx = f"https://download.pytorch.org/whl/{cuda_tag}"
else:
    idx = "https://download.pytorch.org/whl/cpu"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--index-url", idx, "torch", "torchvision",
                       "torchaudio"])

import torch
print("Torch:", torch.__version__, "CUDA:", getattr(torch.version, "cuda", None),
      "avail:", torch.cuda.is_available())
PY

# Execute with Papermill
NOTEBOOK_IN="hvsm_colab.ipynb"
STAMP="$(date +%Y%m%d-%H%M%S)"
NOTEBOOK_OUT="$WORK_DIR/runs/hvsm_executed_${STAMP}.ipynb"

PM_ARGS=(
  -k "${KNAME}"
  -p USE_SPACY true
  -p DATA_DIR "/home/santoshd/hvsm"
  -p TRAIN_CSV "train.csv"
  -p VAL_CSV "val.csv"
  -p TEST_CSV "test.csv"
  -p SAVE_DIR "$WORK_DIR/weights"
  -p MAX_TRAIN_TIME_PER_MODEL_MIN 160
  -p EARLY_DROP_MIN 25
  -p EARLY_DROP_F1 0.68
  -p N_FOLDS 3
  -p MAX_SEQ_LEN 256
  -p TRAIN_BATCH_SIZE 16
  -p EVAL_BATCH_SIZE 64
  -p GRAD_ACCUM_STEPS 2
  -p MIXED_PRECISION "bf16"
  -p GRADIENT_CHECKPOINTING true
  -p PIN_MEMORY true
  -p NUM_WORKERS 4
  -p PREFETCH_FACTOR 2
  -p PERSISTENT_WORKERS true
  -p DATALOADER_MEMORY_EFFICIENT true
  -p KEEP_IN_MEMORY false
  -p SAVE_ONLY_BEST true
  -p SAVE_EVERY_FOLD true
  -p LOG_ARTIFACTS_TO_NOTEBOOK false
  -p PLOT_LEVEL "light"
  -p BASE_MODEL_NAME "distilbert-base-uncased"
  -p TFIDF_MAX_FEATS 50000
  -p TEXT_COL "text"
  -p LABEL_COL "label"
  -p RANDOM_SEED 42
)

echo "Running papermill -> ${NOTEBOOK_OUT}"
papermill "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" "${PM_ARGS[@]}"

echo "Done. Output: ${NOTEBOOK_OUT}"

