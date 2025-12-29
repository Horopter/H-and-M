#!/bin/bash
#SBATCH --job-name=hvsm-notebook
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu           # change if needed
#SBATCH --gpus=1                           # remove if CPU-only
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

############################
# 0) Modules and directories
############################
module purge
module load python3.11-anaconda/2024.02
# If you need NVIDIA GPUs, also load CUDA:
module load cuda/12.1 || true

mkdir -p logs runs weights .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"

############################
# 1) Helpful diagnostics
############################
echo "Host:        $(hostname)"
echo "Date:        $(date -Is)"
echo "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
echo "Python:      $(python -c 'import sys;print(sys.executable)')"
echo "Python ver:  $(python -V)"
echo "CUDA module: ${CUDA_HOME:-none}"
command -v nvidia-smi >/dev/null && nvidia-smi || echo "No nvidia-smi"

############################
# 2) Job-scoped kernel name
############################
KNAME="hvsm-${SLURM_JOB_ID:-$$}"
python -m ipykernel install --user --name "${KNAME}" \
  --display-name "Python 3.11 (${KNAME})"

trap 'jupyter kernelspec remove -y "${KNAME}" 2>/dev/null || true' EXIT

############################
# 3) Caches to fast local disk
############################
export HF_HOME="${SLURM_TMPDIR:-$PWD}/.cache/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

############################
# 4) Threading + BLAS hygiene
############################
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

############################
# 5) Ensure deps in this env
############################
python - <<'PY'
import os, sys, subprocess
pip = [sys.executable, "-m", "pip", "install", "-q"]

# Core libs (match notebook)
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
])

# Pick Torch build based on CUDA presence
def has_cmd(name):
    from shutil import which
    return which(name) is not None

cuda_tag = None
if has_cmd("nvidia-smi"):
    # Heuristic by CUDA module (falls back to runtime probe)
    if os.environ.get("CUDA_HOME","").endswith("/12.1"):
        cuda_tag = "cu121"
    elif os.environ.get("CUDA_HOME","").endswith("/11.8"):
        cuda_tag = "cu118"

idx = None
if cuda_tag:
    idx = f"https://download.pytorch.org/whl/{cuda_tag}"
else:
    # CPU fallback
    idx = "https://download.pytorch.org/whl/cpu"

subprocess.check_call([
  sys.executable, "-m", "pip", "install", "-q",
  "--index-url", idx, "torch", "torchvision", "torchaudio"
])

import torch
print("Torch:", torch.__version__, "CUDA tag:",
      getattr(torch.version, "cuda", None),
      "avail:", torch.cuda.is_available())
PY

############################
# 6) More diagnostics
############################
python - <<'PY'
import torch, platform, sys
print("Platform:", platform.platform())
print("Python:", sys.version)
print("CUDA visible:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

############################
# 7) Execute the notebook
############################
NOTEBOOK_IN="hvsm_colab.ipynb"
STAMP="$(date +%Y%m%d-%H%M%S)"
NOTEBOOK_OUT="runs/hvsm_executed_${STAMP}.ipynb"

# Notebook parameters you wanted timeboxed
PM_ARGS=(
  -k "${KNAME}"
  -p USE_SPACY true
  -p DATA_DIR "."
  -p TRAIN_CSV "train.csv"
  -p VAL_CSV "val.csv"
  -p TEST_CSV "test.csv"
  -p SAVE_DIR "weights"
  -p MAX_TRAIN_TIME_PER_MODEL_MIN 170
  -p EARLY_DROP_MIN 30
  -p EARLY_DROP_F1 0.65
  -p N_FOLDS 5
)

echo "Running papermill -> ${NOTEBOOK_OUT}"
papermill "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" "${PM_ARGS[@]}"

echo "Done at $(date -Is). Output: ${NOTEBOOK_OUT}"

