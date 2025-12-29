#!/bin/bash
#SBATCH --job-name=hvsm_job_3
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail
set -o errtrace
umask 077

module purge
module load python3.11-anaconda/2024.02
module load cuda/12.1 || true

mkdir -p logs .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"

export WORK_DIR="${SLURM_TMPDIR:-$PWD}"
export VENV_DIR="$WORK_DIR/hvsm_venv_${SLURM_JOB_ID:-$$}"
rm -rf "$VENV_DIR"
python -m venv "$VENV_DIR"
rm -f "$VENV_DIR"/bin/Activate.ps1 2>/dev/null || true
source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV_DISABLE_PROMPT=1

mkdir -p "$WORK_DIR/runs" "$WORK_DIR/weights" "$WORK_DIR/.cache"
ln -snf "$WORK_DIR/runs" runs || true
ln -snf "$WORK_DIR/weights" weights || true

export HF_HOME="$WORK_DIR/.cache/hf"
mkdir -p "$HF_HOME"

export MALLOC_TRIM_THRESHOLD_=134217728
export MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:128,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_SERVICE_FORCE_INTEL=1

echo "Host:        $(hostname)"
echo "Date:        $(date -Is)"
echo "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
echo "CUDA_HOME:   ${CUDA_HOME:-none}"
command -v nvidia-smi >/dev/null && nvidia-smi || echo "No nvidia-smi"

python - <<'PY'
import os, sys, subprocess, shutil
def run(*args): print("+", *args, flush=True); subprocess.check_call(list(args))
run(sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "pip>=24.0", "setuptools>=68.0", "wheel>=0.41.0", "packaging>=24.0")
BASE_REQS = [
    "papermill==2.6.0","ipykernel==6.29.5","matplotlib==3.8.4",
    "polars>=0.20.31","numpy>=1.26,<2","tqdm==4.66.4",
    "scipy==1.11.4","seaborn==0.13.2","textblob==0.18.0",
    "pyarrow>=14.0.0",
]
run(sys.executable, "-m", "pip", "install", "-q", *BASE_REQS)
def which(x): return shutil.which(x) is not None
cuda_major=None
if which("nvidia-smi"):
    cuda_home = os.environ.get("CUDA_HOME","")
    if "/12" in cuda_home: cuda_major="cu12"
    elif "/11" in cuda_home: cuda_major="cu11"
if not cuda_major:
    raise RuntimeError("CUDA not detected; cuML/cudf require a GPU node.")
rapids_pkgs = [f"cudf-{cuda_major}", f"cuml-{cuda_major}", f"rmm-{cuda_major}"]
run(sys.executable, "-m", "pip", "install", "-q", "--extra-index-url",
    "https://pypi.nvidia.com", *rapids_pkgs)
cupy_pkg = "cupy-cuda12x" if cuda_major == "cu12" else "cupy-cuda11x"
run(sys.executable, "-m", "pip", "install", "-q", cupy_pkg)
run(sys.executable, "-m", "pip", "check")
for m in ["papermill","ipykernel","polars","numpy","matplotlib","scipy",
          "tqdm","textblob","cudf","cuml","cupy"]:
    __import__(m)
print("All critical imports succeeded.", flush=True)
PY

KNAME="hvsm-${SLURM_JOB_ID:-$$}"
python -m ipykernel install --user --name "${KNAME}"   --display-name "HVSM (${KNAME})"
trap 'jupyter kernelspec remove -y "${KNAME}" 2>/dev/null || true' EXIT

NOTEBOOK_IN="notebooks/hvsm_prod_3.ipynb"
STAMP="$(date +%Y%m%d-%H%M%S)"
NOTEBOOK_OUT="$WORK_DIR/runs/hvsm_executed_${STAMP}.ipynb"

PM_ARGS=(
  -k "${KNAME}"
  -p USE_SPACY true
  -p DATA_DIR "data"
  -p TRAIN_CSV "data/train.csv"
  -p VAL_CSV "data/val.csv"
  -p TEST_CSV "data/test.csv"
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
papermill --log-output "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" "${PM_ARGS[@]}"

echo "Done. Output: ${NOTEBOOK_OUT}"
