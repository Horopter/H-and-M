#!/bin/bash
#SBATCH --job-name=hvsm_job_c
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --requeue
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

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"


mkdir -p logs .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"

export WORK_DIR="${SLURM_TMPDIR:-$PWD}"
export DATA_DIR="${PROJECT_ROOT}/data"
export PYTHONPATH="${PROJECT_ROOT}/tools:${PYTHONPATH:-}"
export HVSM_ENABLE_RMM=1
export HVSM_RMM_CUPY=0
export RMM_MANAGED_MEMORY=1
export RMM_POOL=1


export VENV_DIR="${HVSM_VENV_DIR:-$PROJECT_ROOT/hvsm_venv_shared}"
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
  rm -f "$VENV_DIR"/bin/Activate.ps1 2>/dev/null || true
fi
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
if ! command -v nvidia-smi >/dev/null; then
  echo "No nvidia-smi; GPU node required." >&2
  exit 1
fi
nvidia-smi || { echo "nvidia-smi failed; GPU unavailable." >&2; exit 1; }
GPU_INDICES="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d " " | paste -sd, -)"
if [ -z "$GPU_INDICES" ]; then
  echo "No GPUs detected by nvidia-smi." >&2
  exit 1
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  INVALID=0
  IFS="," read -r -a REQ <<< "${CUDA_VISIBLE_DEVICES}"
  for gid in "${REQ[@]}"; do
    if ! echo ",${GPU_INDICES}," | grep -q ",${gid},"; then
      INVALID=1
      break
    fi
  done
  if [ "$INVALID" -eq 1 ]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} invalid for host GPUs ${GPU_INDICES}; resetting." >&2
    unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
  fi
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_LIST="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
  if [ -n "$GPU_LIST" ]; then
    GPU_LIST_CLEAN="$(echo "$GPU_LIST" | tr -cd "0-9,")"
  else
    GPU_LIST_CLEAN="$GPU_INDICES"
  fi
  if [ -n "$GPU_LIST_CLEAN" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_LIST_CLEAN"
    export NVIDIA_VISIBLE_DEVICES="$GPU_LIST_CLEAN"
  fi
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"


LOCK_FILE="$VENV_DIR/.pip.lock"
if command -v flock >/dev/null; then
  exec 200>"$LOCK_FILE"
  flock -x 200
fi
if [ ! -f "$VENV_DIR/.deps_installed" ]; then
python - <<'PY'
import os, sys, subprocess, shutil
def run(*args): print("+", *args, flush=True); subprocess.check_call(list(args))
run(sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "pip>=24.0", "setuptools>=68.0", "wheel>=0.41.0", "packaging>=24.0")
BASE_REQS = [
    "papermill==2.6.0","ipykernel==6.29.5","matplotlib==3.8.4",
    "pandas>=2.2.2,<3","polars>=0.20.31","numpy>=1.26,<2","scikit-learn==1.5.2",
    "xgboost==2.1.1","tqdm==4.66.4","joblib==1.4.2","scipy==1.11.4",
    "seaborn==0.13.2","ftfy==6.2.3","emoji==2.14.0","textstat==0.7.4",
    "wordfreq==3.1.1","spacy==3.7.5","transformers==4.44.2",
    "datasets>=2.20,<4","pyarrow>=14.0.0","textblob==0.18.0",
]
run(sys.executable, "-m", "pip", "install", "-q", *BASE_REQS)
run(sys.executable, "-m", "pip", "install", "-q",
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl")
def which(x): return shutil.which(x) is not None
cuda_tag=None
cuda_major=None
if which("nvidia-smi"):
    cuda_home = os.environ.get("CUDA_HOME","")
    if cuda_home.endswith("/12.1"):
        cuda_tag="cu121"; cuda_major="cu12"
    elif cuda_home.endswith("/11.8"):
        cuda_tag="cu118"; cuda_major="cu11"
    elif "/12" in cuda_home:
        cuda_major="cu12"
    elif "/11" in cuda_home:
        cuda_major="cu11"
idx=(f"https://download.pytorch.org/whl/{cuda_tag}" if cuda_tag else
    "https://download.pytorch.org/whl/cpu")
run(sys.executable, "-m", "pip", "install", "-q", "--index-url", idx,
    "torch", "torchvision", "torchaudio")
if not cuda_major:
    raise RuntimeError("CUDA not detected; cuML/cudf require a GPU node.")
rapids_pkgs = [f"cudf-{cuda_major}", f"cuml-{cuda_major}", f"rmm-{cuda_major}"]
run(sys.executable, "-m", "pip", "install", "-q", "--extra-index-url",
    "https://pypi.nvidia.com", *rapids_pkgs)
cupy_pkg = "cupy-cuda12x" if cuda_major == "cu12" else "cupy-cuda11x"
run(sys.executable, "-m", "pip", "install", "-q", cupy_pkg)
run(sys.executable, "-m", "pip", "check")
for m in ["papermill","ipykernel","transformers","datasets","sklearn",
          "xgboost","pandas","polars","numpy","matplotlib","spacy",
          "en_core_web_sm","torch","textblob","cudf","cuml","cupy"]:
    __import__(m)
print("All critical imports succeeded.", flush=True)
PY
  touch "$VENV_DIR/.deps_installed"
fi
if command -v flock >/dev/null; then
  flock -u 200
  exec 200>&-
fi

MAX_REQUEUES="${HVSM_MAX_REQUEUES:-2}"
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
if ! python - <<'PY'
import time
try:
    import cupy as cp
except Exception as exc:
    raise SystemExit(f"GPU preflight failed: cupy import error: {exc}")
for attempt in range(3):
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device(0).use()
        x = cp.zeros((1,), dtype=cp.float32)
        _ = float(x.sum().get())
        print("GPU preflight OK", flush=True)
        break
    except Exception as exc:
        print(f"GPU preflight failed (attempt {attempt+1}/3): {exc}", flush=True)
        if attempt == 2:
            raise SystemExit(1)
        time.sleep(10)
PY
then
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null; then
    if [ "${RESTART_COUNT}" -lt "${MAX_REQUEUES}" ]; then
      ATTEMPT=$((RESTART_COUNT + 1))
      echo "GPU preflight failed; requeuing ${SLURM_JOB_ID} (attempt ${ATTEMPT}/${MAX_REQUEUES})." >&2
      if scontrol requeue "${SLURM_JOB_ID}"; then
        exit 0
      fi
      echo "scontrol requeue failed; exiting." >&2
    fi
  fi
  echo "GPU preflight failed; not requeueing." >&2
  exit 1
fi

KNAME="hvsm-${SLURM_JOB_ID:-$$}"
python -m ipykernel install --user --name "${KNAME}"   --display-name "HVSM (${KNAME})"
trap 'jupyter kernelspec remove -y "${KNAME}" 2>/dev/null || true' EXIT

NOTEBOOK_IN="notebooks/hvsm_prod_c.ipynb"
STAMP="$(date +%Y%m%d-%H%M%S)"
NOTEBOOK_OUT="$WORK_DIR/runs/hvsm_executed_${STAMP}.ipynb"

PM_ARGS=(
  -k "${KNAME}"
  -p DATA_DIR "${DATA_DIR}"
  -p TRAIN_CSV "${DATA_DIR}/train.csv"
  -p VAL_CSV "${DATA_DIR}/val.csv"
  -p TEST_CSV "${DATA_DIR}/test.csv"
)

for f in "${DATA_DIR}/train.csv" "${DATA_DIR}/val.csv" "${DATA_DIR}/test.csv"; do
  if [ ! -f "$f" ]; then
    if ls "${f}.part"* >/dev/null 2>&1; then
      echo "Missing $f; chunk parts found (will reassemble in notebook)."
    else
      echo "Missing required data file: $f (no chunk parts found)." >&2
      exit 1
    fi
  fi
done

echo "Running papermill -> ${NOTEBOOK_OUT}"
papermill --log-output "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" "${PM_ARGS[@]}"

echo "Done. Output: ${NOTEBOOK_OUT}"
