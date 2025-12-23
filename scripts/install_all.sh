#!/bin/bash
# Complete installation script - installs everything in the right order
# This is the main script users should run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Complete Dependency Installation"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Step 1: Install core packages (on login node)
echo ""
echo "Step 1: Installing core packages (on login node)..."
echo "=========================================="
bash "$SCRIPT_DIR/install_dependencies.sh"

# Step 2: Check if we're on a login node or compute node
SLURM_SCRIPT="$SCRIPT_DIR/slurm_jobs/slurm_install_large_packages.sh"
if [ -n "$SLURM_JOB_ID" ]; then
    echo ""
    echo "Step 2: Already in compute node, installing large packages..."
    echo "=========================================="
    bash "$SLURM_SCRIPT"
else
    echo ""
    echo "Step 2: Submitting SLURM job for large packages..."
    echo "=========================================="
    echo "Submitting job to install torch and NLP packages..."
    if [ ! -f "$SLURM_SCRIPT" ]; then
        echo "✗ ERROR: SLURM script not found at $SLURM_SCRIPT"
        exit 1
    fi
    JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT")
    echo "✓ Job submitted: $JOB_ID"
    echo ""
    echo "Monitor progress with:"
    echo "  tail -f logs/install_large_packages-${JOB_ID}.out"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "After the job completes, verify installation:"
    echo "  source venv/bin/activate"
    echo "  python -c 'import torch; import transformers; print(\"✓ All packages installed\")'"
fi

echo ""
echo "=========================================="
echo "Installation Process Complete!"
echo "=========================================="

