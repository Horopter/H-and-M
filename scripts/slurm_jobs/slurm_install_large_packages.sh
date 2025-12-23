#!/bin/bash
# SLURM job to install large packages (torch, transformers) in a compute node
# This avoids memory limits on login nodes
#
# Usage:
#   sbatch scripts/slurm_jobs/slurm_install_large_packages.sh

#SBATCH --job-name=install_large_pkgs
#SBATCH --account=eecs442f25_class
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/install_large_packages-%j.out
#SBATCH --error=logs/install_large_packages-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT

set -e

module purge
module load python3.11-anaconda/2024.02

# Get project directory
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/Kaggle_1}"
cd "$PROJECT_DIR" || {
    echo "ERROR: Cannot cd to $PROJECT_DIR"
    exit 1
}

echo "=========================================="
echo "Installing Large Packages in Compute Node"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Activate venv (create if it doesn't exist)
if [ -d "venv" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✓ Activated virtual environment"
    else
        echo "⚠ venv directory exists but activate script not found, recreating..."
        rm -rf venv
        python3 -m venv venv
        source venv/bin/activate
        echo "✓ Created and activated virtual environment"
    fi
else
    echo "⚠ No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Created and activated virtual environment"
fi

# Function to install with logging
install_package() {
    local file=$1
    local name=$2
    echo ""
    echo "Installing: $name"
    echo "----------------------------------------"
    
    if [ ! -f "$file" ]; then
        echo "⚠ WARNING: $file not found, skipping"
        return 0
    fi
    
    if pip install --no-cache-dir --no-build-isolation -r "$file" 2>&1 | tee -a "$PROJECT_DIR/logs/install_large.log"; then
        echo "✓ Successfully installed $name"
        return 0
    else
        echo "✗ Failed to install $name"
        return 1
    fi
}

# Install torch first (largest package)
install_package "$PROJECT_DIR/requirements-torch.txt" "PyTorch" || {
    echo "✗ CRITICAL: PyTorch installation failed"
    exit 1
}

# Install NLP packages (also large)
install_package "$PROJECT_DIR/requirements-nlp.txt" "NLP packages (transformers, sentence-transformers)" || {
    echo "⚠ WARNING: NLP packages installation failed"
}

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Installed packages:"
pip list | grep -E "(torch|transformers|sentence-transformers)" || true
echo ""
echo "Installation log: $PROJECT_DIR/logs/install_large.log"
echo ""

