#!/bin/bash
# Install dependencies in batches to avoid memory issues on login nodes
# This script installs all packages except torch and large NLP packages
# Those should be installed via SLURM job: sbatch scripts/slurm_jobs/slurm_install_large_packages.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Installing Dependencies in Batches"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Activate venv if it exists
if [ -d "venv" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✓ Activated virtual environment"
    else
        echo "⚠ venv directory exists but activate script not found"
    fi
else
    echo "⚠ No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Created and activated virtual environment"
fi

# Function to install with retries and memory-friendly options
install_batch() {
    local file=$1
    local name=$2
    echo ""
    echo "=========================================="
    echo "Installing: $name"
    echo "=========================================="
    
    if [ ! -f "$file" ]; then
        echo "⚠ WARNING: $file not found, skipping"
        return 0
    fi
    
    # Try up to 3 times
    for attempt in 1 2 3; do
        echo "Attempt $attempt/3..."
        if pip install --no-cache-dir --no-build-isolation -r "$file" 2>&1 | tee -a "$PROJECT_DIR/logs/install.log"; then
            echo "✓ Successfully installed $name"
            return 0
        else
            EXIT_CODE=${PIPESTATUS[0]}
            echo "✗ Attempt $attempt failed (exit code: $EXIT_CODE)"
            if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 9 ]; then
                echo "  Process was killed (likely out of memory)"
            fi
            if [ $attempt -lt 3 ]; then
                echo "Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
    
    echo "✗ Failed to install $name after 3 attempts"
    return 1
}

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Install in order (smallest to largest)
echo ""
echo "Starting batch installation..."
echo ""

install_batch "$PROJECT_DIR/requirements-core.txt" "Core packages" || {
    echo "✗ CRITICAL: Core packages installation failed"
    exit 1
}

install_batch "$PROJECT_DIR/requirements-ml.txt" "ML packages" || {
    echo "⚠ WARNING: ML packages installation failed, continuing..."
}

install_batch "$PROJECT_DIR/requirements-stats.txt" "Statistical packages" || {
    echo "⚠ WARNING: Statistical packages installation failed, continuing..."
}

install_batch "$PROJECT_DIR/requirements-viz.txt" "Visualization packages" || {
    echo "⚠ WARNING: Visualization packages installation failed, continuing..."
}

echo ""
echo "=========================================="
echo "Core Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Install large packages in a compute job:"
echo "     sbatch scripts/slurm_jobs/slurm_install_large_packages.sh"
echo ""
echo "  2. (Optional) Install GPU packages via conda:"
echo "     conda install -c rapidsai -c conda-forge -c nvidia cuml cudf python=3.11 cudatoolkit=12.1"
echo ""
echo "Installation log saved to: $PROJECT_DIR/logs/install.log"
echo ""

