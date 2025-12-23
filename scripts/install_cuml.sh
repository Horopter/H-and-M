#!/bin/bash
#
# Install CUML and cuDF for GPU acceleration
# This script handles the notoriously difficult CUML installation
#

set -euo pipefail

echo "=========================================="
echo "CUML/cuDF Installation Script"
echo "=========================================="

# Check if we're in a venv
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "⚠ WARNING: Not in a virtual environment"
    echo "  Activate venv first: source venv/bin/activate"
    exit 1
fi

# Check CUDA version
echo "Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✓ CUDA version: $CUDA_VERSION"
else
    echo "⚠ WARNING: nvcc not found, checking module..."
    if module list 2>&1 | grep -q "cuda"; then
        CUDA_VERSION=$(module list 2>&1 | grep cuda | head -1 | sed 's/.*cuda\/\([0-9]\+\.[0-9]\+\).*/\1/')
        echo "✓ CUDA module version: $CUDA_VERSION"
    else
        echo "⚠ WARNING: CUDA not found, defaulting to CUDA 12.1"
        CUDA_VERSION="12.1"
    fi
fi

# Determine CUDA major version
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

echo ""
echo "Detected CUDA ${CUDA_MAJOR}.${CUDA_MINOR}"
echo ""

# Method 1: Try conda (recommended)
if command -v conda &> /dev/null; then
    echo "Method 1: Installing via conda (recommended)..."
    echo "This is the most reliable method for CUML/cuDF"
    echo ""
    
    # Determine conda channel based on CUDA version
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "Installing CUML/cuDF for CUDA 12.x..."
        conda install -y -c rapidsai -c conda-forge -c nvidia \
            cuml cudf python=3.11 cudatoolkit=12.1 \
            --override-channels || {
            echo "⚠ Conda install failed, trying pip..."
            METHOD="pip"
        }
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        echo "Installing CUML/cuDF for CUDA 11.x..."
        conda install -y -c rapidsai -c conda-forge -c nvidia \
            cuml cudf python=3.11 cudatoolkit=11.8 \
            --override-channels || {
            echo "⚠ Conda install failed, trying pip..."
            METHOD="pip"
        }
    else
        echo "⚠ Unsupported CUDA version, trying pip..."
        METHOD="pip"
    fi
    
    if [ "${METHOD:-conda}" = "conda" ]; then
        echo "✓ CUML/cuDF installed via conda"
        python -c "import cuml; import cudf; print('✓ CUML and cuDF import successfully')"
        exit 0
    fi
else
    echo "⚠ Conda not available, using pip..."
    METHOD="pip"
fi

# Method 2: Try pip with CUDA-specific packages
if [ "${METHOD:-pip}" = "pip" ]; then
    echo ""
    echo "Method 2: Installing via pip..."
    echo "This may take longer and is less reliable"
    echo ""
    
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "Installing cuml-cu12 and cudf-cu12..."
        pip install --no-cache-dir cuml-cu12 cudf-cu12 || {
            echo "✗ pip install failed for CUDA 12.x packages"
            echo ""
            echo "Trying alternative: rapids-cudf and rapids-cuml..."
            pip install --no-cache-dir rapids-cudf rapids-cuml || {
                echo "✗ All pip methods failed"
                echo ""
                echo "RECOMMENDATION: Use conda instead:"
                echo "  conda install -c rapidsai -c conda-forge -c nvidia cuml cudf"
                exit 1
            }
        }
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        echo "Installing cuml-cu11 and cudf-cu11..."
        pip install --no-cache-dir cuml-cu11 cudf-cu11 || {
            echo "✗ pip install failed for CUDA 11.x packages"
            exit 1
        }
    else
        echo "⚠ Unsupported CUDA version for pip install"
        echo "  Try: pip install cuml cudf"
        pip install --no-cache-dir cuml cudf || {
            echo "✗ Generic pip install failed"
            exit 1
        }
    fi
    
    echo "✓ CUML/cuDF installed via pip"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import cuml; print('✓ cuml imported successfully')" || {
    echo "✗ cuml import failed"
    exit 1
}

python -c "import cudf; print('✓ cudf imported successfully')" || {
    echo "✗ cudf import failed"
    exit 1
}

echo ""
echo "=========================================="
echo "✅ CUML/cuDF Installation Complete"
echo "=========================================="
echo ""
echo "Test GPU availability:"
echo "  python -c \"from lib.utils.gpu_utils import check_gpu_availability; print(check_gpu_availability())\""

