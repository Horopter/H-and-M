#!/bin/bash
#
# Quick validation check before running full pipeline
# Run this locally before submitting SLURM job
#

set -euo pipefail

echo "=========================================="
echo "QUICK VALIDATION CHECK"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "✗ ERROR: python3 not found"
    exit 1
fi
echo "✓ python3 found: $(python3 --version)"

# Check data files
if [ ! -f "data/train.csv" ]; then
    echo "✗ ERROR: data/train.csv not found"
    exit 1
fi
echo "✓ data/train.csv exists"

if [ ! -f "data/val.csv" ]; then
    echo "✗ ERROR: data/val.csv not found"
    exit 1
fi
echo "✓ data/val.csv exists"

# Run validation script
if [ -f "scripts/validate_pipeline.py" ]; then
    echo ""
    echo "Running validation script..."
    python3 scripts/validate_pipeline.py
    VALIDATION_EXIT=$?
    
    if [ $VALIDATION_EXIT -ne 0 ]; then
        echo ""
        echo "✗ VALIDATION FAILED"
        echo "Fix errors before submitting SLURM job"
        exit 1
    fi
else
    echo "⚠ WARNING: scripts/validate_pipeline.py not found"
fi

echo ""
echo "=========================================="
echo "✅ QUICK CHECK PASSED"
echo "=========================================="
echo "Ready to submit SLURM job:"
echo "  sbatch scripts/slurm_jobs/slurm_ml_training.sh"

