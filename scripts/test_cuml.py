#!/usr/bin/env python3
"""
Test CUML and cuDF installation
"""
import sys

print("Testing CUML and cuDF installation...")
print("=" * 60)

# Test CUML
try:
    import cuml
    print("✓ cuml imported successfully")
    print(f"  Version: {cuml.__version__ if hasattr(cuml, '__version__') else 'unknown'}")
    
    # Test a simple CUML operation
    from cuml.preprocessing import StandardScaler
    import numpy as np
    scaler = StandardScaler()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler.fit(X)
    print("✓ CUML StandardScaler works")
except Exception as e:
    print(f"✗ cuml failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test cuDF
try:
    import cudf
    print("✓ cudf imported successfully")
    print(f"  Version: {cudf.__version__ if hasattr(cudf, '__version__') else 'unknown'}")
    
    # Test a simple cuDF operation
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print(f"✓ cuDF DataFrame created: shape {df.shape}")
except Exception as e:
    print(f"✗ cudf failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test GPU availability
try:
    from lib.utils.gpu_utils import check_gpu_availability
    gpu_available = check_gpu_availability()
    print(f"✓ GPU check: {gpu_available}")
except Exception as e:
    print(f"⚠ GPU check failed: {e}")

# Test PyTorch (may have version conflicts but might still work)
try:
    import torch
    print(f"✓ torch imported (version: {torch.__version__})")
    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA not available in PyTorch (may be due to version conflicts)")
except Exception as e:
    print(f"⚠ torch import issue: {e}")
    print("  This is OK - neural network model won't work but others will")

print("=" * 60)
print("✅ CUML and cuDF are working!")
print("⚠ PyTorch may have version conflicts but other models will work")

