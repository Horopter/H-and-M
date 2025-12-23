#!/usr/bin/env python3
"""
Pre-flight validation script for the ML pipeline.
Checks imports, data availability, and configuration before expensive training.
"""
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """Check that all critical imports work."""
    print("Checking imports...")
    errors = []
    
    # Core dependencies - test one at a time to catch crashes
    try:
        import polars as pl
        print("✓ polars")
        sys.stdout.flush()
    except ImportError as e:
        errors.append(f"polars: {e}")
        print(f"✗ polars: {e}")
    except Exception as e:
        errors.append(f"polars: CRASH - {type(e).__name__}: {e}")
        print(f"✗ polars: CRASH - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        import numpy as np
        print("✓ numpy")
        sys.stdout.flush()
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print(f"✗ numpy: {e}")
    except Exception as e:
        errors.append(f"numpy: CRASH - {type(e).__name__}: {e}")
        print(f"✗ numpy: CRASH - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        import sklearn
        print("✓ scikit-learn")
        sys.stdout.flush()
    except ImportError as e:
        errors.append(f"scikit-learn: {e}")
        print(f"✗ scikit-learn: {e}")
    except Exception as e:
        errors.append(f"scikit-learn: CRASH - {type(e).__name__}: {e}")
        print(f"✗ scikit-learn: CRASH - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        import scipy
        print("✓ scipy")
        sys.stdout.flush()
    except ImportError as e:
        errors.append(f"scipy: {e}")
        print(f"✗ scipy: {e}")
    except Exception as e:
        errors.append(f"scipy: CRASH - {type(e).__name__}: {e}")
        print(f"✗ scipy: CRASH - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Optional GPU dependencies (should not fail)
    try:
        import cuml
        print("✓ cuml (GPU acceleration available)")
    except ImportError:
        print("⚠ cuml not available (will use CPU fallback)")
    
    try:
        import cudf
        print("✓ cudf (GPU acceleration available)")
    except ImportError:
        print("⚠ cudf not available (will use CPU fallback)")
    
    # Optional ML dependencies
    try:
        import xgboost
        print("✓ xgboost")
    except ImportError:
        print("⚠ xgboost not available")
    
    try:
        import torch
        print("✓ torch")
    except ImportError:
        print("⚠ torch not available")
    
    try:
        import mlflow
        print("✓ mlflow")
    except ImportError:
        print("⚠ mlflow not available (tracking disabled)")
    
    try:
        import duckdb
        print("✓ duckdb")
    except ImportError:
        print("⚠ duckdb not available (reporting disabled)")
    
    return errors

def check_lib_imports():
    """Check that lib modules can be imported."""
    print("\nChecking lib module imports...")
    errors = []
    
    modules = [
        'lib.config',
        'lib.data.loader',
        'lib.training.cv',
        'lib.models.base',
        'lib.utils.gpu_utils'
    ]
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            errors.append(f"{module_name}: {e}")
            print(f"✗ {module_name}: {e}")
    
    return errors

def check_data_files(data_path='data/'):
    """Check that data files exist."""
    print(f"\nChecking data files in {data_path}...")
    errors = []
    
    data_path = Path(data_path)
    required_files = ['train.csv', 'val.csv', 'test.csv']
    
    for filename in required_files:
        filepath = data_path / filename
        if filepath.exists():
            print(f"✓ {filename}")
        else:
            errors.append(f"Missing: {filepath}")
            print(f"✗ {filename} not found at {filepath}")
    
    return errors

def check_cv_stratification():
    """Verify CV uses StratifiedKFold."""
    print("\nChecking CV stratification...")
    try:
        from lib.training.cv import CrossValidator
        from lib.config import Config
        
        # Check source code
        cv_file = Path(__file__).parent.parent / 'lib' / 'training' / 'cv.py'
        with open(cv_file, 'r') as f:
            content = f.read()
        
        if 'StratifiedKFold' in content and 'sklearn.model_selection' in content:
            print("✓ CV uses StratifiedKFold")
            return []
        else:
            return ["CV does not use StratifiedKFold"]
    except Exception as e:
        return [f"Could not verify CV: {e}"]

def check_gpu_fallbacks():
    """Verify GPU code has proper fallbacks."""
    print("\nChecking GPU fallbacks...")
    errors = []
    
    lib_path = Path(__file__).parent.parent / 'lib'
    key_files = [
        'data/preprocessor.py',
        'models/logistic_regression.py',
        'models/svm.py',
        'models/bayesian.py'
    ]
    
    for file_path in key_files:
        full_path = lib_path / file_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
            
            if 'CUML_AVAILABLE' in content and 'except ImportError' in content:
                print(f"✓ {file_path} has GPU fallback")
            else:
                errors.append(f"{file_path} missing GPU fallback")
                print(f"✗ {file_path} missing GPU fallback")
    
    return errors

def main():
    """Run all validation checks."""
    print("=" * 80)
    print("ML PIPELINE VALIDATION")
    print("=" * 80)
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    print("=" * 80)
    print()
    
    all_errors = []
    
    # Run checks with error handling
    try:
        all_errors.extend(check_imports())
    except Exception as e:
        print(f"✗ FATAL: check_imports() crashed: {e}")
        import traceback
        traceback.print_exc()
        all_errors.append(f"check_imports() crashed: {e}")
        return 1
    
    try:
        all_errors.extend(check_lib_imports())
    except Exception as e:
        print(f"✗ FATAL: check_lib_imports() crashed: {e}")
        import traceback
        traceback.print_exc()
        all_errors.append(f"check_lib_imports() crashed: {e}")
        return 1
    
    try:
        all_errors.extend(check_data_files())
    except Exception as e:
        print(f"✗ FATAL: check_data_files() crashed: {e}")
        import traceback
        traceback.print_exc()
        all_errors.append(f"check_data_files() crashed: {e}")
    
    try:
        all_errors.extend(check_cv_stratification())
    except Exception as e:
        print(f"✗ FATAL: check_cv_stratification() crashed: {e}")
        import traceback
        traceback.print_exc()
        all_errors.append(f"check_cv_stratification() crashed: {e}")
    
    try:
        all_errors.extend(check_gpu_fallbacks())
    except Exception as e:
        print(f"✗ FATAL: check_gpu_fallbacks() crashed: {e}")
        import traceback
        traceback.print_exc()
        all_errors.append(f"check_gpu_fallbacks() crashed: {e}")
    
    print("\n" + "=" * 80)
    if all_errors:
        print("VALIDATION FAILED")
        print("=" * 80)
        print("Errors found:")
        for error in all_errors:
            print(f"  - {error}")
        return 1
    else:
        print("✅ VALIDATION PASSED")
        print("=" * 80)
        print("All checks passed. Pipeline is ready to run.")
        return 0

if __name__ == '__main__':
    sys.exit(main())

