#!/usr/bin/env python3
"""
Diagnostic script to identify which package import causes core dumps.
Run this to find the problematic package.
"""
import sys
import os
import traceback
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_import(module_name, import_statement=None):
    """Test importing a module and catch any crashes."""
    if import_statement is None:
        import_statement = f"import {module_name}"
    
    try:
        print(f"Testing: {import_statement}...", end=" ", flush=True)
        exec(import_statement)
        print("✓ OK")
        return True
    except ImportError as e:
        print(f"✗ ImportError: {e}")
        return False
    except Exception as e:
        print(f"✗ Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    except SystemExit:
        print("✗ SystemExit (unexpected)")
        return False
    except:
        print("✗ UNKNOWN ERROR (possible core dump)")
        traceback.print_exc()
        return False

def main():
    """Run diagnostic tests."""
    print("=" * 80)
    print("DIAGNOSTIC: Testing Package Imports")
    print("=" * 80)
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 80)
    print()
    
    # Core packages (test one by one)
    print("Testing CORE packages:")
    print("-" * 80)
    core_packages = [
        ("sys", "import sys"),
        ("os", "import os"),
        ("pathlib", "from pathlib import Path"),
        ("polars", "import polars as pl"),
        ("numpy", "import numpy as np"),
        ("sklearn", "import sklearn"),
        ("scipy", "import scipy"),
        ("joblib", "import joblib"),
        ("pyarrow", "import pyarrow"),
    ]
    
    for name, import_stmt in core_packages:
        if not test_import(name, import_stmt):
            print(f"\n⚠ STOPPING: {name} import failed or crashed")
            return 1
        sys.stdout.flush()
    
    print()
    print("Testing OPTIONAL packages:")
    print("-" * 80)
    
    optional_packages = [
        ("xgboost", "import xgboost"),
        ("mlflow", "import mlflow"),
        ("duckdb", "import duckdb"),
        ("umap", "import umap"),
        ("statsmodels", "import statsmodels"),
        ("matplotlib", "import matplotlib"),
        ("seaborn", "import seaborn"),
        ("tqdm", "import tqdm"),
        ("gensim", "import gensim"),
        ("transformers", "import transformers"),
        ("sentence_transformers", "import sentence_transformers"),
    ]
    
    for name, import_stmt in optional_packages:
        test_import(name, import_stmt)
        sys.stdout.flush()
    
    print()
    print("Testing LIB modules:")
    print("-" * 80)
    
    lib_modules = [
        ("lib.config", "from lib.config import Config"),
        ("lib.data.loader", "from lib.data.loader import DataLoader"),
        ("lib.training.cv", "from lib.training.cv import CrossValidator"),
        ("lib.models.base", "from lib.models.base import BaseModel"),
        ("lib.utils.gpu_utils", "from lib.utils.gpu_utils import check_gpu_availability"),
    ]
    
    for name, import_stmt in lib_modules:
        if not test_import(name, import_stmt):
            print(f"\n⚠ WARNING: {name} import failed")
        sys.stdout.flush()
    
    print()
    print("=" * 80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 80)
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

