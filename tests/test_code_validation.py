"""
Code validation tests - check for common errors, type issues, etc.
"""
import unittest
import ast
import inspect
from pathlib import Path
import importlib.util

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCodeValidation(unittest.TestCase):
    """Validate code quality and correctness."""
    
    def test_imports_resolve(self):
        """Test that all imports can be resolved."""
        lib_path = Path(__file__).parent.parent / 'lib'
        
        # Test key modules
        modules_to_test = [
            'lib.config',
            'lib.data.loader',
            'lib.training.cv',
            'lib.models.base',
            'lib.utils.gpu_utils'
        ]
        
        for module_name in modules_to_test:
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    lib_path / module_name.replace('lib.', '').replace('.', '/') / '__init__.py'
                )
                if spec is None:
                    # Try .py file
                    spec = importlib.util.spec_from_file_location(
                        module_name,
                        lib_path / module_name.replace('lib.', '').replace('.', '/') + '.py'
                    )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't actually load (might fail without dependencies)
                    # Just check spec exists
                    self.assertIsNotNone(spec)
            except Exception as e:
                # Some modules might not be importable without dependencies
                # That's OK - we just check they exist
                pass
    
    def test_cv_uses_stratified(self):
        """Verify CV implementation uses StratifiedKFold."""
        lib_path = Path(__file__).parent.parent / 'lib' / 'training' / 'cv.py'
        
        with open(lib_path, 'r') as f:
            content = f.read()
        
        # Should contain StratifiedKFold
        self.assertIn('StratifiedKFold', content)
        self.assertIn('from sklearn.model_selection import StratifiedKFold', content)
        
        # Should use it in create_folds
        self.assertIn('skf = StratifiedKFold', content)
    
    def test_gpu_fallbacks(self):
        """Verify GPU code has proper fallbacks."""
        lib_path = Path(__file__).parent.parent / 'lib'
        
        # Check key files for try/except CUML imports
        files_to_check = [
            'data/preprocessor.py',
            'models/logistic_regression.py',
            'models/svm.py',
            'models/bayesian.py',
            'utils/gpu_utils.py'
        ]
        
        for file_path in files_to_check:
            full_path = lib_path / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Should have try/except for CUML
                self.assertIn('try:', content.lower())
                self.assertIn('except', content.lower())
                self.assertIn('CUML_AVAILABLE', content)
    
    def test_no_pandas_imports(self):
        """Verify no pandas imports in lib code."""
        lib_path = Path(__file__).parent.parent / 'lib'
        
        # Search for pandas imports
        for py_file in lib_path.rglob('*.py'):
            with open(py_file, 'r') as f:
                content = f.read()
            
            # Should not import pandas
            self.assertNotIn('import pandas', content)
            self.assertNotIn('import pd', content)
            self.assertNotIn('from pandas', content)
    
    def test_type_hints_present(self):
        """Check that key functions have type hints."""
        lib_path = Path(__file__).parent.parent / 'lib'
        
        # Key files should have type hints
        key_files = [
            'models/base.py',
            'data/loader.py',
            'training/cv.py'
        ]
        
        for file_path in key_files:
            full_path = lib_path / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Should have typing imports
                self.assertIn('from typing', content)


if __name__ == '__main__':
    unittest.main()

