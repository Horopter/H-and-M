"""
Test to validate all imports are correct and no NameError occurs.
This catches missing imports like collect_after_operation.
"""
import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestImportValidation(unittest.TestCase):
    """Test that all modules can be imported and all functions are accessible."""
    
    def test_gc_utils_imports(self):
        """Test that gc_utils functions can be imported."""
        from lib.utils.gc_utils import (
            collect_after_chunk,
            collect_after_operation,
            get_gc_manager,
            AggressiveGC
        )
        self.assertTrue(callable(collect_after_chunk))
        self.assertTrue(callable(collect_after_operation))
        self.assertTrue(callable(get_gc_manager))
    
    def test_stage_pipeline_imports(self):
        """Test that stage_pipeline imports all required functions."""
        # This will fail if collect_after_operation is not imported
        from lib.pipeline.stage_pipeline import StagePipeline
        import inspect
        
        # Check that collect_after_operation is available in the module namespace
        import lib.pipeline.stage_pipeline as sp_module
        self.assertTrue(hasattr(sp_module, 'collect_after_operation') or 
                       'collect_after_operation' in dir(sp_module))
    
    def test_rfe_imports(self):
        """Test that rfe imports all required functions."""
        from lib.utils.rfe import RecursiveFeatureElimination
        import lib.utils.rfe as rfe_module
        
        # Check that collect_after_operation is available
        self.assertTrue(hasattr(rfe_module, 'collect_after_operation') or 
                       'collect_after_operation' in dir(rfe_module))
    
    def test_cv_imports(self):
        """Test that cv imports all required functions."""
        from lib.training.cv import CrossValidator
        import lib.training.cv as cv_module
        
        # Check that collect_after_operation is available
        self.assertTrue(hasattr(cv_module, 'collect_after_operation') or 
                       'collect_after_operation' in dir(cv_module))
    
    def test_embeddings_imports(self):
        """Test that embeddings imports all required functions."""
        from lib.features.embeddings import EmbeddingExtractor
        import lib.features.embeddings as emb_module
        
        # Check that collect_after_operation is available
        self.assertTrue(hasattr(emb_module, 'collect_after_operation') or 
                       'collect_after_operation' in dir(emb_module))
    
    def test_all_modules_importable(self):
        """Test that all critical modules can be imported without errors."""
        modules_to_test = [
            'lib.config',
            'lib.data.loader',
            'lib.data.preprocessing_pipeline',
            'lib.features.nlp_features',
            'lib.features.embeddings',
            'lib.features.feature_union',
            'lib.models.neural_network',
            'lib.pipeline.stage_pipeline',
            'lib.training.cv',
            'lib.training.grid_search',
            'lib.utils.rfe',
            'lib.utils.gc_utils',
        ]
        
        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except (ImportError, NameError, AttributeError) as e:
                    self.fail(f"Failed to import {module_name}: {e}")
    
    def test_gc_functions_callable(self):
        """Test that GC functions can be called without errors."""
        from lib.utils.gc_utils import collect_after_chunk, collect_after_operation
        
        # These should not raise NameError
        try:
            collect_after_chunk(0)
            collect_after_operation("test")
        except NameError as e:
            self.fail(f"GC functions not callable: {e}")


if __name__ == '__main__':
    unittest.main()

