"""
Unit tests for garbage collection utilities.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils.gc_utils import (
    AggressiveGC,
    get_gc_manager,
    collect_after_chunk,
    collect_after_operation,
    memory_efficient_chunked_processing
)


class TestGCUtils(unittest.TestCase):
    """Test garbage collection utilities."""
    
    def test_gc_manager_creation(self):
        """Test GC manager creation."""
        gc_manager = AggressiveGC(enabled=True, verbose=False)
        self.assertIsNotNone(gc_manager)
        self.assertTrue(gc_manager.enabled)
    
    def test_collect(self):
        """Test basic collection."""
        gc_manager = AggressiveGC(enabled=True, verbose=False)
        result = gc_manager.collect(generation=2, aggressive=False)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
    
    def test_collect_all(self):
        """Test collecting all generations."""
        gc_manager = AggressiveGC(enabled=True, verbose=False)
        result = gc_manager.collect_all()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
    
    def test_get_gc_manager_singleton(self):
        """Test global GC manager singleton."""
        manager1 = get_gc_manager()
        manager2 = get_gc_manager()
        self.assertIs(manager1, manager2)
    
    def test_collect_after_chunk(self):
        """Test convenience function."""
        # Should not raise
        collect_after_chunk(0, aggressive=True)
        collect_after_chunk(None, aggressive=False)
    
    def test_collect_after_operation(self):
        """Test operation collection."""
        # Should not raise
        collect_after_operation("test_operation", aggressive=True)
    
    def test_memory_efficient_chunked_processing(self):
        """Test chunked processing with GC."""
        def process_chunk(chunk):
            return [x * 2 for x in chunk]
        
        items = list(range(100))
        results = memory_efficient_chunked_processing(
            items,
            process_chunk,
            chunk_size=10,
            collect_after_each=True,
            aggressive_gc=True
        )
        
        self.assertEqual(len(results), 10)  # 100 items / 10 chunk_size = 10 chunks
        self.assertEqual(len(results[0]), 10)
    
    def test_get_memory_stats(self):
        """Test memory statistics."""
        gc_manager = AggressiveGC(enabled=True, verbose=False)
        stats = gc_manager.get_memory_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('gc_counts', stats)
        self.assertIn('gc_threshold', stats)
        self.assertIn('gc_stats', stats)


if __name__ == '__main__':
    unittest.main()

