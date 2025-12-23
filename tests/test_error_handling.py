"""
Unit tests for error handling and edge cases.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import Config
from lib.utils.message_loader import get_message, get_message_safe
from lib.constants import DEFAULT_CHUNK_SIZE


class TestErrorHandling(unittest.TestCase):
    """Test error handling."""
    
    def test_message_loader_missing_category(self):
        """Test message loader handles missing category."""
        msg = get_message('nonexistent_category', 'key')
        self.assertEqual(msg, '[nonexistent_category.key]')
    
    def test_message_loader_missing_key(self):
        """Test message loader handles missing key."""
        msg = get_message('stage', 'nonexistent_key')
        self.assertEqual(msg, '[stage.nonexistent_key]')
    
    def test_message_loader_formatting_error(self):
        """Test message loader handles formatting errors."""
        # Missing format parameter
        msg = get_message('data', 'loaded_rows')
        self.assertIsNotNone(msg)
        # Should return unformatted message if format fails
        msg2 = get_message_safe('data', 'loaded_rows', default='Default: {rows}')
        self.assertIsNotNone(msg2)
    
    def test_message_loader_safe_with_default(self):
        """Test get_message_safe with default."""
        msg = get_message_safe('nonexistent', 'key', default='Default message')
        self.assertEqual(msg, 'Default message')
    
    def test_chunk_size_default(self):
        """Test chunk size default is 1000."""
        config = Config()
        self.assertEqual(config.chunk_size, DEFAULT_CHUNK_SIZE)
        self.assertEqual(config.chunk_size, 1000)
    
    def test_config_handles_missing_env(self):
        """Test config handles missing environment variables."""
        import os
        original = os.environ.get('CHUNK_SIZE')
        try:
            if 'CHUNK_SIZE' in os.environ:
                del os.environ['CHUNK_SIZE']
            config = Config()
            self.assertEqual(config.chunk_size, DEFAULT_CHUNK_SIZE)
        finally:
            if original:
                os.environ['CHUNK_SIZE'] = original


if __name__ == '__main__':
    unittest.main()

