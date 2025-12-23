"""
Unit tests for message loader.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.utils.message_loader import get_message, get_message_safe


class TestMessageLoader(unittest.TestCase):
    """Test message loader."""
    
    def test_get_message_exists(self):
        """Test getting existing message."""
        msg = get_message('stage', 'start')
        self.assertIsNotNone(msg)
        self.assertNotEqual(msg, '[stage.start]')
    
    def test_get_message_with_formatting(self):
        """Test message formatting."""
        msg = get_message('data', 'loaded_rows', rows=100, columns=5)
        self.assertIsNotNone(msg)
        self.assertIn('100', msg)
        self.assertIn('5', msg)
    
    def test_get_message_nested(self):
        """Test nested message keys."""
        msg = get_message('stage', 'stage_1.title')
        self.assertIsNotNone(msg)
        self.assertNotEqual(msg, '[stage.stage_1.title]')
    
    def test_get_message_not_found(self):
        """Test message not found returns placeholder."""
        msg = get_message('nonexistent', 'key')
        self.assertEqual(msg, '[nonexistent.key]')
    
    def test_get_message_safe_with_default(self):
        """Test get_message_safe with default."""
        msg = get_message_safe('nonexistent', 'key', default='Default message')
        self.assertEqual(msg, 'Default message')
    
    def test_get_message_safe_with_formatting(self):
        """Test get_message_safe with formatting."""
        msg = get_message_safe('data', 'loaded_datasets', 
                               default='Train: {train}, Val: {val}',
                               train=100, val=50)
        self.assertIn('100', msg)
        self.assertIn('50', msg)


if __name__ == '__main__':
    unittest.main()

