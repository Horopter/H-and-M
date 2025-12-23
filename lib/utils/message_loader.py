"""
Load and format messages from XML file.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, Any

_MESSAGES_CACHE: Optional[ET.Element] = None


def _load_messages() -> ET.Element:
    """Load messages XML file."""
    global _MESSAGES_CACHE
    if _MESSAGES_CACHE is not None:
        return _MESSAGES_CACHE
    
    # Find messages.xml relative to this file
    current_dir = Path(__file__).parent.parent
    messages_path = current_dir / "messages.xml"
    
    if not messages_path.exists():
        # Fallback: return empty element
        return ET.Element("messages")
    
    tree = ET.parse(messages_path)
    _MESSAGES_CACHE = tree.getroot()
    return _MESSAGES_CACHE


def get_message(category: str, key: str, **kwargs) -> str:
    """
    Get formatted message from XML.
    
    Args:
        category: Message category (e.g., 'stage', 'data', 'error')
        key: Message key (e.g., 'loading_csv', 'complete')
        **kwargs: Format parameters
        
    Returns:
        Formatted message string
    """
    messages = _load_messages()
    category_elem = messages.find(category)
    
    if category_elem is None:
        return f"[{category}.{key}]"
    
    # Handle nested keys (e.g., 'stage_1.title')
    if '.' in key:
        parts = key.split('.')
        elem = category_elem
        for part in parts:
            elem = elem.find(part) if elem is not None else None
            if elem is None:
                break
    else:
        elem = category_elem.find(key)
    
    if elem is None or elem.text is None:
        return f"[{category}.{key}]"
    
    message = elem.text.strip()
    
    # Format with kwargs
    try:
        return message.format(**kwargs)
    except (KeyError, ValueError) as e:
        # Missing format key or invalid format - return unformatted message
        return message


def get_message_safe(category: str, key: str, default: str = "", **kwargs) -> str:
    """
    Get formatted message with fallback to default.
    
    Args:
        category: Message category
        key: Message key
        default: Default message if not found
        **kwargs: Format parameters
        
    Returns:
        Formatted message string
    """
    try:
        msg = get_message(category, key, **kwargs)
        if msg.startswith('[') and msg.endswith(']'):
            try:
                return default.format(**kwargs) if default else msg
            except (KeyError, ValueError):
                return default if default else msg
        return msg
    except (KeyError, ValueError, AttributeError) as e:
        # Format error or missing attribute - use default
        try:
            return default.format(**kwargs) if default else f"[{category}.{key}]"
        except (KeyError, ValueError):
            return default if default else f"[{category}.{key}]"

