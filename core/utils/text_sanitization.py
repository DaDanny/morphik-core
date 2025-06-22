"""
Text sanitization utilities for PostgreSQL database compatibility.

This module provides functions to sanitize text and metadata to remove
null bytes and other characters that PostgreSQL cannot handle.
"""

import re
from typing import Any


def sanitize_text_for_db(text: str) -> str:
    """Remove null bytes and other characters that PostgreSQL can't handle.
    
    Args:
        text: The text string to sanitize
        
    Returns:
        str: Sanitized text with problematic characters removed
    """
    if not text:
        return text
    
    # Remove null bytes and other control characters except newlines and tabs
    sanitized = text.replace('\u0000', '')  # Remove null bytes
    # Remove other problematic control characters (but keep \n, \t, \r)
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
    return sanitized


def sanitize_metadata_for_db(obj: Any) -> Any:
    """Recursively sanitize metadata objects to remove null bytes and problematic characters.
    
    This function works with nested dictionaries, lists, and strings to ensure
    all text values are properly sanitized for PostgreSQL storage.
    
    Args:
        obj: The object to sanitize (can be str, dict, list, or other types)
        
    Returns:
        Any: Sanitized object with all text values cleaned
    """
    if isinstance(obj, str):
        return sanitize_text_for_db(obj)
    elif isinstance(obj, dict):
        return {key: sanitize_metadata_for_db(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_metadata_for_db(item) for item in obj]
    else:
        # For non-string types (int, float, bool, None), return as-is
        return obj 
