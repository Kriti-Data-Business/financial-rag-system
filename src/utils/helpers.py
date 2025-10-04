# Misc helpers
"""
Helper utility functions.
"""
import hashlib
import re
from typing import List, Any
from pathlib import Path


def generate_id(text: str) -> str:
    """
    Generate a unique ID from text using MD5 hash.
    
    Args:
        text: Input text
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.md5(text.encode()).hexdigest()


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\']', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_length:
            last_period = chunk.rfind('.')
            last_question = chunk.rfind('?')
            last_exclamation = chunk.rfind('!')
            
            break_point = max(last_period, last_question, last_exclamation)
            
            if break_point > chunk_size * 0.5:  # At least 50% through
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def format_sources(sources: List[dict]) -> str:
    """
    Format source documents for display.
    
    Args:
        sources: List of source documents
        
    Returns:
        Formatted string
    """
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        formatted.append(f"Source {i}:")
        formatted.append(f"  Document: {source.get('document_name', 'Unknown')}")
        formatted.append(f"  Score: {source.get('score', 0):.4f}")
        formatted.append(f"  Content: {source.get('content', '')[:200]}...")
        formatted.append("")
    
    return "\n".join(formatted)


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """
    Validate file extension.
    
    Args:
        filename: Name of the file
        allowed_types: List of allowed extensions (e.g., ['.pdf', '.txt'])
        
    Returns:
        True if valid, False otherwise
    """
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_types


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."