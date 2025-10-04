# Tests for data processing
"""
Tests for data processing modules.
"""
import pytest
from src.data.processors.text_processor import TextProcessor
from src.utils.helpers import chunk_text, clean_text


def test_text_processor_initialization():
    """Test TextProcessor initialization."""
    processor = TextProcessor(chunk_size=500, chunk_overlap=50)
    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 50


def test_chunk_text():
    """Test text chunking."""
    text = "This is a test. " * 100  # Create long text
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some overflow


def test_clean_text():
    """Test text cleaning."""
    dirty_text = "  This   has   extra   spaces.  "
    clean = clean_text(dirty_text)
    
    assert clean == "This has extra spaces."


def test_process_document():
    """Test document processing."""
    processor = TextProcessor(chunk_size=100, chunk_overlap=20)
    
    document = {
        'content': "This is test content. " * 50,
        'filename': 'test.txt',
        'filepath': '/path/to/test.txt',
        'file_type': '.txt'
    }
    
    chunks = processor.process_document(document)
    
    assert len(chunks) > 0
    assert all('id' in chunk for chunk in chunks)
    assert all('content' in chunk for chunk in chunks)
    assert all(chunk['source_document'] == 'test.txt' for chunk in chunks)


def test_extract_keywords():
    """Test keyword extraction."""
    processor = TextProcessor()
    
    text = "Machine learning is a subset of artificial intelligence. " \
           "Machine learning algorithms build models based on sample data."
    
    keywords = processor.extract_keywords(text, top_k=3)
    
    assert len(keywords) <= 3
    assert isinstance(keywords, list)