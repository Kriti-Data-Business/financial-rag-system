# Tests for models
"""
Tests for ML models.
"""
import pytest
import numpy as np
from src.models.embeddings import EmbeddingModel


def test_embedding_model_initialization():
    """Test embedding model initialization."""
    model = EmbeddingModel()
    assert model.model is not None
    assert model.embedding_dim > 0


def test_embed_text_single():
    """Test embedding single text."""
    model = EmbeddingModel()
    text = "This is a test sentence."
    
    embedding = model.embed_text(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == model.embedding_dim


def test_embed_text_multiple():
    """Test embedding multiple texts."""
    model = EmbeddingModel()
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    
    embeddings = model.embed_text(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == model.embedding_dim


def test_embed_query():
    """Test embedding query."""
    model = EmbeddingModel()
    query = "What is machine learning?"
    
    embedding = model.embed_query(query)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1
    assert embedding.shape[0] == model.embedding_dim


def test_embedding_similarity():
    """Test that similar texts have similar embeddings."""
    model = EmbeddingModel()
    
    text1 = "Machine learning is great."
    text2 = "Machine learning is awesome."
    text3 = "I like pizza and pasta."
    
    emb1 = model.embed_query(text1)
    emb2 = model.embed_query(text2)
    emb3 = model.embed_query(text3)
    
    # Cosine similarity
    sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    # Similar texts should be more similar than dissimilar texts
    assert sim_12 > sim_13