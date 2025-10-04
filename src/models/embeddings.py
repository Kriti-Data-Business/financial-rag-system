"""Embedding model for converting text to vectors."""
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingModel:
    """Wrapper for sentence transformer embedding models."""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding model."""
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
    
    def embed_text(self, text):
        """Generate embeddings for text."""
        if isinstance(text, str):
            text = [text]
        
        return self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    def embed_query(self, query: str):
        """Generate embedding for a query."""
        return self.embed_text(query)[0]
    
    def embed_documents(self, documents):
        """Generate embeddings for multiple documents."""
        return self.embed_text(documents)