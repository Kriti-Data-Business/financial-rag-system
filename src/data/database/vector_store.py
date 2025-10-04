"""Vector database for storing and retrieving document embeddings."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
from src.utils.config import settings
from src.utils.logger import setup_logger
from src.models.embeddings import EmbeddingModel

logger = setup_logger(__name__)


class VectorStore:
    """Vector store using ChromaDB for similarity search."""
    
    def __init__(self):
        """Initialize vector store."""
        self.collection_name = settings.collection_name
        self.persist_directory = settings.vector_db_path
        
        logger.info(f"Initializing vector store: {self.collection_name}")
        
        self.client = chromadb.Client(ChromaSettings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        
        self.embedding_model = EmbeddingModel()
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector store ready. Documents: {self.collection.count()}")
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            ids = [doc['id'] for doc in batch]
            texts = [doc['content'] for doc in batch]
            embeddings = self.embedding_model.embed_documents(texts)
            
            metadatas = [{
                'source': doc.get('source_document', 'unknown')
            } for doc in batch]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
        
        logger.info(f"Successfully added {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            formatted_results = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'score': 1 - results['distances'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        return {
            'name': self.collection_name,
            'document_count': self.collection.count(),
            'persist_directory': self.persist_directory
        }