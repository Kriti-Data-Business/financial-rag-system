# src/data/database/chroma_manager.py
# ChromaDB Manager for Australian Financial RAG System

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import uuid
import json

from ...utils.config import Config
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ChromaDBManager:
    """
    Manages ChromaDB vector database for document storage and retrieval.
    Handles Australian financial documents with metadata filtering.
    """
    
    def __init__(self, config: Config):
        """Initialize ChromaDB manager with configuration."""
        self.config = config
        
        # Database configuration
        db_path = config.get('database.path', 'data/processed/indexes/chroma_db')
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB settings
        self.settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.db_path),
            anonymized_telemetry=False
        )
        
        # Initialize client and collection
        self.client = None
        self.collection = None
        self.collection_name = config.get('database.collection_name', 'financial_knowledge_base')
        
        self._initialize_client()
        logger.info(f"ChromaDBManager initialized with collection: {self.collection_name}")
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=self.settings
            )
            
            # Try to get existing collection or create new one
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Australian Financial RAG Knowledge Base"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents with 'content' and 'metadata' keys
            batch_size: Size of batches for processing
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to ChromaDB collection")
        
        try:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self._add_document_batch(batch)
                logger.debug(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def _add_document_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Add a batch of documents to ChromaDB."""
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for doc in batch:
            # Generate unique ID if not provided
            doc_id = doc.get('id', str(uuid.uuid4()))
            ids.append(doc_id)
            
            # Extract content
            content = doc.get('content', '')
            if not content:
                logger.warning(f"Empty content for document {doc_id}")
                content = "No content available"
            documents.append(content)
            
            # Extract and clean metadata
            metadata = doc.get('metadata', {})
            # Ensure all metadata values are strings (ChromaDB requirement)
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    cleaned_metadata[key] = str(value)
            metadatas.append(cleaned_metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               where: Optional[Dict] = None,
               where_document: Optional[Dict] = None) -> Dict[str, List[Any]]:
        """
        Perform semantic search on the collection.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            Dictionary with search results
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}
        
        try:
            # Perform search with optional filters
            search_kwargs = {
                'query_texts': [query],
                'n_results': min(n_results, 50),  # Cap at 50 results
                'include': ['documents', 'metadatas', 'distances']
            }
            
            if where:
                search_kwargs['where'] = where
            if where_document:
                search_kwargs['where_document'] = where_document
            
            results = self.collection.query(**search_kwargs)
            
            logger.debug(f"Search query '{query[:50]}...' returned {len(results['documents'][0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}
    
    def search_by_category(self, 
                          query: str, 
                          category: str, 
                          n_results: int = 5) -> Dict[str, List[Any]]:
        """
        Search within a specific category.
        
        Args:
            query: Search query string
            category: Category to filter by
            n_results: Number of results to return
            
        Returns:
            Dictionary with filtered search results
        """
        where_filter = {"category": category}
        return self.search(query, n_results, where=where_filter)
    
    def search_financial_topic(self, 
                              query: str, 
                              topics: List[str] = None,
                              n_results: int = 5) -> Dict[str, List[Any]]:
        """
        Search for specific financial topics.
        
        Args:
            query: Search query string
            topics: List of financial topics to include
            n_results: Number of results to return
            
        Returns:
            Dictionary with topic-filtered results
        """
        if topics:
            # ChromaDB supports $in operator for metadata filtering
            where_filter = {"category": {"$in": topics}}
            return self.search(query, n_results, where=where_filter)
        else:
            return self.search(query, n_results)
    
    def get_documents_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all documents from a specific source.
        
        Args:
            source: Source identifier (e.g., 'ABS', 'ASX', 'RBA')
            limit: Maximum number of documents to return
            
        Returns:
            List of documents from the specified source
        """
        try:
            results = self.collection.get(
                where={"source": source},
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            # Convert to list of dictionaries
            documents = []
            for i, doc_id in enumerate(results['ids']):
                documents.append({
                    'id': doc_id,
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            logger.info(f"Retrieved {len(documents)} documents from source: {source}")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents from source {source}: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def update_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Update a document's content and metadata.
        
        Args:
            document_id: ID of document to update
            content: New content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean metadata
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    cleaned_metadata[key] = str(value)
            
            self.collection.update(
                ids=[document_id],
                documents=[content],
                metadatas=[cleaned_metadata]
            )
            
            logger.info(f"Updated document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection metadata and statistics
        """
        try:
            # Get collection count
            count = self.collection.count()
            
            # Get sample documents to analyze categories and sources
            sample_docs = self.collection.get(limit=100, include=['metadatas'])
            
            # Analyze metadata
            categories = set()
            sources = set()
            data_types = set()
            
            for metadata in sample_docs['metadatas']:
                if 'category' in metadata:
                    categories.add(metadata['category'])
                if 'source' in metadata:
                    sources.add(metadata['source'])
                if 'data_type' in metadata:
                    data_types.add(metadata['data_type'])
            
            return {
                'name': self.collection_name,
                'count': count,
                'categories': list(categories),
                'sources': list(sources),
                'data_types': list(data_types),
                'database_path': str(self.db_path),
                'last_updated': None  # ChromaDB doesn't provide this directly
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                'name': self.collection_name,
                'count': 0,
                'error': str(e)
            }
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all document IDs
            all_docs = self.collection.get(include=[])
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                logger.info(f"Cleared {len(all_docs['ids'])} documents from collection")
            else:
                logger.info("Collection is already empty")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Delete and recreate the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing collection
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Australian Financial RAG Knowledge Base"}
            )
            logger.info(f"Created fresh collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def export_collection(self, output_path: str) -> bool:
        """
        Export collection data to JSON file.
        
        Args:
            output_path: Path to save the exported data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents
            all_docs = self.collection.get(include=['documents', 'metadatas'])
            
            # Format for export
            export_data = {
                'collection_name': self.collection_name,
                'export_timestamp': logger.info.__module__,  # Placeholder
                'document_count': len(all_docs['ids']),
                'documents': []
            }
            
            for i, doc_id in enumerate(all_docs['ids']):
                export_data['documents'].append({
                    'id': doc_id,
                    'content': all_docs['documents'][i],
                    'metadata': all_docs['metadatas'][i]
                })
            
            # Save to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(all_docs['ids'])} documents to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return False
    
    def import_collection(self, import_path: str, clear_existing: bool = False) -> bool:
        """
        Import collection data from JSON file.
        
        Args:
            import_path: Path to the import file
            clear_existing: Whether to clear existing documents first
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                logger.error(f"Import file not found: {import_path}")
                return False
            
            # Clear existing data if requested
            if clear_existing:
                self.clear_collection()
            
            # Load import data
            with open(import_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            documents = import_data.get('documents', [])
            if documents:
                self.add_documents(documents)
                logger.info(f"Imported {len(documents)} documents from {import_path}")
                return True
            else:
                logger.warning("No documents found in import file")
                return False
                
        except Exception as e:
            logger.error(f"Error importing collection: {e}")
            return False
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            if self.client:
                # ChromaDB automatically persists data
                logger.debug("ChromaDB client cleanup completed")
        except Exception as e:
            logger.error(f"Error during ChromaDB cleanup: {e}")
