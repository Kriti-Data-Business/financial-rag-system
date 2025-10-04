# Text cleaning/chunking
"""
Text processing and chunking.
"""
from typing import List, Dict
from src.utils.config import settings
from src.utils.helpers import chunk_text, clean_text, generate_id
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextProcessor:
    """Process and chunk text documents."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize text processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap:chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        logger.info(f"Text processor initialized (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
    
    def process_document(self, document: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Process a document into chunks.
        
        Args:
            document: Document dictionary with content
            
        Returns:
            List of chunk dictionaries
        """
        content = document.get('content', '')
        
        if not content:
            logger.warning(f"Empty content in document: {document.get('filename', 'unknown')}")
            return []
        
        # Clean the text
        cleaned_content = clean_text(content)
        
        # Split into chunks
        chunks = chunk_text(cleaned_content, self.chunk_size, self.chunk_overlap)
        
        # Create chunk documents
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            chunk_id = generate_id(f"{document.get('filename', '')}_{i}_{chunk[:50]}")
            
            chunk_doc = {
                'id': chunk_id,
                'content': chunk,
                'chunk_index': i,
                'source_document': document.get('filename', 'unknown'),
                'source_path': document.get('filepath', ''),
                'file_type': document.get('file_type', ''),
                'metadata': {
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                }
            }
            
            chunk_documents.append(chunk_doc)
        
        logger.info(f"Processed document '{document.get('filename', 'unknown')}' into {len(chunks)} chunks")
        return chunk_documents
    
    def process_documents(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Process multiple documents into chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunk dictionaries
        """
        all_chunks = []
        
        for document in documents:
            chunks = self.process_document(document)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extract key terms from text (simple implementation).
        
        Args:
            text: Input text
            top_k: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on word frequency
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            
            if word and word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:top_k]]