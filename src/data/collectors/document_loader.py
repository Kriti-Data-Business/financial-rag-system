"""
Document loader for various file formats.
"""
from typing import List, Dict
from pathlib import Path
import PyPDF2
from docx import Document
from src.utils.logger import setup_logger
from src.utils.helpers import clean_text

logger = setup_logger(__name__)


class DocumentLoader:
    """Load and parse documents from various formats."""
    
    SUPPORTED_FORMATS = ['.pdf', '.txt', '.md', '.docx']
    
    def __init__(self):
        """Initialize document loader."""
        logger.info("Document loader initialized")
    
    def load_document(self, file_path: str) -> Dict[str, any]:
        """
        Load a document from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with document metadata and content
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = path.suffix.lower()
        
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Loading document: {path.name}")
        
        try:
            if file_ext == '.pdf':
                content = self._load_pdf(path)
            elif file_ext == '.docx':
                content = self._load_docx(path)
            elif file_ext in ['.txt', '.md']:
                content = self._load_text(path)
            else:
                raise ValueError(f"Unsupported format: {file_ext}")
            
            document = {
                'filename': path.name,
                'filepath': str(path.absolute()),
                'content': clean_text(content),
                'file_type': file_ext,
                'size': path.stat().st_size
            }
            
            logger.info(f"Document loaded successfully: {path.name} ({len(content)} chars)")
            return document
            
        except Exception as e:
            logger.error(f"Error loading document {path.name}: {e}")
            raise
    
    def _load_pdf(self, path: Path) -> str:
        """Load PDF file."""
        text = []
        with open(path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
    
    def _load_docx(self, path: Path) -> str:
        """Load DOCX file."""
        doc = Document(path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    
    def _load_text(self, path: Path) -> str:
        """Load text file."""
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_directory(self, directory_path: str) -> List[Dict[str, any]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            List of document dictionaries
        """
        path = Path(directory_path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        documents = []
        
        for file_path in path.rglob('*'):
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Skipping {file_path.name}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents