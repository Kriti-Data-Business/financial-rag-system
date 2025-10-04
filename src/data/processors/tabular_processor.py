"""Process CSV and Excel files for RAG system."""
import pandas as pd
from pathlib import Path
from typing import List, Dict
import hashlib
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TabularProcessor:
    """Process CSV and Excel files into text chunks."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
    
    def process_file(self, file_path: str) -> List[Dict]:
        """Process a tabular file into document chunks."""
        path = Path(file_path)
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        logger.info(f"Processing: {path.name}")
        
        # Read file
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        chunks = []
        
        # Create overview chunk
        overview = self._create_overview(df, path.name)
        chunks.append(overview)
        
        # Process rows in batches
        row_chunks = self._create_row_chunks(df, path.name)
        chunks.extend(row_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {path.name}")
        return chunks
    
    def _create_overview(self, df: pd.DataFrame, filename: str) -> Dict:
        """Create an overview chunk with dataset metadata."""
        text = f"""
Dataset: {filename}

Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Columns: {', '.join(df.columns.tolist())}

Summary Statistics:
{df.describe().to_string()}

First Few Rows:
{df.head(3).to_string()}
"""
        
        chunk_id = hashlib.md5(f"{filename}_overview".encode()).hexdigest()
        
        return {
            'id': chunk_id,
            'content': text.strip(),
            'source_document': filename
        }
    
    def _create_row_chunks(self, df: pd.DataFrame, filename: str, 
                           rows_per_chunk: int = 10) -> List[Dict]:
        """Create chunks from groups of rows."""
        chunks = []
        
        for i in range(0, len(df), rows_per_chunk):
            batch = df.iloc[i:i + rows_per_chunk]
            
            text = f"Dataset: {filename}\nRows {i+1} to {i+len(batch)}:\n\n"
            text += batch.to_string(index=False)
            
            chunk_id = hashlib.md5(f"{filename}_rows_{i}".encode()).hexdigest()
            
            chunks.append({
                'id': chunk_id,
                'content': text,
                'source_document': filename
            })
        
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """Process all CSV/Excel files in a directory."""
        path = Path(directory_path)
        all_chunks = []
        
        for file_path in path.glob('**/*'):
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    chunks = self.process_file(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
        
        logger.info(f"Processed {len(all_chunks)} total chunks from directory")
        return all_chunks