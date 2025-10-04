"""
Main application entry point for RAG System with Ollama
"""
import uvicorn
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.server import create_app
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Initialize and run the FastAPI application."""
    logger.info("Starting RAG System API Server...")
    logger.info(f"Host: {settings.api_host}")
    logger.info(f"Port: {settings.api_port}")
    logger.info(f"Ollama Model: {settings.ollama_model}")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()