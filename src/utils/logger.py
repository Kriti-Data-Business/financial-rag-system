"""Logging configuration using loguru."""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(name: str = None):
    """Configure and return a logger instance."""
    from src.utils.config import settings
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
               "<level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # Add file handler
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message}",
        level=settings.log_level,
        rotation="10 MB",
        retention="1 week"
    )
    
    return logger


log = setup_logger()