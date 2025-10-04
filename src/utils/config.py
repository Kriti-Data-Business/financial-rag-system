"""Configuration management using Pydantic settings."""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=False)
    
    # Ollama Configuration
    ollama_model: str = Field(default="mistral")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_temperature: float = Field(default=0.7)
    ollama_max_tokens: int = Field(default=500)
    
    # Embedding Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    
    # Vector Database Configuration
    vector_db_type: str = Field(default="chromadb")
    vector_db_path: str = Field(default="./data/vector_db")
    collection_name: str = Field(default="documents")
    
    # Processing Configuration
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k_results: int = Field(default=5)
    
    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="app.log")
    
    # Data Paths
    data_raw_path: str = Field(default="./data/raw")
    data_processed_path: str = Field(default="./data/processed")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env


settings = Settings()