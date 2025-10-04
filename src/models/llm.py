"""Ollama LLM wrapper for text generation."""
import requests
from typing import Optional
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMModel:
    """Wrapper for Ollama language models."""
    
    def __init__(self):
        """Initialize Ollama LLM."""
        self.model_name = settings.ollama_model
        self.base_url = settings.ollama_base_url
        self.temperature = settings.ollama_temperature
        self.max_tokens = settings.ollama_max_tokens
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Ollama LLM initialized: {self.model_name}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            raise ConnectionError(f"Ollama not accessible: {e}")
    
    def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using Ollama."""
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer based on the context provided. "
                "Be concise and clear."
            )
        
        # Truncate context if too long (prevent timeout)
        max_context_length = 2000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            logger.warning(f"Context truncated to {max_context_length} characters")
        
        prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            logger.info(f"Sending request to Ollama...")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_ctx": 2048  # Context window size
                    }
                },
                timeout=180  # 3 minutes timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "")
            
            if not answer:
                logger.error(f"Empty response from Ollama. Full response: {result}")
                return "I apologize, but I couldn't generate a response. Please try again."
            
            logger.info("Response generated successfully")
            return answer.strip()
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "The request timed out. Please try a simpler question or check if Ollama is responding."
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return "Sorry, there was an error connecting to the AI model. Please ensure Ollama is running."
            
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            return "Received an unexpected response format from the AI model."
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "An unexpected error occurred. Please try again."