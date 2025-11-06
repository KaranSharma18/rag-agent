"""
Ollama LLM client for local model interactions.
"""
import requests
from typing import Optional, Dict, List
from ..utils.logger import setup_logger
from ..utils.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE

logger = setup_logger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        temperature: float = OLLAMA_TEMPERATURE
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use
            temperature: Sampling temperature (0-1)
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.logger = logger
        
        self.logger.info(f"Ollama client initialized: {model} @ {base_url}")
        
        # Verify Ollama is running
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                self.logger.info(f"Ollama connected. Available models: {model_names}")
                
                if self.model not in model_names:
                    self.logger.warning(
                        f"Model '{self.model}' not found. "
                        f"Please run: ollama pull {self.model}"
                    )
            else:
                self.logger.warning("Ollama API responded with non-200 status")
        except Exception as e:
            self.logger.error(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {str(e)}"
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text completion from Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for instructions
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        self.logger.debug(f"Generating completion for prompt: {prompt[:100]}...")
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            self.logger.debug(f"Generated text: {generated_text[:200]}...")
            
            return generated_text
            
        except requests.exceptions.Timeout:
            self.logger.error("Ollama request timed out")
            raise Exception("LLM request timed out")
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> str:
        """
        Chat-style interaction with Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     e.g., [{"role": "user", "content": "Hello"}]
            temperature: Override default temperature
        
        Returns:
            Assistant's response
        """
        self.logger.debug(f"Chat with {len(messages)} messages")
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            
            result = response.json()
            assistant_message = result.get("message", {}).get("content", "")
            
            self.logger.debug(f"Assistant response: {assistant_message[:200]}...")
            
            return assistant_message
            
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and model is loaded.
        
        Returns:
            True if available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
