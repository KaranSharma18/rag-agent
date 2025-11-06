"""
Embedding generation using sentence-transformers.
"""
from typing import List
from sentence_transformers import SentenceTransformer
from ..utils.logger import setup_logger
from ..utils.config import EMBEDDING_MODEL

logger = setup_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.logger = logger
        
        self.logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
        
        Returns:
            List of embedding vectors
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            self.logger.info("Embeddings generated successfully")
            return embeddings.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
