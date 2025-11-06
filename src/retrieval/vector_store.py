"""
Vector store implementation using ChromaDB.
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from .embeddings import EmbeddingGenerator
from ..utils.logger import setup_logger
from ..utils.config import CHROMA_DB_DIR

logger = setup_logger(__name__)


class VectorStore:
    """Manage document storage and retrieval using ChromaDB."""
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize vector store with ChromaDB.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.logger = logger
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize ChromaDB client
        self.logger.info(f"Initializing ChromaDB at {CHROMA_DB_DIR}")
        
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document chunks for RAG"}
        )
        
        self.logger.info(
            f"Vector store initialized. Collection: {collection_name}, "
            f"Documents: {self.collection.count()}"
        )
    
    def add_documents(self, chunks: List[Dict[str, any]]) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata'
        
        Returns:
            Number of chunks added
        """
        if not chunks:
            self.logger.warning("No chunks to add")
            return 0
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract texts for embedding
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{i}_{hash(chunk['content'])}" for i, chunk in enumerate(chunks)]
        documents = texts
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.info(f"Successfully added {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"source": "file.pdf"})
        
        Returns:
            List of result dictionaries:
            [
                {
                    "content": "Chunk text...",
                    "metadata": {...},
                    "similarity": 0.85
                },
                ...
            ]
        """
        self.logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search collection
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results["documents"] and results["documents"][0]:
                for idx in range(len(results["documents"][0])):
                    # ChromaDB returns distances (lower is better), convert to similarity
                    distance = results["distances"][0][idx]
                    similarity = 1 / (1 + distance)  # Convert distance to similarity score
                    
                    result = {
                        "content": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx],
                        "similarity": round(similarity, 4)
                    }
                    formatted_results.append(result)
            
            self.logger.info(f"Found {len(formatted_results)} results")
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            raise
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source document.
        
        Args:
            source: Source document filename
        
        Returns:
            Number of chunks deleted
        """
        self.logger.info(f"Deleting chunks from source: {source}")
        
        try:
            # Get all items with this source
            results = self.collection.get(
                where={"source": source},
                include=[]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                count = len(results["ids"])
                self.logger.info(f"Deleted {count} chunks from {source}")
                return count
            else:
                self.logger.info(f"No chunks found for source: {source}")
                return 0
                
        except Exception as e:
            self.logger.error(f"Error deleting chunks: {str(e)}")
            raise
    
    def clear_all(self):
        """Clear all documents from the collection."""
        self.logger.warning("Clearing all documents from vector store")
        
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for RAG"}
            )
            self.logger.info("Vector store cleared")
        except Exception as e:
            self.logger.error(f"Error clearing vector store: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats
        """
        count = self.collection.count()
        
        # Get unique sources
        all_metadata = self.collection.get(include=["metadatas"])
        sources = set()
        if all_metadata["metadatas"]:
            sources = set(m.get("source", "") for m in all_metadata["metadatas"])
        
        return {
            "total_chunks": count,
            "unique_documents": len(sources),
            "sources": list(sources)
        }
