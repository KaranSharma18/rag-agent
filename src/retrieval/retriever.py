"""
Document retriever for RAG agent.
"""
from typing import List, Dict, Optional
from .vector_store import VectorStore
from ..utils.logger import setup_logger, log_retrieval_results
from ..utils.config import TOP_K_DOCUMENTS, SIMILARITY_THRESHOLD

logger = setup_logger(__name__)


class DocumentRetriever:
    """High-level interface for document retrieval."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize retriever with vector store.
        
        Args:
            vector_store: VectorStore instance (creates new if None)
        """
        self.vector_store = vector_store or VectorStore()
        self.logger = logger
        
        self.logger.info("Document retriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_DOCUMENTS,
        min_similarity: float = SIMILARITY_THRESHOLD,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Maximum number of documents to retrieve
            min_similarity: Minimum similarity threshold (0-1)
            source_filter: Optional source document to filter by
        
        Returns:
            List of retrieved documents with metadata
        """
        self.logger.info(f"Retrieving documents for query: '{query}'")
        
        # Prepare metadata filter
        filter_metadata = {"source": source_filter} if source_filter else None
        
        # Search vector store
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results if r["similarity"] >= min_similarity
        ]
        
        self.logger.info(
            f"Retrieved {len(results)} documents, "
            f"{len(filtered_results)} above threshold ({min_similarity})"
        )
        
        # Log results in detail
        log_retrieval_results(self.logger, query, filtered_results)
        
        return filtered_results
    
    def format_context(self, documents: List[Dict[str, any]]) -> str:
        """
        Format retrieved documents into context string for LLM.
        
        Args:
            documents: List of retrieved documents
        
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for idx, doc in enumerate(documents, 1):
            metadata = doc["metadata"]
            content = doc["content"]
            
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "?")
            
            context_parts.append(
                f"[Document {idx}]\n"
                f"Source: {source}, Page {page}\n"
                f"Content: {content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_citations(self, documents: List[Dict[str, any]]) -> List[str]:
        """
        Extract citations from retrieved documents.
        
        Args:
            documents: List of retrieved documents
        
        Returns:
            List of citation strings
        """
        citations = []
        
        for doc in documents:
            metadata = doc["metadata"]
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "?")
            
            citation = f"[Source: {source}, Page {page}]"
            if citation not in citations:
                citations.append(citation)
        
        return citations
