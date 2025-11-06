"""
Text chunking utilities for splitting documents into smaller pieces.
"""
from typing import List, Dict
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..utils.logger import setup_logger
from ..utils.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = setup_logger(__name__)


class DocumentChunker:
    """Split documents into chunks with metadata preservation."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize chunker with size and overlap settings.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.logger = logger
        self.logger.info(
            f"Chunker initialized: size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def chunk_pages(self, pages: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Split pages into chunks while preserving metadata.
        
        Args:
            pages: List of page dictionaries with 'text', 'page_number', 'source'
        
        Returns:
            List of chunk dictionaries with metadata:
            [
                {
                    "content": "Chunk text...",
                    "metadata": {
                        "source": "filename.pdf",
                        "page": 1,
                        "chunk_index": 0,
                        "total_chunks_in_page": 3
                    }
                },
                ...
            ]
        """
        self.logger.info(f"Chunking {len(pages)} pages")
        
        all_chunks = []
        
        for page in pages:
            page_text = page["text"]
            page_number = page["page_number"]
            source = page["source"]
            
            # Split page into chunks
            chunks = self.text_splitter.split_text(page_text)
            
            self.logger.debug(
                f"Page {page_number}: Split into {len(chunks)} chunks"
            )
            
            # Add metadata to each chunk
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_dict = {
                    "content": chunk_text,
                    "metadata": {
                        "source": source,
                        "page": page_number,
                        "chunk_index": chunk_idx,
                        "total_chunks_in_page": len(chunks)
                    }
                }
                all_chunks.append(chunk_dict)
        
        self.logger.info(
            f"Created {len(all_chunks)} total chunks from {len(pages)} pages"
        )
        
        return all_chunks
    
    def chunk_text(self, text: str, metadata: Dict[str, any] = None) -> List[Dict[str, any]]:
        """
        Split raw text into chunks.
        
        Args:
            text: Raw text to chunk
            metadata: Optional metadata to attach to chunks
        
        Returns:
            List of chunk dictionaries
        """
        chunks = self.text_splitter.split_text(text)
        
        chunk_dicts = []
        for idx, chunk_text in enumerate(chunks):
            chunk_dict = {
                "content": chunk_text,
                "metadata": metadata or {}
            }
            chunk_dict["metadata"]["chunk_index"] = idx
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
