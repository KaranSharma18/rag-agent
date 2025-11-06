"""
Document manager for handling PDF uploads and processing.
"""
import shutil
from typing import List, Dict
from pathlib import Path
from .pdf_parser import PDFParser
from .chunker import DocumentChunker
from ..utils.logger import setup_logger
from ..utils.config import UPLOADS_DIR, MAX_PDF_SIZE_MB

logger = setup_logger(__name__)


class DocumentManager:
    """Manage document uploads and processing pipeline."""
    
    def __init__(self):
        """Initialize document manager with parser and chunker."""
        self.pdf_parser = PDFParser()
        self.chunker = DocumentChunker()
        self.uploads_dir = UPLOADS_DIR
        self.logger = logger
        
        self.logger.info(f"Document manager initialized. Uploads dir: {self.uploads_dir}")
    
    def upload_document(self, file_path: str) -> Path:
        """
        Upload (copy) a document to the uploads directory.
        
        Args:
            file_path: Path to the document to upload
        
        Returns:
            Path to the uploaded document
        
        Raises:
            ValueError: If file is invalid
            FileNotFoundError: If file doesn't exist
        """
        source_path = Path(file_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate PDF
        if not self.pdf_parser.validate_pdf(source_path, MAX_PDF_SIZE_MB):
            raise ValueError(f"Invalid PDF file: {file_path}")
        
        # Copy to uploads directory
        dest_path = self.uploads_dir / source_path.name
        
        # Handle duplicate filenames
        counter = 1
        while dest_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            dest_path = self.uploads_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(source_path, dest_path)
        self.logger.info(f"Document uploaded: {dest_path.name}")
        
        return dest_path
    
    def process_document(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Process a PDF document into chunks with metadata.
        
        Pipeline:
        1. Extract text from PDF (page by page)
        2. Split pages into chunks
        3. Return chunks with metadata
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of chunk dictionaries ready for embedding
        """
        self.logger.info(f"Processing document: {pdf_path.name}")
        
        # Step 1: Extract text from PDF
        pages = self.pdf_parser.extract_text(pdf_path)
        
        if not pages:
            self.logger.warning(f"No text extracted from {pdf_path.name}")
            return []
        
        # Step 2: Chunk pages
        chunks = self.chunker.chunk_pages(pages)
        
        self.logger.info(
            f"Document processing complete: {len(chunks)} chunks created"
        )
        
        return chunks
    
    def list_uploaded_documents(self) -> List[str]:
        """
        List all uploaded documents.
        
        Returns:
            List of uploaded document filenames
        """
        if not self.uploads_dir.exists():
            return []
        
        return [f.name for f in self.uploads_dir.glob("*.pdf")]
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete an uploaded document.
        
        Args:
            filename: Name of the document to delete
        
        Returns:
            True if deleted, False otherwise
        """
        file_path = self.uploads_dir / filename
        
        if file_path.exists():
            file_path.unlink()
            self.logger.info(f"Document deleted: {filename}")
            return True
        
        self.logger.warning(f"Document not found for deletion: {filename}")
        return False
