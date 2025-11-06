"""
PDF parsing utilities for extracting text from PDF documents.
"""
from typing import List, Dict
from pathlib import Path
import pypdf
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFParser:
    """Extract text from PDF files with page-level metadata."""
    
    def __init__(self):
        """Initialize PDF parser."""
        self.logger = logger
    
    def extract_text(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Extract text from PDF file page by page.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of dictionaries with page text and metadata:
            [
                {
                    "text": "Page content...",
                    "page_number": 1,
                    "source": "filename.pdf"
                },
                ...
            ]
        """
        self.logger.info(f"Extracting text from PDF: {pdf_path.name}")
        
        pages = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                self.logger.info(f"PDF has {num_pages} pages")
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Only add pages with actual content
                    if text.strip():
                        pages.append({
                            "text": text,
                            "page_number": page_num + 1,  # 1-indexed
                            "source": pdf_path.name
                        })
                        
                        self.logger.debug(
                            f"Page {page_num + 1}: Extracted {len(text)} characters"
                        )
                
                self.logger.info(
                    f"Successfully extracted text from {len(pages)} pages"
                )
                
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
        
        return pages
    
    def validate_pdf(self, pdf_path: Path, max_size_mb: int = 50) -> bool:
        """
        Validate PDF file before processing.
        
        Args:
            pdf_path: Path to PDF file
            max_size_mb: Maximum allowed file size in MB
        
        Returns:
            True if valid, False otherwise
        """
        # Check if file exists
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        # Check file extension
        if pdf_path.suffix.lower() != '.pdf':
            self.logger.error(f"File is not a PDF: {pdf_path}")
            return False
        
        # Check file size
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            self.logger.error(
                f"PDF file too large: {size_mb:.2f}MB (max: {max_size_mb}MB)"
            )
            return False
        
        # Try to open with pypdf
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                _ = len(pdf_reader.pages)
        except Exception as e:
            self.logger.error(f"Invalid PDF file: {str(e)}")
            return False
        
        self.logger.info(f"PDF validation successful: {pdf_path.name}")
        return True
