"""
Configuration settings for the RAG Agent system.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TEMPERATURE = 0.1  # Low temperature for consistent reasoning

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast and efficient
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Document Processing Configuration
CHUNK_SIZE = 512  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
MAX_PDF_SIZE_MB = 50  # Maximum PDF file size

# Retrieval Configuration
TOP_K_DOCUMENTS = 5  # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score (0-1)

# Agent Configuration
MAX_ITERATIONS = 3  # Maximum reasoning loops
ENABLE_VALIDATION = True  # Enable validation node
ENABLE_GUARDRAILS = True  # Enable guardrails

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_TO_FILE = False
LOG_FILE = BASE_DIR / "agent.log"

# Session Configuration
SESSION_TIMEOUT_MINUTES = 30
