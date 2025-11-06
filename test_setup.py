#!/usr/bin/env python3
"""
Test script to verify RAG Agent setup.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import langchain
        print("✅ langchain")
    except ImportError as e:
        print(f"❌ langchain: {e}")
        return False
    
    try:
        import langgraph
        print("✅ langgraph")
    except ImportError as e:
        print(f"❌ langgraph: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers")
    except ImportError as e:
        print(f"❌ sentence-transformers: {e}")
        return False
    
    try:
        import chromadb
        print("✅ chromadb")
    except ImportError as e:
        print(f"❌ chromadb: {e}")
        return False
    
    try:
        import pypdf
        print("✅ pypdf")
    except ImportError as e:
        print(f"❌ pypdf: {e}")
        return False
    
    try:
        import requests
        print("✅ requests")
    except ImportError as e:
        print(f"❌ requests: {e}")
        return False
    
    return True


def test_ollama_connection():
    """Test Ollama connection."""
    print("\nTesting Ollama connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("✅ Ollama is running")
            
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            print(f"   Available models: {model_names}")
            
            if "llama3.2:3b" in model_names or "llama3.2" in model_names:
                print("✅ llama3.2:3b is installed")
                return True
            else:
                print("⚠️  llama3.2:3b not found")
                print("   Run: ollama pull llama3.2:3b")
                return False
        else:
            print("❌ Ollama responded with error")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False


def test_components():
    """Test that all components can be initialized."""
    print("\nTesting components...")
    
    try:
        from src.utils.config import BASE_DIR
        print(f"✅ Config loaded (BASE_DIR: {BASE_DIR})")
    except Exception as e:
        print(f"❌ Config: {e}")
        return False
    
    try:
        from src.utils.logger import setup_logger
        logger = setup_logger("test")
        print("✅ Logger initialized")
    except Exception as e:
        print(f"❌ Logger: {e}")
        return False
    
    try:
        from src.processing.pdf_parser import PDFParser
        parser = PDFParser()
        print("✅ PDF Parser initialized")
    except Exception as e:
        print(f"❌ PDF Parser: {e}")
        return False
    
    try:
        from src.processing.chunker import DocumentChunker
        chunker = DocumentChunker()
        print("✅ Document Chunker initialized")
    except Exception as e:
        print(f"❌ Document Chunker: {e}")
        return False
    
    try:
        from src.retrieval.embeddings import EmbeddingGenerator
        print("⏳ Loading embedding model (this may take a moment)...")
        embedder = EmbeddingGenerator()
        print("✅ Embedding Generator initialized")
    except Exception as e:
        print(f"❌ Embedding Generator: {e}")
        return False
    
    try:
        from src.retrieval.vector_store import VectorStore
        vs = VectorStore(collection_name="test_collection")
        print("✅ Vector Store initialized")
    except Exception as e:
        print(f"❌ Vector Store: {e}")
        return False
    
    try:
        from src.llm.ollama_client import OllamaClient
        llm = OllamaClient()
        print("✅ Ollama Client initialized")
    except Exception as e:
        print(f"❌ Ollama Client: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("RAG AGENT - Setup Verification")
    print("=" * 70)
    print()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test Ollama
    if not test_ollama_connection():
        print("\n❌ Ollama connection test failed")
        print("Make sure Ollama is running and model is installed")
        sys.exit(1)
    
    # Test components
    if not test_components():
        print("\n❌ Component tests failed")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nYour RAG Agent is ready to use!")
    print("Run: python cli.py")
    print()


if __name__ == "__main__":
    main()
