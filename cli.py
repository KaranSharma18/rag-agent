#!/usr/bin/env python3
"""
Command-line interface for RAG Agent.
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.graph import RAGAgent
from src.processing.document_manager import DocumentManager
from src.utils.logger import setup_logger
from src.utils.config import UPLOADS_DIR

logger = setup_logger(__name__)


class RAGAgentCLI:
    """Command-line interface for RAG Agent."""
    
    def __init__(self):
        """Initialize CLI with agent and document manager."""
        logger.info("=" * 70)
        logger.info("RAG AGENT - Document Q&A System")
        logger.info("=" * 70)
        
        print("\n🚀 Initializing RAG Agent...")
        print("This may take a moment on first run (downloading models)...\n")
        
        try:
            self.agent = RAGAgent()
            self.doc_manager = DocumentManager()
            self.session_id = "cli_session"
            
            print("✅ RAG Agent initialized successfully!\n")
            
        except Exception as e:
            print(f"❌ Error initializing agent: {str(e)}")
            print("\nPlease ensure:")
            print("  1. Ollama is running: ollama serve")
            print("  2. Model is installed: ollama pull llama3.2:3b")
            sys.exit(1)
    
    def upload_document(self, file_path: str):
        """
        Upload and process a PDF document.
        
        Args:
            file_path: Path to PDF file
        """
        print(f"\n📄 Uploading document: {file_path}")
        
        try:
            # Upload document
            uploaded_path = self.doc_manager.upload_document(file_path)
            print(f"✅ Document uploaded: {uploaded_path.name}")
            
            # Process document
            print("⚙️  Processing document (extracting text, chunking, embedding)...")
            chunks = self.doc_manager.process_document(uploaded_path)
            
            if not chunks:
                print("⚠️  Warning: No text extracted from document")
                return
            
            print(f"✅ Extracted {len(chunks)} chunks from document")
            
            # Add to vector store
            print("🔍 Adding to vector database...")
            count = self.agent.vector_store.add_documents(chunks)
            print(f"✅ Added {count} chunks to vector database")
            
            # Show stats
            stats = self.agent.vector_store.get_stats()
            print(f"\n📊 Database stats:")
            print(f"   Total chunks: {stats['total_chunks']}")
            print(f"   Documents: {stats['unique_documents']}")
            print(f"   Sources: {', '.join(stats['sources'])}")
            
        except Exception as e:
            print(f"❌ Error uploading document: {str(e)}")
    
    def ask_question(self, question: str):
        """
        Ask a question to the agent.
        
        Args:
            question: User's question
        """
        print(f"\n❓ Question: {question}")
        print("🤔 Thinking...\n")
        
        try:
            answer = self.agent.query(question, self.session_id)
            
            print("=" * 70)
            print("💡 ANSWER:")
            print("=" * 70)
            print(answer)
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error processing question: {str(e)}")
    
    def list_documents(self):
        """List all uploaded documents."""
        docs = self.doc_manager.list_uploaded_documents()
        
        if not docs:
            print("\n📂 No documents uploaded yet")
        else:
            print(f"\n📂 Uploaded documents ({len(docs)}):")
            for idx, doc in enumerate(docs, 1):
                print(f"   {idx}. {doc}")
    
    def show_stats(self):
        """Show database statistics."""
        stats = self.agent.vector_store.get_stats()
        
        print("\n📊 Vector Database Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Unique documents: {stats['unique_documents']}")
        
        if stats['sources']:
            print(f"   Sources:")
            for source in stats['sources']:
                print(f"      - {source}")
        else:
            print("   No documents in database")
    
    def interactive_mode(self):
        """Run interactive Q&A session."""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("Commands:")
        print("  - Ask any question about your documents")
        print("  - 'upload <path>' - Upload a new document")
        print("  - 'list' - List uploaded documents")
        print("  - 'stats' - Show database statistics")
        print("  - 'help' - Show this help")
        print("  - 'quit' or 'exit' - Exit")
        print("=" * 70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self.interactive_mode()
                    return
                
                elif user_input.lower() == 'list':
                    self.list_documents()
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                
                elif user_input.lower().startswith('upload '):
                    file_path = user_input[7:].strip()
                    self.upload_document(file_path)
                
                else:
                    # Treat as question
                    self.ask_question(user_input)
                
                print()  # Blank line for readability
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Agent - Document Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cli.py
  
  # Upload document and ask question
  python cli.py --upload document.pdf --question "What is the main topic?"
  
  # Just upload
  python cli.py --upload document.pdf
  
  # Just ask (requires documents already uploaded)
  python cli.py --question "What is the revenue?"
        """
    )
    
    parser.add_argument(
        '--upload', '-u',
        type=str,
        help='Path to PDF file to upload'
    )
    
    parser.add_argument(
        '--question', '-q',
        type=str,
        help='Question to ask'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List uploaded documents'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show database statistics'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = RAGAgentCLI()
    
    # Handle commands
    if args.upload:
        cli.upload_document(args.upload)
    
    if args.list:
        cli.list_documents()
    
    if args.stats:
        cli.show_stats()
    
    if args.question:
        cli.ask_question(args.question)
    
    # If no arguments, enter interactive mode
    if not any([args.upload, args.question, args.list, args.stats]):
        cli.interactive_mode()


if __name__ == "__main__":
    main()
