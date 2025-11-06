# RAG Agent - Document Q&A System

An AI agent that answers questions from uploaded PDF documents using Retrieval Augmented Generation (RAG) with LangGraph orchestration.

## 🎯 Features

- **Document Upload**: Process and index PDF documents locally
- **Semantic Search**: Uses sentence-transformers for embeddings + ChromaDB for vector storage
- **Agent Orchestration**: LangGraph-powered agent with reasoning, tool calling, and validation
- **Strict Guardrails**: Only answers from provided documents with source citations
- **Memory**: Maintains conversation history for contextual follow-ups
- **100% Local & Free**: Runs entirely on your machine using Ollama (no API costs)

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER (CLI Interface)                     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    AGENT ORCHESTRATOR                        │
│                      (LangGraph)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Reasoning │→ │Retrieval │→ │Validation│→ │ Answer   │   │
│  │  Node    │  │  Node    │  │  Node    │  │  Node    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼───────┐
│  DOCUMENT      │  │   VECTOR DB     │  │     LLM       │
│  PROCESSOR     │  │   (ChromaDB)    │  │   (Ollama)    │
│                │  │                 │  │               │
│ • PDF Parser   │  │ • Embeddings    │  │ • llama3.2:3b │
│ • Chunker      │  │ • Similarity    │  │ • Local       │
│ • Metadata     │  │   Search        │  │               │
└────────────────┘  └─────────────────┘  └───────────────┘
```

### Agent Flow

```
START
  ↓
[REASONING NODE]
  - Analyzes user question
  - Decides action: RETRIEVE / ANSWER / INSUFFICIENT
  ↓
┌─────────────┴─────────────┐
│                           │
[RETRIEVE]              [ANSWER from history]
  ↓                         ↓
[RETRIEVAL NODE]           END
  - Semantic search
  - Fetch relevant chunks
  ↓
[VALIDATION NODE]
  - Check if docs sufficient
  - Guardrail enforcement
  ↓
┌─────────────┴─────────────┐
│                           │
[SUFFICIENT]          [INSUFFICIENT]
  ↓                         ↓
[ANSWER NODE]         [INSUFFICIENT NODE]
  - Generate response      - Explain limitation
  - Add citations          - Suggest alternatives
  ↓                         ↓
END                        END
```

## 🚀 Setup

### Prerequisites

1. **Python 3.9+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)

### Installation Steps

1. **Clone/Download the repository**

```bash
cd rag-agent
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install and start Ollama**

```bash
# Start Ollama server
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:3b
```

5. **Verify setup**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should show llama3.2:3b in the list
```

## 📖 Usage

### Interactive Mode (Recommended)

```bash
python cli.py
```

Commands in interactive mode:
- Type any question to ask the agent
- `upload <path>` - Upload a PDF document
- `list` - Show uploaded documents
- `stats` - Show database statistics
- `help` - Show help message
- `quit` or `exit` - Exit

### Command-Line Mode

```bash
# Upload a document
python cli.py --upload path/to/document.pdf

# Ask a question
python cli.py --question "What is the main topic?"

# Upload and ask in one command
python cli.py --upload document.pdf --question "What is the revenue?"

# List documents
python cli.py --list

# Show stats
python cli.py --stats
```

### Example Session

```bash
$ python cli.py

🚀 Initializing RAG Agent...
✅ RAG Agent initialized successfully!

======================================================================
INTERACTIVE MODE
======================================================================
Commands:
  - Ask any question about your documents
  - 'upload <path>' - Upload a new document
  - 'list' - List uploaded documents
  - 'stats' - Show database statistics
  - 'quit' or 'exit' - Exit
======================================================================

You: upload financial_report.pdf

📄 Uploading document: financial_report.pdf
✅ Document uploaded: financial_report.pdf
⚙️  Processing document (extracting text, chunking, embedding)...
✅ Extracted 45 chunks from document
🔍 Adding to vector database...
✅ Added 45 chunks to vector database

📊 Database stats:
   Total chunks: 45
   Documents: 1
   Sources: financial_report.pdf

You: What was the Q1 revenue?

❓ Question: What was the Q1 revenue?
🤔 Thinking...

======================================================================
💡 ANSWER:
======================================================================
According to the financial report, Q1 revenue was $150 million, 
representing a 20% increase compared to Q1 of the previous year. 
This growth was primarily driven by expansion in the North American 
market and increased enterprise sales.

[Source: financial_report.pdf, Page 3]
======================================================================

You: How does this compare to Q2?

❓ Question: How does this compare to Q2?
🤔 Thinking...

======================================================================
💡 ANSWER:
======================================================================
Q2 revenue reached $165 million, showing an increase of $15 million 
(10% growth) compared to Q1's $150 million. [Source: financial_report.pdf, 
Page 5]
======================================================================

You: quit

👋 Goodbye!
```

## 🧩 Project Structure

```
rag-agent/
│
├── src/
│   ├── agent/
│   │   ├── graph.py           # LangGraph agent definition
│   │   ├── nodes.py           # Agent node functions
│   │   ├── state.py           # State schema
│   │   └── prompts.py         # Prompt templates
│   │
│   ├── retrieval/
│   │   ├── vector_store.py    # ChromaDB integration
│   │   ├── embeddings.py      # Sentence-transformers
│   │   └── retriever.py       # Retrieval interface
│   │
│   ├── processing/
│   │   ├── pdf_parser.py      # PDF text extraction
│   │   ├── chunker.py         # Text chunking
│   │   └── document_manager.py # Document pipeline
│   │
│   ├── llm/
│   │   └── ollama_client.py   # Ollama API client
│   │
│   └── utils/
│       ├── config.py          # Configuration
│       └── logger.py          # Logging utilities
│
├── data/
│   ├── uploads/               # Uploaded PDFs
│   └── chroma_db/             # Vector database
│
├── cli.py                     # CLI interface
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Edit `src/utils/config.py` to customize:

- **LLM Model**: Change `OLLAMA_MODEL` (e.g., `mistral:7b`, `qwen2.5:7b`)
- **Chunk Size**: Adjust `CHUNK_SIZE` (default: 512)
- **Retrieval**: Modify `TOP_K_DOCUMENTS` (default: 5)
- **Temperature**: Change `OLLAMA_TEMPERATURE` (default: 0.1)
- **Logging**: Set `LOG_LEVEL` (DEBUG, INFO, WARNING, ERROR)

## 🛡️ Guardrails & Safety

The agent implements multiple layers of guardrails:

1. **Reasoning Guardrail**: Agent decides if it should retrieve or decline
2. **Validation Guardrail**: Checks if retrieved docs contain sufficient info
3. **Prompt Guardrail**: Instructions to LLM to only use provided context
4. **Citation Requirement**: All answers must include source references

## 🧠 How It Works

### 1. Document Processing Pipeline

```python
PDF Upload → Text Extraction → Chunking → Embedding → Vector DB
```

- PDFs are parsed page by page
- Text is split into 512-character chunks with 50-char overlap
- Each chunk gets an embedding via sentence-transformers
- Stored in ChromaDB with metadata (source, page number)

### 2. Agent Reasoning Loop

```python
Question → Reasoning → Tool Decision → Retrieval → Validation → Answer
```

The agent:
1. **Reasons** about the question (new query vs follow-up)
2. **Decides** whether to retrieve documents
3. **Retrieves** relevant chunks from vector DB (if needed)
4. **Validates** if documents contain enough information
5. **Generates** answer with citations OR states "cannot answer"

### 3. Tool Calling

The agent has access to:
- **Document Retrieval Tool**: Searches vector database
- **Validation Tool**: Checks answer feasibility

The LLM decides when and how to use these tools based on the question.

## 📊 Logging

Detailed logging tracks every step:

```
====================================================
AGENT STEP: REASONING
====================================================
question: What is the revenue?
iteration: 1
====================================================

[Reasoning output...]

====================================================
AGENT STEP: RETRIEVAL
====================================================
query: revenue Q1 Q2
====================================================

[Retrieved documents...]
```

## 🎓 Assignment Requirements Coverage

✅ **Agent Design & Orchestration**: LangGraph with state machine
✅ **RAG**: Semantic search with ChromaDB + sentence-transformers  
✅ **Memory**: Conversation history in state
✅ **Tool Calling**: Retrieval tool with agent decision-making
✅ **Guardrails**: Multi-layer validation and safety controls
✅ **Real-world Integration**: File system (PDFs), vector DB, local LLM
✅ **Advanced Tech**: LangGraph, modern RAG patterns, validation loops

## 🐛 Troubleshooting

### "Cannot connect to Ollama"
- Ensure Ollama is running: `ollama serve`
- Check if running on correct port: `curl http://localhost:11434/api/tags`

### "Model not found"
- Pull the model: `ollama pull llama3.2:3b`
- Verify: `ollama list`

### "No text extracted from PDF"
- PDF might be scanned images (needs OCR)
- Try a different PDF with selectable text
- Check PDF isn't corrupted

### Slow performance
- First run downloads embedding model (~100MB)
- Reduce `CHUNK_SIZE` in config
- Use smaller model: `llama3.2:1b`

## 📝 Notes

- **First run**: Downloads sentence-transformer model (~100MB)
- **Memory usage**: ~2-4GB RAM depending on model
- **Vector DB**: Persistent across runs (stored in `data/chroma_db/`)
- **Sessions**: Each CLI run creates a new session (no persistence between runs yet)

## 🚧 Future Enhancements

- [ ] Session persistence (save chat history)
- [ ] Multi-document reasoning
- [ ] Table extraction from PDFs
- [ ] Image/chart analysis (with vision models)
- [ ] Query rewriting for better retrieval
- [ ] Re-ranking retrieved documents
- [ ] Streamlit web UI
- [ ] Support for more document types (Word, HTML, etc.)

## 📜 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- **LangGraph**: For agent orchestration framework
- **Ollama**: For local LLM hosting
- **ChromaDB**: For vector storage
- **sentence-transformers**: For embeddings

---

**Built with ❤️ for production-ready LLM applications**
