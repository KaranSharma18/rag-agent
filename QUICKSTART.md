# RAG Agent - Quick Start Guide

Get your RAG Agent up and running in 10 minutes!

## 📋 Prerequisites

1. **Python 3.9+** installed
2. **Ollama** installed from [ollama.ai](https://ollama.ai)

## ⚡ Quick Setup (5 steps)

### Step 1: Extract and Navigate
```bash
cd rag-agent
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Start Ollama & Pull Model

**Terminal 1:**
```bash
ollama serve
```

**Terminal 2:**
```bash
ollama pull llama3.2:3b
```

### Step 5: Verify Setup
```bash
python test_setup.py
```

You should see all green checkmarks ✅

## 🎯 First Use

### Upload a PDF and Ask Questions

```bash
python cli.py
```

Then in the interactive prompt:
```
You: upload /path/to/your/document.pdf
You: What is the main topic of the document?
```

### One-liner Upload + Query
```bash
python cli.py --upload document.pdf --question "What is the revenue?"
```

## 📖 Example Session

```bash
$ python cli.py

🚀 Initializing RAG Agent...
✅ RAG Agent initialized successfully!

You: upload financial_report.pdf
✅ Document uploaded
✅ Added 45 chunks to vector database

You: What was Q1 revenue?
💡 ANSWER: Q1 revenue was $150 million, up 20% year-over-year. 
[Source: financial_report.pdf, Page 3]

You: quit
👋 Goodbye!
```

## 🐛 Troubleshooting

**"Cannot connect to Ollama"**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

**"Model not found"**
```bash
ollama pull llama3.2:3b
```

**"No text extracted from PDF"**
- Make sure PDF has selectable text (not a scanned image)
- Try a different PDF

## 📚 Next Steps

- Read `README.md` for full documentation
- Read `ARCHITECTURE.md` for technical details
- Customize `src/utils/config.py` for your needs

## 💡 Key Commands

```bash
# Interactive mode
python cli.py

# Upload document
python cli.py --upload document.pdf

# Ask question
python cli.py --question "Your question?"

# List documents
python cli.py --list

# Show stats
python cli.py --stats

# Get help
python cli.py --help
```

## ⚙️ Configuration

Edit `src/utils/config.py`:
- Change LLM model (try `mistral:7b` for better quality)
- Adjust chunk size (default: 512)
- Modify retrieval settings (default: top 5 docs)

## 🎓 Understanding the Agent

The system uses **LangGraph** to orchestrate an AI agent that:

1. **Reasons** about your question
2. **Decides** whether to search documents
3. **Retrieves** relevant chunks from vector DB
4. **Validates** if it can answer
5. **Generates** answer with citations

This is NOT a simple RAG pipeline - the LLM actively makes decisions!

## 📊 What Gets Logged

With detailed logging, you'll see:
- Agent's reasoning steps
- Retrieval queries and results
- Validation decisions
- Final answer generation

This helps you understand how the agent works!

## 🔒 Privacy Note

Everything runs **100% locally**:
- Your documents never leave your machine
- No API calls to external services
- Ollama runs locally
- Vector DB is local (ChromaDB)

Perfect for sensitive documents!

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Check `ARCHITECTURE.md` for technical deep-dive
3. Look at code comments - heavily documented!
4. Run `python test_setup.py` to diagnose issues

---

**Ready to go? Run `python cli.py` and start asking questions!** 🚀
