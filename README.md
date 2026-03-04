# Hybrid RAG Document QA System

## Overview

A powerful document question-answering system that combines **Hybrid Search** (semantic + keyword-based) with **Reranking** and **Large Language Models** to provide accurate, context-aware answers from PDF documents.

## Features

- 🔍 **Hybrid Search**: Combines vector-based semantic search and BM25 keyword search for comprehensive document retrieval
- 🎯 **Reranking**: Uses HuggingFace cross-encoders to intelligently rerank retrieved documents for better relevance
- 🤖 **LLM Integration**: Supports both Ollama (open-source) and Cohere LLMs for answer generation
- 📄 **PDF Support**: Efficiently processes and indexes PDF documents
- ⚡ **Performance Metrics**: Tracks retrieval and reranking latency for optimization
- 💬 **Interactive Q&A**: Command-line interface for asking questions about your documents

## Architecture

The system follows a three-stage pipeline:

### 1. **Hybrid Search Setup**
- **Vector Retriever**: Uses HuggingFace embeddings (all-MiniLM-L6-v2) with FAISS for semantic search
- **Keyword Retriever**: Uses BM25 algorithm for traditional keyword-based search
- **Ensemble Retriever**: Combines both retrievers with equal weights for comprehensive results

### 2. **Reranking Stage**
- **Cross-Encoder Model**: Uses BAAI/bge-reranker-base model to rerank retrieved documents
- **Contextual Compression**: Filters and compresses documents to the top-3 most relevant results

### 3. **QA Chain**
- **Document Chain**: Creates prompts with retrieved context
- **LLM Integration**: Generates concise answers using Ollama or Cohere
- **Answer Pipeline**: Combines retrieval and generation for final response

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/VedantPatole/Hybrid-RAG-document-QA-system.git
   cd Hybrid-RAG-document-QA-system
   ```

2. **Install dependencies**
   ```bash
   pip install langchain-community langchain-core pymupdf faiss-cpu sentence-transformers rank_bm25 langchain-cohere
   ```

3. **Set up environment variables** (for Cohere)
   ```bash
   export COHERE_API_KEY="your-cohere-api-key"
   ```
   Or use Ollama (no API key required):
   ```bash
   pip install --upgrade langchain-community
   ```

## Usage

### Basic Usage

```bash
python main.py
```

The system will:
1. Load and process the PDF document
2. Set up hybrid search retrievers
3. Initialize the reranker
4. Start an interactive Q&A session

### Interactive Session

```
what is your question: How do I start a business?
❓ Asking: How do I start a business?
Retrieval Time: 0.1234s
Reranking Time: 0.0456s
🤖 Answer: Based on the document, starting a business involves...
```

Type `exit` to quit the application.

## Configuration

### Customize PDF Document
Edit the document path in `main.py`:
```python
loader = PyMuPDFLoader("your-document.pdf")
```

### Adjust Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for larger chunks
    chunk_overlap=200  # Overlap for context preservation
)
```

### Change Retriever Weights
```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # Adjust weights as needed
)
```

### Use Different LLM
```python
# Option 1: Ollama (local)
llm = Ollama(model="mistral")

# Option 2: Cohere (API-based)
llm = ChatCohere(model="command")
```

## Key Models

- **Embeddings**: `all-MiniLM-L6-v2` (HuggingFace)
- **Reranker**: `BAAI/bge-reranker-base` (HuggingFace)
- **LLM**: `mistral` (Ollama) or `command` (Cohere)

## Project Structure

```
Hybrid-RAG-document-QA-system/
├── main.py                 # Main application script
├── requirement.txt         # Python dependencies
├── data/                   # Data files (vectors, indices)
├── pdfs/                   # Sample PDF documents
└── README.md              # Documentation
```

## Performance Optimization

- **Retrieval Time**: Tracks time to fetch relevant documents
- **Reranking Time**: Monitors reranking latency
- **Device Support**: CPU by default, GPU support available for faster processing

To use GPU:
```python
cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base",
    model_kwargs={"device": "cuda"}
)
```

## Requirements

- langchain-community
- langchain-core
- pymupdf (for PDF processing)
- faiss-cpu (vector database)
- sentence-transformers (embeddings)
- rank_bm25 (keyword search)
- langchain-cohere (Cohere LLM integration)

## Technologies Used

- **LangChain**: Framework for building LLM applications
- **FAISS**: Efficient similarity search library
- **HuggingFace**: Pre-trained models and transformers
- **Ollama**: Local LLM execution
- **Cohere**: API-based LLM services

## Author

**Vedant Patole**

## License

MIT License - Feel free to use and modify this project

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Note**: Ensure you have the necessary PDF documents in the project directory before running the application.
