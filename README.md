# 📚 AI-Powered Document Analyzer

A sophisticated RAG (Retrieval-Augmented Generation) system that enables intelligent querying of PDF documents with support for text, images, and tables. Built with LangChain, ChromaDB, and Ollama.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)

## ✨ Features

- **🔍 Intelligent Search**: Semantic search across text, images, and tables using vector embeddings
- **🤖 AI Agent**: Autonomous agent that decides which tools to use to answer your questions
- **🖼️ Image Analysis**: Automatic image extraction and caption generation using vision models
- **📊 Table Extraction**: Extract and query tables from PDFs with structured data
- **💬 Conversation Memory**: Maintains context across multiple queries (Agent Mode)
- **🌐 Web Interface**: User-friendly Streamlit interface for document interaction
- **📈 Visual Content Filtering**: LLM-based relevance filtering for images

## 🏗️ Architecture

### System Overview

```
┌─────────────┐
│  PDF Files  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  PDF Processing Pipeline    │
│  • Text Extraction          │
│  • Image Extraction         │
│  • Table Extraction         │
│  • Vision-based Captioning  │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  ChromaDB Vector Store      │
│  • Text Chunks (embedded)   │
│  • Image Captions (embedded)│
│  • Table Previews (embedded)│
│  • Metadata & File Paths    │
└──────┬──────────────────────┘
       │
       ├─────────────┬──────────────┐
       ▼             ▼              ▼
┌─────────────┐ ┌─────────┐ ┌─────────────┐
│   Standard  │ │  Agent  │ │  Streamlit  │
│   Search    │ │   Mode  │ │  Interface  │
└─────────────┘ └─────────┘ └─────────────┘
```

### Query Flow

1. **User Query** → Embedded using `qwen3-embedding:latest`
2. **Vector Search** → ChromaDB finds similar content
3. **Content Retrieval** → Text, image paths, and table data
4. **Context Building** → Combined context for LLM
5. **Answer Generation** → LLM generates response using context

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## 🛠️ Technologies Used

### Core Technologies
- **[LangChain](https://python.langchain.com/)**: LLM framework and agent orchestration
- **[ChromaDB](https://www.trychroma.com/)**: Vector database for semantic search
- **[Ollama](https://ollama.ai/)**: Local LLM inference engine
- **[Streamlit](https://streamlit.io/)**: Web interface framework
- **[PDFPlumber](https://github.com/jsvine/pdfplumber)**: PDF text and table extraction
- **[PyMuPDF](https://pymupdf.readthedocs.io/)**: PDF image extraction

### AI Models (Ollama)
- **qwen3-embedding:latest**: Text embeddings for vector search
- **qwen3-vl:8b**: Vision model for image captioning
- **gpt-oss:20b**: LLM for text generation and agent reasoning

### Data Processing
- **Pandas**: Table data manipulation
- **Pillow**: Image processing
- **NumPy**: Numerical operations

### Key Python Dependencies
- **LangChain** (v1.1.0): LLM framework and agent orchestration
- **LangGraph** (v1.0.4): Agent workflow management
- **ChromaDB** (v1.3.5): Vector database
- **Streamlit** (v1.51.0): Web interface
- **PDFPlumber** (v0.11.8): PDF text/table extraction
- **PyMuPDF** (v1.26.6): PDF image extraction
- **sentence-transformers** (v5.1.2): Embedding models


See `requirements.txt` for complete dependency list (187 packages).

## 📋 Requirements

### System Requirements
- Python 3.10+ (tested with 3.10)
- 8GB RAM minimum (16GB recommended for larger models)
- Ollama installed and running locally
- GPU recommended but not required (CPU works but slower)

### Ollama Models
Install required models with:
```bash
ollama pull qwen3-embedding:latest
ollama pull qwen3-vl:8b
ollama pull gpt-oss:20b
```

Alternative smaller models (if memory is limited):
```bash
ollama pull llama3.2:1b  # Smaller LLM
ollama pull llava:7b     # Alternative vision model
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd DocumentAnalyzer
```

### 2. Create Virtual Environment
```bash
python -m venv .venv_da
source .venv_da/bin/activate  # On Windows: .venv_da\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Start Ollama
Follow instructions at [ollama.ai](https://ollama.ai/) to install Ollama, then:
```bash
# Start Ollama service (if not auto-started)
ollama serve

# In another terminal, pull required models
ollama pull qwen3-embedding:latest
ollama pull qwen3-vl:8b
ollama pull gpt-oss:20b
```

### 5. Add Your PDF Files
Place your PDF documents in the `pdf_files/` directory:
```bash
mkdir -p pdf_files
cp your-documents.pdf pdf_files/
```

## 💻 Usage

### Quick Start

#### 1. Build the Vector Database
Process your PDFs and create the vector database:
```bash
python src/app.py
# Select option 1: Create/Rebuild Vector Database
```

This will:
- Extract text, images, and tables from PDFs
- Generate image captions using vision model
- Create embeddings and store in ChromaDB
- Save images to `extracted_images/`

#### 2. Launch Web Interface
```bash
streamlit run src/app.py
# Or select option 2 from the menu
```

The web interface will open at `http://localhost:8501`

### Query Modes

#### 🔍 Standard Search Mode
Direct vector similarity search with customizable results:
- Adjust number of text/image/table results with sliders
- Optional LLM answer generation
- View results in organized tabs

#### 🤖 Agent Mode
Intelligent agent that autonomously decides which tools to use:
- **standard_search**: Search across all content types
- **get_overview**: List available documents
- **summarize_document**: Get document summaries

Features:
- Conversation memory (remembers previous queries)
- Step-by-step reasoning display
- Automatic relevance filtering

### Example Queries

```
"What are the installation instructions?"
"List all available documents"
"How do I configure the BIOS settings?"
"Summarize the user manual"
```

### Web Interface Features

**Sidebar Settings:**
- Query mode selection (Standard/Agent)
- Result count adjustment
- LLM answer toggle
- PDF upload and management
- Conversation memory controls

**Main Display:**
- AI-generated answers
- Organized tabs for text, images, and tables
- Image gallery with captions
- Interactive table display
- Query history

## ⚙️ Configuration

Edit `config.py` to customize behavior:

```python
# Model Selection
EMBED_MODEL = "qwen3-embedding:latest"
VISION_MODEL = "qwen3-vl:8b"
AGENT_MODEL = "gpt-oss:20b"  # Unified model for all LLM operations
AGENT_TEMPERATURE = 0.1

# Performance Tuning
OLLAMA_NUM_CTX = 2048  # Context window size
OLLAMA_TIMEOUT = 240   # Request timeout (seconds)
TOP_K = 6              # Number of search results

# Image Processing
MIN_IMAGE_SIZE = 2000      # Skip small images (bytes)
MAX_IMAGE_SIZE = 5_000_000 # Skip large images (bytes)
CAPTION_RATE_LIMIT_DELAY = 1.5  # Delay between captions

# Agent Settings
ENABLE_IMAGE_FILTERING = True  # LLM-based image relevance filtering
MAX_RELEVANT_IMAGES = 3        # Max images after filtering
```

### Performance Tuning

If experiencing memory issues or crashes:

1. **Reduce context window**: `OLLAMA_NUM_CTX = 1024`
2. **Use smaller models**: `LLM_MODEL = "llama3.2:1b"`
3. **Increase delays**: `CAPTION_RATE_LIMIT_DELAY = 2.0`
4. **Skip more images**: `MIN_IMAGE_SIZE = 5000`

## 📁 Project Structure

```
AI-DocumentAnalyzer/
│
├── src/                          # Source code
│   ├── app.py                    # Main application entry point
│   ├── streamlit_webinterface.py # Streamlit UI implementation
│   ├── rag_backend.py            # RAG system and vector DB
│   ├── agent.py                  # LangChain agent implementation
│   ├── agent_tools.py            # Agent tool definitions
│   ├── llm_singleton.py          # Singleton LLM instances (prevents memory leaks)
│   └── utility/                  # Utility modules
│       ├── pdf_reader.py         # PDF extraction
│       └── caption_images.py     # Image captioning
│
├── pdf_files/                    # Input PDF documents (your PDFs go here)
├── extracted_images/             # Extracted images from PDFs
├── chroma_db/                    # ChromaDB vector database
├── docs/                         # Additional documentation
│   └── ARCHITECTURE.md           # Detailed architecture docs
│
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── .venv_da/                     # Virtual environment (not tracked)
```

## 🔧 Advanced Usage

### Python API

```python
from src.rag_backend import build_vector_db, retriever, query_by_type
from src.agent import query_with_agent

# Build vector database
build_vector_db(text_chunks, image_data, table_data, image_captions)

# Standard retrieval
docs = retriever().invoke("your query")

# Type-specific query
images = query_by_type("query", doc_type="image", k=3)
tables = query_by_type("query", doc_type="table", k=3)

# Agent query
result = query_with_agent("your question")
```

### Database Management

```bash
# Rebuild the vector database
python src/app.py  # Select option 1

# Launch web interface
streamlit run src/app.py  # Or select option 2 from menu
```

## 🐛 Troubleshooting

### Common Issues

**Problem**: Ollama connection error
```bash
# Solution: Check if Ollama is running
ollama list
# If not running:
ollama serve
```

**Problem**: Out of memory errors
```python
# Solution: Reduce context window and timeout in config.py
OLLAMA_NUM_CTX = 1024  # or 512
OLLAMA_TIMEOUT = 300   # Increase timeout if needed
```

**Problem**: No results found
```bash
# Solution: Rebuild vector database
python src/app.py  # Select option 1
```

**Problem**: Image captioning too slow
```python
# Solution: Increase delay or skip more images
CAPTION_RATE_LIMIT_DELAY = 2.0
MIN_IMAGE_SIZE = 5000
```

**Problem**: Agent not using correct tool
- The agent uses LangGraph's ReAct agent with tool calling
- Check system instructions in `src/agent.py` → `_get_system_instructions()`
- Ensure your query clearly indicates what you want (search, overview, or summary)

**Problem**: ImportError or module not found
```bash
# Solution: Ensure virtual environment is activated
source .venv_da/bin/activate  # Linux/Mac
.venv_da\Scripts\activate     # Windows

# Reinstall dependencies if needed
pip install -r requirements.txt
```

## 📌 Known Limitations

- **Large PDFs**: Very large PDFs (100+ pages) may take several minutes to process
- **Complex Tables**: Tables with merged cells or complex formatting may not extract perfectly
- **Image Quality**: Low-resolution images may produce poor captions
- **Memory Usage**: Processing many images simultaneously can consume significant RAM
- **Model Availability**: Requires Ollama models to be pre-downloaded locally

## 🔮 Future Improvements

Potential enhancements for future versions:
- [ ] Support for additional document formats (DOCX, PPTX, HTML)
- [ ] Batch processing for multiple documents
- [ ] Export functionality for search results
- [ ] Advanced table parsing with structure preservation
- [ ] Multi-language support for non-English documents
- [ ] Cloud deployment options (Docker, Kubernetes)
- [ ] API endpoint for programmatic access
- [ ] Citation tracking for answer sources

## 🙏 Acknowledgments

- **LangChain**: Powerful LLM framework
- **ChromaDB**: Efficient vector database
- **Ollama**: Local LLM inference
- **Streamlit**: Easy web interface creation
- **PDFPlumber & PyMuPDF**: Excellent PDF processing libraries

---

**Built with ❤️ for intelligent document analysis**
