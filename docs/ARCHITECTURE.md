# Architecture: How Images and Tables Are Queried and Displayed

## Storage Flow

```
PDF Document
     │
     ├─────────────┬─────────────┬──────────────┐
     │             │             │              │
  Texts        Images        Tables      Image Captions
     │             │             │              │
     │             │             │              │
     ▼             ▼             ▼              ▼
┌────────────────────────────────────────────────────┐
│          ChromaDB Vector Store                     │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Text    │  │  Image   │  │  Table   │          │
│  │ Chunks   │  │ Captions │  │ Previews │          │
│  │          │  │          │  │          │          │
│  │ metadata:│  │ metadata:│  │ metadata:│          │
│  │  type    │  │  type    │  │  type    │          │
│  │  page    │  │  page    │  │  page    │          │
│  │  source  │  │  source  │  │  source  │          │
│  │  content │  │  path    │  │  json    │          │
│  │          │  │  caption │  │  preview │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│                                                    │
│  Embedding Model: qwen3-embedding:latest           │
│  Vision Model: qwen3-vl:8b                         │
│  Agent Model: gpt-oss:20b                          │
└────────────────────────────────────────────────────┘
```

## Query Flow

```
User Query: "Show me the control panel"
           │
           ▼
    ┌──────────────────────┐
    │  Agent (gpt-oss:20b) │ ─── Decides which tool(s) to use
    └──────┬───────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │  standard_search Tool                   │
    │  - Searches text, images, AND tables    │
    │  - Uses qwen3-embedding for similarity  │
    │  - Applies source filtering if needed   │
    └──────┬──────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │  Vector Similarity Search        │
    │  in ChromaDB with type filtering │
    └──────┬───────────────────────────┘
           │
           ├──────────────┬──────────────┬
           ▼              ▼              ▼              
     ┌──────────┐   ┌──────────┐   ┌──────────┐
     │  Text    │   │  Images  │   │  Tables  │
     │  Docs    │   │  (meta)  │   │  (meta)  │
     └──────────┘   └────┬─────┘   └────┬─────┘
                         │              │
                         ▼              │
                    ┌────────────┐      │
                    │  LLM Image │      │
                    │  Filtering │      │
                    │ (relevance)│      │
                    └────┬───────┘      │
                         │              │
                         ▼              ▼
                    ┌──────────┐   ┌──────────┐
                    │  Image   │   │  Table   │
                    │  Files   │   │  JSON    │
                    │  (disk)  │   │  Data    │
                    └──────────┘   └──────────┘
                         │              │
                         ▼              ▼
                    ┌────────────────────────┐
                    │  Display to User       │
                    │  - Streamlit (Web UI)  │
                    │  - PIL Image           │
                    │  - Pandas DataFrame    │
                    └────────────────────────┘
```

## Agent-Based Query System

### Agent Tools (3 Core Tools)

The agent autonomously decides which tool(s) to use based on the user's query.

#### 1. standard_search (Primary Tool)
```python
@tool
def standard_search(query: str) -> str:
    """
    Search for ALL relevant content in documents (text, images, and tables).
    
    This is your PRIMARY tool. Use it for ANY question about document content.
    It automatically searches text, finds relevant images, and locates tables.
    """
    # 1. Search text documents (k=5)
    # 2. Search images (k=5), then filter with LLM for relevance
    # 3. Search tables (k=3)
    # Returns comprehensive results with metadata
```

#### 2. get_overview
```python
@tool
def get_overview() -> str:
    """
    Get an overview of all available documents in the database.
    
    Use this tool ONLY when the user explicitly asks:
    - "What documents are available?"
    - "Show me all documents"
    - "List all PDFs"
    - "List all Documents"
    
    For content questions, use standard_search instead.
    
    Returns:
        List of available PDF documents with file sizes
    """

#### 3. summarize_document
```python
@tool
def summarize_document(document_name: str) -> str:
    """
    Retrieve text content from a specific document for summarization.
    
    Use this tool ONLY when the user explicitly asks for a summary:
    - "Summarize the manual"
    - "Give me an overview of document X"
    - "What is document Y about?"
    
    For specific questions about content, use standard_search instead.
    
    Args:
        document_name: Name of the PDF file (e.g., "manual.pdf" or just "manual")
    
    Returns:
        Text content from the document that you should summarize for the user
    """

### Backend Query Methods

#### Method 1: Type-Specific Query
```python
docs = query_by_type("query", doc_type="image", source="manual.pdf", k=3)
# Returns: Only specified type (text/image/table)
# Supports source filtering
# Distance thresholds: IMAGE_DIST=0.35, TABLE_DIST=0.35
```

#### Method 2: Convenience Function
```python
image_metas, table_metas = get_images_and_tables("query", image_k=3, table_k=3)
# Returns: Both images and tables in one call
# Benefit: Simplest for visual content retrieval
```

#### Method 3: Standard Retriever (Legacy)
```python
retriever_instance = retriever()
docs = retriever_instance.invoke("query")
# Returns: Mix of text, image, and table documents
# Need to: Filter by metadata['type']
```

## Display Pipeline

### Images
```
Image Metadata (from ChromaDB)
    │
    ├─ Get path: metadata['path']
    ├─ Get caption: metadata['caption']
    └─ Get page: metadata['page']
         │
         ▼
Load from filesystem: extracted_images/doc_p3_img1.png
         │
         ▼
Display:
  - Console: Print path, size, caption
  - GUI: PIL Image.show()
  - Web: Streamlit st.image()
```

### Tables
```
Table Metadata (from ChromaDB)
    │
    ├─ Get preview: metadata['preview']
    ├─ Get JSON: metadata['json']
    └─ Get page: metadata['page']
         │
         ▼
Parse JSON string → Python list/dict
         │
         ▼
Convert to DataFrame: pd.DataFrame(table_data)
         │
         ▼
Display:
  - Console: df.to_string()
  - Web: st.dataframe(df)
```

## Agent Processing Flow

```
User Query
     │
     ▼
┌────────────────────────────────────┐
│  LangGraph Agent                   │
│  (gpt-oss:20b)                     │
│                                    │
│  - Maintains conversation memory   │
│  - Decides which tool(s) to call   │
│  - Can make multiple tool calls    │
│  - Synthesizes final answer        │
└────┬───────────────────────────────┘
     │
     ├─────────────┬─────────────────┬──────────────┐
     ▼             ▼                 ▼              ▼
standard_search  get_overview  summarize_doc   (repeat)
     │
     ├─ Searches text (query_by_type)
     ├─ Searches images (get_images_and_tables)
     │  └─ Filters with LLM (_filter_relevant_images)
     └─ Searches tables (get_images_and_tables)
     │
     ▼
Returns formatted string:
=== TEXT CONTENT ===
[Text 1] Source: manual.pdf, Page 5
Content preview...

=== IMAGES ===
[Image 1] ✓ Source: manual.pdf, Page 3
Caption: Control panel with buttons
Path: extracted_images/manual_p3_img1.png

=== TABLES ===
[Table 1] Source: manual.pdf, Page 7
Preview: Button | Function...
     │
     ▼
Agent synthesizes answer using:
- Tool results
- Conversation history
- System prompt guidelines
     │
     ▼
Final answer to user
```

## File Structure

```
DocumentAnalyzer/
│
├── config.py                      # Central configuration
│   ├── Model settings (EMBED_MODEL, VISION_MODEL, AGENT_MODEL)
│   ├── Agent settings (ENABLE_IMAGE_FILTERING, MAX_RELEVANT_IMAGES, AGENT_TEMPERATURE)
│   ├── Image captioning settings (MIN/MAX_IMAGE_SIZE, rate limits)
│   └── Vector DB settings (COLLECTION_NAME, TOP_K, distance thresholds)
│
├── src/
│   ├── rag_backend.py
│   │   ├── get_embedding_model()   # Singleton embedding instance
│   │   ├── get_vector_store()      # Singleton vector store instance
│   │   ├── reset_vector_store()    # Reset after DB updates
│   │   ├── build_vector_db()       # Stores text, images, tables
│   │   ├── retriever()             # Standard LangChain retriever
│   │   ├── query_by_type()         # Type & source filtered queries
│   │   └── get_images_and_tables() # Convenience function for visuals
│   │
│   ├── agent.py
│   │   ├── create_document_agent() # LangGraph agent with tools
│   │   ├── query_with_agent()      # Agent query interface
│   │   ├── get_memory()            # Get conversation history
│   │   ├── save_to_memory()        # Save Q&A to memory
│   │   └── clear_memory()          # Clear conversation history
│   │
│   ├── llm_singleton.py
│   │   └── get_llm()               # Singleton LLM instances (prevents circular imports)
│   │
│   ├── agent_tools.py
│   │   ├── standard_search         # Primary search tool (text + images + tables)
│   │   ├── get_overview            # List available documents
│   │   ├── summarize_document      # Retrieve document content for summarization
│   │   ├── _filter_relevant_images # LLM-based image filtering
│   │   └── _detect_document        # Source detection from query
│   │
│   ├── app.py
│   │   ├── DocumentAnalyserApp     # Main application class
│   │   ├── read_and_process_pdf()  # PDF processing pipeline
│   │   └── create_new_vector_db()  # Vector DB creation
│   │
│   ├── streamlit_webinterface.py
│   │   ├── StreamlitWebInterface   # Web UI class
│   │   ├── _handle_agent_mode()    # Agent query handling
│   │   ├── _stream_agent_events()  # Real-time event streaming
│   │   └── Display methods         # Image/table rendering
│   │
│   └── utility/
│       ├── pdf_reader.py           # PDF extraction (pypdf)
│       └── caption_images.py       # Vision model captioning
│
├── pdf_files/                     # Input PDF documents
├── extracted_images/              # Extracted image files
│   ├── manual_p3_img1.png
│   └── manual_p5_img2.png
│
├── chroma_db/                     # ChromaDB vector store
│   ├── chroma.sqlite3
│   └── 4e84390e-.../              # Collection data
│
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Key Features

✅ **Agent-Based System**: LangGraph agent with autonomous tool selection  
✅ **Intelligent Image Filtering**: LLM evaluates image relevance before returning results  
✅ **Multi-Modal RAG**: Semantic search across text, images, and tables  
✅ **Source Filtering**: Query specific documents with automatic detection  
✅ **Conversation Memory**: Agent maintains context across queries  
✅ **Vision Captioning**: Automatic image description using qwen3-vl:4b  
- **Singleton Pattern**: Memory-efficient model instance management (llm_singleton.py prevents circular imports)
✅ **Streamlit UI**: Real-time agent event streaming and result display

## Performance Considerations

- **Singleton Models**: Prevents memory leaks by reusing LLM/embedding instances
- **Images**: Only captions embedded (not raw images); files stored on disk
- **Tables**: Preview text embedded for search; full JSON in metadata
- **Rate Limiting**: Configurable delays between vision model requests
- **Image Filtering**: LLM pre-filters images (configurable: ENABLE_IMAGE_FILTERING)
- **Context Window**: Configurable OLLAMA_NUM_CTX (default: 2048 tokens)
- **Distance Thresholds**: IMAGE_DIST=0.35, TABLE_DIST=0.35 for relevance
- **Batch Processing**: Image captioning with retry logic and error handling

## Configuration Options (config.py)

### Models
- `EMBED_MODEL`: qwen3-embedding:latest (vector embeddings)
- `VISION_MODEL`: qwen3-vl:8b (image captioning)
- `AGENT_MODEL`: gpt-oss:20b (agent reasoning and all LLM operations)

### Agent Settings
- `ENABLE_IMAGE_FILTERING`: True (LLM filters relevant images)
- `MAX_RELEVANT_IMAGES`: 10 (max images returned to agent)
- `AGENT_TEMPERATURE`: 0.1 (agent temperature - lower = more focused)

### Image Processing
- `MIN_IMAGE_SIZE`: 2000 bytes (skip small icons)
- `MAX_IMAGE_SIZE`: 5MB (skip oversized images)
- `CAPTION_RATE_LIMIT_DELAY`: 1.5s (prevent Ollama overload)
- `CAPTION_MAX_RETRIES`: 3 (retry failed captions)

## How to Use

### 1. Launch Web Interface
```bash
cd week9/DocumentAnalyzer
streamlit run src/streamlit_webinterface.py
```

### 2. Upload & Process PDFs
- Upload PDF files through web interface
- System automatically extracts text, images, and tables
- Vision model generates captions (with progress bar)
- All content embedded in ChromaDB

### 3. Query with Agent
- Enter questions in natural language
- Agent decides which tools to use
- View real-time thinking process (optional)
- Results display text, images, and tables together

### 4. Conversation Mode
- Enable "Use Memory" to maintain context
- Agent remembers previous questions/answers
- Follow-up questions build on conversation history

## Next Steps

1. **Configure**: Adjust settings in `config.py` for your hardware
2. **Launch**: Run `streamlit run src/streamlit_webinterface.py`
3. **Upload**: Add PDF documents through the web interface
4. **Query**: Ask questions and let the agent find answers
5. **Explore**: Try different models and filtering options
