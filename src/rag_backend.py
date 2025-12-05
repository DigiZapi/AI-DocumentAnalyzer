from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from PIL import Image
import chromadb
import pandas as pd
import os


# settings
OLLAMA_EMBED_MODEL = "qwen3-embedding:latest"
COLLECTION_NAME = "pdf_content"

# Use absolute path for DB directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DB_DIR = os.path.join(_PROJECT_ROOT, "chroma_db")

TOP_K = 6
IMAGE_DIST = 0.35
TABLE_DIST = 0.35

# Singleton instances to avoid memory leaks
_EMBED_MODEL_INSTANCE = None
_VECTOR_STORE_INSTANCE = None


def get_embedding_model():
    """Get or create a singleton embedding model instance.
    
    This prevents memory leaks from creating multiple model instances.
    Ollama models are heavy and should be reused.
    """
    global _EMBED_MODEL_INSTANCE
    if _EMBED_MODEL_INSTANCE is None:
        _EMBED_MODEL_INSTANCE = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url="http://localhost:11434"
        )
    return _EMBED_MODEL_INSTANCE


def get_vector_store():
    """Get or create a singleton vector store instance.
    
    This prevents reloading embeddings from disk on every query.
    Call reset_vector_store() after updating the database.
    """
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        embed_model = get_embedding_model()
        _VECTOR_STORE_INSTANCE = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embed_model,
            persist_directory=DB_DIR,
        )
    return _VECTOR_STORE_INSTANCE


def reset_vector_store():
    """Reset the vector store singleton (call after DB updates)."""
    global _VECTOR_STORE_INSTANCE
    _VECTOR_STORE_INSTANCE = None


# ----------------------------------------------------------
# Build Vector DB (LangChain Chroma)
# ----------------------------------------------------------


def build_vector_db(texts, images, tables, captions):
    """Build or update a Chroma vector store using LangChain's wrapper.

    All chunks (text, image captions, table previews) are stored as
    documents with metadata and queried via LangChain interfaces.
    """

    print("Building vector DB (LangChain Chroma)...")

    embed_model = get_embedding_model()

    # Prepare documents and metadatas
    documents = []
    metadatas = []
    ids = []

    # text
    for t in texts:
        documents.append(t["text"])
        metadatas.append({
            "type": "text",
            "page": t["page"],
            "content": t["text"],
            "source": t.get("source"),
        })
        ids.append(t["id"])

    # images – embed via caption text
    for im in images:
        caption_text = captions.get(im["id"], "")
        
        # If caption is empty or too short, create a descriptive fallback
        # This ensures images can be found via semantic search
        if not caption_text or len(caption_text.strip()) < 10:
            source_name = im.get("source", "document")
            page_num = im["page"]
            caption_text = f"Image from {source_name} on page {page_num}. Visual content, diagram, chart, or illustration."
        
        documents.append(caption_text)
        metadatas.append({
            "type": "image",
            "page": im["page"],
            "path": im["path"],
            "caption": caption_text,
            "source": im.get("source"),
        })
        ids.append(im["id"])

    # tables – embed via textual preview
    for tb in tables:
        preview = tb.get("preview", "")
        documents.append(preview)
        
        # Convert table to JSON string properly
        import json
        table_json = json.dumps(tb.get("table", []))
        
        metadatas.append({
            "type": "table",
            "page": tb["page"],
            "json": table_json,
            "preview": preview,
            "source": tb.get("source"),
        })
        ids.append(tb["id"])

    if not documents:
        print("No documents to add to vector DB.")
        return

    # Use LangChain's Chroma wrapper with a persistent directory and
    # deterministic collection name. `ids` ensures stable document IDs.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embed_model,
        persist_directory=DB_DIR,
    )

    # Add or update documents. If some IDs already exist, Chroma will
    # keep the latest version for that ID in the collection.
    vector_store.add_texts(texts=documents, metadatas=metadatas, ids=ids)
    vector_store.persist()
    
    # Reset singleton so next query uses updated data
    reset_vector_store()

    print(f"Added/updated {len(documents)} documents in the vector DB.")


def retriever():
    """Return a LangChain-style retriever backed by Chroma."""
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


def query_by_type(query_text, doc_type=None, source=None, k=TOP_K):
    """
    Query the vector store and optionally filter by document type and/or source.
    
    Args:
        query_text: The search query
        doc_type: Optional filter ('text', 'image', 'table', or None for all)
        source: Optional source PDF filename filter (e.g., 'manual.pdf')
        k: Number of results to return
        
    Returns:
        List of documents with their metadata
    """
    vector_store = get_vector_store()
    
    # Build filter dict - ChromaDB requires $and for multiple conditions
    filter_dict = None
    if doc_type and source:
        # Multiple conditions - use $and
        filter_dict = {
            "$and": [
                {"type": {"$eq": doc_type}},
                {"source": {"$eq": source}}
            ]
        }
    elif doc_type:
        filter_dict = {"type": {"$eq": doc_type}}
    elif source:
        filter_dict = {"source": {"$eq": source}}
    
    # Query with or without filter
    if filter_dict:
        docs = vector_store.similarity_search(
            query_text, 
            k=k,
            filter=filter_dict
        )
    else:
        docs = vector_store.similarity_search(query_text, k=k)
    
    return docs


def get_images_and_tables(query_text, image_k=3, table_k=3, source=None):
    """
    Retrieve only images and tables for a given query.
    
    Args:
        query_text: The search query
        image_k: Number of images to retrieve
        table_k: Number of tables to retrieve
        source: Optional source PDF filename filter
    
    Returns:
        tuple: (image_docs, table_docs) where each is a list of metadata dicts
    """
    # If both are 0, return empty
    if image_k <= 0 and table_k <= 0:
        return [], []
    
    # If only one type requested, use the optimized query_by_type
    if image_k <= 0:
        tables = query_by_type(query_text, doc_type="table", source=source, k=table_k)
        return [], [doc.metadata for doc in tables]
    
    if table_k <= 0:
        images = query_by_type(query_text, doc_type="image", source=source, k=image_k)
        return [doc.metadata for doc in images], []
    
    # Both requested - retrieve more results and split by type (single query)
    # This is faster than 2 separate queries
    total_k = image_k + table_k
    vector_store = get_vector_store()
    
    # Build filter for images and tables only
    filter_dict = None
    if source:
        # Filter by source and type (image OR table)
        filter_dict = {
            "$and": [
                {"source": {"$eq": source}},
                {"type": {"$in": ["image", "table"]}}
            ]
        }
    else:
        # Just filter by type
        filter_dict = {"type": {"$in": ["image", "table"]}}
    
    # Single query for both types
    docs = vector_store.similarity_search(query_text, k=total_k, filter=filter_dict)
    
    # Split results by type
    image_docs = [doc for doc in docs if doc.metadata.get('type') == 'image'][:image_k]
    table_docs = [doc for doc in docs if doc.metadata.get('type') == 'table'][:table_k]
    
    return [doc.metadata for doc in image_docs], [doc.metadata for doc in table_docs]