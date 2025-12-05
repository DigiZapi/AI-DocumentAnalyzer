"""
LangChain Tools for the Document Analyzer Agent.

These tools enable the agent to interact with the RAG infrastructure.
Simplified to 3 core tools: standard search, overview, and summarizing.
"""

from langchain.tools import tool
from typing import Optional, List, Dict
import os
import json
from rag_backend import query_by_type, get_images_and_tables
from config import AGENT_MODEL, ENABLE_IMAGE_FILTERING, MAX_RELEVANT_IMAGES
from llm_singleton import get_llm


def _filter_relevant_images(query: str, image_metas: List[Dict], max_images: int = 3) -> List[Dict]:
    """
    Use LLM to filter images based on relevance to the query.
    
    PERFORMANCE NOTE: This function makes an additional LLM call, which adds latency.
    Consider disabling in config (ENABLE_IMAGE_FILTERING = False) for faster responses.
    
    Args:
        query: The user's search query
        image_metas: List of image metadata dicts with captions
        max_images: Maximum number of relevant images to return
    
    Returns:
        List of relevant image metadata, sorted by relevance
    """
    if not image_metas:
        return []
    
    # Quick return if we already have few images
    if len(image_metas) <= max_images:
        return image_metas
    
    try:
        # Build a compact representation of images for the LLM
        image_list = []
        for idx, img in enumerate(image_metas):
            caption = img.get('caption', 'No description')
            source = img.get('source', 'Unknown')
            page = img.get('page', 'Unknown')
            # Truncate long captions for faster processing
            caption = caption[:100] + "..." if len(caption) > 100 else caption
            image_list.append(f"{idx}: {caption} (from {source}, page {page})")
        
        images_text = "\n".join(image_list[:10])  # Limit to first 10 to reduce LLM overhead
        
        # Ask LLM to select relevant images (use singleton)
        llm = get_llm(AGENT_MODEL, temperature=0.0)  # Use 0 temp for deterministic filtering
        
        # Shorter prompt for faster inference
        prompt = f"""Query: "{query}"

Images:
{images_text}

Return ONLY relevant image numbers (comma-separated) or NONE. Max {max_images}.
Relevant:"""

        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        answer = answer.strip().upper()
        
        # Parse the response
        if answer == "NONE" or not answer:
            return []
        
        # Extract numbers from response
        import re
        numbers = re.findall(r'\d+', answer)
        selected_indices = [int(n) for n in numbers if int(n) < len(image_metas)]
        
        # Return selected images in order, limited to max_images
        relevant_images = [image_metas[i] for i in selected_indices[:max_images]]
        
        return relevant_images if relevant_images else image_metas[:max_images]
        
    except Exception as e:
        print(f"Warning: Image filtering failed ({e}), returning first {max_images} images")
        # Fallback: return first max_images if LLM filtering fails
        return image_metas[:max_images]


def _detect_document(query):
    """Detect if query mentions a specific PDF document."""
    query_lower = query.lower()
    
    # Get available PDFs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    pdf_dir = os.path.join(project_root, "pdf_files")
    
    if not os.path.exists(pdf_dir):
        return None
    
    pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    # Check for document name matches
    for pdf in pdfs:
        pdf_lower = pdf.lower().replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        # Extract key words from PDF name (ignore common words)
        pdf_words = [w for w in pdf_lower.split() if len(w) > 3]
        
        # If multiple key words from PDF name appear in query, it's likely that document
        matches = sum(1 for w in pdf_words if w in query_lower)
        if matches >= 2 or (len(pdf_words) == 1 and pdf_words[0] in query_lower):
            return pdf
    
    return None


@tool
def standard_search(query: str) -> str:
    """
    Search for ALL relevant content in documents (text, images, and tables).
    
    This is your PRIMARY tool. Use it for ANY question about document content.
    It automatically searches text, finds relevant images, and locates tables.
    
    Args:
        query: The search query (e.g., "fan control settings" or "installation steps")
    
    Returns:
        Comprehensive results including text passages, images with captions, and tables
    """
    try:
        # Detect if query mentions a specific document
        source_filter = _detect_document(query)
        
        results = []
        
        # 1. Search text documents
        text_docs = query_by_type(query, doc_type="text", source=source_filter, k=5)
        if text_docs:
            results.append("=== TEXT CONTENT ===")
            for i, doc in enumerate(text_docs, 1):
                page = doc.metadata.get('page', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content[:400]
                results.append(f"\n[Text {i}] Source: {source}, Page {page}\n{content}...")
        
        # 2 & 3. Search images and tables (single call for efficiency)
        image_metas, table_metas = get_images_and_tables(query, image_k=5, table_k=3, source=source_filter)
        
        if image_metas:
            # Filter images by relevance using LLM (if enabled)
            if ENABLE_IMAGE_FILTERING:
                relevant_images = _filter_relevant_images(query, image_metas, max_images=MAX_RELEVANT_IMAGES)
            else:
                relevant_images = image_metas[:MAX_RELEVANT_IMAGES]
            
            if relevant_images:
                results.append("\n\n=== IMAGES ===")
                for i, img in enumerate(relevant_images, 1):
                    path = img.get('path')
                    exists = "✓" if (path and os.path.exists(path)) else "✗"
                    caption = img.get('caption', 'No description')
                    results.append(f"\n[Image {i}] {exists} Source: {img.get('source')}, Page {img.get('page')}")
                    results.append(f"Caption: {caption}")
                    results.append(f"Path: {path}")
        
        # Tables retrieved in same call above
        if table_metas:
            results.append("\n\n=== TABLES ===")
            for i, tb in enumerate(table_metas, 1):
                preview = tb.get('preview', 'No preview')
                if len(preview) > 300:
                    preview = preview[:300] + "..."
                results.append(f"\n[Table {i}] Source: {tb.get('source')}, Page {tb.get('page')}")
                results.append(f"Preview: {preview}")
        
        if not text_docs and not image_metas and not table_metas:
            return "NO_RELEVANT_INFORMATION: No content found in the vectorstore for this query."
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error in standard search: {str(e)}"


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
    try:
        # Get project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        pdf_files_dir = os.path.join(project_root, "pdf_files")
        
        if not os.path.exists(pdf_files_dir):
            return "No pdf_files directory found."
        
        pdf_files = [f for f in os.listdir(pdf_files_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            return "No PDF files available."
        
        result = f"Available documents ({len(pdf_files)}):\n\n"
        for i, pdf_file in enumerate(sorted(pdf_files), 1):
            pdf_path = os.path.join(pdf_files_dir, pdf_file)
            size = os.path.getsize(pdf_path) / 1024  # KB
            result += f"{i}. {pdf_file} ({size:.1f} KB)\n"
        
        return result
    except Exception as e:
        return f"Error retrieving document overview: {str(e)}"


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
    try:
        from rag_backend import get_vector_store
        
        # Ensure .pdf extension
        if not document_name.lower().endswith('.pdf'):
            document_name = document_name + '.pdf'
        
        # Get vector store (cached)
        vector_store = get_vector_store()
        
        # Get text chunks from the database
        # Use a broad query to get documents, then filter by source
        doc_chunks = vector_store.similarity_search(
            query="introduction overview document content",
            k=100,
            filter={"type": "text"}
        )
        
        # Filter to only chunks from the requested document
        # Source field contains just the filename (e.g., 'manual.pdf')
        # Use fuzzy matching to handle variations (e.g., _ vs -, spaces, case)
        def normalize_name(name):
            """Normalize document name for comparison."""
            return name.lower().replace('_', '').replace('-', '').replace(' ', '')
        
        normalized_target = normalize_name(document_name)
        
        filtered_chunks = [
            doc for doc in doc_chunks 
            if normalize_name(doc.metadata.get('source', '')) == normalized_target
        ]
        
        if not filtered_chunks:
            # Try partial match if exact match fails
            filtered_chunks = [
                doc for doc in doc_chunks
                if normalized_target in normalize_name(doc.metadata.get('source', ''))
                or normalize_name(doc.metadata.get('source', '')) in normalized_target
            ]
        
        if not filtered_chunks:
            # Show what documents we actually have
            available_sources = set(doc.metadata.get('source', 'unknown') for doc in doc_chunks)
            sources_str = ", ".join(sorted(available_sources)[:5])
            return f"NO_RELEVANT_INFORMATION: No content found for document '{document_name}'. Available documents in database: {sources_str}. Use get_overview to see all documents."
        
        # Return text chunks for the agent to summarize
        # Limit to avoid token overflow
        combined_text = "\n\n".join([doc.page_content for doc in filtered_chunks[:20]])
        
        return f"=== DOCUMENT CONTENT: {document_name} ===\n\n{combined_text[:4000]}\n\n[Note: Provide a comprehensive summary of this document including main topics, key points, and purpose.]"
        
    except Exception as e:
        return f"Error retrieving document: {str(e)}"



