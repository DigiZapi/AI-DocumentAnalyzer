"""
Configuration file for AI Document Analyser.

Adjust these settings to optimize performance and prevent Ollama crashes.
"""

# ============================================================================
# OLLAMA SERVER SETTINGS
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 240  # seconds
OLLAMA_NUM_CTX = 2048  # context window (lower = less memory)

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Embedding model for vector search
EMBED_MODEL = "qwen3-embedding:latest"

# Vision model for image captioning
VISION_MODEL = "qwen3-vl:8b"

# Agent model for document analysis (handles all LLM operations)

AGENT_MODEL = "gpt-oss:20b"  

# Agent temperature (0.1 = more focused, 0.7 = more creative)
AGENT_TEMPERATURE = 0.1


# ============================================================================
# AGENT SETTINGS
# ============================================================================

# Enable intelligent image filtering (uses LLM to filter relevant images)
ENABLE_IMAGE_FILTERING = True

# Maximum number of images to return to agent
MAX_RELEVANT_IMAGES = 10

# ============================================================================
# IMAGE CAPTIONING SETTINGS
# ============================================================================

# Skip images smaller than this (bytes) - likely icons/buttons
MIN_IMAGE_SIZE = 2000  # 2 KB

# Skip images larger than this (bytes) - too large to process
MAX_IMAGE_SIZE = 5_000_000  # 5 MB

# Delay between image caption requests (seconds) - prevents overwhelming Ollama
CAPTION_RATE_LIMIT_DELAY = 1.5

# Number of retry attempts for failed caption requests
CAPTION_MAX_RETRIES = 3

# Delay before retrying a failed request (seconds)
CAPTION_RETRY_DELAY = 3

# ============================================================================
# VECTOR DATABASE SETTINGS
# ============================================================================

COLLECTION_NAME = "pdf_content"
TOP_K = 6  # Number of results to retrieve
IMAGE_DIST = 0.35  # Distance threshold for image similarity
TABLE_DIST = 0.35  # Distance threshold for table similarity

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# If you're experiencing frequent crashes, try these adjustments:
#
# 1. Reduce context window:
#    OLLAMA_NUM_CTX = 1024  # or even 512
#
# 2. Increase delays:
#    CAPTION_RATE_LIMIT_DELAY = 1.0  # or 2.0
#
# 3. Skip more images:
#    MIN_IMAGE_SIZE = 5000  # 5 KB
#
# 4. Use smaller models:
#    LLM_MODEL = "llama3.2:1b"      # smaller variant
#    VISION_MODEL = "qwen3-vl:4b"   # alternative vision model
#
# 5. Increase timeout:
#    OLLAMA_TIMEOUT = 300  # 5 minutes
