"""
Singleton LLM instances to prevent memory leaks and improve performance.

This module avoids circular imports between agent.py and agent_tools.py.
"""

from langchain_ollama import ChatOllama
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AGENT_MODEL

# Singleton LLM instances to prevent memory leaks
_LLM_INSTANCES = {}


def get_llm(model_name: str = AGENT_MODEL, temperature: float = 0.1):
    """Get or create a singleton LLM instance.
    
    This prevents memory leaks from creating multiple LLM instances.
    Each model+temperature combination gets one shared instance.
    
    Args:
        model_name: Name of the Ollama model
        temperature: Temperature for generation
    
    Returns:
        ChatOllama instance
    """
    key = f"{model_name}_{temperature}"
    if key not in _LLM_INSTANCES:
        _LLM_INSTANCES[key] = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
            timeout=120,
            num_ctx=2048
        )
    return _LLM_INSTANCES[key]
