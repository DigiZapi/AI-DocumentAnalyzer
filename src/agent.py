"""
LangChain Agent for the Document Analyzer.

The agent uses various tools to answer questions about documents.
It can autonomously decide which tools to use.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AGENT_MODEL

from llm_singleton import get_llm
from agent_tools import (
    standard_search,
    get_overview,
    summarize_document
)


# Tools that the agent can use
AGENT_TOOLS = [
    standard_search,
    get_overview,
    summarize_document
]

# Simple conversation memory - list of (question, answer) tuples
_CONVERSATION_HISTORY = []


def get_memory():
    """Get the conversation history as LangChain messages."""
    global _CONVERSATION_HISTORY
    messages = []
    for question, answer in _CONVERSATION_HISTORY:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages


def save_to_memory(question: str, answer: str):
    """Save a question-answer pair to memory."""
    global _CONVERSATION_HISTORY
    _CONVERSATION_HISTORY.append((question, answer))


def clear_memory():
    """Clear the conversation memory."""
    global _CONVERSATION_HISTORY
    _CONVERSATION_HISTORY = []


def create_document_agent(model_name: str = AGENT_MODEL, temperature: float = 0.1):
    """
    Create a Document Analyzer agent.
    
    Args:
        model_name: Name of the Ollama model (default: from config.AGENT_MODEL)
        temperature: Temperature for LLM generation (default: 0.1)
    
    Returns:
        Agent: The configured agent
    """
    # ChatOllama for Tool-Calling (supports bind_tools)
    llm = get_llm(model_name, temperature)
    
    # Create agent with LangGraph (latest LangChain version)
    # Note: create_react_agent doesn't support state_modifier directly
    # We'll inject the system prompt into each query instead
    agent = create_react_agent(
        model=llm,
        tools=AGENT_TOOLS
    )
    
    return agent


def _get_system_instructions() -> str:
    """
    Get system instructions for the agent.
    
    Returns:
        System instructions string
    """
    return """You are a document analysis assistant with 3 tools:

    1. standard_search - Use this for ANY content question (automatically searches text, images, and tables)
    2. get_overview - Use ONLY when user asks "what documents are available?"
    3. summarize_document - Use ONLY when user asks for a summary of a specific document

    CRITICAL INSTRUCTIONS:
    
    FOR get_overview tool:
    - Return the tool output EXACTLY as your final answer
    - Do NOT add any introduction or formatting
    - Just return it verbatim
    
    FOR summarize_document tool:
    - The tool returns document content
    - YOU must generate a comprehensive summary based on that content
    - Include main topics, key points, and purpose
    - Format: "Summary of '[document_name]':\\n\\n[your summary]"
    
    FOR standard_search tool:
    - Answer the question directly and concisely
    - Extract and present ONLY the specific information requested
    - Do NOT create structured summaries, numbered lists, or overviews unless explicitly asked
    - Do NOT add introductions like "Here's a summary" or "Here's what I found"
    - If tool says "NO_RELEVANT_INFORMATION" → Say you don't have that information
    - Always base your answer ONLY on what the tool returned
    - Be brief and to the point"""


def _prepare_agent_query(question: str, use_memory: bool):
    """
    Prepare the agent query by handling system instructions and message history.
    
    Args:
        question: The user's question
        use_memory: Whether to use conversation memory
    
    Returns:
        tuple: (enhanced_question, messages)
    """
    # Get system instructions
    system_instructions = _get_system_instructions()
    
    # Create enhanced question with system instructions
    enhanced_question = f"""[System Instructions: {system_instructions}]

User Question: {question}"""
    
    # Build messages with history
    messages = []
    if use_memory:
        history_messages = get_memory()
        messages.extend(history_messages)
    
    # Add current question
    messages.append(("user", enhanced_question))
    
    return enhanced_question, messages


def query_with_agent(question: str, model_name: str = AGENT_MODEL, use_memory: bool = True) -> dict:
    """
    Ask a question to the agent.
    
    Args:
        question: The user's question
        model_name: The LLM model to use
        use_memory: Whether to use conversation memory (default: True)
    
    Returns:
        dict with 'output' and 'messages'
    """
    result = {"output": "", "messages": []}
    
    # Prepare query using shared helper
    enhanced_question, messages = _prepare_agent_query(question, use_memory)
    
    # Execute with agent
    agent = create_document_agent(model_name=model_name)
    agent_result = agent.invoke({"messages": messages})
    
    # Extract the answer from the result
    if "messages" in agent_result:
        last_message = agent_result["messages"][-1]
        output = last_message.content if hasattr(last_message, 'content') else str(last_message)
    else:
        output = str(agent_result)
    
    result["output"] = output
    result["messages"] = agent_result.get("messages", [])
    
    # Save to memory if enabled
    if use_memory:
        save_to_memory(question, output)
    
    return result


def stream_agent(question: str, model_name: str = AGENT_MODEL, use_memory: bool = True):
    """
    Stream the agent's response step by step (for Streamlit).
    
    Args:
        question: The user's question
        model_name: The LLM model to use
        use_memory: Whether to use conversation memory (default: True)
    
    Yields:
        dict with 'type' ('tool_call', 'tool_result', 'thinking', 'answer') and relevant data
    """
    # Prepare query using shared helper
    enhanced_question, messages = _prepare_agent_query(question, use_memory)
    
    # Execute with agent and stream results
    agent = create_document_agent(model_name=model_name)
    
    # Track which messages we've already seen
    seen_messages = 0
    has_used_tools = False
    final_answer = None
    last_tool_used = None
    last_tool_result = None
    
    # Stream through all steps - stream_mode="values" returns the entire state
    for chunk in agent.stream({"messages": messages}, stream_mode="values"):
        # chunk contains the complete state with all messages
        if "messages" in chunk:
            chunk_messages = chunk["messages"]
            
            # Only process new messages
            new_messages = chunk_messages[seen_messages:]
            seen_messages = len(chunk_messages)
            
            for msg in new_messages:
                msg_type = getattr(msg, '__class__', None)
                
                # AIMessage with tool_calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    has_used_tools = True
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        last_tool_used = tool_name
                        yield {
                            'type': 'tool_call',
                            'tool': tool_name,
                            'input': tool_call.get('args', {}),
                            'id': tool_call.get('id')
                        }
                
                # ToolMessage (result of a tool call)
                elif msg_type and msg_type.__name__ == 'ToolMessage':
                    last_tool_result = msg.content
                    yield {
                        'type': 'tool_result',
                        'content': msg.content,
                        'tool_call_id': getattr(msg, 'tool_call_id', None)
                    }
                
                # AIMessage without tool_calls (thinking or final answer)
                elif msg_type and msg_type.__name__ == 'AIMessage':
                    if hasattr(msg, 'content') and msg.content:
                        # Only if there are no tool_calls
                        if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                            # If we've used tools before, this is the final answer
                            if has_used_tools:
                                # For get_overview, use tool output directly
                                # For summarize_document, agent generates summary from content
                                if last_tool_used == 'get_overview' and last_tool_result:
                                    final_answer = last_tool_result
                                    yield {
                                        'type': 'answer',
                                        'content': last_tool_result
                                    }
                                else:
                                    # For other tools, use agent's response
                                    final_answer = msg.content
                                    yield {
                                        'type': 'answer',
                                        'content': msg.content
                                    }
                            else:
                                # Otherwise it's thinking
                                yield {
                                    'type': 'thinking',
                                    'content': msg.content
                                }
    
    # Save to memory if enabled
    if use_memory and final_answer:
        save_to_memory(question, final_answer)


