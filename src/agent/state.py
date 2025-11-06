"""
Agent state definition for LangGraph.
"""
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """
    State schema for the RAG agent.
    
    This state flows through all nodes in the graph and accumulates
    information as the agent processes the query.
    """
    # Input
    question: str  # Current user question
    session_id: str  # Session identifier for memory
    
    # Chat history (managed by LangGraph)
    messages: Annotated[List[Dict[str, str]], add_messages]
    
    # Agent reasoning
    agent_thought: str  # Agent's current reasoning
    next_action: str  # Decided action: "retrieve", "answer", "insufficient"
    
    # Retrieval
    retrieval_query: str  # Query to use for retrieval
    retrieved_docs: List[Dict[str, any]]  # Documents from vector store
    formatted_context: str  # Formatted context for LLM
    
    # Validation
    has_sufficient_info: bool  # Whether docs have enough info
    validation_reason: str  # Why validation passed/failed
    
    # Output
    final_answer: str  # Final answer to return to user
    citations: List[str]  # Source citations
    
    # Control flow
    iteration_count: int  # Number of reasoning iterations
    should_continue: bool  # Whether to continue processing
