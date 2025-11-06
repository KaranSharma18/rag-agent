"""
Agent nodes for LangGraph RAG agent.
"""
from typing import Dict, Any
from .state import AgentState
from .prompts import (
    REASONING_PROMPT,
    VALIDATION_PROMPT,
    ANSWER_PROMPT,
    INSUFFICIENT_INFO_PROMPT,
    CHAT_HISTORY_ANSWER_PROMPT,
    SYSTEM_PROMPT
)
from ..llm.ollama_client import OllamaClient
from ..retrieval.retriever import DocumentRetriever
from ..utils.logger import setup_logger, log_agent_step
from ..utils.config import MAX_ITERATIONS

logger = setup_logger(__name__)


class AgentNodes:
    """Collection of node functions for the RAG agent graph."""
    
    def __init__(self, llm_client: OllamaClient, retriever: DocumentRetriever):
        """
        Initialize agent nodes with required dependencies.
        
        Args:
            llm_client: Ollama client for LLM calls
            retriever: Document retriever for RAG
        """
        self.llm = llm_client
        self.retriever = retriever
        self.logger = logger
    
    def reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Agent analyzes the question and decides next action.
        
        This is where the agent "thinks" about what to do.
        """
        log_agent_step(self.logger, "REASONING", {
            "question": state["question"],
            "iteration": state.get("iteration_count", 0)
        })
        
        # Format chat history
        messages = state.get("messages", [])
        chat_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages[-4:]  # Last 4 messages for context
        ]) if messages else "No previous conversation"
        
        # Generate reasoning prompt
        prompt = REASONING_PROMPT.format(
            question=state["question"],
            chat_history=chat_history
        )
        
        # Get agent's reasoning
        response = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        
        self.logger.info(f"Agent reasoning:\n{response}")
        
        # Parse response
        action = "retrieve"  # default
        query = state["question"]  # default
        
        if "ACTION: RETRIEVE" in response.upper():
            action = "retrieve"
            # Extract query if provided
            for line in response.split("\n"):
                if line.startswith("QUERY:"):
                    query = line.replace("QUERY:", "").strip()
        elif "ACTION: ANSWER" in response.upper():
            action = "answer"
        elif "ACTION: INSUFFICIENT" in response.upper():
            action = "insufficient"
        
        return {
            "agent_thought": response,
            "next_action": action,
            "retrieval_query": query,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
    
    def retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute document retrieval based on agent's query.
        """
        query = state["retrieval_query"]
        
        log_agent_step(self.logger, "RETRIEVAL", {
            "query": query
        })
        
        # Retrieve documents
        docs = self.retriever.retrieve(query)
        
        # Format context
        formatted_context = self.retriever.format_context(docs)
        
        # Extract citations
        citations = self.retriever.get_citations(docs)
        
        self.logger.info(f"Retrieved {len(docs)} documents")
        
        return {
            "retrieved_docs": docs,
            "formatted_context": formatted_context,
            "citations": citations
        }
    
    def validation_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate if retrieved documents have sufficient information.
        
        This is a key guardrail - checks if we can actually answer.
        """
        log_agent_step(self.logger, "VALIDATION", {
            "question": state["question"],
            "num_docs": len(state.get("retrieved_docs", []))
        })
        
        context = state.get("formatted_context", "")
        
        if not context or context == "No relevant documents found.":
            return {
                "has_sufficient_info": False,
                "validation_reason": "No relevant documents found in the database"
            }
        
        # Ask LLM to validate
        prompt = VALIDATION_PROMPT.format(
            question=state["question"],
            context=context
        )
        
        response = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        
        self.logger.info(f"Validation result:\n{response}")
        
        # Parse validation result
        sufficient = "SUFFICIENT: YES" in response.upper()
        
        # Extract reason
        reason = ""
        for line in response.split("\n"):
            if line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
        
        return {
            "has_sufficient_info": sufficient,
            "validation_reason": reason or response
        }
    
    def answer_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate final answer with citations.
        """
        log_agent_step(self.logger, "ANSWER GENERATION", {
            "question": state["question"],
            "has_context": bool(state.get("formatted_context"))
        })
        
        # Check if we're answering from history or from retrieved docs
        if state.get("next_action") == "answer" and not state.get("formatted_context"):
            # Answer from chat history
            messages = state.get("messages", [])
            chat_history = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in messages[-6:]
            ])
            
            prompt = CHAT_HISTORY_ANSWER_PROMPT.format(
                chat_history=chat_history,
                question=state["question"]
            )
        else:
            # Answer from retrieved documents
            context = state.get("formatted_context", "No context available")
            
            prompt = ANSWER_PROMPT.format(
                question=state["question"],
                context=context
            )
        
        # Generate answer
        answer = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        
        self.logger.info(f"Generated answer:\n{answer[:200]}...")
        
        # Update messages
        new_messages = [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer}
        ]
        
        return {
            "final_answer": answer,
            "messages": new_messages,
            "should_continue": False
        }
    
    def insufficient_info_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle cases where we cannot answer from documents.
        """
        log_agent_step(self.logger, "INSUFFICIENT INFO", {
            "reason": state.get("validation_reason", "Unknown")
        })
        
        reason = state.get("validation_reason", "Information not found in documents")
        
        prompt = INSUFFICIENT_INFO_PROMPT.format(
            question=state["question"],
            reason=reason
        )
        
        answer = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        
        self.logger.info(f"Insufficient info response:\n{answer}")
        
        # Update messages
        new_messages = [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer}
        ]
        
        return {
            "final_answer": answer,
            "messages": new_messages,
            "should_continue": False
        }
    
    def should_continue(self, state: AgentState) -> str:
        """
        Routing function to determine next node.
        
        This is called after reasoning_node to route to appropriate node.
        """
        action = state.get("next_action", "retrieve")
        iteration = state.get("iteration_count", 0)
        
        # Safety: prevent infinite loops
        if iteration >= MAX_ITERATIONS:
            self.logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached")
            return "answer"
        
        self.logger.info(f"Routing decision: {action}")
        
        return action
    
    def should_answer_or_retry(self, state: AgentState) -> str:
        """
        Routing function after validation.
        
        Decides whether to answer or declare insufficient info.
        """
        sufficient = state.get("has_sufficient_info", False)
        
        if sufficient:
            self.logger.info("Validation passed - proceeding to answer")
            return "answer"
        else:
            self.logger.info("Validation failed - insufficient information")
            return "insufficient"
