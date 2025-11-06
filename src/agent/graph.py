"""
LangGraph definition for RAG agent.
"""
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import AgentNodes
from ..llm.ollama_client import OllamaClient
from ..retrieval.retriever import DocumentRetriever
from ..retrieval.vector_store import VectorStore
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGAgent:
    """RAG Agent orchestrator using LangGraph."""
    
    def __init__(
        self,
        llm_client: OllamaClient = None,
        vector_store: VectorStore = None
    ):
        """
        Initialize RAG agent with LangGraph.
        
        Args:
            llm_client: Ollama client (creates default if None)
            vector_store: Vector store (creates default if None)
        """
        self.logger = logger
        self.logger.info("Initializing RAG Agent")
        
        # Initialize components
        self.llm_client = llm_client or OllamaClient()
        self.vector_store = vector_store or VectorStore()
        self.retriever = DocumentRetriever(self.vector_store)
        
        # Initialize nodes
        self.nodes = AgentNodes(self.llm_client, self.retriever)
        
        # Build graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        self.logger.info("RAG Agent initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Graph structure:
        
        START
          ↓
        reasoning_node → [decide action]
          ↓              ↓             ↓
        retrieve    answer    insufficient
          ↓
        validation_node → [sufficient?]
          ↓              ↓
        answer    insufficient
          ↓              ↓
        END           END
        """
        self.logger.info("Building agent graph")
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("reasoning", self.nodes.reasoning_node)
        workflow.add_node("retrieval", self.nodes.retrieval_node)
        workflow.add_node("validation", self.nodes.validation_node)
        workflow.add_node("answer", self.nodes.answer_node)
        workflow.add_node("insufficient", self.nodes.insufficient_info_node)
        
        # Set entry point
        workflow.set_entry_point("reasoning")
        
        # Add conditional edges from reasoning
        workflow.add_conditional_edges(
            "reasoning",
            self.nodes.should_continue,
            {
                "retrieve": "retrieval",
                "answer": "answer",
                "insufficient": "insufficient"
            }
        )
        
        # Retrieval always goes to validation
        workflow.add_edge("retrieval", "validation")
        
        # Add conditional edges from validation
        workflow.add_conditional_edges(
            "validation",
            self.nodes.should_answer_or_retry,
            {
                "answer": "answer",
                "insufficient": "insufficient"
            }
        )
        
        # Terminal nodes
        workflow.add_edge("answer", END)
        workflow.add_edge("insufficient", END)
        
        self.logger.info("Agent graph built successfully")
        
        return workflow
    
    def query(self, question: str, session_id: str = "default") -> str:
        """
        Process a query through the agent.
        
        Args:
            question: User's question
            session_id: Session identifier for chat history
        
        Returns:
            Agent's answer
        """
        self.logger.info(f"Processing query: '{question}'")
        
        # Initialize state
        initial_state = {
            "question": question,
            "session_id": session_id,
            "messages": [],
            "agent_thought": "",
            "next_action": "",
            "retrieval_query": "",
            "retrieved_docs": [],
            "formatted_context": "",
            "has_sufficient_info": False,
            "validation_reason": "",
            "final_answer": "",
            "citations": [],
            "iteration_count": 0,
            "should_continue": True
        }
        
        try:
            # Run the graph
            result = self.app.invoke(initial_state)
            
            answer = result.get("final_answer", "I apologize, but I couldn't generate an answer.")
            
            self.logger.info("Query processed successfully")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error processing your question: {str(e)}"
    
    def get_graph_visualization(self) -> str:
        """
        Get a text representation of the graph structure.
        
        Returns:
            Graph structure as string
        """
        return """
RAG Agent Graph Structure:

    START
      ↓
    [REASONING NODE]
    - Analyzes question
    - Decides action
      ↓
    ┌─────────┴─────────┐
    ↓                   ↓
[RETRIEVE]          [ANSWER from history]
    ↓                   ↓
[RETRIEVAL NODE]       END
    ↓
[VALIDATION NODE]
- Checks if docs sufficient
    ↓
    ┌─────────┴─────────┐
    ↓                   ↓
[ANSWER]         [INSUFFICIENT]
    ↓                   ↓
   END                 END
"""
