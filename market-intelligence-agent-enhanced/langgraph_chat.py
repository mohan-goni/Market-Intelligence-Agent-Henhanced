import os
import uuid
from typing import TypedDict, List, Annotated, Sequence, Tuple

from langgraph.graph import StateGraph, END
# For now, we'll manage memory explicitly via Supabase calls, not using SqliteSaver or a full BaseChatMessageHistory implementation.
# from langgraph.checkpoint.sqlite import SqliteSaver 

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Supabase client functions for chat history
from market_intelligence_agent_enhanced.supabase_client import (
    save_chat_message, 
    get_chat_history_for_session,
    # supabase # Direct client usage if needed, but prefer specific functions
)

# For Agent and LLM - these will be initialized in app.py and accessed/passed here.
# This avoids circular imports if app.py needs to import run_chat_graph.
# A setup function could be used to pass these from app.py to this module.
# For now, call_agent_node will directly use these assuming they are loaded in app.py
from market_intelligence_agent_enhanced.app import llm, create_market_intelligence_agent 

# --- Define Graph State ---
class ChatGraphState(TypedDict):
    user_id: str
    session_id: str
    input_query: str
    chat_history: List[Tuple[str, str]] # List of (type, content) tuples e.g. [("human", "hi"), ("ai", "hello")]
    agent_response: str
    # error: Optional[str] # For potential error handling within graph

# --- Graph Nodes ---

async def load_history_node(state: ChatGraphState) -> ChatGraphState:
    print(f"--- Langgraph: Loading history for user {state['user_id']}, session {state['session_id']} ---")
    history_records = await get_chat_history_for_session(
        user_id=state['user_id'], 
        session_id=state['session_id'],
        limit=20 # Get recent history
    )
    
    loaded_history = []
    if history_records:
        for record in history_records:
            # Assuming record has 'message_type' ('human', 'ai') and 'message_content'
            loaded_history.append((record.get('message_type', 'unknown'), record.get('message_content', '')))
    
    state['chat_history'] = loaded_history
    print(f"--- Langgraph: Loaded history: {loaded_history} ---")
    return state

async def call_agent_node(state: ChatGraphState) -> ChatGraphState:
    print(f"--- Langgraph: Calling agent for user {state['user_id']}, session {state['session_id']} ---")
    if not llm: # LLM should be initialized in app.py's startup
        print("--- Langgraph Error: LLM not initialized! ---")
        state['agent_response'] = "Error: The underlying language model is not available."
        # state['error'] = "LLM not initialized"
        return state

    agent_executor = create_market_intelligence_agent(llm, user_id=state['user_id'])
    if not agent_executor:
        print("--- Langgraph Error: Could not create market intelligence agent! ---")
        state['agent_response'] = "Error: Could not initialize the market intelligence agent."
        # state['error'] = "Agent creation failed"
        return state

    # Simple history formatting for context (can be improved)
    history_for_prompt = "\n".join([f"{msg_type.capitalize()}: {msg_content}" for msg_type, msg_content in state.get('chat_history', [])])
    
    # Construct a prompt that includes history if available.
    # The agent itself might have its own memory, but this ensures context.
    # This is a simplified way to pass history. A more robust way is if the agent itself handles a memory object.
    # For now, we prepend history to the input query.
    
    prompt_with_history = state['input_query']
    if history_for_prompt:
        prompt_with_history = f"Previous conversation:\n{history_for_prompt}\n\nNew user query: {state['input_query']}"
    
    print(f"--- Langgraph: Agent prompt (with history):\n{prompt_with_history}\n---")

    try:
        # Use agent_executor.arun for async execution if the agent supports it
        response = await agent_executor.arun(prompt_with_history)
        state['agent_response'] = response
    except Exception as e:
        print(f"--- Langgraph Error: Agent execution failed: {e} ---")
        state['agent_response'] = f"Error during agent execution: {str(e)}"
        # state['error'] = str(e)
    
    print(f"--- Langgraph: Agent response: {state['agent_response'][:200]}... ---")
    return state

async def save_messages_node(state: ChatGraphState) -> ChatGraphState:
    print(f"--- Langgraph: Saving messages for user {state['user_id']}, session {state['session_id']} ---")
    
    # Save user message
    await save_chat_message(
        user_id=state['user_id'],
        session_id=state['session_id'],
        message_type="human",
        message_content=state['input_query']
        # metadata can be added if needed
    )
    
    # Save AI response (only if there's no error message that implies a failure before response)
    # if not state.get('error') and state.get('agent_response'):
    if state.get('agent_response'): # Save even if agent response indicates an error from its side
        await save_chat_message(
            user_id=state['user_id'],
            session_id=state['session_id'],
            message_type="ai",
            message_content=state['agent_response']
            # metadata can be added if needed
        )
    print(f"--- Langgraph: Messages saved. User: '{state['input_query']}', AI: '{state['agent_response'][:100]}...' ---")
    return state

# --- Build the Graph ---
workflow = StateGraph(ChatGraphState)

workflow.add_node("load_history", load_history_node)
workflow.add_node("call_agent", call_agent_node)
workflow.add_node("save_messages", save_messages_node)

workflow.set_entry_point("load_history")

workflow.add_edge("load_history", "call_agent")
workflow.add_edge("call_agent", "save_messages")
workflow.add_edge("save_messages", END)

# Compile the graph
# No persistent checkpointing for now, memory is explicitly managed by nodes.
# memory_checkpointer = SqliteSaver.from_conn_string(":memory:") # Example if using built-in checkpointing
chat_graph = workflow.compile() # checkpointer=memory_checkpointer (if using)

# --- Main Invocation Function ---
async def run_chat_graph(user_id: str, session_id: str, input_query: str) -> str:
    """
    Runs the Langgraph chat flow for a given user, session, and query.
    """
    initial_state: ChatGraphState = {
        "user_id": user_id,
        "session_id": session_id,
        "input_query": input_query,
        "chat_history": [],
        "agent_response": ""
        # "error": None
    }
    
    print(f"--- Langgraph: Invoking graph for user {user_id}, session {session_id}, query: '{input_query}' ---")
    
    # The `ainvoke` method takes the initial state and returns the final state.
    # We need to ensure the agent and LLM are available when `call_agent_node` is executed.
    # This means `llm` and `create_market_intelligence_agent` from app.py must be initialized.
    final_state = await chat_graph.ainvoke(initial_state)
    
    print(f"--- Langgraph: Graph invocation complete. Final state agent_response: {final_state['agent_response'][:200]}... ---")
    
    # if final_state.get('error'):
    #     return f"An error occurred in the chat flow: {final_state['error']}"
        
    return final_state['agent_response']

# Example of how to potentially pass LLM and agent creator if not using direct imports from app
# _llm_instance = None
# _agent_creator_func = None
# def setup_langgraph_dependencies(llm_instance, agent_creator_func):
#     global _llm_instance, _agent_creator_func
#     _llm_instance = llm_instance
#     _agent_creator_func = agent_creator_func
#     print("--- Langgraph: Dependencies (LLM, Agent Creator) set up. ---")

```
