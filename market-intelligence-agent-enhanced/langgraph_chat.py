import os
import re # For regex-based source extraction
import uuid
import json # For safely parsing JSON strings if analysis_output is a stringified JSON
from typing import TypedDict, List, Annotated, Sequence, Tuple, Optional, Dict, Any

from langgraph.graph import StateGraph, END
# For now, we'll manage memory explicitly via Supabase calls, not using SqliteSaver or a full BaseChatMessageHistory implementation.
# from langgraph.checkpoint.sqlite import SqliteSaver 

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Supabase client functions for chat history and analysis results
from market_intelligence_agent_enhanced.supabase_client import (
    save_chat_message, 
    get_chat_history_for_session,
    get_analysis_result_by_id, 
    # supabase # Direct client usage if needed, but prefer specific functions
)

# For Agent and LLM - these will be initialized in app.py and accessed/passed here.
from market_intelligence_agent_enhanced.app import llm, create_market_intelligence_agent 

# --- Define Graph State ---
class ChatGraphState(TypedDict):
    user_id: str
    session_id: str
    input_query: str
    chat_history: List[Tuple[str, str]] 
    agent_response: str
    analysis_id: Optional[str] = None
    analysis_type: Optional[str] = None
    analysis_context_content: Optional[str] = None
    extracted_sources: List[str] # New field for sources
    # error: Optional[str] # For potential error handling within graph

# --- Graph Nodes ---

async def load_history_node(state: ChatGraphState) -> ChatGraphState:
    print(f"--- Langgraph: Loading history for user {state['user_id']}, session {state['session_id']} ---")
    history_records = await get_chat_history_for_session(
        user_id=state['user_id'], 
        session_id=state['session_id'],
        limit=20 
    )
    
    loaded_history = []
    if history_records:
        for record in history_records:
            loaded_history.append((record.get('message_type', 'unknown'), record.get('message_content', '')))
    
    state['chat_history'] = loaded_history
    print(f"--- Langgraph: Loaded history: {loaded_history} ---")
    return state

async def load_analysis_context_node(state: ChatGraphState) -> ChatGraphState:
    print(f"--- Langgraph: Loading analysis context for user {state['user_id']}, session {state['session_id']} ---")
    analysis_id = state.get('analysis_id')
    user_id = state.get('user_id') 
    
    if analysis_id and user_id: 
        print(f"--- Langgraph: analysis_id '{analysis_id}' found, attempting to fetch context for user '{user_id}'. ---")
        try:
            analysis_result = await get_analysis_result_by_id(user_id=user_id, result_id=analysis_id)
            
            if analysis_result:
                result_data = analysis_result.get('result_data')
                context_content_str = None
                if isinstance(result_data, dict):
                    if 'analysis_output' in result_data and result_data['analysis_output']:
                        context_content_str = str(result_data['analysis_output'])
                    elif 'executive_summary' in result_data and result_data['executive_summary']:
                        context_content_str = str(result_data['executive_summary'])
                    else:
                        context_content_str = json.dumps(result_data) 
                elif isinstance(result_data, str):
                    try:
                        parsed_result_data = json.loads(result_data)
                        if 'analysis_output' in parsed_result_data and parsed_result_data['analysis_output']:
                            context_content_str = str(parsed_result_data['analysis_output'])
                        elif 'executive_summary' in parsed_result_data and parsed_result_data['executive_summary']:
                            context_content_str = str(parsed_result_data['executive_summary'])
                        else: 
                            context_content_str = json.dumps(parsed_result_data)
                    except json.JSONDecodeError:
                        context_content_str = result_data 
                
                if context_content_str:
                    state['analysis_context_content'] = context_content_str
                    print(f"--- Langgraph: Analysis context loaded successfully for ID '{analysis_id}'. Length: {len(state['analysis_context_content'])} ---")
                else:
                    state['analysis_context_content'] = None
                    print(f"--- Langgraph: Analysis context found for ID '{analysis_id}', but relevant content is missing or empty. ---")
            else:
                state['analysis_context_content'] = None
                print(f"--- Langgraph: No analysis context found for ID '{analysis_id}'. ---")
        except Exception as e:
            print(f"--- Langgraph Error: Failed to load analysis context for ID '{analysis_id}': {e} ---")
            state['analysis_context_content'] = None 
    else:
        state['analysis_context_content'] = None
        if not analysis_id:
            print(f"--- Langgraph: No analysis_id provided. Skipping context load. ---")
        if not user_id: 
            print(f"--- Langgraph Warning: user_id missing in state. Skipping context load. ---")
            
    return state

def _extract_sources_from_text(text: str) -> List[str]:
    """Helper function to extract URLs and 'Source: [filename]' patterns."""
    if not text:
        return []
    
    # Regex for URLs
    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    found_urls = re.findall(url_pattern, text)
    
    # Regex for "Source: [filename]" (captures filename)
    # Filename can contain alphanumeric, dots, underscores, hyphens.
    source_file_pattern = r'Source:\s*([\w\._-]+)' 
    found_file_sources_matches = re.findall(source_file_pattern, text, re.IGNORECASE)
    # Format them back to "Source: [filename]"
    found_file_sources = [f"Source: {name}" for name in found_file_sources_matches]

    # Combine and ensure uniqueness
    all_found = list(set(found_urls + found_file_sources))
    
    # Simple heuristic: if RAG tool's "No relevant documents found..." is in text, don't count it as a source.
    # This also means genuine sources might be missed if the agent phrases its response poorly.
    # A more robust solution would be for tools to return structured source info.
    no_docs_message = "No relevant documents found for your query in uploaded files."
    error_retrieving_message = "Error retrieving documents."
    
    # Filter out known non-source phrases that might be picked up by regex if they contain "Source:"
    # or if they are just part of the agent's conversational fluff.
    filtered_sources = [
        src for src in all_found 
        if src not in [no_docs_message, error_retrieving_message] and 
           not src.startswith("Source: Unknown") # From RAG tool if metadata 'source' is missing
    ]

    print(f"--- Langgraph Source Extraction: Found raw sources: {all_found}, Filtered sources: {filtered_sources} ---")
    return filtered_sources


async def call_agent_node(state: ChatGraphState) -> ChatGraphState:
    print(f"--- Langgraph: Calling agent for user {state['user_id']}, session {state['session_id']} ---")
    if not llm: 
        print("--- Langgraph Error: LLM not initialized! ---")
        state['agent_response'] = "Error: The underlying language model is not available."
        state['extracted_sources'] = []
        return state

    agent_executor = create_market_intelligence_agent(llm, user_id=state['user_id'])
    if not agent_executor:
        print("--- Langgraph Error: Could not create market intelligence agent! ---")
        state['agent_response'] = "Error: Could not initialize the market intelligence agent."
        state['extracted_sources'] = []
        return state

    history_for_prompt = "\n".join([f"{msg_type.capitalize()}: {msg_content}" for msg_type, msg_content in state.get('chat_history', [])])
    analysis_context_content = state.get('analysis_context_content')
    analysis_id = state.get('analysis_id')
    analysis_type = state.get('analysis_type')

    prompt_parts = []
    if analysis_context_content and analysis_id: 
        context_header = f"Relevant Analysis Context (Type: {analysis_type or 'N/A'}, ID: {analysis_id}):"
        prompt_parts.append(context_header)
        prompt_parts.append(analysis_context_content)
        prompt_parts.append("\n--- End of Analysis Context ---\n") 

    if history_for_prompt:
        prompt_parts.append("Previous conversation:")
        prompt_parts.append(history_for_prompt)
        prompt_parts.append("\n--- End of Previous Conversation ---\n") 
    
    prompt_parts.append(f"New user query: {state['input_query']}")
    final_prompt = "\n\n".join(prompt_parts) 
    print(f"--- Langgraph: Final agent prompt:\n{final_prompt}\n---")

    try:
        response = await agent_executor.arun(final_prompt) 
        state['agent_response'] = response
        # Extract sources from the agent's response
        state['extracted_sources'] = _extract_sources_from_text(response)
    except Exception as e:
        print(f"--- Langgraph Error: Agent execution failed: {e} ---")
        state['agent_response'] = f"Error during agent execution: {str(e)}"
        state['extracted_sources'] = [] # Ensure empty list on error
    
    print(f"--- Langgraph: Agent response: {state['agent_response'][:200]}... ---")
    print(f"--- Langgraph: Extracted sources: {state['extracted_sources']} ---")
    return state

async def save_messages_node(state: ChatGraphState) -> ChatGraphState:
    print(f"--- Langgraph: Saving messages for user {state['user_id']}, session {state['session_id']} ---")
    
    await save_chat_message(
        user_id=state['user_id'],
        session_id=state['session_id'],
        message_type="human",
        message_content=state['input_query']
    )
    
    if state.get('agent_response'): 
        await save_chat_message(
            user_id=state['user_id'],
            session_id=state['session_id'],
            message_type="ai",
            message_content=state['agent_response'],
            # Optionally, save extracted_sources as metadata if your Supabase schema supports it
            # metadata={"sources": state.get('extracted_sources', [])} 
        )
    print(f"--- Langgraph: Messages saved. User: '{state['input_query']}', AI: '{state.get('agent_response', '')[:100]}...' ---")
    return state

# --- Build the Graph ---
workflow = StateGraph(ChatGraphState)

workflow.add_node("load_history", load_history_node)
workflow.add_node("load_analysis_context", load_analysis_context_node) 
workflow.add_node("call_agent", call_agent_node)
workflow.add_node("save_messages", save_messages_node)

workflow.set_entry_point("load_history")

workflow.add_edge("load_history", "load_analysis_context") 
workflow.add_edge("load_analysis_context", "call_agent")   
workflow.add_edge("call_agent", "save_messages")
workflow.add_edge("save_messages", END)

chat_graph = workflow.compile() 

# --- Main Invocation Function ---
async def run_chat_graph(
    user_id: str, 
    session_id: str, 
    input_query: str, 
    analysis_id: Optional[str] = None, 
    analysis_type: Optional[str] = None 
) -> Dict[str, Any]: # Changed return type
    """
    Runs the Langgraph chat flow for a given user, session, and query,
    optionally including context from a specific analysis.
    Returns a dictionary containing the agent's response and extracted sources.
    """
    initial_state: ChatGraphState = {
        "user_id": user_id,
        "session_id": session_id,
        "input_query": input_query,
        "chat_history": [],
        "agent_response": "",
        "analysis_id": analysis_id, 
        "analysis_type": analysis_type, 
        "analysis_context_content": None, 
        "extracted_sources": [] # Initialize new field
        # "error": None
    }
    
    print(f"--- Langgraph: Invoking graph for user {user_id}, session {session_id}, query: '{input_query}', analysis_id: {analysis_id}, analysis_type: {analysis_type} ---")
    
    final_state = await chat_graph.ainvoke(initial_state)
    
    print(f"--- Langgraph: Graph invocation complete. Final state agent_response: {final_state['agent_response'][:200]}... ---")
    print(f"--- Langgraph: Final extracted sources: {final_state['extracted_sources']} ---")
        
    return {
        "response": final_state['agent_response'],
        "sources": final_state.get('extracted_sources', []) # Ensure sources list is always returned
    }
