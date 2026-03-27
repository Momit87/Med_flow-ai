"""
Checkpoint Manager - Load and search previous conversations
Enables continuation of patient encounters across sessions
"""

import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.config import settings
from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer():
    """Get SqliteSaver instance for reading checkpoints."""
    conn = sqlite3.connect(settings.CHECKPOINT_DB, check_same_thread=False)
    return SqliteSaver(conn), conn


def get_all_conversations() -> List[Dict[str, Any]]:
    """
    Get list of all saved conversations from checkpoint database.

    Returns:
        List of conversation metadata dictionaries
    """
    try:
        checkpointer, conn = get_checkpointer()
        cursor = conn.cursor()

        # Get all unique thread_ids
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC LIMIT 50")
        thread_ids = [row[0] for row in cursor.fetchall()]

        conversations = []
        for thread_id in thread_ids:
            try:
                # Use checkpointer to get the latest checkpoint for this thread
                config = {"configurable": {"thread_id": thread_id}}
                checkpoint_tuple = checkpointer.get_tuple(config)

                if checkpoint_tuple and checkpoint_tuple.checkpoint:
                    state = checkpoint_tuple.checkpoint.get("channel_values", {})

                    patient_id = state.get("patient_id")
                    if not patient_id:
                        continue

                    conversations.append({
                        "patient_id": patient_id,
                        "thread_id": thread_id,
                        "chief_complaint": state.get("chief_complaint", "N/A"),
                        "triage_level": state.get("triage_level", "N/A"),
                        "created_at": state.get("created_at", "N/A"),
                        "last_updated_at": state.get("last_updated_at", "N/A"),
                        "status": state.get("status", "N/A")
                    })
            except Exception as e:
                # Skip problematic checkpoints
                continue

        conn.close()
        return conversations

    except Exception as e:
        print(f"Error getting conversations: {e}")
        return []


def search_conversations(search_term: str) -> List[Dict[str, Any]]:
    """
    Search conversations by patient_id, thread_id, or chief complaint.

    Args:
        search_term: Search string (patient_id, thread_id, or keywords)

    Returns:
        List of matching conversation metadata
    """
    all_conversations = get_all_conversations()

    if not search_term:
        return all_conversations

    search_lower = search_term.lower()

    # Filter conversations matching search term
    filtered = []
    for conv in all_conversations:
        if (search_lower in str(conv.get("patient_id", "")).lower() or
            search_lower in str(conv.get("thread_id", "")).lower() or
            search_lower in str(conv.get("chief_complaint", "")).lower()):
            filtered.append(conv)

    return filtered


def load_conversation(thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a specific conversation state by thread_id.

    Args:
        thread_id: Thread identifier

    Returns:
        Conversation state dictionary with messages and metadata
    """
    try:
        checkpointer, conn = get_checkpointer()

        # Get the latest checkpoint for this thread
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = checkpointer.get_tuple(config)

        if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
            conn.close()
            return None

        # Extract state data
        state = checkpoint_tuple.checkpoint.get("channel_values", {})

        # Reconstruct messages from checkpoint
        messages = []
        for msg in state.get("messages", []):
            # LangChain messages are objects, extract role and content
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                if msg.type == "human":
                    messages.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    messages.append({"role": "assistant", "content": msg.content})

        conn.close()

        return {
            "patient_id": state.get("patient_id"),
            "thread_id": thread_id,
            "messages": messages,
            "encounter_state": {
                "chief_complaint": state.get("chief_complaint"),
                "symptoms": state.get("symptoms", []),
                "medications": state.get("medications", []),
                "allergies": state.get("allergies", []),
                "triage_level": state.get("triage_level"),
                "triage_reasoning": state.get("triage_reasoning"),
                "status": state.get("status"),
                "current_agent": state.get("current_agent"),
                "created_at": state.get("created_at"),
                "last_updated_at": state.get("last_updated_at")
            }
        }

    except Exception as e:
        print(f"Error loading conversation: {e}")
        import traceback
        traceback.print_exc()
        return None
