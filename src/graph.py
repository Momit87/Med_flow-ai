
"""
Main encounter orchestration graph.
This is a simplified initial version - agents will be expanded in subsequent iterations.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import EncounterState, InputState, OutputState, create_initial_state
from src.config import get_llm, settings
from datetime import datetime
import sqlite3


# ============================================================================
# Node Functions (Simplified for MVP)
# ============================================================================

def intake_node(state: EncounterState) -> dict:
    """
    Simplified intake node - conducts initial patient interview.
    Full implementation will be a sub-graph with adaptive questioning.
    """
    llm = get_llm(provider=settings.DEFAULT_LLM_PROVIDER)

    # System prompt for intake
    system_prompt = """You are a medical intake assistant. Your role is to:
1. Gather chief complaint and symptoms
2. Ask about current medications
3. Ask about allergies
4. Collect relevant medical history

Be empathetic, clear, and thorough. Use simple language.
Focus on gathering information needed for triage and clinical assessment.

IMPORTANT: This is a demonstration system. Always remind patients that this is not a substitute for professional medical advice."""

    # Build message history
    messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])

    # Generate response
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "current_agent": "intake",
        "last_updated_at": datetime.now().isoformat(),
        "status": "intake"
    }


def extract_clinical_data(state: EncounterState) -> dict:
    """
    Extract structured clinical data from conversation.
    Uses LLM to parse unstructured chat into structured fields.
    """
    llm = get_llm()

    # Get conversation history
    conversation = "\n".join([
        f"{msg.type}: {msg.content if isinstance(msg.content, str) else str(msg.content)}"
        for msg in state.get("messages", [])
        if isinstance(msg, (HumanMessage, AIMessage))
    ])

    extraction_prompt = f"""Based on this conversation, extract the following information:

Conversation:
{conversation}

Extract and return in this format:
CHIEF_COMPLAINT: [main reason for visit]
SYMPTOMS: [list of symptoms, comma-separated]
MEDICATIONS: [list of current medications with doses if mentioned]
ALLERGIES: [list of allergies]
MEDICAL_HISTORY: [relevant past medical history]

If information is not mentioned, write "Not provided".
"""

    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    extracted = response.content if isinstance(response.content, str) else str(response.content)

    # Parse extracted data (simplified - in production would use structured output)
    updates = {
        "current_agent": "extraction",
        "last_updated_at": datetime.now().isoformat()
    }

    # Simple parsing
    if "CHIEF_COMPLAINT:" in extracted:
        chief = extracted.split("CHIEF_COMPLAINT:")[1].split("\n")[0].strip()
        if chief and chief != "Not provided":
            updates["chief_complaint"] = chief

    return updates


def triage_node(state: EncounterState) -> dict:
    """
    Simplified triage classification using ESI (Emergency Severity Index).
    Full implementation will include vitals analysis and clinical decision support.
    """
    llm = get_llm()

    chief_complaint = state.get("chief_complaint", "")
    conversation = "\n".join([
        msg.content if isinstance(msg.content, str) else str(msg.content)
        for msg in state.get("messages", [])[-5:]
        if isinstance(msg, (HumanMessage, AIMessage))
    ])

    triage_prompt = f"""You are an emergency triage nurse. Classify this patient using the Emergency Severity Index (ESI):

ESI-1: Immediate life-threatening (requires immediate intervention)
ESI-2: High risk or severe pain/distress (10 min wait acceptable)
ESI-3: Stable but needs multiple resources (30 min wait acceptable)
ESI-4: Stable, needs one simple resource
ESI-5: Non-urgent, may not need resources

Chief Complaint: {chief_complaint}

Recent conversation:
{conversation}

Provide:
1. ESI Level (1-5)
2. Brief reasoning (2-3 sentences)

Format:
LEVEL: ESI-[number]
REASONING: [your reasoning]
"""

    response = llm.invoke([HumanMessage(content=triage_prompt)])
    triage_text = response.content if isinstance(response.content, str) else str(response.content)

    # Parse triage level
    triage_level = "ESI-3"  # default
    if "ESI-1" in triage_text:
        triage_level = "ESI-1"
    elif "ESI-2" in triage_text:
        triage_level = "ESI-2"
    elif "ESI-3" in triage_text:
        triage_level = "ESI-3"
    elif "ESI-4" in triage_text:
        triage_level = "ESI-4"
    elif "ESI-5" in triage_text:
        triage_level = "ESI-5"

    reasoning = triage_text.split("REASONING:")[-1].strip() if "REASONING:" in triage_text else triage_text

    return {
        "triage_level": triage_level,
        "triage_reasoning": reasoning,
        "current_agent": "triage",
        "status": "analyzing",
        "last_updated_at": datetime.now().isoformat()
    }


def should_continue(state: EncounterState) -> str:
    """
    Routing logic - determines next step based on state.
    """
    status = state.get("status", "intake")

    if status == "intake":
        # Check if we have enough information
        chief_complaint = state.get("chief_complaint", "")
        message_count = len(state.get("messages", []))

        # Need at least 3 messages or a clear chief complaint
        if message_count >= 4 or chief_complaint:
            return "extract"
        return "continue_intake"

    elif status == "analyzing":
        return "complete"

    return "complete"


# ============================================================================
# Build Graph
# ============================================================================

def create_encounter_graph():
    """
    Create the main encounter orchestration graph.
    This is a simplified version - will be expanded with full agent sub-graphs.
    """

    # Initialize graph with state schema
    graph = StateGraph(
        state_schema=EncounterState,
        input=InputState,
        output=OutputState
    )

    # Add nodes
    graph.add_node("intake", intake_node)
    graph.add_node("extract_data", extract_clinical_data)
    graph.add_node("triage", triage_node)

    # Add edges
    graph.add_edge("__start__", "intake")
    graph.add_conditional_edges(
        "intake",
        should_continue,
        {
            "continue_intake": "intake",
            "extract": "extract_data",
            "complete": END
        }
    )
    graph.add_edge("extract_data", "triage")
    graph.add_edge("triage", END)

    return graph


# ============================================================================
# Compiled Application
# ============================================================================

# Initialize checkpoint saver
def get_checkpointer():
    """Get SQLite checkpointer for state persistence."""
    conn = sqlite3.connect(settings.CHECKPOINT_DB, check_same_thread=False)
    return SqliteSaver(conn)


# Compile the graph
workflow = create_encounter_graph()
app = workflow.compile(
    checkpointer=get_checkpointer(),
    # interrupt_before=["provider_review"]  # Will add HITL in later iteration
)


# ============================================================================
# Helper Functions
# ============================================================================

def run_encounter(patient_id: str, thread_id: str, user_message: str):
    """
    Run a single turn of the encounter.

    Args:
        patient_id: Patient identifier
        thread_id: Thread identifier for conversation continuity
        user_message: User's message

    Returns:
        Final state after processing
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "patient_id": patient_id
        }
    }

    input_state = {
        "messages": [HumanMessage(content=user_message)],
        "patient_id": patient_id
    }

    result = app.invoke(input_state, config)
    return result


def stream_encounter(patient_id: str, thread_id: str, user_message: str):
    """
    Stream encounter processing with real-time updates.

    Args:
        patient_id: Patient identifier
        thread_id: Thread identifier
        user_message: User's message

    Yields:
        State updates as they occur
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "patient_id": patient_id
        }
    }

    input_state = {
        "messages": [HumanMessage(content=user_message)],
        "patient_id": patient_id
    }

    for event in app.stream(input_state, config, stream_mode="updates"):
        yield event
