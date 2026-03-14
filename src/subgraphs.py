"""
Sub-graphs for MedFlow AI multi-agent system.
Includes Intake Agent, Safety Sentinel, Evidence Synthesizer, and other specialized agents.
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, Send
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.config import get_llm
from src.state import EncounterState, Medication, SafetyAlert
from src.tools import query_drug_interactions, analyze_medication_image
from datetime import datetime


# ============================================================================
# Intake Agent Sub-graph
# ============================================================================

class IntakeState(TypedDict):
    """State for intake sub-graph."""
    messages: list
    chief_complaint: str
    symptoms: list[str]
    medications: list[Medication]
    allergies: list[str]
    medical_history: str
    intake_complete: bool
    follow_up_needed: bool


def intake_gather_info(state: IntakeState) -> dict:
    """
    Gather initial patient information with adaptive questioning.
    """
    llm = get_llm()

    # Get recent messages
    recent_messages = state.get("messages", [])[-3:]

    # Check what information we already have
    has_complaint = bool(state.get("chief_complaint"))
    has_symptoms = len(state.get("symptoms", [])) > 0
    has_medications = len(state.get("medications", [])) > 0

    # Determine what to ask next
    system_prompt = """You are a compassionate medical intake assistant.
Your goal is to gather complete information efficiently while being empathetic."""

    if not has_complaint:
        question = "I'm here to help. Can you tell me what brought you in today? What's your main concern?"
    elif not has_symptoms:
        question = f"I understand you're experiencing {state.get('chief_complaint')}. Can you describe your symptoms in detail? When did they start?"
    elif not has_medications:
        question = "Are you currently taking any medications, including over-the-counter drugs or supplements?"
    else:
        question = "Do you have any known allergies to medications? Any relevant medical history I should know about?"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    ai_response = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "messages": [AIMessage(content=ai_response)]
    }


def intake_extract_info(state: IntakeState) -> dict:
    """
    Extract structured information from conversation.
    """
    llm = get_llm()

    # Get full conversation
    conversation = "\n".join([
        f"{msg.type}: {msg.content if isinstance(msg.content, str) else str(msg.content)}"
        for msg in state.get("messages", [])
        if isinstance(msg, (HumanMessage, AIMessage))
    ])

    extraction_prompt = f"""Extract clinical information from this conversation:

{conversation}

Return in this exact format:
CHIEF_COMPLAINT: [main concern]
SYMPTOMS: [comma-separated list]
MEDICATIONS: [comma-separated list with doses if mentioned]
ALLERGIES: [comma-separated list or "None mentioned"]
MEDICAL_HISTORY: [relevant history or "None mentioned"]
COMPLETE: [yes/no - is intake complete?]
"""

    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    extracted = response.content if isinstance(response.content, str) else str(response.content)

    # Parse response
    updates = {}

    if "CHIEF_COMPLAINT:" in extracted:
        complaint = extracted.split("CHIEF_COMPLAINT:")[1].split("\n")[0].strip()
        if complaint and complaint.lower() not in ["not mentioned", "none"]:
            updates["chief_complaint"] = complaint

    if "SYMPTOMS:" in extracted:
        symptoms_text = extracted.split("SYMPTOMS:")[1].split("\n")[0].strip()
        if symptoms_text and symptoms_text.lower() not in ["not mentioned", "none"]:
            updates["symptoms"] = [s.strip() for s in symptoms_text.split(",")]

    if "MEDICATIONS:" in extracted:
        meds_text = extracted.split("MEDICATIONS:")[1].split("\n")[0].strip()
        if meds_text and meds_text.lower() not in ["not mentioned", "none", "none mentioned"]:
            # Create Medication objects
            medications = []
            for med in meds_text.split(","):
                medications.append(Medication(
                    name=med.strip(),
                    dosage="As prescribed",
                    frequency="Regular",
                    source="patient_reported"
                ))
            updates["medications"] = medications

    if "COMPLETE:" in extracted:
        is_complete = "yes" in extracted.split("COMPLETE:")[1].split("\n")[0].lower()
        updates["intake_complete"] = is_complete

    return updates


def should_continue_intake(state: IntakeState) -> Literal["gather_more", "complete"]:
    """
    Decide if intake needs more information.
    """
    if state.get("intake_complete", False):
        return "complete"

    # Check if we have minimum required info
    has_complaint = bool(state.get("chief_complaint"))
    has_some_detail = bool(state.get("symptoms") or state.get("medications"))

    if has_complaint and has_some_detail:
        return "complete"

    return "gather_more"


# Build intake sub-graph
def create_intake_graph():
    """Create the intake agent sub-graph."""
    intake_graph = StateGraph(IntakeState)

    # Add nodes
    intake_graph.add_node("gather_info", intake_gather_info)
    intake_graph.add_node("extract_info", intake_extract_info)

    # Add edges
    intake_graph.add_edge(START, "gather_info")
    intake_graph.add_edge("gather_info", "extract_info")
    intake_graph.add_conditional_edges(
        "extract_info",
        should_continue_intake,
        {
            "gather_more": "gather_info",
            "complete": END
        }
    )

    return intake_graph.compile()


# ============================================================================
# Safety Sentinel Sub-graph with Send() Map-Reduce
# ============================================================================

class SafetyCheckState(TypedDict):
    """State for individual safety check."""
    drug_pair: tuple[str, str]
    interaction_found: bool
    alert: SafetyAlert | None


class SafetySentinelState(TypedDict):
    """State for safety sentinel sub-graph."""
    medications: list[Medication]
    safety_alerts: list[SafetyAlert]
    critical_alerts_count: int
    check_complete: bool


def create_safety_checks(state: SafetySentinelState) -> list[Send]:
    """
    Create parallel safety check tasks using Send() for map-reduce pattern.

    Returns a Send for each drug pair to check.
    """
    medications = state.get("medications", [])

    if len(medications) < 2:
        return []

    # Generate all drug pairs
    sends = []
    for i, med1 in enumerate(medications):
        for med2 in medications[i+1:]:
            # Create a Send for each pair
            sends.append(
                Send(
                    "check_drug_pair",
                    {
                        "drug_pair": (med1.name, med2.name),
                        "interaction_found": False,
                        "alert": None
                    }
                )
            )

    return sends


def check_drug_pair(state: SafetyCheckState) -> dict:
    """
    Check a single drug pair for interactions.
    This runs in parallel for all pairs.
    """
    drug_pair = state["drug_pair"]

    # Query for interactions
    interactions = query_drug_interactions.invoke({
        "medications": list(drug_pair),
        "severity_filter": None
    })

    if interactions:
        # Create safety alert for the first/most severe interaction
        interaction = interactions[0]
        alert = SafetyAlert(
            alert_type="drug_interaction",
            severity=interaction.get("severity", "moderate"),
            description=f"{drug_pair[0]} + {drug_pair[1]}: {interaction.get('description', 'Potential interaction')}",
            recommendation="Consult pharmacist or provider before combining these medications",
            source="knowledge_base",
            timestamp=datetime.now().isoformat()
        )

        return {
            "interaction_found": True,
            "alert": alert
        }

    return {
        "interaction_found": False,
        "alert": None
    }


def aggregate_safety_results(state: SafetySentinelState) -> dict:
    """
    Aggregate results from all parallel safety checks.
    """
    # This node collects all the results from parallel check_drug_pair calls
    alerts = state.get("safety_alerts", [])

    # Count critical alerts
    critical_count = sum(1 for alert in alerts if alert.severity == "critical")

    return {
        "critical_alerts_count": critical_count,
        "check_complete": True
    }


# Build safety sentinel sub-graph
def create_safety_sentinel_graph():
    """Create the safety sentinel sub-graph with Send() map-reduce."""
    safety_graph = StateGraph(SafetySentinelState)

    # Add nodes
    safety_graph.add_node("check_drug_pair", check_drug_pair)
    safety_graph.add_node("aggregate", aggregate_safety_results)

    # Fan-out using Send() from START
    safety_graph.add_conditional_edges(
        START,
        create_safety_checks,
        ["check_drug_pair"]
    )

    # Fan-in to aggregate
    safety_graph.add_edge("check_drug_pair", "aggregate")
    safety_graph.add_edge("aggregate", END)

    return safety_graph.compile()


# ============================================================================
# Evidence Synthesizer Sub-graph (ReAct Pattern)
# ============================================================================

class EvidenceState(TypedDict):
    """State for evidence synthesizer sub-graph."""
    query: str
    search_iterations: int
    evidence_collected: list[dict]
    synthesis_complete: bool
    final_synthesis: str


def evidence_search(state: EvidenceState) -> dict:
    """
    Search for clinical evidence (ReAct pattern - Action step).
    """
    from src.tools import search_clinical_guidelines

    query = state.get("query", "")
    iterations = state.get("search_iterations", 0)

    # Search guidelines
    guidelines = search_clinical_guidelines.invoke({
        "condition": query,
        "care_setting": "emergency"
    })

    evidence = state.get("evidence_collected", [])
    evidence.extend(guidelines)

    return {
        "evidence_collected": evidence,
        "search_iterations": iterations + 1
    }


def evidence_evaluate(state: EvidenceState) -> dict:
    """
    Evaluate if we have enough evidence (ReAct pattern - Thought/Reasoning step).
    """
    llm = get_llm()

    evidence = state.get("evidence_collected", [])
    query = state.get("query", "")

    # Create evidence summary
    evidence_summary = "\n\n".join([
        f"Source: {e.get('source', 'unknown')}\n{e.get('protocol', e.get('content', ''))[:500]}"
        for e in evidence[:5]
    ])

    evaluation_prompt = f"""Query: {query}

Evidence collected:
{evidence_summary}

Is this evidence sufficient to provide clinical guidance? Respond with:
SUFFICIENT: yes/no
REASONING: [explain why]
SYNTHESIS: [if sufficient, provide 2-3 sentence synthesis]
"""

    response = llm.invoke([HumanMessage(content=evaluation_prompt)])
    result = response.content if isinstance(response.content, str) else str(response.content)

    is_sufficient = "SUFFICIENT: yes" in result

    # Extract synthesis if available
    synthesis = ""
    if "SYNTHESIS:" in result:
        synthesis = result.split("SYNTHESIS:")[1].strip()

    return {
        "synthesis_complete": is_sufficient,
        "final_synthesis": synthesis if is_sufficient else ""
    }


def should_continue_evidence(state: EvidenceState) -> Literal["search_more", "complete"]:
    """
    Decide if more evidence search is needed.
    """
    if state.get("synthesis_complete", False):
        return "complete"

    if state.get("search_iterations", 0) >= 3:
        return "complete"  # Max iterations reached

    return "search_more"


# Build evidence synthesizer sub-graph
def create_evidence_synthesizer_graph():
    """Create the evidence synthesizer sub-graph with ReAct pattern."""
    evidence_graph = StateGraph(EvidenceState)

    # Add nodes
    evidence_graph.add_node("search", evidence_search)
    evidence_graph.add_node("evaluate", evidence_evaluate)

    # Add edges (ReAct loop)
    evidence_graph.add_edge(START, "search")
    evidence_graph.add_edge("search", "evaluate")
    evidence_graph.add_conditional_edges(
        "evaluate",
        should_continue_evidence,
        {
            "search_more": "search",
            "complete": END
        }
    )

    return evidence_graph.compile()


# Export compiled graphs
intake_agent = create_intake_graph()
safety_sentinel = create_safety_sentinel_graph()
evidence_synthesizer = create_evidence_synthesizer_graph()
