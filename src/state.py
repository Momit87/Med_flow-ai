"""
State schemas for MedFlow AI encounter orchestration.
Defines the core data structures used across all agents.
"""

from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from datetime import datetime


# ============================================================================
# Pydantic Models for Structured Data
# ============================================================================

class Medication(BaseModel):
    """Structured medication information."""
    name: str = Field(description="Drug name (generic or brand)")
    dose: str = Field(description="Dosage e.g. 500mg, 10ml")
    frequency: str = Field(description="e.g. BID (twice daily), QD (once daily), PRN (as needed)")
    route: str = Field(default="oral", description="Administration route")
    start_date: str | None = Field(default=None, description="When patient started taking")


class SafetyAlert(BaseModel):
    """Drug interaction or safety warning."""
    severity: Literal["critical", "moderate", "low"] = Field(
        description="Severity level of the alert"
    )
    drug_pair: tuple[str, str] = Field(
        description="The two drugs involved in the interaction"
    )
    description: str = Field(
        description="Human-readable description of the interaction"
    )
    source: str = Field(
        description="Evidence source (e.g., 'openFDA', 'ChromaDB clinical guidelines')"
    )
    evidence_score: float = Field(
        description="Confidence score from RAG retrieval (0-1)",
        ge=0.0,
        le=1.0
    )


class RAGContext(BaseModel):
    """Retrieved evidence from RAG system."""
    source: str = Field(description="Source document or database")
    content: str = Field(description="Retrieved text content")
    score: float = Field(description="Similarity/relevance score", ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class SOAPNote(BaseModel):
    """SOAP note structure for clinical documentation."""
    subjective: str = Field(
        description="Patient's reported symptoms and history"
    )
    objective: str = Field(
        description="Observable findings, vitals, physical exam"
    )
    assessment: str = Field(
        description="Clinical interpretation, differential diagnoses"
    )
    plan: str = Field(
        description="Treatment plan, medications, follow-up instructions"
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of generation"
    )


# ============================================================================
# Custom State Reducers
# ============================================================================

def safety_reducer(existing: list[SafetyAlert], new: list[SafetyAlert]) -> list[SafetyAlert]:
    """
    Accumulates safety alerts from parallel drug checks.
    Deduplicates by drug pair and sorts by severity.
    """
    # Convert to dict for deduplication
    seen = {}
    for alert in existing:
        key = tuple(sorted([alert.drug_pair[0], alert.drug_pair[1]]))
        seen[key] = alert

    # Add new alerts
    for alert in new:
        key = tuple(sorted([alert.drug_pair[0], alert.drug_pair[1]]))
        if key not in seen:
            seen[key] = alert
        elif alert.evidence_score > seen[key].evidence_score:
            # Keep the higher-confidence alert
            seen[key] = alert

    # Sort by severity
    severity_order = {"critical": 0, "moderate": 1, "low": 2}
    sorted_alerts = sorted(
        seen.values(),
        key=lambda a: severity_order[a.severity]
    )

    return sorted_alerts


# ============================================================================
# TypedDict State Schemas
# ============================================================================

class EncounterState(TypedDict):
    """
    Complete internal state for a clinical encounter.
    This is the full state used internally by the graph.
    """
    # Core conversation (uses LangGraph's add_messages reducer)
    messages: Annotated[list, add_messages]

    # Patient identification
    patient_id: str
    thread_id: str

    # Intake data (populated by Intake Agent)
    chief_complaint: str
    symptoms: list[str]
    medications: list[Medication]
    allergies: list[str]
    vitals: dict  # {bp, hr, temp, rr, spo2}
    medical_history: str

    # Safety analysis (populated by Safety Sentinel with custom reducer)
    safety_alerts: Annotated[list[SafetyAlert], safety_reducer]

    # Evidence (populated by Evidence Synthesizer via RAG)
    rag_context: list[RAGContext]
    evidence_summary: str

    # Triage (populated by Router)
    triage_level: Literal["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"]
    triage_reasoning: str

    # Documentation (populated by Brief Generator)
    clinical_brief: SOAPNote | None
    patient_instructions: str

    # Workflow control
    status: Literal["intake", "analyzing", "reviewing", "complete", "halted"]
    current_agent: str
    attempt_count: int

    # Timestamps
    encounter_started_at: str
    last_updated_at: str


class InputState(TypedDict):
    """
    Minimal input schema for starting an encounter.
    This is what external callers provide.
    """
    messages: Annotated[list, add_messages]
    patient_id: str


class OutputState(TypedDict):
    """
    Output schema returned to external callers.
    Only includes clinically relevant final outputs.
    """
    clinical_brief: SOAPNote | None
    triage_level: str
    safety_alerts: list[SafetyAlert]
    patient_instructions: str
    status: str


# ============================================================================
# Helper Functions
# ============================================================================

def create_initial_state(patient_id: str, thread_id: str) -> dict:
    """Create initial encounter state with defaults."""
    return {
        "messages": [],
        "patient_id": patient_id,
        "thread_id": thread_id,
        "chief_complaint": "",
        "symptoms": [],
        "medications": [],
        "allergies": [],
        "vitals": {},
        "medical_history": "",
        "safety_alerts": [],
        "rag_context": [],
        "evidence_summary": "",
        "triage_level": "ESI-5",
        "triage_reasoning": "",
        "clinical_brief": None,
        "patient_instructions": "",
        "status": "intake",
        "current_agent": "intake",
        "attempt_count": 0,
        "encounter_started_at": datetime.now().isoformat(),
        "last_updated_at": datetime.now().isoformat(),
    }
