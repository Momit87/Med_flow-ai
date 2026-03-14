"""
Chains and agents for MedFlow AI.
Includes SOAP note generation, patient education, and documentation workflows.
"""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import get_llm, settings
from src.state import EncounterState, SOAPNote
from datetime import datetime


# ============================================================================
# SOAP Note Generation Chain
# ============================================================================

def collect_encounter_data(state: EncounterState) -> Dict[str, Any]:
    """
    Step 1: Collect all relevant data from encounter state.
    """
    # Extract conversation summary
    messages = state.get("messages", [])
    conversation_summary = "\n".join([
        f"{msg.type}: {msg.content if isinstance(msg.content, str) else str(msg.content)}"
        for msg in messages[-10:]  # Last 10 messages
    ])

    encounter_data = {
        "chief_complaint": state.get("chief_complaint", "Not documented"),
        "symptoms": ", ".join(state.get("symptoms", [])) or "None documented",
        "medications": ", ".join([med.name for med in state.get("medications", [])]) or "None reported",
        "allergies": ", ".join(state.get("allergies", [])) or "None reported",
        "medical_history": state.get("medical_history", "Not documented"),
        "triage_level": state.get("triage_level", "Not triaged"),
        "triage_reasoning": state.get("triage_reasoning", ""),
        "safety_alerts": state.get("safety_alerts", []),
        "conversation": conversation_summary,
        "assessment_notes": state.get("assessment_notes", ""),
        "differential_diagnosis": state.get("differential_diagnosis", [])
    }

    return encounter_data


def generate_soap_sections(encounter_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Step 2: Generate each SOAP section using LLM.
    """
    llm = get_llm()

    # Subjective section
    subjective_prompt = f"""Generate the SUBJECTIVE section of a SOAP note from this patient encounter:

Chief Complaint: {encounter_data['chief_complaint']}
Symptoms: {encounter_data['symptoms']}
Conversation: {encounter_data['conversation'][:1000]}

Write in professional medical documentation style. Include HPI (History of Present Illness) in narrative format.
"""

    subjective_response = llm.invoke([HumanMessage(content=subjective_prompt)])
    subjective = subjective_response.content if isinstance(subjective_response.content, str) else str(subjective_response.content)

    # Objective section
    objective_prompt = f"""Generate the OBJECTIVE section of a SOAP note:

Current Medications: {encounter_data['medications']}
Allergies: {encounter_data['allergies']}
Triage Level: {encounter_data['triage_level']}
Safety Alerts: {len(encounter_data['safety_alerts'])} alerts identified

Note: Vital signs would be recorded by clinical staff. For this AI-assisted documentation,
note that vital signs should be obtained and documented by healthcare provider.

Write in professional medical documentation style.
"""

    objective_response = llm.invoke([HumanMessage(content=objective_prompt)])
    objective = objective_response.content if isinstance(objective_response.content, str) else str(objective_response.content)

    # Assessment section
    assessment_prompt = f"""Generate the ASSESSMENT section of a SOAP note:

Chief Complaint: {encounter_data['chief_complaint']}
Symptoms: {encounter_data['symptoms']}
Triage Level: {encounter_data['triage_level']}
Reasoning: {encounter_data['triage_reasoning']}
Medical History: {encounter_data['medical_history']}

Provide differential diagnosis and clinical impression. Use ICD-10 terminology where appropriate.
"""

    assessment_response = llm.invoke([HumanMessage(content=assessment_prompt)])
    assessment = assessment_response.content if isinstance(assessment_response.content, str) else str(assessment_response.content)

    # Plan section
    plan_prompt = f"""Generate the PLAN section of a SOAP note:

Chief Complaint: {encounter_data['chief_complaint']}
Triage Level: {encounter_data['triage_level']}
Safety Alerts: {len(encounter_data['safety_alerts'])} medication safety alerts

Include:
1. Diagnostic workup needed
2. Treatment recommendations
3. Patient education topics
4. Follow-up instructions
5. Safety considerations

Write in professional medical documentation style.
"""

    plan_response = llm.invoke([HumanMessage(content=plan_prompt)])
    plan = plan_response.content if isinstance(plan_response.content, str) else str(plan_response.content)

    return {
        "subjective": subjective,
        "objective": objective,
        "assessment": assessment,
        "plan": plan
    }


def validate_soap_note(soap_sections: Dict[str, str]) -> Dict[str, Any]:
    """
    Step 3: Validate SOAP note completeness and quality.
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }

    # Check all sections exist and are not empty
    required_sections = ["subjective", "objective", "assessment", "plan"]
    for section in required_sections:
        if not soap_sections.get(section) or len(soap_sections[section].strip()) < 20:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Section '{section}' is missing or too short")

    # Check for minimum content quality
    if validation_results["is_valid"]:
        # Warn if subjective doesn't mention chief complaint
        if "subjective" in soap_sections and len(soap_sections["subjective"]) < 50:
            validation_results["warnings"].append("Subjective section may lack detail")

        # Warn if assessment doesn't have diagnosis
        if "assessment" in soap_sections and "diagnosis" not in soap_sections["assessment"].lower():
            validation_results["warnings"].append("Assessment may lack clear diagnosis")

    return validation_results


def format_soap_note(soap_sections: Dict[str, str], encounter_data: Dict[str, Any]) -> SOAPNote:
    """
    Step 4: Format final SOAP note with metadata.
    """
    soap_note = SOAPNote(
        subjective=soap_sections["subjective"],
        objective=soap_sections["objective"],
        assessment=soap_sections["assessment"],
        plan=soap_sections["plan"],
        provider_name="AI Assistant (requires provider review)",
        timestamp=datetime.now().isoformat(),
        encounter_id=encounter_data.get("encounter_id", "unknown")
    )

    return soap_note


def generate_soap_note_chain(state: EncounterState) -> SOAPNote:
    """
    Complete SOAP note generation chain (linear workflow).

    Steps:
    1. Collect encounter data
    2. Generate SOAP sections
    3. Validate note
    4. Format and return

    Args:
        state: Current encounter state

    Returns:
        Complete SOAP note object
    """
    # Step 1: Collect data
    encounter_data = collect_encounter_data(state)

    # Step 2: Generate sections
    soap_sections = generate_soap_sections(encounter_data)

    # Step 3: Validate
    validation = validate_soap_note(soap_sections)

    if not validation["is_valid"]:
        # Add validation errors to plan section
        soap_sections["plan"] += f"\n\n⚠️ Note Generation Warnings:\n" + "\n".join(validation["errors"])

    # Step 4: Format and return
    soap_note = format_soap_note(soap_sections, encounter_data)

    return soap_note


# ============================================================================
# Patient Education Agent with Dynamic Configuration
# ============================================================================

class PatientEducationAgent:
    """
    Dynamic patient education agent that adapts to care setting, language, and reading level.
    """

    def __init__(self, care_setting: str = "emergency", reading_level: str = "8th_grade", language: str = "en"):
        """
        Initialize patient education agent with configuration.

        Args:
            care_setting: Care setting (emergency, primary_care, pediatrics)
            reading_level: Target reading level (6th_grade, 8th_grade, college)
            language: Language code (en, es, etc.)
        """
        self.care_setting = care_setting
        self.reading_level = reading_level
        self.language = language
        self.llm = get_llm()

        # Get assistant config
        self.config = settings.ASSISTANTS.get(
            care_setting,
            settings.ASSISTANTS["ed"]
        )

    def reconfigure(self, care_setting: str = None, reading_level: str = None, language: str = None):
        """
        Dynamically reconfigure the agent.
        """
        if care_setting:
            self.care_setting = care_setting
            self.config = settings.ASSISTANTS.get(care_setting, settings.ASSISTANTS["ed"])

        if reading_level:
            self.reading_level = reading_level

        if language:
            self.language = language

    def generate_education_material(self, topic: str, key_points: List[str] = None) -> Dict[str, Any]:
        """
        Generate patient education material for a topic.

        Args:
            topic: Health topic or condition
            key_points: Specific points to cover (optional)

        Returns:
            Dictionary with educational content
        """
        # Query existing education materials
        from src.tools import get_patient_education

        existing_materials = get_patient_education.invoke({
            "topic": topic,
            "reading_level": self.reading_level
        })

        # Create tailored prompt based on care setting
        system_message = self.config["system_prompt"]

        education_prompt = f"""Create patient education material about: {topic}

Reading Level: {self.reading_level}
Care Setting: {self.care_setting}

Requirements:
- Use simple, clear language appropriate for {self.reading_level} reading level
- Focus on practical, actionable information
- Include what to watch for and when to seek help
- Keep tone {self._get_tone_guidance()}

{f"Make sure to cover these key points: {', '.join(key_points)}" if key_points else ""}

Existing materials for reference:
{existing_materials.get('content', '')[:500]}

Generate a comprehensive but concise education sheet.
"""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=education_prompt)
        ]

        response = self.llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        return {
            "topic": topic,
            "content": content,
            "reading_level": self.reading_level,
            "care_setting": self.care_setting,
            "language": self.language,
            "key_points": key_points or existing_materials.get("key_points", []),
            "generated_at": datetime.now().isoformat()
        }

    def _get_tone_guidance(self) -> str:
        """Get tone guidance based on care setting."""
        tone_map = {
            "emergency": "urgent but reassuring",
            "ed": "urgent but reassuring",
            "primary_care": "warm and supportive",
            "primary": "warm and supportive",
            "pediatrics": "friendly and parent-focused"
        }
        return tone_map.get(self.care_setting, "professional and empathetic")

    def create_discharge_instructions(self, encounter_state: EncounterState) -> Dict[str, Any]:
        """
        Create discharge instructions based on encounter.

        Args:
            encounter_state: Current encounter state

        Returns:
            Discharge instructions with care instructions, red flags, and follow-up
        """
        chief_complaint = encounter_state.get("chief_complaint", "your condition")
        medications = encounter_state.get("medications", [])
        triage_level = encounter_state.get("triage_level", "ESI-3")

        # Generate comprehensive discharge instructions
        discharge_prompt = f"""Create discharge instructions for a patient with:

Chief Complaint: {chief_complaint}
Triage Level: {triage_level}
Medications: {', '.join([med.name for med in medications]) if medications else 'None prescribed'}

Include:
1. Home care instructions
2. Medication instructions (if any)
3. Warning signs to watch for
4. When to return to emergency/call doctor
5. Follow-up recommendations

Reading Level: {self.reading_level}
Tone: {self._get_tone_guidance()}
"""

        messages = [
            SystemMessage(content=self.config["system_prompt"]),
            HumanMessage(content=discharge_prompt)
        ]

        response = self.llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        return {
            "title": f"Discharge Instructions: {chief_complaint}",
            "content": content,
            "medications": [{"name": med.name, "dosage": med.dosage, "frequency": med.frequency} for med in medications],
            "reading_level": self.reading_level,
            "care_setting": self.care_setting,
            "generated_at": datetime.now().isoformat()
        }


# ============================================================================
# Message Trimming and Summarization
# ============================================================================

def trim_messages(messages: list, max_tokens: int = 4000) -> list:
    """
    Trim message history to stay within token limit.

    Args:
        messages: List of messages
        max_tokens: Maximum token count (approximate)

    Returns:
        Trimmed message list
    """
    # Simple character-based approximation (4 chars ≈ 1 token)
    max_chars = max_tokens * 4

    # Always keep first message (system) and last few messages
    if len(messages) <= 5:
        return messages

    # Calculate total characters
    total_chars = sum(len(str(msg.content)) for msg in messages)

    if total_chars <= max_chars:
        return messages

    # Keep first message and recent messages
    trimmed = [messages[0]]  # System message

    # Add recent messages until we hit limit
    chars_used = len(str(messages[0].content))
    for msg in reversed(messages[1:]):
        msg_chars = len(str(msg.content))
        if chars_used + msg_chars > max_chars:
            break
        trimmed.insert(1, msg)  # Insert at beginning (after system message)
        chars_used += msg_chars

    return trimmed


def summarize_conversation(messages: list) -> str:
    """
    Create a summary of conversation for returning patients.

    Args:
        messages: Full message history

    Returns:
        2-3 sentence summary
    """
    llm = get_llm()

    conversation_text = "\n".join([
        f"{msg.type}: {str(msg.content)[:200]}"
        for msg in messages[-20:]  # Last 20 messages
    ])

    summary_prompt = f"""Summarize this patient encounter in 2-3 sentences for medical records:

{conversation_text}

Focus on: chief complaint, key symptoms, diagnosis/plan if mentioned.
"""

    response = llm.invoke([HumanMessage(content=summary_prompt)])
    summary = response.content if isinstance(response.content, str) else str(response.content)

    return summary[:500]  # Max 500 characters
