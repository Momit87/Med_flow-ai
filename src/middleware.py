"""
Middleware for MedFlow AI.
Includes PII redaction, input guardrails, output disclaimers, and safety checks.
"""

import re
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.config import settings


# ============================================================================
# PII Redaction Middleware
# ============================================================================

class PIIRedactor:
    """
    Redact personally identifiable information from text.
    """

    # PII patterns
    PATTERNS = {
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "date_of_birth": r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        "mrn": r'\bMRN:?\s*\d{6,10}\b',
    }

    @classmethod
    def redact(cls, text: str, redaction_char: str = "X") -> tuple[str, list[str]]:
        """
        Redact PII from text.

        Args:
            text: Input text
            redaction_char: Character to use for redaction

        Returns:
            Tuple of (redacted_text, list of PII types found)
        """
        if not text:
            return text, []

        redacted = text
        pii_found = []

        for pii_type, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pii_found.append(pii_type)
                # Replace each match with X's of same length
                for match in matches:
                    redacted = redacted.replace(match, redaction_char * len(match))

        return redacted, pii_found

    @classmethod
    def redact_messages(cls, messages: list) -> tuple[list, list[str]]:
        """
        Redact PII from all messages in a list.

        Args:
            messages: List of messages

        Returns:
            Tuple of (redacted_messages, list of PII types found across all messages)
        """
        redacted_messages = []
        all_pii_found = set()

        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                redacted_content, pii_found = cls.redact(content)
                all_pii_found.update(pii_found)

                # Create new message with redacted content
                if isinstance(msg, HumanMessage):
                    redacted_messages.append(HumanMessage(content=redacted_content))
                else:
                    redacted_messages.append(AIMessage(content=redacted_content))
            else:
                redacted_messages.append(msg)

        return redacted_messages, list(all_pii_found)


# ============================================================================
# Input Guardrails
# ============================================================================

class InputGuardrails:
    """
    Validate and filter user input for safety and appropriateness.
    """

    # Emergency keywords that trigger immediate escalation
    EMERGENCY_KEYWORDS = settings.EMERGENCY_KEYWORDS

    # Inappropriate content patterns
    INAPPROPRIATE_PATTERNS = [
        r'\b(hack|crack|exploit|bypass)\b',
        r'\b(illegal|unlawful)\s+(drug|substance)\b',
        r'\b(how\s+to\s+make|synthesize)\s+(drug|explosive|weapon)\b',
    ]

    @classmethod
    def check_emergency(cls, text: str) -> tuple[bool, str]:
        """
        Check if text contains emergency indicators.

        Args:
            text: User input text

        Returns:
            Tuple of (is_emergency, matched_keyword)
        """
        text_lower = text.lower()

        for keyword in cls.EMERGENCY_KEYWORDS:
            if keyword in text_lower:
                return True, keyword

        return False, ""

    @classmethod
    def check_inappropriate(cls, text: str) -> tuple[bool, str]:
        """
        Check if text contains inappropriate content.

        Args:
            text: User input text

        Returns:
            Tuple of (is_inappropriate, reason)
        """
        for pattern in cls.INAPPROPRIATE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "Request appears to involve illegal or harmful content"

        # Check for very short or empty input
        if len(text.strip()) < 3:
            return True, "Input too short or empty"

        # Check for excessive length (potential spam/abuse)
        if len(text) > 5000:
            return True, "Input exceeds maximum length"

        return False, ""

    @classmethod
    def validate_input(cls, text: str) -> Dict[str, Any]:
        """
        Comprehensive input validation.

        Args:
            text: User input text

        Returns:
            Dictionary with validation results
        """
        result = {
            "is_valid": True,
            "is_emergency": False,
            "emergency_keyword": "",
            "is_inappropriate": False,
            "rejection_reason": "",
            "warnings": []
        }

        # Check for emergency
        is_emergency, keyword = cls.check_emergency(text)
        if is_emergency:
            result["is_emergency"] = True
            result["emergency_keyword"] = keyword
            result["warnings"].append(f"Emergency indicator detected: '{keyword}'")

        # Check for inappropriate content
        is_inappropriate, reason = cls.check_inappropriate(text)
        if is_inappropriate:
            result["is_valid"] = False
            result["is_inappropriate"] = True
            result["rejection_reason"] = reason

        return result


# ============================================================================
# Output Disclaimers
# ============================================================================

class OutputDisclaimer:
    """
    Add appropriate disclaimers and warnings to AI responses.
    """

    GENERAL_DISCLAIMER = """

---
⚠️ **Medical Disclaimer**: This AI assistant provides general health information only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read here. If you think you may have a medical emergency, call your doctor, 911 (or your local emergency number) immediately.
"""

    EMERGENCY_DISCLAIMER = """

🚨 **EMERGENCY NOTICE**: If you are experiencing a medical emergency, please call 911 (or your local emergency number) immediately or go to the nearest emergency room. This AI system cannot provide emergency medical care.
"""

    MEDICATION_DISCLAIMER = """

💊 **Medication Safety**: The medication information provided is for educational purposes only. Do not start, stop, or change any medications without consulting your healthcare provider or pharmacist. Report any adverse drug reactions to your doctor immediately.
"""

    @classmethod
    def add_disclaimer(cls, text: str, disclaimer_type: str = "general") -> str:
        """
        Add appropriate disclaimer to response text.

        Args:
            text: AI response text
            disclaimer_type: Type of disclaimer (general, emergency, medication)

        Returns:
            Text with disclaimer appended
        """
        if disclaimer_type == "emergency":
            return text + cls.EMERGENCY_DISCLAIMER
        elif disclaimer_type == "medication":
            return text + cls.MEDICATION_DISCLAIMER
        else:
            return text + cls.GENERAL_DISCLAIMER

    @classmethod
    def format_with_citations(cls, text: str, sources: List[Dict[str, Any]]) -> str:
        """
        Add citations and sources to response.

        Args:
            text: AI response text
            sources: List of source dictionaries with 'title', 'url', etc.

        Returns:
            Text with formatted citations
        """
        if not sources:
            return text

        citations = "\n\n---\n**Sources:**\n"
        for i, source in enumerate(sources[:5], 1):  # Max 5 citations
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            citations += f"{i}. {title}"
            if url:
                citations += f" - {url}"
            citations += "\n"

        return text + citations


# ============================================================================
# Emergency Detection and Routing
# ============================================================================

class EmergencyDetector:
    """
    Detect emergency situations and route appropriately.
    """

    @classmethod
    def assess_urgency(cls, text: str, symptoms: List[str], triage_level: str = None) -> Dict[str, Any]:
        """
        Assess urgency level of patient situation.

        Args:
            text: Patient input text
            symptoms: List of reported symptoms
            triage_level: ESI triage level if available

        Returns:
            Dictionary with urgency assessment
        """
        assessment = {
            "urgency_level": "routine",  # routine, urgent, emergency, critical
            "requires_immediate_attention": False,
            "call_911": False,
            "emergency_keywords_found": [],
            "reasoning": ""
        }

        # Check triage level
        if triage_level in ["ESI-1", "ESI-2"]:
            assessment["urgency_level"] = "critical" if triage_level == "ESI-1" else "emergency"
            assessment["requires_immediate_attention"] = True
            assessment["call_911"] = triage_level == "ESI-1"
            assessment["reasoning"] = f"Triage level {triage_level} indicates immediate medical attention needed"
            return assessment

        # Check for emergency keywords
        text_lower = text.lower()
        symptoms_text = " ".join(symptoms).lower()
        combined_text = f"{text_lower} {symptoms_text}"

        for keyword in settings.EMERGENCY_KEYWORDS:
            if keyword in combined_text:
                assessment["emergency_keywords_found"].append(keyword)

        if assessment["emergency_keywords_found"]:
            assessment["urgency_level"] = "emergency"
            assessment["requires_immediate_attention"] = True
            assessment["call_911"] = True
            assessment["reasoning"] = f"Emergency indicators found: {', '.join(assessment['emergency_keywords_found'])}"

        return assessment

    @classmethod
    def create_emergency_response(cls, urgency_assessment: Dict[str, Any]) -> str:
        """
        Create appropriate emergency response message.

        Args:
            urgency_assessment: Output from assess_urgency()

        Returns:
            Emergency response message
        """
        if urgency_assessment["call_911"]:
            return """🚨 **IMMEDIATE EMERGENCY - CALL 911 NOW**

Based on what you've described, this appears to be a medical emergency that requires immediate professional care.

**CALL 911 or go to the nearest emergency room immediately.**

Do NOT:
- Wait to see if symptoms improve
- Drive yourself (call an ambulance)
- Delay seeking emergency care

Emergency services can begin treatment on the way to the hospital. Your safety is the top priority.

If you are with someone experiencing these symptoms, call 911 for them immediately.
"""
        elif urgency_assessment["urgency_level"] == "emergency":
            return """⚠️ **URGENT MEDICAL ATTENTION NEEDED**

The symptoms you've described require urgent medical evaluation. Please:

1. Go to the nearest emergency room, OR
2. Call your doctor immediately for same-day evaluation

Do not wait. While this may not require calling 911, you should seek medical care right away.

If symptoms worsen or you develop any of these warning signs, call 911:
- Difficulty breathing or severe shortness of breath
- Chest pain or pressure
- Sudden confusion or difficulty speaking
- Severe bleeding that won't stop
- Loss of consciousness
"""
        else:
            return """

This appears to require medical attention. Please contact your healthcare provider to schedule an appointment or visit an urgent care clinic if your symptoms worsen.
"""


# ============================================================================
# Content Filtering
# ============================================================================

def filter_tool_messages(messages: list) -> list:
    """
    Remove tool-related messages before showing to patient.

    Args:
        messages: Full message list including tool calls

    Returns:
        Filtered message list with only human-friendly messages
    """
    from langchain_core.messages import ToolMessage

    return [
        msg for msg in messages
        if not isinstance(msg, ToolMessage)
    ]


def sanitize_medical_terminology(text: str, reading_level: str = "8th_grade") -> str:
    """
    Replace complex medical terms with simpler alternatives.

    Args:
        text: Text with medical terminology
        reading_level: Target reading level

    Returns:
        Text with simplified terminology
    """
    if reading_level == "college":
        return text  # Keep technical terms

    # Simple replacement dictionary (expand as needed)
    replacements = {
        "myocardial infarction": "heart attack",
        "cerebrovascular accident": "stroke",
        "hypertension": "high blood pressure",
        "hypotension": "low blood pressure",
        "tachycardia": "fast heart rate",
        "bradycardia": "slow heart rate",
        "dyspnea": "shortness of breath",
        "hemoptysis": "coughing up blood",
        "syncope": "fainting",
        "vertigo": "dizziness",
        "nausea": "feeling sick to your stomach",
        "emesis": "vomiting",
        "pyrexia": "fever",
        "analgesic": "pain reliever",
        "antipyretic": "fever reducer"
    }

    sanitized = text
    for medical_term, simple_term in replacements.items():
        sanitized = re.sub(
            r'\b' + medical_term + r'\b',
            simple_term,
            sanitized,
            flags=re.IGNORECASE
        )

    return sanitized
