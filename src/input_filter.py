"""
Input Filter - Detect medical/healthcare-related queries
Prevents unnecessary API calls for irrelevant questions
"""

import re
from typing import Dict, Any

# Medical/healthcare-related keywords
MEDICAL_KEYWORDS = {
    # Symptoms
    "pain", "ache", "hurt", "sore", "fever", "cough", "cold", "flu", "nausea", "vomit",
    "dizzy", "headache", "migraine", "fatigue", "tired", "weak", "bleeding", "blood",
    "swelling", "rash", "itch", "burn", "chest", "breath", "breathing", "asthma",
    "sneez", "congestion", "diarrhea", "constipation", "stomach", "abdom", "cramp",

    # Body parts
    "head", "neck", "shoulder", "arm", "hand", "finger", "leg", "foot", "toe",
    "back", "spine", "chest", "heart", "lung", "liver", "kidney", "stomach",
    "throat", "ear", "eye", "nose", "mouth", "tooth", "teeth", "skin", "joint",
    "muscle", "bone", "knee", "ankle", "wrist", "elbow", "hip",

    # Medical terms
    "disease", "infection", "virus", "bacteria", "cancer", "tumor", "diabetes",
    "pressure", "hypertension", "stroke", "seizure", "allergy", "allergic",
    "medication", "medicine", "drug", "pill", "prescription", "dose", "dosage",
    "treatment", "therapy", "surgery", "operation", "procedure", "diagnosis",
    "test", "lab", "xray", "x-ray", "mri", "ct scan", "ultrasound",

    # Medical scenarios
    "doctor", "hospital", "clinic", "emergency", "ambulance", "911",
    "appointment", "visit", "checkup", "exam", "physical", "vaccine",
    "immunization", "shot", "injury", "wound", "broken", "fracture",
    "sprain", "strain", "sick", "ill", "health", "medical",

    # Common conditions
    "covid", "corona", "pneumonia", "bronchitis", "sinusitis", "arthritis",
    "depression", "anxiety", "insomnia", "sleep", "mental", "stress",
    "pregnant", "pregnancy", "baby", "infant", "child", "pediatric",

    # Medications (common ones)
    "aspirin", "tylenol", "advil", "ibuprofen", "acetaminophen", "penicillin",
    "antibiotic", "insulin", "inhaler", "warfarin", "metformin", "statin",
}

# Non-medical phrases that should be rejected
NON_MEDICAL_PHRASES = [
    "weather", "time", "date", "joke", "story", "game", "movie", "song",
    "recipe", "food", "restaurant", "sports", "football", "basketball",
    "politics", "news", "stock", "market", "price", "buy", "sell",
    "homework", "essay", "math", "calculation", "code", "programming",
]


def is_medical_query(text: str) -> Dict[str, Any]:
    """
    Check if the input text is medical/healthcare-related.

    Returns:
        dict: {
            "is_medical": bool,
            "confidence": float,
            "matched_keywords": list,
            "reason": str
        }
    """
    if not text or len(text.strip()) < 3:
        return {
            "is_medical": False,
            "confidence": 0.0,
            "matched_keywords": [],
            "reason": "Input too short"
        }

    text_lower = text.lower()

    # Check for non-medical phrases first
    for phrase in NON_MEDICAL_PHRASES:
        if phrase in text_lower:
            return {
                "is_medical": False,
                "confidence": 0.9,
                "matched_keywords": [],
                "reason": f"Detected non-medical topic: '{phrase}'"
            }

    # Check for medical keywords
    matched_keywords = []
    for keyword in MEDICAL_KEYWORDS:
        # Use word boundary matching to avoid false positives
        pattern = r'\b' + re.escape(keyword)
        if re.search(pattern, text_lower):
            matched_keywords.append(keyword)

    # Calculate confidence based on number of matches
    if len(matched_keywords) == 0:
        confidence = 0.0
    elif len(matched_keywords) == 1:
        confidence = 0.6
    elif len(matched_keywords) == 2:
        confidence = 0.8
    else:
        confidence = 0.95

    # Consider it medical if confidence > 0.5
    is_medical = confidence > 0.5

    return {
        "is_medical": is_medical,
        "confidence": confidence,
        "matched_keywords": matched_keywords[:5],  # Return top 5
        "reason": f"Found {len(matched_keywords)} medical keywords" if is_medical else "No medical keywords detected"
    }


def get_non_medical_response() -> str:
    """
    Return a polite message for non-medical queries.
    """
    return """I'm MedFlow AI, a specialized medical assistant designed to help with healthcare-related questions.

I can help you with:
- Medical symptoms and conditions
- Medication information and interactions
- Clinical guidance and patient education
- Health-related concerns

For non-medical questions, please use a general-purpose AI assistant.

**If you're experiencing a medical emergency, please call 911 or go to your nearest emergency room immediately.**"""


# Testing examples
if __name__ == "__main__":
    test_cases = [
        "I have a headache and fever",
        "What's the weather today?",
        "Tell me a joke",
        "My chest hurts and I can't breathe",
        "Is aspirin safe with warfarin?",
        "Who won the game last night?",
        "I need help with my diabetes medication",
        "What time is it?",
        "My child has a rash",
        "Calculate 2+2",
    ]

    print("Testing Input Filter:\n")
    for test in test_cases:
        result = is_medical_query(test)
        print(f"Query: '{test}'")
        print(f"  Medical: {result['is_medical']} (confidence: {result['confidence']:.2f})")
        print(f"  Reason: {result['reason']}")
        if result['matched_keywords']:
            print(f"  Keywords: {result['matched_keywords']}")
        print()
