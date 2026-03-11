"""
Configuration for LLM providers, embeddings, and system settings.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


# ============================================================================
# LLM Configuration
# ============================================================================

def get_llm(provider: str = "gemini", temperature: float = 0.1):
    """
    Get LLM instance based on provider.

    Args:
        provider: Either "gemini" or "groq"
        temperature: Temperature for generation (0.0-1.0)

    Returns:
        ChatModel instance
    """
    if provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    else:  # default to gemini
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )


def get_vision_llm():
    """Get multimodal LLM for image analysis."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,  # Deterministic for image analysis
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_embeddings():
    """Get embedding model for RAG."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ============================================================================
# System Settings
# ============================================================================

class Settings:
    """Global system settings."""

    # Directories
    CHROMA_DB_DIR = "./chroma_db"
    DATA_DIR = "./data"
    CHECKPOINT_DB = "./encounters.db"

    # LLM Settings
    DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    DEFAULT_TEMPERATURE = 0.1  # Low temperature for clinical accuracy

    # RAG Settings
    RAG_SIMILARITY_THRESHOLD = 0.75
    RAG_TOP_K = 5
    RAG_CHUNK_SIZE = 1000
    RAG_CHUNK_OVERLAP = 200

    # Streaming Settings
    STREAM_FIRST_TOKEN_TIMEOUT_MS = 500
    MAX_STREAM_TOKENS = 2000

    # Safety Settings
    EMERGENCY_KEYWORDS = [
        "can't breathe", "chest pain", "stroke", "seizure",
        "unconscious", "severe bleeding", "overdose", "suicidal",
        "heart attack", "choking"
    ]

    # LangSmith Settings
    LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "medflow-ai")

    # Assistant Configurations
    ASSISTANTS = {
        "ed": {
            "name": "Emergency Department Assistant",
            "reading_level": "college",
            "care_setting": "ed",
            "system_prompt": """You are an AI assistant for emergency department clinical intake.
            Focus on rapid triage, identifying time-critical conditions, and emergency protocols.
            Always prioritize patient safety and escalate urgent cases immediately.""",
        },
        "primary_care": {
            "name": "Primary Care Assistant",
            "reading_level": "8th_grade",
            "care_setting": "primary",
            "system_prompt": """You are an AI assistant for primary care intake.
            Focus on preventive care, lifestyle factors, and comprehensive health history.
            Use clear, accessible language appropriate for general patient education.""",
        },
        "pediatrics": {
            "name": "Pediatrics Assistant",
            "reading_level": "6th_grade",
            "care_setting": "pediatrics",
            "system_prompt": """You are an AI assistant for pediatric care intake.
            Focus on age-appropriate assessment, growth/development, and parent-friendly communication.
            Always consider weight-based dosing and developmental milestones.""",
        },
    }


settings = Settings()
