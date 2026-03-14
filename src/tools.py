"""
Custom tools for MedFlow AI agents.
Includes medication image analysis, drug interaction checking, and clinical lookups.
"""

from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from src.config import get_vision_llm
from src.rag import rag_manager
import base64
import re
import requests
from datetime import datetime, timedelta


@tool
def analyze_medication_image(image_data: str, image_format: str = "jpeg") -> Dict[str, Any]:
    """
    Analyze a medication bottle/package photo to extract drug information.

    Args:
        image_data: Base64-encoded image data or file path
        image_format: Image format (jpeg, png, etc.)

    Returns:
        Dictionary with extracted medication info: name, dosage, instructions, NDC code
    """
    try:
        vision_llm = get_vision_llm()

        # Create multimodal message
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Analyze this medication bottle/package image and extract:
1. Medication name (brand and generic if visible)
2. Dosage strength (e.g., 500mg, 10mg)
3. Dosage form (tablet, capsule, liquid, etc.)
4. Instructions (if visible)
5. NDC code (if visible)
6. Expiration date (if visible)

Return in this format:
MEDICATION_NAME: [name]
GENERIC_NAME: [generic name or 'Not visible']
DOSAGE: [strength and form]
INSTRUCTIONS: [dosing instructions or 'Not visible']
NDC: [code or 'Not visible']
EXPIRATION: [date or 'Not visible']
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{image_data}"
                    }
                }
            ]
        )

        response = vision_llm.invoke([message])
        extracted_text = response.content if isinstance(response.content, str) else str(response.content)

        # Parse the response
        medication_info = {
            "medication_name": _extract_field(extracted_text, "MEDICATION_NAME"),
            "generic_name": _extract_field(extracted_text, "GENERIC_NAME"),
            "dosage": _extract_field(extracted_text, "DOSAGE"),
            "instructions": _extract_field(extracted_text, "INSTRUCTIONS"),
            "ndc_code": _extract_field(extracted_text, "NDC"),
            "expiration_date": _extract_field(extracted_text, "EXPIRATION"),
            "source": "image_analysis",
            "confidence": "high" if "not visible" not in extracted_text.lower() else "medium"
        }

        return medication_info

    except Exception as e:
        return {
            "error": str(e),
            "medication_name": "Error analyzing image",
            "source": "image_analysis"
        }


@tool
def query_drug_interactions(medications: List[str], severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Check for drug-drug interactions among a list of medications.

    Args:
        medications: List of medication names to check
        severity_filter: Optional filter for severity level (critical, moderate, minor)

    Returns:
        List of interaction warnings with severity and clinical guidance
    """
    if not medications or len(medications) < 2:
        return []

    interactions = []

    # Check each pair of medications
    for i, med1 in enumerate(medications):
        for med2 in medications[i+1:]:
            # Query ChromaDB for interaction info
            query = f"interaction between {med1} and {med2}"
            results = rag_manager.query_drug_interactions(query, k=3)

            if results:
                for result in results:
                    interaction = {
                        "drug_pair": [med1, med2],
                        "description": result.get("content", ""),
                        "source": result.get("source", "knowledge_base"),
                        "severity": _extract_severity(result.get("content", "")),
                        "metadata": result.get("metadata", {})
                    }

                    # Apply severity filter if specified
                    if severity_filter is None or interaction["severity"] == severity_filter:
                        interactions.append(interaction)

    return interactions


@tool
def search_clinical_guidelines(condition: str, care_setting: str = "emergency") -> List[Dict[str, Any]]:
    """
    Search clinical practice guidelines for a specific condition.

    Args:
        condition: Medical condition or chief complaint
        care_setting: Care setting (emergency, primary_care, pediatrics)

    Returns:
        List of relevant clinical guidelines with protocols and recommendations
    """
    results = rag_manager.query_guidelines(condition, k=5)

    guidelines = []
    for result in results:
        guideline = {
            "condition": result.get("metadata", {}).get("condition", condition),
            "protocol": result.get("content", ""),
            "source": result.get("source", "clinical_guidelines"),
            "care_setting": care_setting,
            "evidence_level": result.get("metadata", {}).get("evidence_level", "standard_practice")
        }
        guidelines.append(guideline)

    return guidelines


@tool
def lookup_icd10_code(diagnosis: str) -> Dict[str, Any]:
    """
    Look up ICD-10 diagnosis code for a condition.

    Args:
        diagnosis: Clinical diagnosis or condition description

    Returns:
        Dictionary with ICD-10 code, description, and category
    """
    # Simplified ICD-10 lookup (in production, use official API or database)
    common_codes = {
        "chest pain": {"code": "R07.9", "description": "Chest pain, unspecified", "category": "Symptoms"},
        "shortness of breath": {"code": "R06.02", "description": "Shortness of breath", "category": "Symptoms"},
        "abdominal pain": {"code": "R10.9", "description": "Abdominal pain, unspecified", "category": "Symptoms"},
        "headache": {"code": "R51", "description": "Headache", "category": "Symptoms"},
        "fever": {"code": "R50.9", "description": "Fever, unspecified", "category": "Symptoms"},
        "hypertension": {"code": "I10", "description": "Essential (primary) hypertension", "category": "Circulatory"},
        "diabetes": {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "category": "Endocrine"},
        "asthma": {"code": "J45.909", "description": "Unspecified asthma, uncomplicated", "category": "Respiratory"},
    }

    # Find best match
    diagnosis_lower = diagnosis.lower()
    for key, value in common_codes.items():
        if key in diagnosis_lower:
            return value

    return {
        "code": "R69",
        "description": "Illness, unspecified",
        "category": "Symptoms",
        "note": "Exact match not found - using general code"
    }


@tool
def get_patient_education(topic: str, reading_level: str = "8th_grade") -> Dict[str, Any]:
    """
    Retrieve patient education materials on a specific health topic.

    Args:
        topic: Health topic or condition
        reading_level: Reading level (6th_grade, 8th_grade, college)

    Returns:
        Dictionary with educational content tailored to reading level
    """
    # Query patient education collection
    results = rag_manager.query_education(topic, k=3)

    if not results:
        return {
            "topic": topic,
            "content": f"General information about {topic} is being prepared. Please consult with your healthcare provider for specific guidance.",
            "reading_level": reading_level,
            "source": "system_default"
        }

    # Combine and format results
    education_content = []
    for result in results:
        education_content.append(result.get("content", ""))

    return {
        "topic": topic,
        "content": "\n\n".join(education_content),
        "reading_level": reading_level,
        "source": "patient_education_library",
        "key_points": _extract_key_points("\n\n".join(education_content))
    }


@tool
def search_pubmed_live(query: str, max_results: int = 5, recency_days: int = 365) -> List[Dict[str, Any]]:
    """
    Search PubMed for recent medical literature and research articles.

    Args:
        query: Medical search query
        max_results: Maximum number of results to return (default 5)
        recency_days: Only include articles from last N days (default 365)

    Returns:
        List of research articles with title, abstract, authors, and publication info
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=recency_days)
        date_range = f"{start_date.year}/{start_date.month}/{start_date.day}:{end_date.year}/{end_date.month}/{end_date.day}"

        # Step 1: Search for PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "datetype": "pdat",
            "mindate": start_date.strftime("%Y/%m/%d"),
            "maxdate": end_date.strftime("%Y/%m/%d"),
            "retmode": "json"
        }

        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        search_data = search_response.json()

        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return [{
                "title": "No recent articles found",
                "abstract": f"No PubMed articles found for '{query}' in the last {recency_days} days.",
                "source": "pubmed_api",
                "pmid": None
            }]

        # Step 2: Fetch article details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json"
        }

        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
        fetch_response.raise_for_status()
        fetch_data = fetch_response.json()

        # Parse results
        articles = []
        result_data = fetch_data.get("result", {})

        for pmid in pmids:
            if pmid in result_data:
                article_data = result_data[pmid]

                # Get authors
                authors = []
                for author in article_data.get("authors", [])[:3]:  # First 3 authors
                    authors.append(author.get("name", ""))

                articles.append({
                    "pmid": pmid,
                    "title": article_data.get("title", "No title"),
                    "abstract": article_data.get("abstract", "Abstract not available"),
                    "authors": authors,
                    "journal": article_data.get("source", ""),
                    "pub_date": article_data.get("pubdate", ""),
                    "doi": article_data.get("elocationid", ""),
                    "source": "pubmed_api",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })

        return articles

    except Exception as e:
        return [{
            "title": "Error searching PubMed",
            "abstract": f"Unable to search PubMed: {str(e)}",
            "source": "pubmed_api",
            "error": str(e)
        }]


# Helper functions

def _extract_field(text: str, field_name: str) -> str:
    """Extract a specific field from formatted text."""
    pattern = rf"{field_name}:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        return value if value and value.lower() != "not visible" else "Not provided"
    return "Not provided"


def _extract_severity(text: str) -> str:
    """Extract severity level from interaction description."""
    text_lower = text.lower()
    if any(word in text_lower for word in ["critical", "severe", "life-threatening", "contraindicated"]):
        return "critical"
    elif any(word in text_lower for word in ["moderate", "caution", "monitor"]):
        return "moderate"
    elif any(word in text_lower for word in ["minor", "mild", "low risk"]):
        return "minor"
    return "moderate"  # default


def _extract_key_points(text: str) -> List[str]:
    """Extract key points from educational content."""
    key_points = []

    # Look for bullet points or numbered lists
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        # Match bullet points or numbered items
        if line.startswith("-") or line.startswith("•") or re.match(r"^\d+\.", line):
            key_points.append(line.lstrip("-•0123456789. "))

    # If no bullet points found, extract first 3 sentences
    if not key_points:
        sentences = re.split(r'[.!?]+', text)
        key_points = [s.strip() for s in sentences[:3] if s.strip()]

    return key_points[:5]  # Return max 5 key points


# Export all tools as a list for easy binding
ALL_TOOLS = [
    analyze_medication_image,
    query_drug_interactions,
    search_clinical_guidelines,
    lookup_icd10_code,
    get_patient_education,
    search_pubmed_live
]
