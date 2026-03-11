"""
Data ingestion script for populating ChromaDB with medical knowledge.
Fetches data from openFDA, creates synthetic clinical guidelines and patient education content.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from langchain_core.documents import Document
from src.rag import rag_manager
from src.config import settings
import json
import time


def ingest_drug_interactions(limit: int = 50):
    """
    Fetch drug interaction warnings from openFDA and index into ChromaDB.

    Args:
        limit: Number of drug labels to fetch
    """
    print(f"\n=== Ingesting Drug Interaction Data ===")
    print(f"Fetching {limit} drug labels from openFDA...")

    try:
        url = "https://api.fda.gov/drug/label.json"
        params = {
            "limit": limit,
            "search": "warnings:interaction OR drug_interactions:[* TO *]"
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])

        print(f"Retrieved {len(results)} drug labels")

        documents = []
        for r in results:
            # Extract drug name
            drug_names = r.get("openfda", {}).get("brand_name", ["Unknown"])
            drug_name = drug_names[0] if drug_names else "Unknown"

            # Extract warnings and interactions
            warnings = r.get("warnings", [])
            drug_interactions = r.get("drug_interactions", [])

            # Combine text
            text = ""
            if warnings:
                text += "WARNINGS:\n" + "\n".join(warnings) + "\n\n"
            if drug_interactions:
                text += "DRUG INTERACTIONS:\n" + "\n".join(drug_interactions)

            if text.strip():
                # Chunk the text
                chunks = rag_manager.chunk_text(
                    text,
                    chunk_size=1000,
                    chunk_overlap=200,
                    metadata={
                        "drug_name": drug_name,
                        "source": "openFDA",
                        "type": "drug_interaction"
                    }
                )
                documents.extend(chunks)

        if documents:
            print(f"Adding {len(documents)} chunks to drug_interactions collection...")
            rag_manager.add_documents("drug_interactions", documents)
            print(f"✓ Successfully ingested {len(documents)} drug interaction chunks")
        else:
            print("⚠ No drug interaction data found")

    except Exception as e:
        print(f"✗ Error fetching from openFDA: {e}")
        print("Adding synthetic drug interaction data instead...")
        ingest_synthetic_drug_interactions()


def ingest_synthetic_drug_interactions():
    """Add synthetic drug interaction data for demo purposes."""
    synthetic_interactions = [
        {
            "drug_pair": ("Warfarin", "Aspirin"),
            "severity": "critical",
            "description": "Concurrent use of warfarin and aspirin significantly increases bleeding risk. Monitor INR closely and consider dose adjustments.",
            "evidence": "Multiple clinical studies demonstrate 2-3x increased bleeding risk with combination therapy."
        },
        {
            "drug_pair": ("Metformin", "Contrast Dye"),
            "severity": "critical",
            "description": "Metformin should be held before and after contrast studies due to risk of lactic acidosis, especially in patients with renal impairment.",
            "evidence": "FDA Black Box Warning: Hold metformin for 48 hours after IV contrast administration."
        },
        {
            "drug_pair": ("Lisinopril", "Potassium Supplements"),
            "severity": "moderate",
            "description": "ACE inhibitors like lisinopril increase potassium retention. Adding potassium supplements may cause hyperkalemia.",
            "evidence": "ACE inhibitors reduce aldosterone, leading to potassium retention. Monitor serum potassium levels."
        },
        {
            "drug_pair": ("Simvastatin", "Grapefruit Juice"),
            "severity": "moderate",
            "description": "Grapefruit juice inhibits CYP3A4 metabolism of simvastatin, increasing blood levels and risk of myopathy.",
            "evidence": "Grapefruit juice can increase simvastatin levels by 3-fold, increasing rhabdomyolysis risk."
        },
        {
            "drug_pair": ("Levothyroxine", "Calcium Carbonate"),
            "severity": "moderate",
            "description": "Calcium supplements can reduce levothyroxine absorption. Separate administration by at least 4 hours.",
            "evidence": "Calcium binds to levothyroxine in the GI tract, reducing bioavailability by up to 40%."
        },
        {
            "drug_pair": ("Fluoxetine", "Tramadol"),
            "severity": "critical",
            "description": "SSRIs combined with tramadol increase risk of serotonin syndrome. Monitor for agitation, hyperthermia, and hyperreflexia.",
            "evidence": "Both drugs increase serotonergic activity. Case reports of severe serotonin syndrome documented."
        },
        {
            "drug_pair": ("Prednisone", "Ibuprofen"),
            "severity": "moderate",
            "description": "Combining corticosteroids with NSAIDs significantly increases GI bleeding risk.",
            "evidence": "Synergistic effect on gastric mucosa. Consider PPI prophylaxis if combination necessary."
        },
    ]

    documents = []
    for interaction in synthetic_interactions:
        text = f"""
DRUG INTERACTION: {interaction['drug_pair'][0]} + {interaction['drug_pair'][1]}

SEVERITY: {interaction['severity'].upper()}

DESCRIPTION: {interaction['description']}

EVIDENCE: {interaction['evidence']}

CLINICAL RECOMMENDATION: Evaluate risk vs benefit. Consider alternative medications or enhanced monitoring.
"""
        chunks = rag_manager.chunk_text(
            text,
            metadata={
                "drug_a": interaction['drug_pair'][0],
                "drug_b": interaction['drug_pair'][1],
                "severity": interaction['severity'],
                "source": "synthetic_demo_data",
                "type": "drug_interaction"
            }
        )
        documents.extend(chunks)

    if documents:
        rag_manager.add_documents("drug_interactions", documents)
        print(f"✓ Added {len(documents)} synthetic drug interaction chunks")


def ingest_clinical_guidelines():
    """Add synthetic clinical guidelines for common emergency conditions."""
    print(f"\n=== Ingesting Clinical Guidelines ===")

    guidelines = [
        {
            "condition": "Chest Pain",
            "content": """
CHIEF COMPLAINT: Chest Pain - Emergency Assessment Protocol

IMMEDIATE ASSESSMENT (ESI Level 1-2 if):
- Severe, crushing substernal chest pressure
- Radiation to left arm, jaw, or back
- Associated dyspnea, diaphoresis, nausea
- Vital sign instability

DIFFERENTIAL DIAGNOSIS:
1. Acute Coronary Syndrome (ACS)
   - ST-elevation MI (STEMI): Immediate cath lab activation
   - Non-ST-elevation MI (NSTEMI): Serial troponins, cardiology consult
   - Unstable angina: Risk stratification

2. Pulmonary Embolism (PE)
   - Risk factors: Recent surgery, DVT, prolonged immobility
   - Wells criteria and D-dimer if low/moderate risk
   - CT pulmonary angiography if high clinical suspicion

3. Aortic Dissection
   - Tearing/ripping pain, radiating to back
   - Blood pressure differential between arms
   - Widened mediastinum on chest X-ray
   - CT angiography for diagnosis

INITIAL WORKUP:
- 12-lead ECG within 10 minutes
- Cardiac biomarkers (troponin) at 0 and 3 hours
- Chest X-ray
- Basic metabolic panel, CBC
- Consider D-dimer if PE suspected

TREATMENT:
- Oxygen if hypoxic (SpO2 < 94%)
- Aspirin 325mg PO (unless contraindicated)
- Sublingual nitroglycerin if BP permits
- IV access and continuous monitoring
"""
        },
        {
            "condition": "Shortness of Breath",
            "content": """
CHIEF COMPLAINT: Dyspnea - Emergency Assessment

TRIAGE CLASSIFICATION:
- ESI-1: Severe respiratory distress, SpO2 < 90%, altered mental status
- ESI-2: Moderate distress, wheezing, accessory muscle use
- ESI-3: Mild dyspnea with normal vitals

DIFFERENTIAL DIAGNOSIS:
1. Asthma Exacerbation / COPD
   - Wheezing, prolonged expiration
   - Peak flow < 50% predicted = severe
   - Treatment: Albuterol, ipratropium, steroids

2. Pulmonary Embolism
   - Sudden onset, pleuritic pain
   - Risk factors: DVT, surgery, malignancy
   - Wells criteria + D-dimer → CT angiography

3. Congestive Heart Failure
   - Orthopnea, paroxysmal nocturnal dyspnea
   - Bilateral crackles, peripheral edema
   - BNP > 400 suggests CHF
   - Treatment: Diuretics, nitroglycerin

4. Pneumonia
   - Fever, productive cough, pleuritic pain
   - Chest X-ray for diagnosis
   - Antibiotics based on CURB-65 score

INITIAL ASSESSMENT:
- Vital signs including SpO2
- Assess work of breathing, accessory muscle use
- Lung auscultation
- Oxygen supplementation to maintain SpO2 > 94%
"""
        },
        {
            "condition": "Abdominal Pain",
            "content": """
CHIEF COMPLAINT: Abdominal Pain - Emergency Evaluation

SURGICAL EMERGENCIES (ESI-1/2):
- Ruptured AAA: Pulsatile mass, hypotension, back pain
- Perforated viscus: Rigid abdomen, free air on imaging
- Bowel obstruction: Distension, absent bowel sounds, vomiting
- Acute appendicitis: RLQ tenderness, fever, elevated WBC

LOCATION-BASED DIFFERENTIAL:
RUQ: Cholecystitis, hepatitis
- Murphy's sign, RUQ ultrasound

RLQ: Appendicitis, ovarian pathology
- McBurney's point tenderness, CT abdomen/pelvis

LUQ: Splenic injury, pancreatitis
- Trauma history, elevated lipase

LLQ: Diverticulitis, colitis
- Fever, change in bowel habits, CT with contrast

INITIAL WORKUP:
- Complete blood count, comprehensive metabolic panel
- Lipase if epigastric pain
- Urinalysis and pregnancy test (females)
- Imaging: Start with ultrasound or CT based on presentation
- NPO if surgical abdomen suspected
"""
        }
    ]

    documents = []
    for guideline in guidelines:
        chunks = rag_manager.chunk_text(
            guideline["content"],
            chunk_size=800,
            chunk_overlap=150,
            metadata={
                "condition": guideline["condition"],
                "source": "clinical_guidelines",
                "type": "emergency_protocol"
            }
        )
        documents.extend(chunks)

    if documents:
        rag_manager.add_documents("clinical_guidelines", documents)
        print(f"✓ Added {len(documents)} clinical guideline chunks")


def ingest_patient_education():
    """Add patient education materials."""
    print(f"\n=== Ingesting Patient Education Materials ===")

    education_materials = [
        {
            "topic": "Chest Pain - When to Seek Emergency Care",
            "reading_level": "8th_grade",
            "content": """
WHEN TO CALL 911 FOR CHEST PAIN

Call 911 immediately if you experience:
- Severe pressure, squeezing, or pain in your chest
- Pain that spreads to your jaw, left arm, or back
- Chest discomfort with shortness of breath, sweating, or nausea
- Sudden severe chest pain (could be heart attack or other emergency)

DO NOT DRIVE YOURSELF - Call an ambulance so treatment can begin on the way to the hospital.

WHAT TO EXPECT:
- Paramedics will give you oxygen and take an EKG
- You'll receive aspirin (unless you're allergic)
- The hospital will do blood tests and imaging
- Treatment depends on the cause of your chest pain

FOLLOW-UP CARE:
- Take all prescribed medications as directed
- Attend all follow-up appointments with your doctor
- Learn your risk factors: high blood pressure, diabetes, high cholesterol
- Lifestyle changes: quit smoking, exercise, healthy diet
"""
        },
        {
            "topic": "Taking Multiple Medications Safely",
            "reading_level": "6th_grade",
            "content": """
HOW TO TAKE MULTIPLE MEDICATIONS SAFELY

KEEP A MEDICATION LIST:
Write down all your medicines, including:
- Prescription medications
- Over-the-counter drugs (like Tylenol or Advil)
- Vitamins and supplements
- Herbal products

For each medicine, write:
- Name of the medicine
- Why you take it
- How much to take (dose)
- When to take it
- What it looks like

TELL YOUR DOCTORS:
- Show your medication list to every doctor and pharmacist
- Tell them about allergies or side effects
- Ask if new medicines interact with what you already take

TIPS FOR STAYING ORGANIZED:
- Use a pill organizer box
- Set phone alarms as reminders
- Take medicines at the same time each day
- Keep medicines in their original bottles
- Store them in a cool, dry place (not the bathroom)

NEVER:
- Share your medications with others
- Take someone else's medication
- Stop taking medicine without asking your doctor
- Take expired medications
"""
        },
        {
            "topic": "When to Go to the Emergency Room vs Urgent Care",
            "reading_level": "8th_grade",
            "content": """
EMERGENCY ROOM VS URGENT CARE: MAKING THE RIGHT CHOICE

GO TO THE EMERGENCY ROOM (CALL 911) FOR:
- Chest pain or pressure
- Difficulty breathing or shortness of breath
- Sudden severe headache
- Loss of consciousness or fainting
- Severe bleeding that won't stop
- Broken bones or severe injuries
- Signs of stroke: Face drooping, arm weakness, speech difficulty
- Poisoning or drug overdose
- Severe allergic reactions

GO TO URGENT CARE FOR:
- Minor cuts that need stitches
- Sprains and strains
- Mild fever or flu symptoms
- Ear infections or sore throat
- Urinary tract infections
- Minor burns
- Rashes or minor allergic reactions
- X-rays for possible broken bones (non-severe)

CALL YOUR DOCTOR'S OFFICE FOR:
- Medication refills
- Routine follow-up questions
- Non-urgent health concerns
- Scheduling annual check-ups

REMEMBER: When in doubt, it's better to be safe. If you're unsure, call 911 or go to the emergency room.
"""
        }
    ]

    documents = []
    for material in education_materials:
        chunks = rag_manager.chunk_text(
            material["content"],
            chunk_size=600,
            chunk_overlap=100,
            metadata={
                "topic": material["topic"],
                "reading_level": material["reading_level"],
                "source": "patient_education",
                "type": "consumer_health"
            }
        )
        documents.extend(chunks)

    if documents:
        rag_manager.add_documents("patient_education", documents)
        print(f"✓ Added {len(documents)} patient education chunks")


def main():
    """Main ingestion workflow."""
    print("\n" + "=" * 70)
    print("MedFlow AI - Medical Knowledge Ingestion")
    print("=" * 70)

    # Ingest all data sources
    ingest_drug_interactions(limit=50)
    ingest_clinical_guidelines()
    ingest_patient_education()

    # Print summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Drug Interactions: {rag_manager.get_collection_count('drug_interactions')} documents")
    print(f"Clinical Guidelines: {rag_manager.get_collection_count('clinical_guidelines')} documents")
    print(f"Patient Education: {rag_manager.get_collection_count('patient_education')} documents")
    print("\n✓ Data ingestion complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
