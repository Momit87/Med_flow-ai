"""
MedFlow AI - Streamlit Application
Multi-Agent Clinical Encounter Orchestrator
"""

import streamlit as st
import uuid
from datetime import datetime
from src.graph import app as encounter_app, stream_encounter
from src.config import settings
from src.input_filter import is_medical_query, get_non_medical_response
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="MedFlow AI - Clinical Encounter Orchestrator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e40af;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .disclaimer {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .esi-1 { background-color: #fee2e2; border-left: 4px solid #dc2626; padding: 0.5rem; }
    .esi-2 { background-color: #fed7aa; border-left: 4px solid #ea580c; padding: 0.5rem; }
    .esi-3 { background-color: #fef08a; border-left: 4px solid #ca8a04; padding: 0.5rem; }
    .esi-4 { background-color: #d9f99d; border-left: 4px solid #65a30d; padding: 0.5rem; }
    .esi-5 { background-color: #dbeafe; border-left: 4px solid #2563eb; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "patient_id" not in st.session_state:
        st.session_state.patient_id = f"DEMO-{str(uuid.uuid4())[:8]}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "encounter_started" not in st.session_state:
        st.session_state.encounter_started = False

    if "encounter_state" not in st.session_state:
        st.session_state.encounter_state = {}

    if "assistant_type" not in st.session_state:
        st.session_state.assistant_type = "ed"


init_session_state()


# ============================================================================
# Sidebar - Configuration & Status
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Assistant selector
    assistant_type = st.selectbox(
        "Care Setting",
        options=["ed", "primary_care", "pediatrics"],
        format_func=lambda x: settings.ASSISTANTS[x]["name"],
        key="assistant_selector"
    )

    if assistant_type != st.session_state.assistant_type:
        st.session_state.assistant_type = assistant_type
        st.rerun()

    st.divider()

    # Session Info
    st.markdown("### 📋 Session Info")
    st.text(f"Patient ID: {st.session_state.patient_id}")
    st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")

    # New encounter button
    if st.button("🔄 New Encounter", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.patient_id = f"DEMO-{str(uuid.uuid4())[:8]}"
        st.session_state.messages = []
        st.session_state.encounter_state = {}
        st.session_state.encounter_started = False
        st.rerun()

    st.divider()

    # Current state display
    if st.session_state.encounter_state:
        st.markdown("### 📊 Encounter Status")

        state = st.session_state.encounter_state

        # Triage level
        if "triage_level" in state:
            triage_level = state["triage_level"]
            esi_class = triage_level.lower().replace("-", "-")
            st.markdown(f'<div class="{esi_class}"><strong>Triage:</strong> {triage_level}</div>',
                       unsafe_allow_html=True)

        # Current agent
        if "current_agent" in state:
            st.info(f"**Current Agent:** {state['current_agent']}")

        # Status
        if "status" in state:
            st.success(f"**Status:** {state['status']}")


# ============================================================================
# Main Content Area
# ============================================================================

# Header
st.markdown('<p class="main-header">🏥 MedFlow AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Agent Clinical Encounter Orchestrator</p>',
            unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>DEMONSTRATION SYSTEM ONLY</strong><br>
    This is an educational demonstration of AI agent architecture. It is NOT intended for actual clinical use.
    Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Chat Interface
# ============================================================================

st.markdown("### 💬 Patient Intake Interview")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Welcome message if first interaction
if not st.session_state.encounter_started:
    with st.chat_message("assistant"):
        welcome_msg = f"""Hello! I'm your {settings.ASSISTANTS[st.session_state.assistant_type]['name']}.

I'll help gather information about your visit today. Please remember:
- This is a demonstration system for educational purposes
- For actual emergencies, call 911 or go to your nearest emergency room
- This system cannot diagnose or provide medical treatment

To begin, please tell me: **What brings you in today?**"""
        st.markdown(welcome_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
        st.session_state.encounter_started = True

# Chat input
if prompt := st.chat_input("Describe your symptoms..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if query is medical-related before invoking agents
    filter_result = is_medical_query(prompt)

    # Process with agent system
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_placeholder = st.empty()

        # If not medical, return polite message without calling agents
        if not filter_result["is_medical"]:
            full_response = get_non_medical_response()
            response_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

            # Show filter info in sidebar (optional debug info)
            with st.sidebar:
                with st.expander("🔍 Input Filter Debug", expanded=False):
                    st.write(f"**Medical Query:** {filter_result['is_medical']}")
                    st.write(f"**Confidence:** {filter_result['confidence']:.2f}")
                    st.write(f"**Reason:** {filter_result['reason']}")

        else:
            # Medical query - proceed with full agent workflow
            try:
                # Show processing status
                with status_placeholder.status("🤔 Processing...", expanded=True) as status:
                    st.write("Analyzing your input...")

                    # Show filter info
                    if filter_result['matched_keywords']:
                        st.write(f"Detected medical keywords: {', '.join(filter_result['matched_keywords'][:3])}")

                    # Stream response
                    full_response = ""
                    latest_state = {}

                    for event in stream_encounter(
                        patient_id=st.session_state.patient_id,
                        thread_id=st.session_state.thread_id,
                        user_message=prompt
                    ):
                        # Update state tracking
                        for node_name, node_state in event.items():
                            st.write(f"✓ {node_name} complete")
                            latest_state.update(node_state)

                            # Extract assistant message if present
                            if "messages" in node_state:
                                for msg in node_state["messages"]:
                                    if isinstance(msg, AIMessage):
                                        full_response = msg.content

                    # Update stored state
                    st.session_state.encounter_state = latest_state

                    status.update(label="✅ Complete!", state="complete")

                # Clear status and show response
                status_placeholder.empty()
                response_placeholder.markdown(full_response)

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

                # Force rerun to update sidebar
                st.rerun()

            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                st.exception(e)


# ============================================================================
# Encounter Summary (if triage complete)
# ============================================================================

if st.session_state.encounter_state.get("triage_level"):
    st.divider()
    st.markdown("### 📋 Encounter Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Triage Level", st.session_state.encounter_state["triage_level"])

    with col2:
        chief = st.session_state.encounter_state.get("chief_complaint", "N/A")
        st.metric("Chief Complaint", chief if chief else "N/A")

    with col3:
        status = st.session_state.encounter_state.get("status", "N/A")
        st.metric("Status", status.title())

    # Triage reasoning
    if "triage_reasoning" in st.session_state.encounter_state:
        with st.expander("📊 Triage Assessment", expanded=True):
            st.write(st.session_state.encounter_state["triage_reasoning"])

    # Clinical data
    if st.session_state.encounter_state.get("chief_complaint"):
        with st.expander("🔍 Clinical Data Extracted"):
            state = st.session_state.encounter_state

            if "chief_complaint" in state and state["chief_complaint"]:
                st.write(f"**Chief Complaint:** {state['chief_complaint']}")

            if "symptoms" in state and state["symptoms"]:
                st.write(f"**Symptoms:** {', '.join(state['symptoms'])}")

            if "medications" in state and state["medications"]:
                st.write(f"**Medications:** {', '.join([m.name for m in state['medications']])}")

            if "allergies" in state and state["allergies"]:
                st.write(f"**Allergies:** {', '.join(state['allergies'])}")


# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem;">
    <p>MedFlow AI v1.0 | Portfolio Demonstration Project</p>
    <p>Powered by Google Gemini • LangChain • LangGraph • ChromaDB</p>
    <p>🚫 NOT FOR CLINICAL USE | Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)
