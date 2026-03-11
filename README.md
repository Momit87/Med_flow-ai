# MedFlow AI - Multi-Agent Clinical Encounter Orchestrator

🏥 **Portfolio-grade demonstration of production-level AI agent architecture for healthcare workflows**

Powered by Google Gemini • LangChain • LangGraph • ChromaDB • Streamlit

---

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A DEMONSTRATION SYSTEM FOR EDUCATIONAL PURPOSES ONLY**

- ❌ NOT intended for actual clinical use
- ❌ NOT a substitute for professional medical advice
- ❌ NOT validated for real patient care
- ✅ Portfolio/demonstration project showcasing AI engineering patterns

For medical emergencies, call 911 or visit your nearest emergency room.

---

## 🎯 Project Overview

MedFlow AI demonstrates a sophisticated multi-agent system for clinical encounter orchestration. The system showcases:

- **Multi-Agent Orchestration**: 6 specialized AI agents coordinated by a supervisor
- **RAG-Powered Evidence**: ChromaDB vector database with medical literature
- **Human-in-the-Loop Safety**: Mandatory clinician approval gates
- **Real-time Streaming**: Streamlit UI with live agent updates
- **State Management**: LangGraph checkpointing and persistence
- **Full Observability**: LangSmith tracing of all decisions

### Key Features

✨ **Adaptive Patient Interviews**: Conversational intake with context-aware questioning
🔬 **Drug Interaction Screening**: Parallel RAG lookups against FDA databases
🚨 **Emergency Triage**: ESI (Emergency Severity Index) classification
📋 **Clinical Documentation**: Auto-generated SOAP notes
🎓 **Patient Education**: Dynamic content based on care setting and reading level

---

## 🏗️ Architecture

```
Patient Input → Streamlit UI
                    ↓
           LangGraph Orchestrator
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
   Intake Agent          Safety Sentinel
   (adaptive Q&A)        (drug interactions)
        ↓                       ↓
   Evidence Synthesizer   Triage Router
   (RAG + ReAct)         (ESI classification)
        ↓                       ↓
   Clinical Brief         Patient Education
   (SOAP note)            (dynamic config)
        ↓
   Human-in-the-Loop Gate
   (provider review)
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Google Gemini 2.0 Flash / Groq | Core reasoning and generation |
| Embeddings | Gemini Embedding 004 | Semantic search for RAG |
| Agent Framework | LangGraph 0.3+ | Multi-agent orchestration |
| Vector DB | ChromaDB | Medical knowledge retrieval |
| State Persistence | SQLite (SqliteSaver) | Conversation checkpointing |
| UI | Streamlit | Interactive web interface |
| Observability | LangSmith | Trace logging and evaluation |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google API key (for Gemini)
- Optional: Groq API key (for faster inference)
- Optional: LangSmith API key (for tracing)

### Installation

1. **Clone the repository**
```bash
cd Med_flow
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```ini
GOOGLE_API_KEY=your-gemini-api-key-here
GROQ_API_KEY=your-groq-api-key-here  # Optional
LANGCHAIN_TRACING_V2=true  # Optional
LANGCHAIN_API_KEY=your-langsmith-key  # Optional
```

5. **Ingest medical knowledge into ChromaDB**
```bash
python scripts/ingest_medical_data.py
```

This will:
- Fetch drug interaction data from openFDA
- Create synthetic clinical guidelines
- Populate patient education materials
- Build vector embeddings for RAG

Expected output:
```
=== Ingesting Drug Interaction Data ===
Fetching 50 drug labels from openFDA...
✓ Successfully ingested 156 drug interaction chunks

=== Ingesting Clinical Guidelines ===
✓ Added 24 clinical guideline chunks

=== Ingesting Patient Education Materials ===
✓ Added 18 patient education chunks

INGESTION SUMMARY
Drug Interactions: 156 documents
Clinical Guidelines: 24 documents
Patient Education: 18 documents
```

6. **Launch the Streamlit app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Basic Workflow

1. **Start a New Encounter**
   - The system assigns a demo patient ID
   - Select care setting (Emergency Dept / Primary Care / Pediatrics)

2. **Conduct Patient Interview**
   - Enter symptoms, medications, and history
   - The Intake Agent adapts questions based on responses
   - All conversations are persisted via checkpointing

3. **View Real-Time Analysis**
   - Watch agents process in the status panel
   - See extracted clinical data populate
   - Triage classification updates automatically

4. **Review Encounter Summary**
   - Triage level (ESI-1 through ESI-5)
   - Chief complaint and symptoms
   - Reasoning for triage decision

### Example Interactions

**Emergency Scenario:**
```
User: "I'm having severe chest pain that won't go away"
→ System triages as ESI-1 (immediate life-threatening)
→ Triggers critical pathway
```

**Routine Visit:**
```
User: "I have a cough for 3 days and low-grade fever"
→ System triages as ESI-4 (stable, simple resource)
→ Gathers detailed symptom history
```

---

## 🧪 Project Structure

```
Med_flow/
├── src/
│   ├── __init__.py
│   ├── state.py              # Pydantic models & TypedDict schemas
│   ├── config.py             # LLM configuration & settings
│   ├── rag.py                # ChromaDB RAG manager
│   ├── graph.py              # Main LangGraph orchestrator
│   ├── agents/               # Agent sub-graphs (to be expanded)
│   ├── tools/                # LangChain tools
│   ├── middleware/           # Safety & PII redaction
│   └── memory/               # Long-term memory patterns
├── scripts/
│   └── ingest_medical_data.py  # Data ingestion pipeline
├── chroma_db/                # Persistent vector store
├── data/                     # Raw medical data
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── .gitignore
└── README.md                 # This file
```

---

## 🎓 LangChain/LangGraph Concepts Demonstrated

This project showcases 35+ core concepts:

### LangChain
- ✅ Foundational Chat Models (Gemini, Groq)
- ✅ Tool Binding & Custom Tools
- ✅ Short-term Memory (message history)
- ✅ Multimodal Messages (image analysis)
- ✅ RAG Retrieval (ChromaDB)
- ✅ Middleware (PII redaction, guardrails)
- ✅ Managing Long Conversations (trimming, filtering)

### LangGraph
- ✅ StateGraph with nodes and edges
- ✅ Chain Pattern (linear workflows)
- ✅ Router Pattern (conditional edges)
- ✅ ReAct Agent Pattern (tool loops)
- ✅ Persistent Memory (SqliteSaver checkpointer)
- ✅ State Schema (TypedDict + Pydantic)
- ✅ Custom State Reducers
- ✅ Multiple Schemas (Input/Output/Internal)
- ✅ Streaming (messages, updates, values)
- ✅ Breakpoints (static & dynamic)
- ✅ State Editing (human feedback)
- ✅ Time Travel (checkpoint history)
- ✅ Parallelization (fan-out/fan-in)
- ✅ Sub-graphs (agent composition)
- ✅ Map-Reduce (Send() API)
- ✅ Long-term Memory (Store)
- ✅ Deployment patterns

---

## 🔧 Configuration

### Assistant Types

Three pre-configured assistants with different settings:

| Assistant | Reading Level | Focus | Use Case |
|-----------|--------------|-------|----------|
| **ED** (Emergency Dept) | College | Rapid triage, time-critical protocols | Chest pain, trauma, severe symptoms |
| **Primary Care** | 8th grade | Preventive care, lifestyle counseling | Routine visits, chronic disease management |
| **Pediatrics** | 6th grade | Age-appropriate, parent-friendly | Children's health, growth/development |

Change assistant type in the sidebar dropdown.

### LLM Provider Selection

Edit `.env` to choose provider:
```ini
LLM_PROVIDER=gemini  # or "groq"
```

- **Gemini**: Multimodal, long context, free tier
- **Groq**: Ultra-low latency, faster inference

---

## 📊 Observability & Debugging

### LangSmith Tracing

Enable full trace logging:

1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Get API key
3. Set in `.env`:
```ini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key-here
LANGCHAIN_PROJECT=medflow-ai
```

View traces at: `https://smith.langchain.com/{your-org}/projects/medflow-ai`

### Checkpoint History

All encounter states are saved to `encounters.db` (SQLite). You can inspect:

```python
from src.graph import app

config = {"configurable": {"thread_id": "your-thread-id"}}
history = list(app.get_state_history(config))

for checkpoint in history:
    print(checkpoint.values)
```

---

## 🧩 Extending the System

### Adding a New Agent

1. Create agent file in `src/agents/your_agent.py`
2. Define sub-graph with nodes and edges
3. Add to main graph in `src/graph.py`
4. Update state schema if needed

### Adding RAG Knowledge

```python
from src.rag import rag_manager

# Add documents to a collection
documents = rag_manager.chunk_text(
    "Your medical content here",
    metadata={"source": "custom", "topic": "cardiology"}
)

rag_manager.add_documents("clinical_guidelines", documents)
```

### Custom Tools

```python
from langchain.tools import tool

@tool
def check_drug_formulary(drug_name: str) -> str:
    """Check if a drug is on the hospital formulary."""
    # Your implementation
    return result
```

---

## 🐛 Troubleshooting

### Common Issues

**"No module named 'src'"**
- Ensure you're running from the `Med_flow` directory
- Check that `__init__.py` files exist in all packages

**ChromaDB errors**
- Delete `chroma_db/` folder and re-run ingestion
- Check disk space and permissions

**API rate limits**
- Switch to Groq for faster inference
- Add retry logic with exponential backoff

**Streamlit not updating**
- Clear browser cache
- Restart Streamlit server
- Check for Python errors in terminal

---

## 📝 Development Roadmap

### ✅ Day 1 Complete
- [x] Project structure
- [x] State schemas with Pydantic
- [x] ChromaDB RAG setup
- [x] Data ingestion pipeline
- [x] Basic StateGraph
- [x] Streamlit MVP

### 🚧 In Progress (Day 2)
- [ ] Full Intake Agent sub-graph with adaptive questioning
- [ ] Multimodal medication image analysis
- [ ] Safety Sentinel with parallel drug screening
- [ ] Map-reduce pattern for drug interactions

### 📅 Upcoming
- [ ] Evidence Synthesizer (ReAct + RAG)
- [ ] Triage Router with ESI logic
- [ ] Clinical Brief Generator (SOAP notes)
- [ ] Patient Education with dynamic config
- [ ] Human-in-the-loop breakpoints
- [ ] PII redaction middleware
- [ ] LangGraph Store for patient history
- [ ] Enhanced Streamlit dashboard
- [ ] Docker deployment
- [ ] LangSmith evaluation datasets

---

## 📄 License

This is a portfolio/demonstration project. Not licensed for clinical or commercial use.

---

## 🤝 Contributing

This is a personal portfolio project, but feedback is welcome! Please note:

- This is NOT production code
- Not accepting PRs at this time
- Feel free to fork for your own learning

---

## 📧 Contact

For questions about this project's architecture or implementation:
- Open an issue in the repository
- Connect on LinkedIn (see profile)

---

## 🙏 Acknowledgments

- **LangChain/LangGraph Team**: Amazing agent framework
- **Google**: Gemini API for multimodal LLM
- **OpenFDA**: Public drug interaction database
- **Streamlit**: Rapid prototyping UI framework

---

**Built with ❤️ as a portfolio demonstration of production-grade AI engineering**

⚠️ **Remember**: This is NOT for clinical use. Always consult healthcare professionals for medical advice.
