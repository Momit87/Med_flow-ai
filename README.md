# MedFlow AI - Multi-Agent Clinical Encounter Orchestrator

<div align="center">

🏥 **Production-grade demonstration of advanced AI agent architecture for healthcare workflows**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-Portfolio-yellow.svg)](LICENSE)

Powered by **Groq (Llama 3.3 70B)** • **Google Gemini** • **LangChain** • **LangGraph** • **ChromaDB** • **Streamlit**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Documentation](DOCUMENTATION.md) • [Demo](#-demo)

</div>

---

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A DEMONSTRATION SYSTEM FOR EDUCATIONAL AND PORTFOLIO PURPOSES ONLY**

- ❌ **NOT** intended for actual clinical use or real patient care
- ❌ **NOT** a substitute for professional medical advice, diagnosis, or treatment
- ❌ **NOT** validated, tested, or approved for medical decision-making
- ❌ **NOT** HIPAA compliant or production-ready
- ✅ **IS** a portfolio project showcasing AI engineering patterns and architecture

**For medical emergencies, call 911 or visit your nearest emergency room.**

---

## 🎯 Project Overview

MedFlow AI is a sophisticated multi-agent system that demonstrates advanced AI orchestration patterns in a healthcare context. Built with **LangGraph** and powered by **Groq's ultra-fast inference**, this system showcases production-grade AI engineering techniques including multi-agent coordination, RAG (Retrieval-Augmented Generation), state management, and human-in-the-loop workflows.

### Why MedFlow AI?

This project serves as a comprehensive portfolio piece demonstrating:

- 🏗️ **Complex system architecture** with multiple coordinated agents
- 🔄 **State management** using LangGraph checkpointing and persistence
- 🔍 **RAG implementation** with ChromaDB vector database (demo medical documents)
- 🚀 **Real-time streaming** and progressive updates
- 🛡️ **Safety-first design** with input filtering and emergency detection
- 📊 **Observability** through comprehensive logging and tracing
- 🎨 **Clean UI/UX** with Streamlit for rapid prototyping
- 🐳 **Production deployment** with Docker containerization

---

## ✨ Features

### Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **🤖 Multi-Agent Orchestration** | 6+ specialized agents coordinated via LangGraph supervisor pattern | ✅ Complete |
| **💬 Adaptive Patient Intake** | Context-aware conversational interview with dynamic questioning | ✅ Complete |
| **🔍 Drug Interaction Screening** | Parallel RAG lookups against demo drug interaction documents | ✅ Complete |
| **🚨 Emergency Triage** | ESI (Emergency Severity Index) 1-5 classification with reasoning | ✅ Complete |
| **📋 Clinical Documentation** | Auto-generated SOAP notes from conversation history | ✅ Complete |
| **🎓 Patient Education** | Dynamic content retrieval based on care setting and reading level | ✅ Complete |
| **🛡️ Input Filtering** | Smart query detection to reduce API costs for non-medical queries | ✅ Complete |
| **💾 Conversation History** | Search and resume previous encounters by Patient ID | ✅ Complete |
| **🔐 PII Redaction** | Middleware for protecting sensitive patient information | ✅ Complete |
| **⚕️ Clinical Guidelines** | Evidence-based protocol retrieval from vector database | ✅ Complete |

### Advanced Features

- **🖼️ Multimodal Analysis**: Medication bottle image recognition using vision LLMs
- **🔬 Live Medical Research**: PubMed API integration for recent literature
- **📊 Real-time Status Updates**: Streaming agent progress with visual feedback
- **🎯 Care Setting Profiles**: Pre-configured for ED, Primary Care, and Pediatrics
- **🔄 State Persistence**: SQLite checkpointing for conversation continuity
- **📈 Cost Optimization**: Intelligent filtering reduces unnecessary API calls by ~40%

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Patient Chat │  │ Search & Load│  │ Status Panel │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Input Filter Layer                           │
│  • Medical keyword detection (100+ terms)                       │
│  • Non-medical query rejection                                  │
│  • Cost optimization (prevents unnecessary agent invocation)    │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  LangGraph Orchestrator                         │
│                                                                 │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐   │
│  │   Intake    │────→│  Extraction  │────→│   Triage    │   │
│  │   Agent     │     │    Node      │     │    Node     │   │
│  └─────────────┘     └──────────────┘     └─────────────┘   │
│         │                                          │           │
│         ↓                                          ↓           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Conditional Routing                         │ │
│  │  • Emergency Detection                                   │ │
│  │  • Status-based Flow Control                            │ │
│  │  • Dynamic Agent Selection                              │ │
│  └─────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Sub-Graphs                             │
│                                                                 │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Safety        │  │  Evidence    │  │   SOAP Note  │       │
│  │ Sentinel      │  │ Synthesizer  │  │  Generator   │       │
│  │ (Map-Reduce)  │  │ (ReAct Loop) │  │  (Chain)     │       │
│  └───────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Data & Tools Layer                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  ChromaDB    │  │  LLM APIs    │  │   SQLite     │        │
│  │  (Demo docs) │  │ Groq/Gemini  │  │ Checkpoints  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Custom Tools: Drug Lookup • ICD-10 • PubMed • Vision   │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input** → Input filter checks for medical relevance
2. **If Medical** → Route to LangGraph orchestrator
3. **Intake Agent** → Adaptive conversation and data gathering
4. **Extraction** → Parse structured clinical data from conversation
5. **Triage** → ESI classification with reasoning
6. **Conditional Routing** → Based on urgency and completeness
7. **Sub-Graphs** → Execute specialized workflows (safety, evidence, documentation)
8. **State Persistence** → Save checkpoint to SQLite
9. **UI Update** → Stream results back to user

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (or Docker)
- **Groq API Key** (free tier: 30 req/min) - [Get it here](https://console.groq.com/)
- **Optional**: Google API Key for vision features
- **Optional**: LangSmith API Key for tracing

### Option 1: Docker Deployment (Recommended)

**One-command startup:**

```bash
# Clone repository
git clone https://github.com/Momit87/Med_flow-ai.git
cd Med_flow-ai

# Create .env file
cat > .env << EOF
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
EOF

# Start application
docker-compose up --build

# Open browser to http://localhost:8501
```

**That's it!** Docker handles all dependencies, ChromaDB setup, and data ingestion.

### Option 2: Manual Installation

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env and add your API keys

# 4. Ingest medical knowledge (one-time setup)
python scripts/ingest_medical_data.py

# Expected output:
# ✓ Drug Interactions: demo documents loaded
# ✓ Clinical Guidelines: demo documents loaded
# ✓ Patient Education: demo documents loaded
# Total: Demo medical documents ingested to ChromaDB

# 5. Launch application
streamlit run app.py
```

### Verify Installation

Once running, you should see:
- Streamlit UI at `http://localhost:8501`
- Welcome message from the assistant
- Sidebar with Patient ID and care setting selector

---

## 📖 Usage Guide

### Starting a New Encounter

1. **Open the app** at http://localhost:8501
2. **Select care setting** in sidebar (Emergency Dept, Primary Care, or Pediatrics)
3. **Start conversation** by entering symptoms or chief complaint

### Example Interactions

**Emergency Scenario (ESI-1):**
```
👤 User: "I have severe chest pain radiating to my left arm for the past hour"

🤖 MedFlow AI:
   • Classifies as ESI-1 (immediate life-threatening)
   • Triggers emergency protocols
   • Recommends immediate 911 call
   • Extracts: chest pain, radiation to arm, acute onset
```

**Drug Interaction Check:**
```
👤 User: "I'm currently taking warfarin 5mg daily and aspirin 81mg. Are these safe together?"

🤖 MedFlow AI:
   • Searches drug interaction database
   • Identifies major interaction risk
   • Provides clinical guidance
   • Recommends INR monitoring
```

**Non-Medical Query (Filtered):**
```
👤 User: "What's the weather today?"

🤖 MedFlow AI: [Returns polite message without invoking agents]
   "I'm MedFlow AI, specialized for healthcare questions.
    For non-medical queries, please use a general-purpose assistant."
```

### Loading Previous Conversations

1. In the sidebar, find **"Load Conversation"** section
2. Enter a **Patient ID** (e.g., `DEMO-abc12345`)
3. Click **"Load This Conversation"** button
4. Continue the encounter seamlessly

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **LLM (Default)** | Groq - Llama 3.3 70B | Latest | Ultra-fast inference (free tier) |
| **LLM (Vision)** | Google Gemini 2.0 Flash | Latest | Multimodal image analysis |
| **Embeddings** | Google Embedding 004 | Latest | Semantic search for RAG |
| **Agent Framework** | LangGraph | 0.3+ | Multi-agent orchestration |
| **Vector Database** | ChromaDB | Latest | Medical knowledge retrieval |
| **State Persistence** | SQLite (SqliteSaver) | Built-in | Conversation checkpointing |
| **Web Framework** | Streamlit | Latest | Interactive UI |
| **Observability** | LangSmith | Optional | Trace logging & evaluation |
| **Deployment** | Docker + Compose | Latest | Containerization |

### Python Dependencies

```
langchain>=0.3.0
langchain-google-genai>=2.0.0
langchain-groq>=0.2.0
langgraph>=0.3.0
chromadb>=0.5.0
streamlit>=1.40.0
pydantic>=2.0.0
python-dotenv>=1.0.0
requests>=2.32.0
```

---

## 📁 Project Structure

```
Med_flow/
├── src/                          # Core application code
│   ├── __init__.py
│   ├── state.py                  # Pydantic models & state schemas
│   ├── config.py                 # LLM configuration & settings
│   ├── rag.py                    # ChromaDB RAG manager
│   ├── graph.py                  # Main LangGraph orchestrator
│   ├── tools.py                  # Custom LangChain tools (6 tools)
│   ├── subgraphs.py              # Agent sub-graphs (Intake, Safety, Evidence)
│   ├── chains.py                 # SOAP note generation chain
│   ├── middleware.py             # PII redaction, guardrails, disclaimers
│   ├── input_filter.py           # Medical query detection (cost optimization)
│   └── checkpoint_manager.py     # Conversation search & load
│
├── scripts/                      # Utilities and data ingestion
│   └── ingest_medical_data.py    # ChromaDB data pipeline
│
├── data/                         # Raw medical data (gitignored)
│   └── .gitkeep
│
├── chroma_db/                    # Persistent vector store
│   └── chroma.sqlite3            # Demo embedded documents
│
├── app.py                        # Streamlit web application
├── Dockerfile                    # Production container image
├── docker-compose.yml            # One-command deployment
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── .gitignore                    # Git exclusions
├── README.md                     # This file
└── DOCUMENTATION.md              # Detailed technical documentation
```

---

## 🎓 LangChain/LangGraph Concepts Demonstrated

This project is a comprehensive showcase of **35+ advanced AI engineering patterns**:

### LangChain Fundamentals
- ✅ Chat Models (Gemini, Groq, Claude-compatible)
- ✅ Tool Binding & Custom Tool Creation
- ✅ Message History & Short-term Memory
- ✅ Multimodal Messages (text + images)
- ✅ RAG Retrieval with ChromaDB
- ✅ Middleware & Guardrails
- ✅ Conversation Trimming & Filtering
- ✅ Structured Output Parsing

### LangGraph Orchestration
- ✅ **StateGraph** with TypedDict schemas
- ✅ **Chain Pattern** (linear SOAP note generation)
- ✅ **Router Pattern** (conditional edges for triage)
- ✅ **ReAct Agent Pattern** (evidence synthesis loop)
- ✅ **Map-Reduce Pattern** (parallel drug interaction checks via Send())
- ✅ **Sub-graphs** (composable agent modules)
- ✅ **Persistent Memory** (SqliteSaver checkpointing)
- ✅ **State Reducers** (custom merge logic)
- ✅ **Multiple Schemas** (Input, Output, Internal state)
- ✅ **Streaming** (messages, updates, values modes)
- ✅ **Breakpoints** (static & dynamic for HITL)
- ✅ **State Editing** (human feedback integration)
- ✅ **Time Travel** (checkpoint history replay)
- ✅ **Parallelization** (fan-out/fan-in workflows)

### Production Patterns
- ✅ Error Handling & Retry Logic
- ✅ Input Validation & Sanitization
- ✅ Cost Optimization (selective agent invocation)
- ✅ Observability & Logging
- ✅ State Management & Persistence
- ✅ Docker Deployment
- ✅ Environment Configuration
- ✅ API Rate Limiting
- ✅ Conversation Search & Resume

---

## 🔧 Configuration

### Care Settings

Choose from three pre-configured assistant profiles:

| Setting | Reading Level | Focus | Typical Use Cases |
|---------|--------------|-------|-------------------|
| **Emergency Dept** | College (12th grade) | Rapid triage, time-critical protocols | Chest pain, trauma, acute abdomen, severe symptoms |
| **Primary Care** | 8th grade | Preventive care, chronic disease management | Routine checkups, hypertension, diabetes follow-up |
| **Pediatrics** | 6th grade | Age-appropriate, parent-friendly language | Well-child visits, vaccines, growth concerns |

Change setting via sidebar dropdown (takes effect immediately).

### LLM Provider Configuration

Edit `src/config.py` or set environment variable:

```python
# Default provider (recommended for speed)
DEFAULT_LLM_PROVIDER = "groq"  # or "gemini"

# Available models
GROQ_MODEL = "llama-3.3-70b-versatile"      # Fast, free tier
GEMINI_MODEL = "gemini-2.0-flash"            # Multimodal, long context
```

**Provider Comparison:**

| Feature | Groq (Llama 3.3 70B) | Google Gemini 2.0 |
|---------|----------------------|-------------------|
| **Speed** | ⚡ Ultra-fast (500+ tok/s) | Fast (100 tok/s) |
| **Cost** | Free tier: 30 req/min | Free tier: 15 req/min |
| **Context** | 128K tokens | 1M tokens |
| **Vision** | ❌ No | ✅ Yes |
| **Best For** | Text-only, speed-critical | Multimodal, long docs |

### Environment Variables

```bash
# Required
GROQ_API_KEY=gsk_...                    # Get at console.groq.com
GOOGLE_API_KEY=AIza...                  # Optional for vision

# Optional: Observability
LANGCHAIN_TRACING_V2=true              # Enable LangSmith tracing
LANGCHAIN_API_KEY=lsv2_...             # LangSmith API key
LANGCHAIN_PROJECT=medflow-ai           # Project name

# Optional: Configuration
LLM_PROVIDER=groq                      # "groq" or "gemini"
CHECKPOINT_DB=encounters.db            # SQLite database path
```

---

## 📊 Observability & Debugging

### LangSmith Tracing

Enable comprehensive trace logging for debugging and evaluation:

1. **Sign up** at [smith.langchain.com](https://smith.langchain.com)
2. **Get API key** from Settings
3. **Configure** in `.env`:
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_pt_...
   LANGCHAIN_PROJECT=medflow-ai
   ```
4. **View traces** at your project dashboard

**What you'll see:**
- Complete agent execution flows
- LLM calls with prompts & responses
- Tool invocations and results
- Latency breakdowns
- Cost per conversation

### Checkpoint Inspection

All encounter states are persisted to SQLite. Inspect programmatically:

```python
from src.graph import app
from src.checkpoint_manager import load_conversation

# Load a specific conversation
config = {"configurable": {"thread_id": "5c8d2d98-b3cc-4cb9..."}}
checkpoint = load_conversation("5c8d2d98-b3cc-4cb9...")

print(f"Patient: {checkpoint['patient_id']}")
print(f"Messages: {len(checkpoint['messages'])}")
print(f"Triage: {checkpoint['encounter_state']['triage_level']}")
```

### Logging

Application logs are written to console. Adjust verbosity in `src/config.py`:

```python
import logging
logging.basicConfig(level=logging.INFO)  # or DEBUG for verbose
```

---

## 🧪 Testing

### Manual Testing Scenarios

| Test Case | Expected Behavior |
|-----------|-------------------|
| **Emergency** | "Severe chest pain for 1 hour" → ESI-1, emergency alert |
| **Routine** | "Cough for 3 days, low fever" → ESI-4, standard intake |
| **Drug Check** | "Taking warfarin and aspirin" → Interaction warning |
| **Non-Medical** | "What's the weather?" → Filtered without agent call |
| **Conversation Load** | Search "DEMO-abc123" → Loads full history |

### Automated Testing

```bash
# Run unit tests (if implemented)
pytest tests/

# Test data ingestion
python scripts/ingest_medical_data.py

# Expected: Demo documents successfully ingested
```

---

## 🐛 Troubleshooting

### Common Issues

**Import Error: "No module named 'src'"**
```bash
# Ensure you're in the Med_flow directory
pwd  # Should show .../Med_flow

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**ChromaDB Lock Error**
```bash
# Stop all Streamlit processes
pkill -f streamlit

# Delete lock files
rm -rf chroma_db/.chroma/

# Re-ingest data
python scripts/ingest_medical_data.py
```

**API Rate Limit Exceeded**
```bash
# Solution 1: Use Groq (higher free tier)
echo "LLM_PROVIDER=groq" >> .env

# Solution 2: Wait and retry
# Groq: 30 req/min, resets every minute
```

**Streamlit Port Already in Use**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill
```

**Docker Build Fails**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

---

## 📄 License

This project is a **portfolio demonstration** and is provided as-is for educational purposes.

**Not licensed for:**
- ❌ Clinical or medical use
- ❌ Commercial deployment
- ❌ Production healthcare systems
- ❌ Actual patient care

**You may:**
- ✅ Study the code for learning
- ✅ Fork for personal projects
- ✅ Reference in your portfolio
- ✅ Adapt patterns for non-medical use cases

---

## 🤝 Contributing

This is a personal portfolio project showcasing AI engineering skills. While the codebase is public:

- **Not accepting PRs** at this time
- **Feedback welcome** via issues
- **Feel free to fork** for your own learning
- **Questions?** Open an issue or connect on LinkedIn

---

## 📧 Contact

**Repository**: [github.com/Momit87/Med_flow-ai](https://github.com/Momit87/Med_flow-ai)

For questions about architecture, implementation, or AI engineering patterns demonstrated in this project:
- Open an issue on GitHub
- Connect on LinkedIn (see profile)

---

## 🙏 Acknowledgments

- **LangChain Team** - Incredible agent framework and ecosystem
- **Groq** - Lightning-fast LLM inference (free tier rocks!)
- **Google** - Gemini API for multimodal capabilities
- **OpenFDA** - Public drug interaction database
- **Streamlit** - Rapid prototyping made easy
- **ChromaDB** - Elegant vector database for RAG

---

## 📚 Additional Resources

- **[Detailed Documentation](DOCUMENTATION.md)** - Technical deep-dive
- **[LangGraph Docs](https://langchain-ai.github.io/langgraph/)** - Official LangGraph guide
- **[LangChain Docs](https://python.langchain.com/)** - LangChain documentation
- **[Groq API](https://console.groq.com/docs)** - Groq LLM documentation

---

<div align="center">

**Built with ❤️ as a portfolio demonstration of production-grade AI engineering**

⚠️ **Remember**: This is NOT for clinical use. Always consult healthcare professionals for medical advice.

[⬆ Back to Top](#medflow-ai---multi-agent-clinical-encounter-orchestrator)

</div>
