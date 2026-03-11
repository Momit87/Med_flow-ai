# MedFlow AI — Project Brief

---

## Project Title

**MedFlow AI** — Multi-Agent Clinical Encounter Orchestrator

---

## Problem Statement

Emergency department physicians make over 10,000 cognitive decisions per shift, yet critical drug interactions are caught late or missed entirely due to fragmented intake workflows, manual data entry, and information overload. Existing clinical intake systems lack intelligent triage, real-time safety screening, and evidence-grounded decision support — leading to delayed care, preventable adverse drug events, and provider burnout. There is no open-source, multi-agent system that orchestrates the full patient encounter lifecycle from intake through provider-approved clinical documentation with built-in safety gates.

---

## Overview

MedFlow AI is a multi-agent clinical encounter orchestrator built with LangGraph and LangChain that coordinates six specialist AI agents — Intake, Safety Sentinel, Evidence Synthesizer, Triage Router, Clinical Brief Generator, and Patient Education — to handle the full patient encounter workflow. The system uses RAG (Retrieval-Augmented Generation) over ChromaDB medical knowledge bases to ground every recommendation in verified evidence, requires mandatory human-in-the-loop clinician approval before any AI output reaches the patient, and streams the entire workflow in real-time through a Streamlit dashboard. Powered by Google Gemini (or Groq as a low-latency alternative), it demonstrates 37 LangChain/LangGraph concepts in a single cohesive healthcare application.

---

## Tech Stack

### LangChain Components Used
- **ChatGoogleGenerativeAI / ChatGroq** — Provider-agnostic foundational model abstraction with config-driven swap
- **@tool decorator + bind_tools()** — 6 custom tools (FDA drug lookup, PubMed search, ChromaDB retrievers, ICD-10 lookup, multimodal image analysis, patient profile retrieval)
- **HumanMessage with multimodal content** — Medication bottle photo upload → Gemini vision extraction
- **trim_messages** — Token-based conversation trimming for long intake interviews
- **RecursiveCharacterTextSplitter** — Document chunking for ChromaDB ingestion
- **Chroma (as_retriever)** — Similarity search + MMR retrieval over 3 medical collections

### LangGraph Components Used
- **StateGraph** — Top-level supervisor graph with 8 nodes and conditional/parallel edges
- **Sub-graphs** — Each of 6 agents is a self-contained compiled sub-graph
- **add_messages reducer** — Message accumulation across conversation turns
- **Custom state reducers** — `safety_reducer` for deduplicating parallel drug interaction alerts
- **Multiple schemas** — `InputState` / `OutputState` / `EncounterState` (internal) for clean API boundaries
- **Conditional edges (add_conditional_edges)** — Triage router: ESI-1/2 → fast track, ESI-3 → full workup, ESI-4/5 → standard
- **Send() API** — Map-reduce fan-out for parallel pairwise drug interaction checks
- **Parallel fan-out / fan-in** — Safety Sentinel + Evidence Synthesizer + Patient Education run concurrently
- **SqliteSaver checkpointer** — Persistent encounter state across browser refreshes
- **interrupt() + interrupt_before** — Static breakpoint at provider review; dynamic interrupt for critical safety alerts
- **update_state()** — Provider edits triage level, dismisses alerts, modifies care plan mid-workflow
- **get_state_history()** — Time travel for QA audit of any past decision point
- **InMemoryStore** — Long-term patient memory with profile pattern (demographics) + collection pattern (encounter history)
- **Configurable assistants** — 3 variants (ED, Primary Care, Pediatrics) sharing one graph with different runtime configs
- **stream_mode (messages/updates/values)** — Token streaming, agent activity updates, and structured data updates
- **multitask_strategy: "interrupt"** — Double-texting handling

### Other Tools / APIs
- **Google Gemini 2.0 Flash** — Primary LLM (free tier, multimodal, fast inference)
- **Groq (Llama 3.3 70B)** — Low-latency LLM alternative (~10x faster inference)
- **ChromaDB** — Persistent vector database with 3 collections: `drug_interactions`, `clinical_guidelines`, `patient_education`
- **GoogleGenerativeAIEmbeddings** — `gemini-embedding-exp-03-07` for vector embeddings
- **openFDA API** — Drug labels, adverse event reports (free, 240 req/min)
- **PubMed E-utilities API** — Medical literature search and abstract retrieval (free, 10 req/sec)
- **RxNorm API** — Drug name normalization and interaction data (free, 20 req/sec)
- **NLM Clinical Tables API** — ICD-10 code lookup (free, unlimited)
- **LangSmith** — Full observability: tracing, evaluation datasets, latency monitoring
- **Streamlit** — All-Python frontend with chat UI, dashboard, and provider review panel
- **Docker** — Containerized deployment

---

## Key Features

### 1. 🧠 Adaptive Clinical Intake with Multimodal Support
The Intake Agent conducts a conversational interview that adapts follow-up questions based on patient responses. Patients can upload photos of medication bottles — Gemini's vision model extracts drug names, dosages, and frequencies automatically. Conversation memory persists across turns via LangGraph checkpointing, and long conversations are automatically trimmed to stay within context limits.

### 2. 🛡️ Parallel Drug Safety Screening (Map-Reduce RAG)
The Safety Sentinel uses LangGraph's `Send()` API to fan out parallel interaction checks for every medication pair (6 meds = 15 simultaneous checks). Each check queries ChromaDB's `drug_interactions` collection via RAG and cross-references with the openFDA adverse events API. Results are merged via a custom `safety_reducer` that deduplicates and ranks alerts by severity. Critical interactions trigger a dynamic `interrupt()` that halts the workflow immediately for provider attention.

### 3. 📚 Evidence-Grounded Clinical Reasoning (ReAct + RAG)
The Evidence Synthesizer is a ReAct agent that iteratively searches ChromaDB's `clinical_guidelines` collection and PubMed's live API, evaluates relevance, refines queries, and synthesizes findings — up to 3 search-evaluate-refine cycles. Every clinical recommendation cites its source with retrieval scores, eliminating unsourced LLM hallucination. This grounds the entire system in peer-reviewed medical evidence rather than parametric model memory.

### 4. 👨‍⚕️ Mandatory Human-in-the-Loop Provider Review
No AI-generated content reaches the patient without explicit clinician approval. A static breakpoint pauses every encounter at the provider review stage, where the clinician can edit the triage classification, dismiss false-positive alerts, modify the SOAP note, and adjust the care plan — all via `update_state()`. Time travel via `get_state_history()` enables post-hoc QA audit of any decision point in the encounter.

### 5. 🔄 Multi-Tenant Dynamic Assistants
Three pre-configured assistant modes (Emergency Dept, Primary Care, Pediatrics) share the same underlying graph but differ in system prompts, reading levels, tool permissions, and clinical protocols. The Patient Education agent dynamically reconfigures at runtime based on care setting, language preference, and health literacy level — demonstrating production-grade multi-tenant agent architecture from a single deployment.

---

## Architecture

### Agent Flow (Nodes → Edges → State)

```
[START]
   │
   ▼
┌──────────────────┐
│  PII Redaction    │ ← Middleware (pre-processing)
│  + Guardrails     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  INTAKE AGENT    │ ← Sub-graph: greet → collect_complaint → collect_symptoms
│  (Chain Pattern)  │   → collect_medications → collect_history → summarize
│                   │   Tools: analyze_medication_image (multimodal)
│                   │   Memory: checkpointer (short-term), trim_messages
└──────┬───────────┘
       │
       ▼ (Fan-out: 3 parallel edges)
       ┌───────────────────────────────────────┐
       │                 │                      │
       ▼                 ▼                      ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
│   SAFETY    │  │  EVIDENCE    │  │ PATIENT EDUCATION│
│  SENTINEL   │  │ SYNTHESIZER  │  │    AGENT         │
│             │  │              │  │                  │
│ Map-Reduce: │  │ ReAct Loop:  │  │ Dynamic Config:  │
│ Send() per  │  │ search →     │  │ care_setting,    │
│ drug pair → │  │ evaluate →   │  │ language,        │
│ check →     │  │ refine →     │  │ reading_level    │
│ reduce      │  │ synthesize   │  │                  │
│             │  │              │  │ RAG: patient_    │
│ RAG: drug_  │  │ RAG: clinical│  │ education        │
│ interactions│  │ _guidelines  │  │ collection       │
│ + openFDA   │  │ + PubMed API │  │                  │
└──────┬──────┘  └──────┬───────┘  └────────┬─────────┘
       │                │                    │
       └────────────────┼────────────────────┘
                        │
                        ▼ (Fan-in: all 3 merge via reducers)
               ┌─────────────────┐
               │  TRIAGE ROUTER  │ ← Conditional edges:
               │                 │   ESI-1/2 → fast_track
               │  ESI 5-level    │   ESI-3   → full_workup
               │  classification │   ESI-4/5 → standard
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ BRIEF GENERATOR │ ← Chain pattern (linear):
               │                 │   collect → generate_soap
               │  SOAP Note:     │   → validate → format
               │  S/O/A/P        │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ PROVIDER REVIEW │ ← BREAKPOINT (interrupt_before)
               │                 │
               │ Human edits:    │   Dynamic interrupt() if
               │ • triage level  │   critical drug interaction
               │ • safety alerts │   detected earlier
               │ • care plan     │
               │ • SOAP note     │   update_state() → resume
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   FINALIZE      │ ← Save to LangGraph Store:
               │                 │   Profile memory (demographics)
               │  + Disclaimer   │   Collection memory (encounter log)
               └────────┬────────┘
                        │
                        ▼
                     [END]
```

### State Management

- **State Schema:** `EncounterState(TypedDict)` with 15+ typed fields (messages, medications, safety_alerts, rag_context, triage_level, clinical_brief, etc.)
- **Input/Output Boundaries:** `InputState` (just messages + patient_id) → `OutputState` (clinical_brief, triage, alerts, instructions) — internal state hidden from callers
- **Reducers:** `add_messages` for conversation, custom `safety_reducer` for parallel alert accumulation
- **Persistence:** `SqliteSaver` checkpointer for short-term (per-thread), `InMemoryStore` for long-term (cross-session patient profiles + encounter history)
- **Summarization:** Returning patients get prior encounters condensed into a 2-3 sentence system message via LLM summarization

---

## Expected Output

The final product is a **Streamlit web application** with three pages:

### Page 1: 🏥 Encounter (Main Interface)
- **Chat panel** with `st.chat_message` — token-by-token streaming responses from the Intake Agent
- **Sidebar dashboard** showing: active agent indicator with spinner, triage level metric (color-coded ESI badge), safety alerts count, assistant mode selector (ED / Primary Care / Pediatrics), LLM provider toggle (Gemini / Groq)
- **Medication photo uploader** — `st.file_uploader` with image preview and extracted medication JSON
- **Persistent disclaimer banner** — "AI-powered educational demo. Not for clinical use."

### Page 2: 👨‍⚕️ Provider Review (HITL Dashboard)
- **SOAP Note editor** — Tabbed `st.text_area` for Subjective / Objective / Assessment / Plan with pre-filled AI content
- **Safety alerts panel** — Expandable `st.expander` with severity icons, evidence details, and dismiss checkboxes
- **Triage override** — `st.selectbox` to adjust ESI level with clinical justification
- **Action buttons** — Approve & Finalize / Request Revision / Halt Encounter

### Page 3: 🔍 Debug (Time Travel + Observability)
- **Checkpoint history** — Scrollable list of all graph state snapshots with node name, timestamp, and state diff
- **State inspector** — JSON viewer for any selected checkpoint
- **LangSmith link** — Direct link to the full trace for the current encounter
- **Rewind & replay** — Select a past checkpoint and re-run the graph from that point

### Demo Scenario (3 minutes)
A patient describes chest tightness, uploads a photo of 3 pill bottles (metoprolol, warfarin, ibuprofen). The system conducts an adaptive interview, flags a **critical warfarin + ibuprofen interaction** via RAG, retrieves 2 PubMed abstracts on ACS management, classifies as **ESI-2 (Emergent)**, generates a SOAP note, and pauses for provider approval. The provider edits the plan, approves, and the encounter finalizes — all visible in the Streamlit UI with real-time streaming and a complete LangSmith trace.

---

## Timeline

| Day | Focus | Deliverable |
|---|---|---|
| **Day 1** | Foundation | State schemas, top-level StateGraph, ChromaDB setup + data ingestion, SqliteSaver, LangSmith tracing. Working terminal flow. |
| **Day 2** | Intake + Safety | Intake sub-graph with multimodal image tool, Safety Sentinel with Send() map-reduce, ChromaDB drug retriever, custom safety_reducer. |
| **Day 3** | Evidence + Triage | ReAct evidence sub-graph with ChromaDB + PubMed tools, Triage Router with conditional edges and ESI classification. |
| **Day 4** | Brief + Education + Memory | SOAP chain, dynamic Patient Education agent, LangGraph Store (profile + collection patterns), conversation summarization, message trimming. |
| **Day 5** | HITL + Middleware + Streaming | Static/dynamic breakpoints, state editing, PII redaction, guardrails, disclaimers, 3 streaming modes, double-texting config. |
| **Day 6** | Streamlit Frontend | Multi-page app: Encounter chat, Provider Review dashboard, Debug/Time Travel page. Assistant selector, photo upload, agent activity indicators. |
| **Day 7** | Polish + Deploy | Docker deployment, LangSmith eval datasets (drug interactions, triage, RAG quality), README with architecture diagram, 3-min demo video. |

**Total: 7 days**
