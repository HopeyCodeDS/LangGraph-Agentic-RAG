<h1 align="center">LangGraph Agentic RAG</h1>

<p align="center">
  <em>A self-correcting Retrieval-Augmented Generation system that reasons, grades, and rewrites — built with LangGraph.</em>
</p>

<p align="center">
  <img src="images/img_3.png" alt="Agentic RAG — Streamlit UI with reasoning trace" width="900">
</p>

<p align="center">
  <a href="https://github.com/langchain-ai/langgraph"><img alt="Built with LangGraph" src="https://img.shields.io/badge/Built%20with-LangGraph-7c3aed"></a>
  <a href="https://streamlit.io/"><img alt="Streamlit UI" src="https://img.shields.io/badge/UI-Streamlit-ff4b4b"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776ab">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

Instead of blindly stuffing retrieved documents into a prompt, the agent **reasons** about when to retrieve, **grades** the relevance of what it gets back, and **rewrites** the query to try again when the retrieved context is weak.

Supports multiple LLM and embedding providers out of the box — **OpenAI, Ollama, Groq, Google Gemini, and HuggingFace** — so you can run it fully local, fully cloud, or mix and match.

Ships with both a **CLI** entry point and a clean **Streamlit UI** for interactive use, knowledge-base management, and live graph inspection.

---

## Screenshots

### Chat with full reasoning trace
Every answer comes with an expandable trace of what the graph actually did — agent routing, relevance grading, retrieval, and generation — plus the exact source chunks used.

<p align="center">
  <img src="images/img_3.png" alt="Chat with reasoning trace and sources" width="850">
</p>

### Knowledge Base — ingest URLs or files
Drop in URLs (one per line) or upload PDF / TXT / Markdown files. The app splits, embeds, and adds them to the FAISS index live, then lists every ingested source with its chunk count.

<p align="center">
  <img src="images/img_5.png" alt="Knowledge Base with ingested source" width="850">
</p>

### Graph & State — live workflow inspector
Visualize the compiled LangGraph and inspect the raw `GraphState` (messages, reasoning trace, tool calls) after each turn. Great for debugging agent behavior.

<p align="center">
  <img src="images/img_7.png" alt="Graph and State inspector with JSON snapshot" width="850">
</p>

### Provider config in the sidebar
Swap LLM provider, model, temperature, and embedding backend without touching code. Status badges at the bottom confirm what's actually wired up.

<p align="center">
  <img src="images/img_1.png" alt="Sidebar provider configuration and welcome screen" width="850">
</p>

---

## Why "Agentic" RAG?

Classic RAG is a fixed pipeline: `retrieve → stuff → generate`. It fails silently when the retriever returns irrelevant chunks, and it has no way to recover.

This project models retrieval as a **state machine** where the LLM is in the driver's seat:

```
        ┌──────────┐
        │  Agent   │ ◄────────────┐
        └────┬─────┘              │
             │ tool_call?         │
       ┌─────┴─────┐              │
       ▼           ▼              │
   ┌────────┐    [END]            │
   │Retrieve│                     │
   └───┬────┘                     │
       │                          │
       ▼                          │
   ┌────────┐  no   ┌──────────┐  │
   │ Grade  ├──────►│ Rewrite  ├──┘
   └───┬────┘       └──────────┘
       │ yes
       ▼
   ┌────────┐
   │Generate│ ──► [END]
   └────────┘
```

- **Agent** — decides whether the question needs retrieval or can be answered directly.
- **Retrieve** — calls the FAISS retriever tool.
- **Grade** — a structured-output LLM judge gives a strict `yes`/`no` on whether the retrieved chunks actually answer the question.
- **Rewrite** — reformulates the query for better recall, then loops back to the agent. Budgeted (default: 2 attempts) so the graph cannot spin forever.
- **Generate** — produces the final answer grounded **only** in the retrieved context.

---

## Features

- **Self-correcting retrieval** — relevance grading + bounded query rewrites.
- **Multi-provider support** — swap LLMs and embeddings via environment variables, no code changes.
- **Local-first option** — run end-to-end on your machine with Ollama + HuggingFace, zero API keys.
- **FAISS vector store** — fast in-memory similarity search.
- **Web ingestion** — point it at a list of URLs and it builds the index for you.
- **Structured output grading** — uses Pydantic schemas with `with_structured_output` for reliable judge calls.
- **Bounded loops** — `MAX_REWRITES` and `recursion_limit` prevent runaway agents.

---

## Project Structure

```
src/
├── main.py                 # CLI entry — ingests URLs, builds graph, runs sample questions
├── retriever.py            # WebBaseLoader → splitter → FAISS → retriever tool
├── agents/
│   ├── graph.py            # LangGraph StateGraph wiring
│   ├── nodes.py            # agent / rewrite / generate nodes
│   └── edges.py            # grade_documents + route_after_agent conditional edges
└── config/
    ├── settings.py         # env-driven provider + model selection
    ├── openai.py           # OpenAI LLM + embeddings
    ├── ollama.py           # Ollama LLM + embeddings (local)
    ├── groq.py             # Groq LLM
    ├── gemini.py           # Google Gemini LLM
    └── huggingface.py      # HuggingFace embeddings (local)

ui/
├── app.py                  # Streamlit app entry point
├── state.py                # session-state helpers
├── components/             # chat, knowledge-base, graph-inspector UI
└── services/               # ingestion, retrieval, graph orchestration glue
```

---

## Requirements

- Python **3.10+**
- An API key for your chosen provider, **or** a local Ollama installation
- ~500MB disk for HuggingFace embedding model (if used locally)

---

## Installation

```bash
git clone https://github.com/HopeyCodeDS/LangGraph-Agentic-RAG.git
cd LangGraph-Agentic-RAG

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

This installs the project (declared in [pyproject.toml](pyproject.toml)) along with `langchain`, `langgraph`, `faiss-cpu`, and the provider SDKs.

---

## Configuration

Create a `.env` file in the project root:

```dotenv
# --- choose your providers ---
LLM_PROVIDER=openai            # openai | ollama | groq | google
EMBEDDINGS_PROVIDER=openai     # openai | huggingface | ollama

# --- credentials (only the ones you need) ---
OPENAI_API_KEY=...
GROQ_API_KEY=...
GOOGLE_API_KEY=...

# --- optional model overrides ---
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GROQ_MODEL=llama-3.1-8b-instant
GEMINI_MODEL=gemini-2.0-flash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Common configurations

| Setup | `LLM_PROVIDER` | `EMBEDDINGS_PROVIDER` | Needs |
| --- | --- | --- | --- |
| All-OpenAI (default) | `openai` | `openai` | `OPENAI_API_KEY` |
| Fully local | `ollama` | `huggingface` | Ollama running locally |
| Fast + cheap | `groq` | `huggingface` | `GROQ_API_KEY` |
| Gemini + local embeddings | `google` | `huggingface` | `GOOGLE_API_KEY` |

---

## Running

### Streamlit UI (recommended)

```bash
streamlit run ui/app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`) and you'll land on the screens shown in [Screenshots](#screenshots) above. Workflow:

1. Pick your provider/model in the sidebar and click **Apply & Rebuild Graph**.
2. Go to **Knowledge Base** and ingest one or more URLs or upload PDF/TXT/MD files.
3. Switch to **Chat** and ask away — each answer expands into a reasoning trace and source chunks.
4. Open **Graph & State** to view the compiled LangGraph and the raw state from the last turn.

### CLI

```bash
python -m src.main
```

You'll see the agent ingest two Lilian Weng blog posts, build the FAISS index, then answer two sample questions while streaming each node's output:

```
🚀 Initializing Agentic RAG System...
📚 Ingesting documents into FAISS vector store...
✅ Documents ingested successfully!
🔧 Building LangGraph state machine...
✅ Graph built successfully!

============================================================
💬 Interactive mode — ask a question, or type 'quit' / 'exit' to leave.       
============================================================

> who are VCs and what do they really think?

============================================================
❓ Question: who are VCs and what do they really think?
============================================================
🔄 Processing...

📍 Node: agent
🔧 Tool Call: [{'name': 'retrieve_document', 'args': {'query': 'VCs opinions'}, 'id': '63761ab9-14b0-43ed-990a-aa5f70b913bb', 'type': 'tool_call'}]

🔍 Grading: Yes
💭 Reasoning: The document contains direct quotes from Newton and other investors that provide insight into what VCs really think about ARR misrepresentations...

📍 Node: retrieve
💭 Output:
Newton, whose legal AI startup Clio was valued at $5 billion last fall, also ...

📍 Node: generate
💬 Output:
According to the context, VCs (Venture Capitalists) are aware of startups inflation ...

============================================================
✨ FINAL ANSWER:
============================================================
According to the context, VCs (Venture Capitalists) are aware of startups inflating their ARR (Annual Recurring Revenue) numbers but often choose not to expose them. In fact, some VCs even support or overlook this practice...
```

### Use your own documents

Edit the `urls` list in [src/main.py](src/main.py):

```python
urls = [
    "https://hostname/topic-1",
    "https://hostname/topic-2",
]
```

The ingestion pipeline uses `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200) and stores everything in an in-memory FAISS index. Swap `WebBaseLoader` in [src/retriever.py](src/retriever.py) for `PyPDFLoader`, `DirectoryLoader`, etc. to ingest other sources.

---

## How the graph behaves

| Situation | What happens |
| --- | --- |
| Question doesn't need retrieval (small talk, math, etc.) | Agent answers directly → `END` |
| Retrieved chunks are relevant | Grader returns `yes` → `generate` → `END` |
| Retrieved chunks are weak | Grader returns `no` → `rewrite` → back to agent |
| Rewrites hit the budget (`MAX_REWRITES = 2`) | Forced `generate` with whatever context is available |
| Agent loops too long | `recursion_limit=10` aborts the run |

Tune the strictness of the grader by editing `grade_prompt` in [src/agents/edges.py](src/agents/edges.py), or change `MAX_REWRITES` to allow more attempts.

---

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph): stateful agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain): model + tool abstractions
- [FAISS](https://github.com/facebookresearch/faiss): vector similarity search
- [Pydantic](https://docs.pydantic.dev/): structured output schemas
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/): HTML parsing for web ingestion

---

## License

MIT — feel free to use, modify, and ship.
