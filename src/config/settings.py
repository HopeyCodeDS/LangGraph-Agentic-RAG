import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- provider selection ---
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# --- credentials ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- model names (overridable via env) ---
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

def _require(value, env_name: str, provider: str) -> str:
    if not value:
        raise ValueError(
            f"{env_name} is required when the active provider is '{provider}'."
        )
    return value

def require_openai_key() -> str:
    return _require(OPENAI_API_KEY, "OPENAI_API_KEY", "openai")


def require_groq_key() -> str:
    return _require(GROQ_API_KEY, "GROQ_API_KEY", "groq")


def require_google_key() -> str:
    return _require(GOOGLE_API_KEY, "GOOGLE_API_KEY", "google")