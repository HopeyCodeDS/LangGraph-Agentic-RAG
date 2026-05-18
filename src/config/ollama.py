from langchain_ollama import ChatOllama, OllamaEmbeddings
from .settings import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_LLM_MODEL,
)


def get_llm(model_name: str | None = None, temperature: float = 0.0):
    return ChatOllama(
        model=model_name or OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


def get_embeddings():
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
