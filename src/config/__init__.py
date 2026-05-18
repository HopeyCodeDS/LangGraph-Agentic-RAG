from .settings import EMBEDDINGS_PROVIDER, LLM_PROVIDER


def get_llm(model_name: str | None = None, temperature: float = 0.0):
    provider = LLM_PROVIDER
    if provider == "openai":
        from . import openai as _p
    elif provider == "ollama":
        from . import ollama as _p
    elif provider == "groq":
        from . import groq as _p
    elif provider in ("google", "gemini"):
        from . import gemini as _p
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Expected: openai | ollama | groq | google."
        )
    return _p.get_llm(model_name=model_name, temperature=temperature)


def get_embeddings():
    provider = EMBEDDINGS_PROVIDER
    if provider == "openai":
        from . import openai as _p
    elif provider == "huggingface":
        from . import huggingface as _p
    elif provider == "ollama":
        from . import ollama as _p
    else:
        raise ValueError(
            f"Unknown EMBEDDINGS_PROVIDER '{provider}'. "
            "Expected: openai | huggingface | ollama."
        )
    return _p.get_embeddings()


__all__ = ["get_llm", "get_embeddings"]
