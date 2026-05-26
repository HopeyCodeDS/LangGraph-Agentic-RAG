"""Runtime provider/model switching by mutating os.environ + reloading src.config."""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field

LLM_PROVIDERS = ["openai", "ollama", "groq", "google"]
EMBEDDING_PROVIDERS = ["openai", "huggingface", "ollama"]

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
    "groq": "llama-3.1-8b-instant",
    "google": "gemini-2.0-flash",
}

DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "ollama": "nomic-embed-text",
}


@dataclass
class ProviderConfig:
    llm_provider: str = "openai"
    embeddings_provider: str = "openai"
    llm_model: str = ""
    embedding_model: str = ""
    temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"
    max_rewrites: int = 2
    api_keys: dict = field(default_factory=dict)


def current_config() -> ProviderConfig:
    """Read the current effective configuration from the environment."""
    llm = os.getenv("LLM_PROVIDER", "openai").lower()
    emb = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()

    llm_model_env = {
        "openai": os.getenv("OPENAI_LLM_MODEL", DEFAULT_MODELS["openai"]),
        "ollama": os.getenv("OLLAMA_LLM_MODEL", DEFAULT_MODELS["ollama"]),
        "groq": os.getenv("GROQ_MODEL", DEFAULT_MODELS["groq"]),
        "google": os.getenv("GEMINI_MODEL", DEFAULT_MODELS["google"]),
    }
    emb_model_env = {
        "openai": os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODELS["openai"]),
        "huggingface": os.getenv("HF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODELS["huggingface"]),
        "ollama": os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODELS["ollama"]),
    }

    return ProviderConfig(
        llm_provider=llm,
        embeddings_provider=emb,
        llm_model=llm_model_env.get(llm, ""),
        embedding_model=emb_model_env.get(emb, ""),
        temperature=0.0,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        max_rewrites=int(os.getenv("MAX_REWRITES", "2")),
        api_keys={
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "groq": os.getenv("GROQ_API_KEY", ""),
            "google": os.getenv("GOOGLE_API_KEY", ""),
        },
    )


def apply_provider_settings(cfg: ProviderConfig) -> None:
    """Mutate os.environ then reload src.config so factories reflect the new values."""
    os.environ["LLM_PROVIDER"] = cfg.llm_provider
    os.environ["EMBEDDINGS_PROVIDER"] = cfg.embeddings_provider
    os.environ["MAX_REWRITES"] = str(cfg.max_rewrites)
    os.environ["OLLAMA_BASE_URL"] = cfg.ollama_base_url

    if cfg.llm_model:
        if cfg.llm_provider == "openai":
            os.environ["OPENAI_LLM_MODEL"] = cfg.llm_model
        elif cfg.llm_provider == "ollama":
            os.environ["OLLAMA_LLM_MODEL"] = cfg.llm_model
        elif cfg.llm_provider == "groq":
            os.environ["GROQ_MODEL"] = cfg.llm_model
        elif cfg.llm_provider == "google":
            os.environ["GEMINI_MODEL"] = cfg.llm_model

    if cfg.embedding_model:
        if cfg.embeddings_provider == "openai":
            os.environ["OPENAI_EMBEDDING_MODEL"] = cfg.embedding_model
        elif cfg.embeddings_provider == "huggingface":
            os.environ["HF_EMBEDDING_MODEL"] = cfg.embedding_model
        elif cfg.embeddings_provider == "ollama":
            os.environ["OLLAMA_EMBEDDING_MODEL"] = cfg.embedding_model

    for key, value in cfg.api_keys.items():
        if not value:
            continue
        env_key = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
        }.get(key)
        if env_key:
            os.environ[env_key] = value

    _reload_config_modules()


def _reload_config_modules() -> None:
    import src.config.settings as settings_mod
    import src.config as config_pkg
    importlib.reload(settings_mod)
    importlib.reload(config_pkg)
    for name in ("openai", "ollama", "groq", "gemini", "huggingface"):
        full = f"src.config.{name}"
        mod = importlib.import_module(full)
        importlib.reload(mod)
    # Reload edges so MAX_REWRITES picks up the new env value.
    import src.agents.edges as edges_mod
    importlib.reload(edges_mod)


def default_model_for(provider: str, kind: str = "llm") -> str:
    if kind == "embedding":
        return DEFAULT_EMBEDDING_MODELS.get(provider, "")
    return DEFAULT_MODELS.get(provider, "")
