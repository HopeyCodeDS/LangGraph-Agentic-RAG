"""Sidebar: provider/model controls + status pills + reset actions."""
from __future__ import annotations

import streamlit as st

from ui import state as ui_state
from ui.components.styles import status_chips
from ui.services import vectorstore as vs_service
from ui.services.providers import (
    EMBEDDING_PROVIDERS,
    LLM_PROVIDERS,
    ProviderConfig,
    apply_provider_settings,
    default_model_for,
)


def _api_key_field(provider: str, current: dict) -> str:
    label_map = {
        "openai": "OpenAI API key",
        "groq": "Groq API key",
        "google": "Google API key",
    }
    if provider not in label_map:
        return current.get(provider, "")
    placeholder = "•••• already set" if current.get(provider) else "paste key"
    value = st.text_input(
        label_map[provider],
        type="password",
        placeholder=placeholder,
        key=f"api_key_{provider}",
    )
    return value or current.get(provider, "")


def render_sidebar() -> None:
    ss = st.session_state
    cfg: ProviderConfig = ss.provider_cfg

    with st.sidebar:
        st.markdown(
            '<h2 style="margin-bottom:0.2rem">⚙️ Providers</h2>',
            unsafe_allow_html=True,
        )

        llm_provider = st.selectbox(
            "LLM provider",
            LLM_PROVIDERS,
            index=LLM_PROVIDERS.index(cfg.llm_provider)
            if cfg.llm_provider in LLM_PROVIDERS
            else 0,
        )
        llm_model = st.text_input(
            "LLM model",
            value=cfg.llm_model or default_model_for(llm_provider, "llm"),
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.5, float(cfg.temperature), 0.05
        )

        emb_provider = st.selectbox(
            "Embeddings provider",
            EMBEDDING_PROVIDERS,
            index=EMBEDDING_PROVIDERS.index(cfg.embeddings_provider)
            if cfg.embeddings_provider in EMBEDDING_PROVIDERS
            else 0,
        )
        emb_model = st.text_input(
            "Embedding model",
            value=cfg.embedding_model or default_model_for(emb_provider, "embedding"),
        )

        with st.expander("Advanced", expanded=False):
            ollama_url = st.text_input("Ollama base URL", value=cfg.ollama_base_url)
            max_rewrites = st.number_input(
                "Max rewrites", min_value=0, max_value=10, value=int(cfg.max_rewrites)
            )
            new_keys = {
                "openai": _api_key_field("openai", cfg.api_keys),
                "groq": _api_key_field("groq", cfg.api_keys),
                "google": _api_key_field("google", cfg.api_keys),
            }

        if st.button("Apply & Rebuild Graph", use_container_width=True, type="primary"):
            new_cfg = ProviderConfig(
                llm_provider=llm_provider,
                embeddings_provider=emb_provider,
                llm_model=llm_model.strip(),
                embedding_model=emb_model.strip(),
                temperature=float(temperature),
                ollama_base_url=ollama_url.strip(),
                max_rewrites=int(max_rewrites),
                api_keys=new_keys,
            )
            try:
                apply_provider_settings(new_cfg)
                ss.provider_cfg = new_cfg
                if ss.vectorstore is not None:
                    ui_state.refresh_app_from_store()
                st.success("Settings applied. Graph rebuilt.")
            except Exception as exc:
                st.error(f"Failed to apply settings: {exc}")

        # ── Status ────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-section"><h2>📊 Status</h2></div>',
            unsafe_allow_html=True,
        )
        size = vs_service.index_size(ss.vectorstore) if ss.vectorstore is not None else 0
        chips = [
            ("LLM", f"{cfg.llm_provider} · {cfg.llm_model or '–'}"),
            ("Embeddings", f"{cfg.embeddings_provider} · {cfg.embedding_model or '–'}"),
            ("Chunks", str(size)),
            ("Rewrite budget", str(cfg.max_rewrites)),
        ]
        st.markdown(status_chips(chips), unsafe_allow_html=True)

        # ── Actions ───────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-section"><h2>🧹 Actions</h2></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear chat", use_container_width=True):
                ss.messages = []
                ss.last_state = None
                st.rerun()
        with c2:
            if st.button("Reset KB", use_container_width=True):
                vs_service.reset_index()
                ui_state.reload_vectorstore()
                st.success("Knowledge base wiped.")
                st.rerun()
