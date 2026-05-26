"""Session state schema for the Streamlit UI."""
from __future__ import annotations

import streamlit as st

from ui.services import vectorstore as vs_service
from ui.services.agent_runner import build_app_from_store
from ui.services.providers import current_config


def init_state() -> None:
    """Initialize keys in st.session_state if not already set."""
    ss = st.session_state

    if "provider_cfg" not in ss:
        ss.provider_cfg = current_config()

    if "vectorstore" not in ss:
        ss.vectorstore = vs_service.load_index() if vs_service.index_exists() else None

    if "app" not in ss:
        ss.app = build_app_from_store(ss.vectorstore) if ss.vectorstore is not None else None

    if "messages" not in ss:
        # Chat history: list of dicts {role, content, trace?: list[event]}
        ss.messages = []

    if "last_state" not in ss:
        ss.last_state = None  # most recent GraphState snapshot for inspection

    if "manifest" not in ss:
        ss.manifest = vs_service.read_manifest()


def refresh_app_from_store() -> None:
    """Rebuild the LangGraph app after the vectorstore changes."""
    ss = st.session_state
    if ss.vectorstore is None:
        ss.app = None
    else:
        ss.app = build_app_from_store(ss.vectorstore)


def reload_vectorstore() -> None:
    """Reload FAISS from disk into session state."""
    ss = st.session_state
    ss.vectorstore = vs_service.load_index() if vs_service.index_exists() else None
    ss.manifest = vs_service.read_manifest()
    refresh_app_from_store()
