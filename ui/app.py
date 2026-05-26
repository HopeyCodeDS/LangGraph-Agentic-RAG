"""Streamlit entrypoint for the LangGraph Agentic RAG UI.

Run from repo root:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when Streamlit launches this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui import state as ui_state  # noqa: E402
from ui.components import chat, graph_view, ingest, settings, styles  # noqa: E402


def main() -> None:
    st.set_page_config(
        page_title="Agentic RAG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    styles.inject()
    ui_state.init_state()

    st.markdown(
        """
        <div class="app-header">
            <div style="font-size:1.85rem">🧠</div>
            <h1 class="app-header-title">Agentic RAG</h1>
        </div>
        <div class="app-header-sub">
            A self-correcting LangGraph agent — retrieves, grades, rewrites,
            and answers from your knowledge base.
        </div>
        """,
        unsafe_allow_html=True,
    )

    settings.render_sidebar()

    chat_tab, kb_tab, graph_tab = st.tabs(["💬 Chat", "📚 Knowledge Base", "🗺️ Graph & State"])

    with chat_tab:
        chat.render()
    with kb_tab:
        ingest.render()
    with graph_tab:
        graph_view.render()


if __name__ == "__main__":
    main()
