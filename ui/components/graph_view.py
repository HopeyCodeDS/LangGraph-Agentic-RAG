"""Graph & State tab: Mermaid diagram + JSON state inspector."""
from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

_FALLBACK_MERMAID = """
flowchart TD
    Start([__start__]) --> agent
    agent -->|tool_calls| retrieve
    agent -->|no tool| End([__end__])
    retrieve -->|relevant| generate
    retrieve -->|not relevant| rewrite
    rewrite --> agent
    generate --> End
""".strip()


def _mermaid_source(app) -> str:
    if app is None:
        return _FALLBACK_MERMAID
    try:
        return app.get_graph().draw_mermaid()
    except Exception:
        return _FALLBACK_MERMAID


def _render_mermaid(diagram: str, height: int = 520) -> None:
    safe = html.escape(diagram)
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            body {{ background: #11141c; color: #e6e8ee; font-family: sans-serif;
                    margin: 0; padding: 14px; }}
            .mermaid {{ background: transparent; }}
        </style>
    </head>
    <body>
        <pre class="mermaid">{safe}</pre>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: "dark",
                themeVariables: {{
                    primaryColor: "#7c3aed",
                    primaryTextColor: "#e6e8ee",
                    primaryBorderColor: "#7c3aed",
                    lineColor: "#5a6275",
                    secondaryColor: "#161922",
                    tertiaryColor: "#0b0d12",
                    background: "#11141c"
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_doc, height=height, scrolling=True)


def _message_to_dict(msg: Any) -> dict:
    out: dict = {
        "type": type(msg).__name__,
        "content": getattr(msg, "content", None),
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None),
                "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None),
            }
            for tc in tool_calls
        ]
    name = getattr(msg, "name", None)
    if name:
        out["name"] = name
    return out


def _state_to_dict(state: dict | None) -> dict:
    if not state:
        return {}
    out: dict = {}
    for key, value in state.items():
        if key == "messages" and isinstance(value, list):
            out["messages"] = [_message_to_dict(m) for m in value]
        else:
            try:
                json.dumps(value)
                out[key] = value
            except TypeError:
                out[key] = repr(value)
    return out


def render() -> None:
    ss = st.session_state

    st.markdown("### 🗺️ Graph & State")

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("#### LangGraph workflow")
        diagram = _mermaid_source(ss.app)
        st.markdown('<div class="mermaid-container">', unsafe_allow_html=True)
        _render_mermaid(diagram)
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("Mermaid source", expanded=False):
            st.code(diagram, language="mermaid")

    with right:
        st.markdown("#### Last turn — GraphState")
        if ss.last_state is None:
            st.info("Run a chat turn to see the state snapshot here.")
        else:
            st.json(_state_to_dict(ss.last_state))
        if ss.messages:
            with st.expander("Latest reasoning trace (raw)", expanded=False):
                last_assistant = next(
                    (m for m in reversed(ss.messages) if m["role"] == "assistant"),
                    None,
                )
                if last_assistant and last_assistant.get("trace"):
                    st.json(last_assistant["trace"])
                else:
                    st.caption("No trace recorded yet.")
