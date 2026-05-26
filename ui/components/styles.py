"""Custom CSS injection for the Streamlit UI."""
from __future__ import annotations

import streamlit as st

NODE_COLORS = {
    "agent": "#7c3aed",
    "retrieve": "#3b82f6",
    "grade": "#f59e0b",
    "rewrite": "#fb923c",
    "generate": "#10b981",
}

_CSS = """
<style>
/* Page width + spacing */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.4rem;
}
.app-header-title {
    font-size: 1.85rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0;
}
.app-header-sub {
    color: #9aa0ad;
    font-size: 0.93rem;
    margin-top: -0.2rem;
    margin-bottom: 1.1rem;
}

/* Status pill row */
.status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.4rem 0 0.5rem 0;
}
.status-chip {
    background: #161922;
    border: 1px solid #2a2f3d;
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    color: #cdd2dd;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
.status-chip strong {
    color: #ffffff;
    font-weight: 600;
}
.status-chip .dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: #10b981;
    display: inline-block;
}

/* Node trace pills */
.node-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin-right: 0.4rem;
}
.node-pill .swatch {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    display: inline-block;
}
.node-agent    { background: rgba(124, 58, 237, 0.16); color: #c4b5fd; }
.node-agent    .swatch { background: #7c3aed; }
.node-retrieve { background: rgba(59, 130, 246, 0.16); color: #93c5fd; }
.node-retrieve .swatch { background: #3b82f6; }
.node-grade    { background: rgba(245, 158, 11, 0.16); color: #fcd34d; }
.node-grade    .swatch { background: #f59e0b; }
.node-rewrite  { background: rgba(251, 146, 60, 0.16); color: #fdba74; }
.node-rewrite  .swatch { background: #fb923c; }
.node-generate { background: rgba(16, 185, 129, 0.16); color: #6ee7b7; }
.node-generate .swatch { background: #10b981; }

/* Source cards */
.source-card {
    background: #11141c;
    border: 1px solid #232838;
    border-left: 3px solid #7c3aed;
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    color: #cdd2dd;
}
.source-card .source-title {
    font-weight: 600;
    color: #e6e8ee;
    margin-bottom: 0.2rem;
    word-break: break-word;
}
.source-card .source-snippet {
    color: #9aa0ad;
    font-size: 0.8rem;
    line-height: 1.45;
}

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background: #0b0d12;
    border-right: 1px solid #1c1f2a;
}
section[data-testid="stSidebar"] h2 {
    margin-top: 0.4rem;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
}
section[data-testid="stSidebar"] .sidebar-section {
    border-top: 1px solid #1c1f2a;
    margin-top: 0.9rem;
    padding-top: 0.9rem;
}

/* Chat input bottom */
[data-testid="stChatInput"] {
    border-radius: 14px;
}

/* Mermaid container */
.mermaid-container {
    background: #11141c;
    border: 1px solid #232838;
    border-radius: 12px;
    padding: 1rem;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.01em;
}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def node_pill(node: str) -> str:
    """Return HTML for an inline node pill."""
    cls = f"node-{node}" if node in NODE_COLORS else "node-agent"
    return (
        f'<span class="node-pill {cls}">'
        f'<span class="swatch"></span>{node}</span>'
    )


def status_chips(items: list[tuple[str, str]]) -> str:
    parts = ['<div class="status-row">']
    for label, value in items:
        parts.append(
            f'<span class="status-chip"><span class="dot"></span>'
            f'{label} <strong>{value}</strong></span>'
        )
    parts.append("</div>")
    return "".join(parts)
