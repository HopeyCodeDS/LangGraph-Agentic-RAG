"""Chat tab: chat history + streaming reasoning trace renderer."""
from __future__ import annotations

import html
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ui.components.styles import node_pill
from ui.services.agent_runner import run_agent


# ── Trace event helpers ─────────────────────────────────────────────────────


def _split_retrieved_chunks(content: str) -> list[str]:
    """Heuristic split of the retriever ToolMessage content into k chunks."""
    if not content:
        return []
    # create_retriever_tool joins docs with two newlines by default.
    parts = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(parts) <= 1:
        return [content.strip()]
    return parts


def _render_chunk_card(idx: int, text: str) -> str:
    preview = html.escape(text[:600])
    if len(text) > 600:
        preview += "…"
    return (
        f'<div class="source-card">'
        f'<div class="source-title">Chunk #{idx + 1}</div>'
        f'<div class="source-snippet">{preview}</div>'
        f'</div>'
    )


def _render_agent_event(value: dict) -> None:
    msg = value["messages"][-1] if value.get("messages") else None
    tool_calls = getattr(msg, "tool_calls", None) if msg else None
    if tool_calls:
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            st.markdown(
                f"🔧 **Tool call:** `{name}`  \nArgs: `{args}`"
            )
    elif msg is not None and getattr(msg, "content", ""):
        st.markdown(msg.content)


def _render_retrieve_event(value: dict) -> list[str]:
    msg = value["messages"][-1] if value.get("messages") else None
    if not isinstance(msg, ToolMessage):
        return []
    chunks = _split_retrieved_chunks(msg.content or "")
    cards = "".join(_render_chunk_card(i, c) for i, c in enumerate(chunks))
    st.markdown(cards, unsafe_allow_html=True)
    st.caption(f"{len(chunks)} chunk(s) retrieved.")
    return chunks


def _render_grade_event(event: dict) -> None:
    score = event.get("score", "?")
    reasoning = event.get("reasoning", "")
    if score == "yes":
        st.success(f"**Relevant ✓**  \n{reasoning}")
    elif score == "no":
        st.warning(f"**Not relevant ✗**  \n{reasoning}")
    else:
        st.info(reasoning or "Forcing generate (rewrite budget exhausted).")


def _render_rewrite_event(value: dict) -> None:
    msg = value["messages"][-1] if value.get("messages") else None
    rewrites = value.get("rewrites", "?")
    if isinstance(msg, HumanMessage):
        st.markdown(f"**Rewritten query** (attempt {rewrites}):")
        st.markdown(f"> {msg.content}")


def _render_generate_event(value: dict) -> str:
    msg = value["messages"][-1] if value.get("messages") else None
    text = getattr(msg, "content", "") if msg is not None else ""
    if text:
        st.markdown(text)
    return text


# ── Trace replay (for prior turns) ──────────────────────────────────────────


def _replay_trace(trace: list[dict]) -> None:
    for ev in trace:
        node = ev["node"]
        label = ev.get("label") or node
        with st.status(label, state="complete", expanded=False):
            st.markdown(node_pill(node), unsafe_allow_html=True)
            kind = ev.get("kind", "state")
            if kind == "grade":
                _render_grade_event(ev)
            elif kind == "tool_call":
                args = ev.get("args", {})
                st.markdown(f"🔧 Tool call: `{ev.get('tool_name','?')}`  \nArgs: `{args}`")
            elif kind == "retrieve":
                cards = "".join(
                    _render_chunk_card(i, c) for i, c in enumerate(ev.get("chunks", []))
                )
                st.markdown(cards, unsafe_allow_html=True)
                st.caption(f"{len(ev.get('chunks', []))} chunk(s) retrieved.")
            elif kind == "rewrite":
                st.markdown(
                    f"**Rewritten query** (attempt {ev.get('rewrites','?')}):  \n"
                    f"> {ev.get('text','')}"
                )
            elif kind == "text":
                st.markdown(ev.get("text", ""))


# ── Live streaming ──────────────────────────────────────────────────────────


def _stream_turn(app, question: str) -> tuple[str, list[dict], dict[str, Any] | None]:
    """Stream a single turn, rendering live status blocks. Returns
    (final_answer, trace, last_state)."""
    trace: list[dict] = []
    final_answer = ""
    last_state: dict[str, Any] | None = None
    sources: list[str] = []

    for event in run_agent(app, question):
        if "state" in event:
            node = event["node"]
            value = event["state"]
            last_state = value if isinstance(value, dict) else last_state

            if node == "agent":
                with st.status(f"agent — routing", expanded=False) as box:
                    st.markdown(node_pill("agent"), unsafe_allow_html=True)
                    _render_agent_event(value)
                msg = value["messages"][-1] if value.get("messages") else None
                tool_calls = getattr(msg, "tool_calls", None) if msg else None
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                        trace.append({"node": "agent", "kind": "tool_call",
                                      "label": "agent — tool call",
                                      "tool_name": name, "args": args})
                else:
                    trace.append({"node": "agent", "kind": "text",
                                  "label": "agent — direct response",
                                  "text": getattr(msg, "content", "") if msg else ""})

            elif node == "retrieve":
                with st.status("retrieve — pulling chunks", expanded=False):
                    st.markdown(node_pill("retrieve"), unsafe_allow_html=True)
                    chunks = _render_retrieve_event(value)
                sources = chunks
                trace.append({"node": "retrieve", "kind": "retrieve",
                              "label": "retrieve — pulled chunks", "chunks": chunks})

            elif node == "rewrite":
                with st.status("rewrite — refining query", expanded=False):
                    st.markdown(node_pill("rewrite"), unsafe_allow_html=True)
                    _render_rewrite_event(value)
                msg = value["messages"][-1] if value.get("messages") else None
                trace.append({"node": "rewrite", "kind": "rewrite",
                              "label": "rewrite — refined query",
                              "rewrites": value.get("rewrites", "?"),
                              "text": getattr(msg, "content", "") if msg else ""})

            elif node == "generate":
                with st.status("generate — final answer", state="complete", expanded=True):
                    st.markdown(node_pill("generate"), unsafe_allow_html=True)
                    text = _render_generate_event(value)
                final_answer = text
                trace.append({"node": "generate", "kind": "text",
                              "label": "generate — final answer", "text": text})

            else:
                trace.append({"node": node, "kind": "state",
                              "label": node, "text": str(value)[:300]})

        elif event.get("node") == "grade":
            with st.status("grade — relevance check", expanded=False):
                st.markdown(node_pill("grade"), unsafe_allow_html=True)
                _render_grade_event(event)
            trace.append({"node": "grade", "kind": "grade",
                          "label": "grade — relevance check",
                          "score": event.get("score"),
                          "reasoning": event.get("reasoning", "")})

    if sources and final_answer:
        st.markdown("---")
        st.markdown("**Sources used:**")
        cards = "".join(_render_chunk_card(i, c) for i, c in enumerate(sources))
        st.markdown(cards, unsafe_allow_html=True)

    return final_answer, trace, last_state


# ── Public entry ────────────────────────────────────────────────────────────


def render() -> None:
    ss = st.session_state

    if ss.app is None:
        st.info(
            "👋 Welcome! To start chatting, head to the **Knowledge Base** tab "
            "and ingest a URL or upload a file. The agent needs documents to "
            "search through."
        )
        return

    # Replay previous turns
    for turn in ss.messages:
        with st.chat_message(turn["role"]):
            if turn["role"] == "assistant" and turn.get("trace"):
                with st.expander("Reasoning trace", expanded=False):
                    _replay_trace(turn["trace"])
            st.markdown(turn["content"])

    question = st.chat_input("Ask the agent...")
    if not question:
        return

    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.expander("Reasoning trace", expanded=True):
            try:
                final_answer, trace, last_state = _stream_turn(ss.app, question)
            except Exception as exc:
                st.error(f"Agent failed: {exc}")
                final_answer, trace, last_state = "", [], None

        if final_answer:
            st.markdown("### 💬 Answer")
            st.markdown(final_answer)
        elif not trace:
            st.warning("The agent produced no output.")

    ss.messages.append({
        "role": "assistant",
        "content": final_answer or "_(no answer)_",
        "trace": trace,
    })
    ss.last_state = last_state
