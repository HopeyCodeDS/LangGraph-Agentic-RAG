"""Agent runner: wraps app.stream() and yields typed events for the UI.

Captures stdout during the stream so grader output (which comes from a conditional
edge, not a node — see src/agents/edges.py) can be surfaced in the trace.
"""
from __future__ import annotations

import io
import re
import sys
from typing import Iterator

from langchain_core.messages import HumanMessage

from src.agents.graph import build_graph
from src.retriever import get_retriever_tools


_GRADE_PATTERN = re.compile(r"Grading:\s*(yes|no)", re.IGNORECASE)
_REASONING_PATTERN = re.compile(r"Reasoning:\s*(.+)")
_BUDGET_PATTERN = re.compile(r"Rewrite budget exhausted\s*\((\d+)/(\d+)\)")


class _TeeStream(io.TextIOBase):
    """A text stream that mirrors writes to a buffer AND a sink stream."""

    def __init__(self, sink):
        self._sink = sink
        self._buffer: list[str] = []

    def write(self, s: str) -> int:
        self._sink.write(s)
        self._buffer.append(s)
        return len(s)

    def flush(self) -> None:
        self._sink.flush()

    def drain(self) -> str:
        text = "".join(self._buffer)
        self._buffer.clear()
        return text


def build_app_from_store(vectorstore):
    """Build a fresh LangGraph app from a FAISS vectorstore."""
    tool = get_retriever_tools(vectorstore)
    return build_graph([tool])


def _parse_grader_events(text: str) -> list[dict]:
    events: list[dict] = []
    score: str | None = None
    reasoning: str | None = None
    budget: tuple[int, int] | None = None

    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        m = _GRADE_PATTERN.search(clean)
        if m:
            score = m.group(1).lower()
            continue
        m = _REASONING_PATTERN.search(clean)
        if m:
            reasoning = m.group(1).strip()
            continue
        m = _BUDGET_PATTERN.search(clean)
        if m:
            budget = (int(m.group(1)), int(m.group(2)))

    if budget is not None:
        events.append({
            "node": "grade",
            "score": "forced_generate",
            "reasoning": f"Rewrite budget exhausted ({budget[0]}/{budget[1]}). Forcing generate.",
        })
    elif score is not None:
        events.append({
            "node": "grade",
            "score": score,
            "reasoning": reasoning or "",
        })

    return events


def run_agent(app, question: str, recursion_limit: int = 10) -> Iterator[dict]:
    """Stream agent events. Yields dicts with shape:
        {"node": <name>, "state": <state-dict>}            (from app.stream)
        {"node": "grade", "score": "yes"|"no"|..., "reasoning": str}  (synthetic)
    """
    inputs = {"messages": [HumanMessage(content=question)]}

    original_stdout = sys.stdout
    tee = _TeeStream(original_stdout)
    sys.stdout = tee
    try:
        for chunk in app.stream(inputs, {"recursion_limit": recursion_limit}):
            captured = tee.drain()
            if captured:
                for ev in _parse_grader_events(captured):
                    yield ev
            for node, value in chunk.items():
                yield {"node": node, "state": value}
        trailing = tee.drain()
        if trailing:
            for ev in _parse_grader_events(trailing):
                yield ev
    finally:
        sys.stdout = original_stdout
