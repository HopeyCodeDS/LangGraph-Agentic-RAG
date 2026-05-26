"""Knowledge Base tab: URL + file ingestion + manifest table."""
from __future__ import annotations

import streamlit as st

from ui import state as ui_state
from ui.services import vectorstore as vs_service


def _parse_urls(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.replace(",", "\n").splitlines():
        line = line.strip()
        if line:
            parts.append(line)
    return parts


def render() -> None:
    ss = st.session_state

    st.markdown("### 📚 Knowledge Base")
    st.caption(
        "Ingest URLs or files into the FAISS index."
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Add URLs")
        url_text = st.text_area(
            "One URL per line (or comma-separated)",
            placeholder=(
                "https://jamesclear.com/five-step-creative-process\n"
                "https://www.nirandfar.com/timeboxing-success/"
            ),
            height=130,
            key="url_text",
        )
        if st.button("Ingest URLs", type="primary", use_container_width=True):
            urls = _parse_urls(url_text)
            if not urls:
                st.warning("Paste at least one URL.")
            else:
                with st.spinner(f"Ingesting {len(urls)} URL(s)..."):
                    result = vs_service.add_urls(urls)
                ui_state.reload_vectorstore()
                if result["added"]:
                    st.success(
                        f"Added {result['added']} source(s), "
                        f"{result['chunks']} chunks."
                    )
                for err in result["errors"]:
                    st.error(err)
                st.rerun()

    with right:
        st.markdown("#### Upload Files")
        uploads = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key="file_uploads",
        )
        if st.button(
            "Ingest Files",
            type="primary",
            use_container_width=True,
            disabled=not uploads,
        ):
            with st.spinner(f"Ingesting {len(uploads)} file(s)..."):
                result = vs_service.add_files(uploads)
            ui_state.reload_vectorstore()
            if result["added"]:
                st.success(
                    f"Added {result['added']} file(s), {result['chunks']} chunks."
                )
            for err in result["errors"]:
                st.error(err)
            st.rerun()

    st.divider()
    st.markdown("#### Ingested sources")

    manifest = ss.manifest or []
    if not manifest:
        st.info("No sources ingested yet. Add URLs or files above to get started.")
        return

    for i, entry in enumerate(manifest):
        c1, c2, c3, c4 = st.columns([0.5, 6, 1.4, 1.2])
        icon = "🌐" if entry["type"] == "url" else "📄"
        c1.markdown(f"### {icon}")
        c2.markdown(
            f"**{entry['value']}**  \n"
            f"<span style='color:#9aa0ad;font-size:0.8rem'>"
            f"added {entry.get('added_at','—')}</span>",
            unsafe_allow_html=True,
        )
        c3.metric("chunks", entry["chunks"], label_visibility="collapsed")
        c3.caption(f"{entry['chunks']} chunks")
        if c4.button("Remove", key=f"remove_{i}", use_container_width=True):
            with st.spinner("Rebuilding index without that source..."):
                vs_service.remove_source(entry["value"])
            ui_state.reload_vectorstore()
            st.rerun()
