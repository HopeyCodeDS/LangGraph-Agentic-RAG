"""FAISS vectorstore service: persistence, ingestion, manifest tracking."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_embeddings

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "faiss_index"
MANIFEST_PATH = DATA_DIR / "sources.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def read_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def write_manifest(entries: list[dict]) -> None:
    _ensure_data_dir()
    MANIFEST_PATH.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_manifest(entry: dict) -> None:
    entries = read_manifest()
    entries.append(entry)
    write_manifest(entries)


def index_exists() -> bool:
    return (INDEX_DIR / "index.faiss").exists()


def load_index(embeddings=None) -> FAISS | None:
    if not index_exists():
        return None
    embeddings = embeddings or get_embeddings()
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def save_index(vectorstore: FAISS) -> None:
    _ensure_data_dir()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))


def reset_index() -> None:
    """Wipe the FAISS index directory and clear the manifest."""
    if INDEX_DIR.exists():
        for child in INDEX_DIR.iterdir():
            child.unlink()
        INDEX_DIR.rmdir()
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_or_extend(splits: list[Document], embeddings) -> FAISS:
    existing = load_index(embeddings)
    if existing is None:
        return FAISS.from_documents(splits, embedding=embeddings)
    existing.add_documents(splits)
    return existing


def add_urls(urls: list[str]) -> dict:
    """Ingest a list of URLs. Returns a summary dict."""
    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        return {"added": 0, "chunks": 0, "errors": []}

    embeddings = get_embeddings()
    splitter = _splitter()
    errors: list[str] = []
    total_chunks = 0
    added = 0

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for d in docs:
                d.metadata.setdefault("source", url)
            splits = splitter.split_documents(docs)
            if not splits:
                errors.append(f"{url}: no content extracted")
                continue
            vs = _build_or_extend(splits, embeddings)
            save_index(vs)
            _append_manifest({
                "type": "url",
                "value": url,
                "chunks": len(splits),
                "added_at": _now_iso(),
            })
            total_chunks += len(splits)
            added += 1
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    return {"added": added, "chunks": total_chunks, "errors": errors}


def _load_file_docs(path: Path, original_name: str) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix in (".txt", ".md", ".markdown"):
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    for d in docs:
        d.metadata["source"] = original_name
    return docs


def add_files(files: Iterable) -> dict:
    """Ingest uploaded files (Streamlit UploadedFile-like). Returns a summary dict."""
    files = list(files)
    if not files:
        return {"added": 0, "chunks": 0, "errors": []}

    embeddings = get_embeddings()
    splitter = _splitter()
    errors: list[str] = []
    total_chunks = 0
    added = 0

    for uploaded in files:
        name = getattr(uploaded, "name", "upload")
        try:
            suffix = Path(name).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer() if hasattr(uploaded, "getbuffer") else uploaded.read())
                tmp_path = Path(tmp.name)
            try:
                docs = _load_file_docs(tmp_path, name)
                splits = splitter.split_documents(docs)
                if not splits:
                    errors.append(f"{name}: no content extracted")
                    continue
                vs = _build_or_extend(splits, embeddings)
                save_index(vs)
                _append_manifest({
                    "type": "file",
                    "value": name,
                    "chunks": len(splits),
                    "added_at": _now_iso(),
                })
                total_chunks += len(splits)
                added += 1
            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    return {"added": added, "chunks": total_chunks, "errors": errors}


def remove_source(value: str) -> bool:
    """Remove a source by `value` from the manifest and rebuild the FAISS index
    from scratch using the remaining manifest entries."""
    entries = read_manifest()
    remaining = [e for e in entries if e["value"] != value]
    if len(remaining) == len(entries):
        return False

    write_manifest([])
    if INDEX_DIR.exists():
        for child in INDEX_DIR.iterdir():
            child.unlink()
        INDEX_DIR.rmdir()

    urls = [e["value"] for e in remaining if e["type"] == "url"]
    if urls:
        add_urls(urls)
    return True


def index_size(vectorstore: FAISS | None = None) -> int:
    vs = vectorstore if vectorstore is not None else load_index()
    if vs is None:
        return 0
    try:
        return vs.index.ntotal
    except AttributeError:
        return 0
