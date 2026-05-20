"""
RAG adapter for server use.
Lazy-loads and caches the embedder + Chroma client on first call.
Wraps the logic from scripts/rag_query.py.
"""
from __future__ import annotations

import os
import sys
import json
import re
import warnings
from pathlib import Path
from typing import Any, Generator

import requests

# Allow importing from scripts/
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rag_query import (
    load_manifest,
    expand_query,
    chroma_query,
    lexical_search,
    rerank_results,
    build_context,
    load_local_records,
    generate_answer,
)

# ── Config from env ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _resolve(env_key: str, default: Path) -> Path:
    val = os.getenv(env_key)
    if not val:
        return default
    p = Path(val)
    return p if p.is_absolute() else PROJECT_ROOT / p

VECTOR_DIR = _resolve("VECTOR_DIR", SCRIPTS_DIR / "vector_db")
DATA_DIR   = _resolve("DATA_DIR",   SCRIPTS_DIR)
OLLAMA_BASE  = os.getenv("OLLAMA_BASE",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

PROP_K    = int(os.getenv("RAG_PROP_K",    "8"))
CHUNK_K   = int(os.getenv("RAG_CHUNK_K",   "10"))
LEXICAL_K = int(os.getenv("RAG_LEXICAL_K", "12"))
FINAL_K   = int(os.getenv("RAG_FINAL_K",   "8"))
MAX_CHARS = int(os.getenv("RAG_MAX_CHARS", "7000"))

# ── Lazy-loaded singletons ───────────────────────────────────────────────────
_embedder     = None
_chroma_client = None
_datasets     = None
_local_records: list[dict[str, Any]] | None = None


def _load_runtime() -> None:
    global _embedder, _chroma_client, _datasets, _local_records

    if _embedder is not None:
        return  # already loaded

    warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    manifest = load_manifest(VECTOR_DIR)
    model_name = manifest["embedding_model"]

    try:
        _embedder = SentenceTransformer(model_name, local_files_only=True)
    except TypeError:
        _embedder = SentenceTransformer(model_name)

    _chroma_client = chromadb.PersistentClient(
        path=str(VECTOR_DIR / "chroma"),
        settings=Settings(anonymized_telemetry=False),
    )
    _datasets = manifest["datasets"]
    _local_records = load_local_records(DATA_DIR)


# ── Public API ───────────────────────────────────────────────────────────────

def retrieve(query: str) -> tuple[list[dict[str, Any]], str]:
    """Return (ranked_results, context_string) for a query."""
    _load_runtime()

    expanded = expand_query(query)
    vec = _embedder.encode(
        [expanded],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32").tolist()

    prop_results   = chroma_query(client=_chroma_client, collection_name=_datasets["propositions"]["collection"], query_embedding=vec, top_k=PROP_K,    dataset="propositions")
    chunk_results  = chroma_query(client=_chroma_client, collection_name=_datasets["main_chunks"]["collection"],  query_embedding=vec, top_k=CHUNK_K,   dataset="main_chunks")
    lexical        = lexical_search(query, _local_records, LEXICAL_K)

    merged  = rerank_results(query, prop_results + chunk_results + lexical, FINAL_K)
    context = build_context(merged, MAX_CHARS)
    return merged, context


def query_rag(query: str) -> str:
    """Retrieve context and generate a full answer (blocking)."""
    _, context = retrieve(query)
    return generate_answer(
        query=query,
        context=context,
        ollama_base=OLLAMA_BASE,
        ollama_model=OLLAMA_MODEL,
        timeout=OLLAMA_TIMEOUT,
    )


def stream_rag(query: str) -> Generator[str, None, None]:
    """Retrieve context then stream tokens from Ollama."""
    _, context = retrieve(query)

    prompt = f"""You are the ADCET college chatbot.

Answer the user question using only the retrieved context below.
The user may use short forms, typos, or casual wording. Interpret common college terms normally:
- department, branch, course, and program can refer to offered programs.
- AIDS means Artificial Intelligence and Data Science.
- IoT means Internet of Things.
- CSE means Computer Science Engineering.
- previous year or last year usually means 2024-25 when the context contains 2025-26, 2024-25, and 2023-24.
If the retrieved context contains the answer under an equivalent term, answer it.
If the context truly does not contain the answer, say: "I do not have that information in the current knowledge base."
Keep the answer concise and factual.

Retrieved context:
{context}

User question: {query}

Answer:"""

    response = requests.post(
        f"{OLLAMA_BASE.rstrip('/')}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 350},
        },
        stream=True,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("response", "")
        if token:
            yield token
        if chunk.get("done"):
            break
