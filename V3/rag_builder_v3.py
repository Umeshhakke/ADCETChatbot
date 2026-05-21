"""
college_rag_builder.py  ·  v3  (Robustness Edition)
=====================================================
Converts a structured college JSON into a production-grade, robustness-first
knowledge base for RAG + GraphRAG pipelines.

═══════════════════════════════════════════════════════════════════════════════
WHY THE PREVIOUS VERSION STILL FAILED FOR REAL USERS
═══════════════════════════════════════════════════════════════════════════════
Research (arXiv 2504.08231, 2507.06956) shows that RAG recall drops 15-40%
when queries are informal, grammatically broken, or phrased differently to how
the data was indexed.  The root cause is always in the DATA, not the pipeline:

  • The index has no surface for "kitni fees hai", "kya cutoff chahiye",
    "placement kaisi hai", "hostel milega kya" — a student's actual words.
  • Embeddings for short, broken queries land in a different vector region
    than the polished paragraph they should retrieve.
  • Without entity relationships stored explicitly, multi-hop questions
    ("which branch has both good placement AND low cutoff?") fail because
    no single chunk contains both facts.

═══════════════════════════════════════════════════════════════════════════════
V3 ENHANCEMENTS  (data-side robustness fixes)
═══════════════════════════════════════════════════════════════════════════════

1.  EMBEDDINGS REMOVED from chunks
    ─────────────────────────────────
    `embedding` and `embedding_model` fields are gone.  Vector DBs own that.
    Only `id` links chunks to vectors.  No file bloat.

2.  WHAT TO EMBED  (instruction comment in every chunk)
    ───────────────────────────────────────────────────
    Each chunk carries `_embed_this` = the field names the embedder MUST use:
      • `contextualised_text`  → primary semantic vector
      • each item in `propositions[]` → separate factoid vector
      • each item in `hypothetical_questions[]` → HyDE query vectors

3.  ALIASES & SURFACE FORMS  (fixes name-mismatch retrieval failures)
    ────────────────────────────────────────────────────────────────────
    `aliases`: common abbreviations and misspellings of the college / entity
    e.g. ["ADCET", "Annasaheb Dange", "adcet ashta", "dange college ashta"]
    These are injected into bm25_tokens so BM25 finds them on exact match.

4.  COLLOQUIAL QUERY SURFACE FORMS  (core robustness addition)
    ────────────────────────────────────────────────────────────
    `colloquial_variants[]`: 6-8 natural, broken-English, Hinglish-style
    questions a non-technical student would actually type.
    e.g. "fees kitni hai", "cutoff kya tha last year", "hostel milega kya",
         "placement acchi hai kya", "konsa branch lena chahiye"
    These are embedded as SEPARATE light-weight documents in a parallel
    "query surface" index.  When a user's query matches one of these, the
    parent chunk is fetched.  This bridges the vocabulary gap proven in
    arXiv 2504.08231 to cause most real-world RAG failures.

5.  INTENT TAGS  (metadata-filtered routing)
    ──────────────────────────────────────────
    `intent_tags[]`: 2-4 intent labels from a controlled vocabulary:
    ["fees", "admission", "placement", "hostel", "transport", "program_info",
     "eligibility", "company_info", "documents", "comparison"]
    Used to route queries to the right namespace/collection before retrieval.

6.  NORMALIZED TEXT  (fuzzy / BM25 exact-match surface)
    ──────────────────────────────────────────────────────
    `normalized_text`: lowercase, punctuation-stripped version of the main
    text.  Feed to a trigram or fuzzy index alongside BM25.

7.  QUESTION VARIANTS  (multi-phrasing index)
    ─────────────────────────────────────────
    `question_variants[]`: 5-6 formal English rephrasings of the most likely
    questions for this chunk.  Different from hypothetical_questions (which
    are for HyDE embeddings) — these cover synonym / angle variation so the
    chunk is reachable from multiple formal query routes.

8.  KNOWLEDGE GRAPH NODES & EDGES  (GraphRAG-ready)
    ───────────────────────────────────────────────
    `graph_nodes[]`: list of entity nodes extracted for a KG
      { "id", "label", "type", "properties" }
    `graph_edges[]`: directed relationships between nodes
      { "source_id", "target_id", "relation", "weight" }
    Compatible with Neo4j, NetworkX, or any triple-store.
    Enables: multi-hop queries, entity disambiguation, relational answers.

9.  CROSS-CHUNK RELATIONSHIPS  (for graph traversal)
    ───────────────────────────────────────────────
    `related_section_ids[]`: chunk ids of semantically adjacent chunks.
    Allows graph traversal: "CSE cutoff" → "CSE placement" → "CSE companies"
    without re-querying the vector index.

10. MULTI-FORMAT OUTPUTS
    ──────────────────────
    knowledge_base.jsonl        – full enriched chunks  (embed contextualised_text)
    propositions.jsonl          – atomic facts          (embed each proposition)
    query_surfaces.jsonl        – colloquial variants   (embed each variant)
    graph_nodes.json            – KG nodes              (load into Neo4j / NetworkX)
    graph_edges.json            – KG edges              (load into Neo4j / NetworkX)
    summary_index.json          – one-line summaries    (two-stage retrieval)
    qa_pairs.json               – 4-type Q&A pairs      (eval / fine-tuning)

Outputs
-------
  knowledge_base.jsonl
  propositions.jsonl
  query_surfaces.jsonl
  graph_nodes.json
  graph_edges.json
  summary_index.json
  qa_pairs.json

Usage
-----
  python college_rag_builder.py
  python college_rag_builder.py --input output.json
  python college_rag_builder.py --input output.json --model qwen2.5:7b-instruct
  python college_rag_builder.py --input output.json --skip-qa
"""

import argparse
import json
import logging
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL   = "qwen2.5:7b"
OLLAMA_BASE     = "http://localhost:11434"
OLLAMA_TIMEOUT  = 180
MAX_RETRIES     = 3
RETRY_DELAY     = 5
MIN_CHUNK_WORDS = 40

# Intent tag vocabulary (controlled set for metadata routing)
INTENT_VOCAB = {
    "fees", "admission", "placement", "hostel", "transport",
    "program_info", "eligibility", "company_info", "documents", "comparison",
}

# Category → primary intents map (seed; LLM may add more)
CATEGORY_INTENTS = {
    "fees":                ["fees", "admission"],
    "hostel":              ["hostel", "fees"],
    "transport":           ["transport"],
    "cutoffs":             ["admission", "eligibility", "comparison"],
    "eligibility":         ["eligibility", "admission"],
    "placements":          ["placement", "comparison"],
    "company_visits":      ["company_info", "placement"],
    "programs_offered":    ["program_info", "admission"],
    "admission_documents": ["documents", "admission"],
}

DOMAIN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "for",
    "to", "and", "or", "at", "by", "on", "with", "from", "this", "that",
    "it", "its", "be", "has", "have", "had", "not", "as", "also",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Ollama helpers
# =============================================================================

def ollama_generate(
    prompt: str,
    model: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 900,
) -> str:
    payload: dict[str, Any] = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9, "num_predict": max_tokens},
    }
    if system:
        payload["system"] = system
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            return clean_text(resp.json().get("response", "").strip())
        except requests.exceptions.ConnectionError:
            log.error("Ollama not reachable. Start it with: ollama serve")
            sys.exit(1)
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return ""


def check_ollama(model: str) -> None:
    try:
        tags = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10).json()
        available = [m["name"] for m in tags.get("models", [])]
    except Exception:
        log.error("Cannot reach Ollama at %s. Run: ollama serve", OLLAMA_BASE)
        sys.exit(1)
    if not any(model in m for m in available):
        log.warning("Model '%s' not found. Pull: ollama pull %s", model, model)
        log.warning("Available: %s", available or "none")


def clean_text(text: str) -> str:
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"^(Answer|Response|Output|Result|Text)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_json_parse(raw: str) -> Any:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    for sc, ec in [("[", "]"), ("{", "}")]:
        idx = raw.find(sc)
        if idx != -1:
            depth, end_idx = 0, -1
            for i, ch in enumerate(raw[idx:], idx):
                if ch == sc: depth += 1
                elif ch == ec:
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            if end_idx > idx:
                try:
                    return json.loads(raw[idx:end_idx])
                except json.JSONDecodeError:
                    pass
    objects = []
    for m in re.finditer(r'\{[^{}]+\}', raw, re.DOTALL):
        try:
            objects.append(json.loads(m.group()))
        except json.JSONDecodeError:
            pass
    return objects if objects else None


# =============================================================================
# Shared system prompts
# =============================================================================

SYSTEM_WRITER = """\
You are an expert knowledge-base writer specialising in Indian engineering
college admissions, placements, and campus facilities.
Write clear, factual, human-readable paragraphs — no bullet points, no
markdown headers, no numbered lists unless explicitly asked.
Every sentence must be independently informative.
Do NOT invent or extrapolate numbers; use only the data provided.\
"""

SYSTEM_ENRICHER = """\
You are a RAG data-preparation specialist. Your job is to generate structured
enrichment data (JSON) from college knowledge text.
Return ONLY valid JSON — no preamble, no markdown fences.\
"""

# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_inr(value: Any) -> str:
    if value is None:
        return "not specified"
    try:
        v = float(str(value).replace(",", "").replace("/-", "").strip())
        return f"Rs.{v:,.0f}"
    except (ValueError, AttributeError):
        return str(value)

def fmt_lpa(value: Any) -> str:
    return "not disclosed" if value is None else f"{value} LPA"


# =============================================================================
# ENRICHMENT LAYER  — all new v3 fields
# =============================================================================

# ── 1. Context header  (Anthropic contextual retrieval) ────────────────────

def generate_context_header(chunk_text: str, doc_desc: str, model: str) -> str:
    """
    2-3 sentence situating blurb.  Reduces retrieval failure 49% (Anthropic 2024).
    Prepend to text before embedding.
    """
    prompt = (
        f"Document: {doc_desc}\n\nChunk:\n{chunk_text[:600]}\n\n"
        "In 2-3 sentences, state exactly what aspect of the document this chunk covers "
        "and what key facts it contains. Be concrete (e.g. 'CSE branch MHT-CET cutoffs'). "
        "Output only those sentences."
    )
    return ollama_generate(prompt, model, system=SYSTEM_WRITER, max_tokens=120)


# ── 2. Propositions  (atomic fact chunking) ────────────────────────────────

def extract_propositions(chunk_text: str, model: str) -> list[str]:
    """
    Atomic, self-contained fact sentences.  Embed separately for factoid queries.
    (NCBI Bioengineering 2025: best for precise factual retrieval.)
    """
    prompt = (
        "Break the following text into atomic fact statements. "
        "Each must be self-contained (full subject, no 'it'/'they'). "
        "JSON array of strings, max 10.\n\n"
        f"Text:\n{chunk_text}\n\nOutput only the JSON array."
    )
    raw = ollama_generate(prompt, model, system=SYSTEM_ENRICHER, temperature=0.1, max_tokens=600)
    result = safe_json_parse(raw)
    return [str(p).strip() for p in result if str(p).strip()] if isinstance(result, list) else []


# ── 3. Hypothetical questions  (HyDE embedding prep) ──────────────────────

def generate_hypothetical_questions(chunk_text: str, category: str, model: str) -> list[str]:
    """
    3 formal English questions whose ideal answer IS this chunk.
    Embed separately alongside the chunk to improve HyDE-style retrieval.
    NOTE: HyDE helps semantic recall; avoid for exact numerical lookups
    (T2-RAGBench 2026 finding).
    """
    prompt = (
        f"Category: {category}\n\nText:\n{chunk_text[:700]}\n\n"
        "Write exactly 3 specific questions a student might ask whose ideal answer "
        "is in the text. JSON array of 3 strings only."
    )
    raw = ollama_generate(prompt, model, system=SYSTEM_ENRICHER, temperature=0.4, max_tokens=300)
    result = safe_json_parse(raw)
    return [str(q).strip() for q in result[:3] if str(q).strip()] if isinstance(result, list) else []


# ── 4. Colloquial query variants  (THE KEY ROBUSTNESS ADDITION) ────────────

def generate_colloquial_variants(
    chunk_text: str,
    category: str,
    college: str,
    model: str,
) -> list[str]:
    """
    6-8 queries written the way a NON-TECHNICAL Indian student would actually type:
      • broken English ("fees kitni hai", "hostel milega kya")
      • Hinglish ("cutoff kya tha", "placement acchi hai")
      • short/vague ("how much fee", "any hostel", "last year placement")
      • misspelled ("placment", "addmission", "colg fees")
      • question from a different angle ("can i get admission in cse")

    These are embedded as a SEPARATE `query_surfaces` index.
    When a user query closely matches one of these, the parent chunk is
    retrieved directly — bridging the vocabulary gap that causes 15-40%
    recall loss for informal queries (arXiv 2504.08231).
    """
    prompt = (
        f"College: {college} | Section: {category}\n\n"
        f"Knowledge:\n{chunk_text[:500]}\n\n"
        "Generate 8 realistic queries that an Indian college student with BASIC English "
        "might type to find this information. Include:\n"
        "  - 2 in broken/informal English (e.g. 'fees kitni', 'hostel milega')\n"
        "  - 2 in simple short English (e.g. 'how much fee', 'any hostel available')\n"
        "  - 2 with common misspellings (e.g. 'addmission', 'placment', 'cutof')\n"
        "  - 2 from a different angle (e.g. 'can i get cse', 'is hostel cheap')\n"
        "JSON array of 8 strings only. Queries must be realistic, not academic."
    )
    raw = ollama_generate(prompt, model, system=SYSTEM_ENRICHER, temperature=0.6, max_tokens=400)
    result = safe_json_parse(raw)
    return [str(q).strip() for q in result[:8] if str(q).strip()] if isinstance(result, list) else []


# ── 5. Formal question variants  (multi-phrasing index) ───────────────────

def generate_question_variants(chunk_text: str, category: str, model: str) -> list[str]:
    """
    5-6 formal English rephrasings covering synonym/angle variation.
    Different from hypothetical_questions (HyDE) — these ensure the chunk
    is reachable from multiple formal query phrasings.
    """
    prompt = (
        f"Category: {category}\n\nText:\n{chunk_text[:600]}\n\n"
        "Write 6 different formal English questions that this text answers. "
        "Vary vocabulary and phrasing — cover different angles. "
        "JSON array of 6 strings only."
    )
    raw = ollama_generate(prompt, model, system=SYSTEM_ENRICHER, temperature=0.5, max_tokens=400)
    result = safe_json_parse(raw)
    return [str(q).strip() for q in result[:6] if str(q).strip()] if isinstance(result, list) else []


# ── 6. Aliases  (name-mismatch fix) ────────────────────────────────────────

def generate_aliases(
    chunk_text: str,
    category: str,
    college: str,
    college_short: str,
    model: str,
) -> list[str]:
    """
    Abbreviations, alternate names, and common misspellings of the college
    and any key entities mentioned in this chunk.
    Injected into bm25_tokens for exact-match retrieval on any variant.
    """
    base_aliases = [
        college,
        college_short,
        college_short.lower(),
        college.lower(),
    ]
    prompt = (
        f"College full name: {college}\n"
        f"College short name: {college_short}\n"
        f"Chunk category: {category}\n\n"
        f"Text:\n{chunk_text[:400]}\n\n"
        "List all abbreviations, alternate spellings, short forms, and common "
        "misspellings of: (a) the college name, (b) any branch names, "
        "(c) any key terms in this chunk. "
        "JSON array of strings only. Include 6-10 items."
    )
    raw = ollama_generate(prompt, model, system=SYSTEM_ENRICHER, temperature=0.2, max_tokens=250)
    result = safe_json_parse(raw)
    llm_aliases = [str(a).strip() for a in result if str(a).strip()] if isinstance(result, list) else []
    all_aliases = list({a for a in base_aliases + llm_aliases if a})
    return all_aliases[:15]


# ── 7. Intent tags  (metadata routing) ────────────────────────────────────

def assign_intent_tags(category: str, chunk_text: str, model: str) -> list[str]:
    """
    2-4 intent labels from INTENT_VOCAB.
    Seed from category map, then ask LLM to confirm/add.
    Used for namespace routing (only search the 'fees' collection for fee queries).
    """
    seed = CATEGORY_INTENTS.get(category, [])
    prompt = (
        f"Available intent labels: {sorted(INTENT_VOCAB)}\n\n"
        f"Text:\n{chunk_text[:400]}\n\n"
        "Pick 2-4 labels from the list that best describe the user intent for "
        "finding this text. JSON array of strings only, using ONLY labels from the list."
    )
    raw = ollama_generate(prompt, model, system=SYSTEM_ENRICHER, temperature=0.1, max_tokens=100)
    result = safe_json_parse(raw)
    llm_tags = [str(t).strip().lower() for t in result if str(t).strip().lower() in INTENT_VOCAB] \
        if isinstance(result, list) else []
    combined = list(dict.fromkeys(seed + llm_tags))  # preserve order, dedupe
    return combined[:4]


# ── 8. Named entity extraction  (metadata filter + KG nodes) ──────────────

def extract_entities(chunk_text: str, model: str) -> dict[str, list[str]]:
    """
    Typed entity lists for metadata-filtered vector search and KG node creation.
    """
    prompt = (
        "Extract named entities. Return JSON object with these exact keys "
        "(empty list if none):\n"
        '  "branches": [branch names],\n'
        '  "years": [academic years like "2024-25"],\n'
        '  "companies": [company names],\n'
        '  "amounts_inr": [fee/salary amounts as strings],\n'
        '  "categories": [reservation categories OPEN/OBC/SC/ST etc.],\n'
        '  "locations": [city/route names]\n\n'
        f"Text:\n{chunk_text[:800]}\n\nOutput only the JSON object."
    )
    raw = ollama_generate(prompt, model, system=SYSTEM_ENRICHER, temperature=0.1, max_tokens=400)
    result = safe_json_parse(raw)
    keys = ["branches", "years", "companies", "amounts_inr", "categories", "locations"]
    if isinstance(result, dict):
        return {k: [str(v) for v in result.get(k, []) if v] for k in keys}
    return {k: [] for k in keys}


# ── 9. Knowledge graph nodes + edges  (GraphRAG) ──────────────────────────

def build_graph_elements(
    chunk_id: str,
    chunk_text: str,
    entities: dict[str, list[str]],
    category: str,
    college: str,
    year: str,
    model: str,
) -> tuple[list[dict], list[dict]]:
    """
    Build KG nodes and edges for GraphRAG.

    Node types: College, Branch, Company, Fee, ReservationCategory,
                AcademicYear, HostelName, BusRoute, AdmissionType
    Edge types: OFFERS_PROGRAM, HAS_CUTOFF, PLACED_IN, VISITED_BY,
                CHARGES_FEE, HAS_HOSTEL, OPERATES_ROUTE, REQUIRES_DOC

    Research basis: GraphRAG (Microsoft 2024) + RobustGraphRAG (arXiv 2603.05698)
    show graph traversal handles multi-hop queries that vector search fails
    (e.g. "which branch has both good placement AND low cutoff?").
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    # College node (always present)
    college_node_id = f"college::{college.replace(' ', '_').lower()}"
    nodes.append({
        "id":         college_node_id,
        "label":      college,
        "type":       "College",
        "properties": {"academic_year": year},
        "source_chunk_id": chunk_id,
    })

    # Branch nodes + OFFERS_PROGRAM edges
    for branch in entities.get("branches", []):
        nid = f"branch::{branch.replace(' ', '_').lower()}"
        nodes.append({"id": nid, "label": branch, "type": "Branch",
                      "properties": {"category": category},
                      "source_chunk_id": chunk_id})
        edges.append({"source_id": college_node_id, "target_id": nid,
                      "relation": "OFFERS_PROGRAM", "weight": 1.0,
                      "source_chunk_id": chunk_id})

    # Company nodes + VISITED_BY edges
    for company in entities.get("companies", [])[:20]:
        nid = f"company::{company.replace(' ', '_').lower()}"
        nodes.append({"id": nid, "label": company, "type": "Company",
                      "properties": {}, "source_chunk_id": chunk_id})
        edges.append({"source_id": college_node_id, "target_id": nid,
                      "relation": "VISITED_BY", "weight": 1.0,
                      "source_chunk_id": chunk_id})

    # Reservation category nodes + CHARGES_FEE / HAS_CUTOFF edges
    rel = "HAS_CUTOFF" if category == "cutoffs" else "CHARGES_FEE"
    for cat in entities.get("categories", []):
        nid = f"category::{cat.replace('/', '_').lower()}"
        nodes.append({"id": nid, "label": cat, "type": "ReservationCategory",
                      "properties": {}, "source_chunk_id": chunk_id})
        # Link via branch if present, else via college
        sources = [f"branch::{b.replace(' ', '_').lower()}" for b in entities.get("branches", [])] \
                  or [college_node_id]
        for src in sources:
            edges.append({"source_id": src, "target_id": nid,
                          "relation": rel, "weight": 1.0,
                          "source_chunk_id": chunk_id})

    # Year nodes
    for yr in entities.get("years", []):
        nid = f"year::{yr.replace('-', '_')}"
        nodes.append({"id": nid, "label": yr, "type": "AcademicYear",
                      "properties": {}, "source_chunk_id": chunk_id})
        edges.append({"source_id": college_node_id, "target_id": nid,
                      "relation": "IN_YEAR", "weight": 1.0,
                      "source_chunk_id": chunk_id})

    # Location nodes (bus routes, hostels)
    for loc in entities.get("locations", []):
        nid = f"location::{loc.replace(' ', '_').lower()}"
        nodes.append({"id": nid, "label": loc, "type": "Location",
                      "properties": {"section": category},
                      "source_chunk_id": chunk_id})
        edges.append({"source_id": college_node_id, "target_id": nid,
                      "relation": "OPERATES_ROUTE" if category == "transport" else "LOCATED_IN",
                      "weight": 1.0, "source_chunk_id": chunk_id})

    # Ask LLM for any additional important relationship not captured above
    prompt = (
        f"From this college knowledge text, extract 3-5 important relationships "
        f"not already in: {[e['relation'] for e in edges[:5]]}\n\n"
        f"Text:\n{chunk_text[:500]}\n\n"
        'Return JSON array: [{"source":"entity A","relation":"RELATION","target":"entity B"},...]\n'
        "Use UPPERCASE_UNDERSCORE for relation names. Output only JSON array."
    )
    raw = ollama_generate(prompt, model, system=SYSTEM_ENRICHER, temperature=0.2, max_tokens=400)
    extra = safe_json_parse(raw)
    if isinstance(extra, list):
        for rel_obj in extra[:5]:
            if not isinstance(rel_obj, dict):
                continue
            src_label = str(rel_obj.get("source", "")).strip()
            tgt_label = str(rel_obj.get("target", "")).strip()
            rel_name  = str(rel_obj.get("relation", "RELATES_TO")).strip().upper().replace(" ", "_")
            if not src_label or not tgt_label:
                continue
            src_id = f"entity::{src_label.replace(' ', '_').lower()}"
            tgt_id = f"entity::{tgt_label.replace(' ', '_').lower()}"
            if not any(n["id"] == src_id for n in nodes):
                nodes.append({"id": src_id, "label": src_label, "type": "Entity",
                              "properties": {}, "source_chunk_id": chunk_id})
            if not any(n["id"] == tgt_id for n in nodes):
                nodes.append({"id": tgt_id, "label": tgt_label, "type": "Entity",
                              "properties": {}, "source_chunk_id": chunk_id})
            edges.append({"source_id": src_id, "target_id": tgt_id,
                          "relation": rel_name, "weight": 0.8,
                          "source_chunk_id": chunk_id})

    # Deduplicate nodes by id
    seen_ids: set[str] = set()
    unique_nodes = []
    for n in nodes:
        if n["id"] not in seen_ids:
            seen_ids.add(n["id"])
            unique_nodes.append(n)

    return unique_nodes, edges


# ── 10. One-line summary  (two-stage retrieval) ────────────────────────────

def generate_chunk_summary(chunk_text: str, model: str) -> str:
    prompt = (
        "Summarise the following text in ONE sentence of 20-30 words. "
        "Include key numbers or names if present.\n\n"
        f"Text:\n{chunk_text[:700]}\n\nOutput only the sentence."
    )
    return ollama_generate(prompt, model, system=SYSTEM_WRITER, temperature=0.2, max_tokens=80)


# ── 11. BM25 tokens  (keyword index) ──────────────────────────────────────

def build_bm25_tokens(text: str, extra: list[str]) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    all_tokens = words + [k.lower() for k in extra]
    filtered = [w for w in all_tokens if w not in DOMAIN_STOPWORDS and len(w) >= 2]
    seen: set[str] = set()
    return [t for t in filtered if not (seen.add(t) or t in seen - {t})]  # type: ignore[misc]


def _dedup_list(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ── 12. Normalized text  (fuzzy / trigram index) ───────────────────────────

def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ── 13. Quality score  ─────────────────────────────────────────────────────

def compute_quality_score(text: str, propositions: list[str]) -> float:
    words     = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    nums      = len(re.findall(r'\d+', text))
    return round(
        0.30 * min(len(words) / 200, 1.0) +
        0.25 * min(len(sentences) / 8, 1.0) +
        0.25 * min(len(propositions) / 6, 1.0) +
        0.20 * min(nums / 5, 1.0),
        3
    )


# =============================================================================
# Central chunk builder  — assembles ALL v3 fields
# =============================================================================

def build_chunk(
    *,
    text: str,
    category: str,
    source_section: str,
    metadata: dict,
    keywords: list[str],
    college: str,
    college_short: str,
    year: str,
    model: str,
    parent_id: str | None = None,
    run_enrichment: bool = True,
) -> tuple[dict | None, list[dict], list[dict]]:
    """
    Returns (chunk_dict, graph_nodes, graph_edges).
    graph_nodes/edges are empty if run_enrichment=False.
    """
    if len(text.split()) < MIN_CHUNK_WORDS:
        return None, [], []

    chunk_id = str(uuid.uuid4())
    doc_desc = f"{college} college data for {year}, section: {category}."

    if run_enrichment:
        log.debug("  Enriching: %s", source_section)
        context_header      = generate_context_header(text, doc_desc, model)
        propositions        = extract_propositions(text, model)
        hyp_questions       = generate_hypothetical_questions(text, category, model)
        colloquial_variants = generate_colloquial_variants(text, category, college, model)
        question_variants   = generate_question_variants(text, category, model)
        aliases             = generate_aliases(text, category, college, college_short, model)
        intent_tags         = assign_intent_tags(category, text, model)
        entities            = extract_entities(text, model)
        one_line_summary    = generate_chunk_summary(text, model)
        graph_nodes, graph_edges = build_graph_elements(
            chunk_id, text, entities, category, college, year, model
        )
    else:
        context_header      = ""
        propositions        = []
        hyp_questions       = []
        colloquial_variants = []
        question_variants   = []
        aliases             = [college, college_short]
        intent_tags         = CATEGORY_INTENTS.get(category, [])
        entities            = {}
        one_line_summary    = text[:120] + "..."
        graph_nodes, graph_edges = [], []

    contextualised_text = f"{context_header}\n\n{text}".strip() if context_header else text
    bm25_tokens = _dedup_list(build_bm25_tokens(text, keywords + aliases))
    quality     = compute_quality_score(text, propositions)

    chunk: dict = {
        # ── Identity ──────────────────────────────────────────────────────
        "id":               chunk_id,
        "parent_id":        parent_id,
        "category":         category,
        "source_section":   source_section,

        # ── Embedding instructions (NO vectors stored here) ────────────────
        # IMPORTANT — tell the embedder exactly what to embed:
        #   1. Embed `contextualised_text`           → primary semantic index
        #   2. Embed each item in `propositions`     → factoid index
        #   3. Embed each item in `hypothetical_questions` → HyDE index
        #   4. Embed each item in `colloquial_variants`    → query surface index
        #   5. Embed each item in `question_variants`      → multi-phrasing index
        # Use chunk `id` as the link key in all indexes.
        "_embed_fields": [
            "contextualised_text",
            "propositions",
            "hypothetical_questions",
            "colloquial_variants",
            "question_variants",
        ],

        # ── Core text fields ───────────────────────────────────────────────
        "text":                   text,               # human-readable
        "context_header":         context_header,     # Anthropic contextual blurb
        "contextualised_text":    contextualised_text, # EMBED THIS as primary
        "normalized_text":        normalize_text(text), # for fuzzy / trigram index

        # ── Retrieval surface fields  (THE ROBUSTNESS LAYER) ──────────────
        "aliases":             aliases,            # name variants → BM25 exact match
        "colloquial_variants": colloquial_variants, # Hinglish / broken English queries
        "question_variants":   question_variants,  # formal synonym rephrasings
        "hypothetical_questions": hyp_questions,   # HyDE embedding prep
        "propositions":        propositions,       # atomic facts, embed separately

        # ── Metadata & routing ─────────────────────────────────────────────
        "intent_tags":  intent_tags,   # for namespace/collection routing
        "keywords":     keywords,
        "bm25_tokens":  bm25_tokens,
        "entities":     entities,
        "metadata": {
            **metadata,
            "college":    college,
            "year":       year,
            "word_count": len(text.split()),
            "char_count": len(text),
        },

        # ── Summary (two-stage retrieval) ──────────────────────────────────
        "one_line_summary": one_line_summary,

        # ── Quality signal ─────────────────────────────────────────────────
        "quality_score":    quality,
        "low_quality_flag": quality < 0.4,
    }

    return chunk, graph_nodes, graph_edges


# =============================================================================
# Section processors  (identical logic to v2, now call new build_chunk)
# =============================================================================

def _bc(text, category, source_section, metadata, keywords, college, college_short,
        year, parent_id=None, run_enrichment=True):
    """Shorthand wrapper around build_chunk for use in section processors."""
    return build_chunk(
        text=text, category=category, source_section=source_section,
        metadata=metadata, keywords=keywords,
        college=college, college_short=college_short, year=year,
        model=_MODEL, parent_id=parent_id, run_enrichment=run_enrichment,
    )


def describe_programs(programs_sheet: dict, college: str, cs: str, year: str):
    chunks, nodes, edges = [], [], []
    label_map = {"btech": "B.Tech", "ug_other": "UG (Non-Engineering)", "mtech": "M.Tech"}
    for key, progs in programs_sheet.items():
        if not isinstance(progs, list) or not progs:
            continue
        label = label_map.get(key, key)
        rows = "\n".join(
            f"  - {p.get('program','?')} (Intake: {p.get('sanctioned_intake','?')}, "
            f"Since: {p.get('year_of_starting','?')})" for p in progs
        )
        prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"The following {label} programmes are offered:\n{rows}\n\n"
            "Write a 280-360 word paragraph about these programmes. Mention each intake "
            "size and year of establishment. Highlight newer specialisations like AI, "
            "IoT, Robotics if present."
        )
        text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
        if not text:
            continue
        c, n, e = _bc(text, "programs_offered", f"programs_offered/{key}",
                      {"label": label, "programs": [p.get("program") for p in progs],
                       "program_count": len(progs)},
                      ["programme", "course", label.lower(), "intake", "engineering",
                       "B.Tech", "M.Tech", "specialisation"], college, cs, year)
        if c:
            chunks.append(c); nodes.extend(n); edges.extend(e)
    return chunks, nodes, edges


def describe_tuition_fees(fee_list: list, college: str, cs: str, year: str):
    if not fee_list:
        return [], [], []
    rows = "\n".join(
        f"  {f.get('category','?')}: FY={fmt_inr(f.get('fy_fee'))} | DSE={fmt_inr(f.get('dse_fee'))}"
        for f in fee_list
    )
    prompt = (
        f"College: {college}  |  Academic Year: {year}\n\n"
        f"Annual tuition fee structure:\n{rows}\n\n"
        "Write 280-360 words explaining the fee structure for prospective students. "
        "Cover FY (First Year) and DSE (Direct Second Year) admissions. "
        "Briefly explain each reservation category (OPEN, EBC, OBC, SC/ST, TFWS, EWS). "
        "Highlight the highest and lowest fee."
    )
    text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
    if not text:
        return [], [], []
    c, n, e = _bc(text, "fees", "fy27_cutoff_bus_hostel_fees/tuition_fees",
                  {"year": year, "fee_rows": fee_list,
                   "categories": [f.get("category") for f in fee_list]},
                  ["tuition fee", "annual fee", "reservation", "OPEN", "SC", "ST",
                   "OBC", "EBC", "EWS", "TFWS", "fee structure", "FY", "DSE"],
                  college, cs, year)
    return ([c], n, e) if c else ([], [], [])


def describe_hostels(hostel_list: list, college: str, cs: str, year: str):
    if not hostel_list:
        return [], [], []
    rows = "\n".join(
        f"  {h.get('hostel','?')}: {fmt_inr(h.get('annual_fee','?'))} per year"
        for h in hostel_list
    )
    prompt = (
        f"College: {college}  |  Academic Year: {year}\n\nHostel options:\n{rows}\n\n"
        "Write 220-300 words about the hostel options. Mention each hostel's name, "
        "gender it serves if apparent, annual cost, and benefits for outstation students."
    )
    text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
    if not text:
        return [], [], []
    c, n, e = _bc(text, "hostel", "fy27_cutoff_bus_hostel_fees/hostels",
                  {"year": year, "hostels": hostel_list, "count": len(hostel_list)},
                  ["hostel", "accommodation", "ladies hostel", "boarding",
                   "annual fee", "campus stay", "residential"],
                  college, cs, year)
    return ([c], n, e) if c else ([], [], [])


def describe_bus_routes(bus_routes: dict, college: str, cs: str, year: str):
    all_chunks, all_nodes, all_edges = [], [], []
    route_chunks_for_parent = []
    all_route_lines = []

    for route_name, stops in bus_routes.items():
        if not stops:
            continue
        rows = "\n".join(
            f"  {s.get('stop','?')}: {fmt_inr(s.get('fee',0))} per month" for s in stops
        )
        all_route_lines.append(f"Route '{route_name}': {len(stops)} stops")
        prompt = (
            f"College: {college}  |  Route: {route_name}\n\nStops:\n{rows}\n\n"
            "Write 220-300 words for commuting students. Mention route name, key stops, "
            "monthly fee range (cheapest to costliest), and who this route serves."
        )
        text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
        if not text:
            continue
        c, n, e = _bc(text, "transport",
                      f"fy27_cutoff_bus_hostel_fees/bus_routes/{route_name}",
                      {"route": route_name, "stops": stops, "stop_count": len(stops)},
                      ["bus", "transport", "route", route_name, "commute", "monthly fee"],
                      college, cs, year)
        if c:
            route_chunks_for_parent.append(c)
            all_nodes.extend(n); all_edges.extend(e)

    if len(route_chunks_for_parent) > 1:
        ov_text = ollama_generate(
            f"College: {college}  |  Year: {year}\n\n"
            f"Bus routes:\n" + "\n".join(all_route_lines) + "\n\n"
            "Write 200-260 word overview of the bus transport network. "
            "Mention total routes, geographic coverage, distance-based monthly fees.",
            _MODEL, system=SYSTEM_WRITER
        )
        if ov_text:
            parent_id = str(uuid.uuid4())
            pc, pn, pe = _bc(ov_text, "transport",
                             "fy27_cutoff_bus_hostel_fees/bus_routes/overview",
                             {"is_parent": True, "route_count": len(bus_routes),
                              "routes": list(bus_routes.keys())},
                             ["bus routes", "transport overview", "commute", "college bus"],
                             college, cs, year, run_enrichment=False)
            if pc:
                pc["id"] = parent_id
                for rc in route_chunks_for_parent:
                    rc["parent_id"] = parent_id
                all_chunks.append(pc)
                all_nodes.extend(pn); all_edges.extend(pe)

    all_chunks.extend(route_chunks_for_parent)
    return all_chunks, all_nodes, all_edges


def describe_cutoffs(cutoff_list: list, college: str, cs: str, year: str):
    all_chunks, all_nodes, all_edges = [], [], []
    branch_chunks = []
    summary_lines = []

    for entry in cutoff_list:
        course  = entry.get("course", "Unknown")
        cutoffs = entry.get("cutoff", {})
        rows = []
        for cat, seats in cutoffs.items():
            g  = seats.get("G", {}) or {}
            l  = seats.get("L", {}) or {}
            gm = f"{g['merit_marks']:.2f}" if isinstance(g.get("merit_marks"), float) else "N/A"
            lm = f"{l['merit_marks']:.2f}" if isinstance(l.get("merit_marks"), float) else "N/A"
            rows.append(f"  {cat}: General={gm} marks | Ladies={lm} marks")
            if cat == "OPEN" and gm != "N/A":
                summary_lines.append(f"  {course}: OPEN General={gm} marks")
        if not rows:
            continue
        prompt = (
            f"College: {college}  |  Year: {year}\n\n"
            f"MHT-CET cutoff for {course} branch:\n" + "\n".join(rows) + "\n\n"
            "Write 300-400 words explaining these cutoffs. Clarify G (General) and L (Ladies) seats. "
            "Cover each category: VJ, NT-1/2/3, OBC, SEBC, OPEN, SC, ST."
        )
        text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
        if not text:
            continue
        c, n, e = _bc(text, "cutoffs",
                      f"fy27_cutoff_bus_hostel_fees/cutoffs/{course}",
                      {"course": course, "year": year, "categories": list(cutoffs.keys())},
                      ["cutoff", "merit", "MHT-CET", course, "admission", "marks",
                       "category", "OPEN", "OBC", "SC", "ST", "VJ", "NT", "SEBC"],
                      college, cs, year)
        if c:
            branch_chunks.append(c)
            all_nodes.extend(n); all_edges.extend(e)

    if len(branch_chunks) > 1 and summary_lines:
        ov_text = ollama_generate(
            f"College: {college}  |  Year: {year}\n\n"
            "OPEN category General seat cutoffs across branches:\n"
            + "\n".join(summary_lines) + "\n\n"
            "Write 260-340 word comparative overview. Highlight most competitive branches "
            "and more accessible ones to help students choose.",
            _MODEL, system=SYSTEM_WRITER
        )
        if ov_text:
            parent_id = str(uuid.uuid4())
            pc, pn, pe = _bc(ov_text, "cutoffs",
                             "fy27_cutoff_bus_hostel_fees/cutoffs/overview",
                             {"is_parent": True, "branches": [e.get("course") for e in cutoff_list]},
                             ["cutoff overview", "all branches", "MHT-CET", "comparison"],
                             college, cs, year, run_enrichment=False)
            if pc:
                pc["id"] = parent_id
                for bc in branch_chunks:
                    bc["parent_id"] = parent_id
                all_chunks.append(pc)
                all_nodes.extend(pn); all_edges.extend(pe)

    all_chunks.extend(branch_chunks)
    return all_chunks, all_nodes, all_edges


def describe_qualifying_criteria(criteria: Any, college: str, cs: str, year: str):
    if not criteria:
        return [], [], []
    raw = json.dumps(criteria, indent=2, ensure_ascii=False)
    prompt = (
        f"College: {college}  |  Year: {year}\n\n"
        f"12th-standard qualifying criteria:\n{raw}\n\n"
        "Write 220-300 words summarising minimum 12th-grade marks needed. "
        "Be specific about percentage thresholds per subject combination and category."
    )
    text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
    if not text:
        return [], [], []
    c, n, e = _bc(text, "eligibility",
                  "fy27_cutoff_bus_hostel_fees/qualifying_criteria",
                  {"year": year},
                  ["eligibility", "qualifying", "12th", "HSC", "minimum marks",
                   "percentage", "physics", "maths"],
                  college, cs, year)
    return ([c], n, e) if c else ([], [], [])


def describe_placement_summary(summary_sheet: dict, college: str, cs: str, year: str):
    all_chunks, all_nodes, all_edges = [], [], []
    by_branch = summary_sheet.get("placement_summary_by_branch", [])
    totals    = summary_sheet.get("overall_totals", {})
    companies = summary_sheet.get("companies_visited", {})

    total_lines = [
        f"  {yr}: {t.get('total_students','?')} students, {t.get('students_placed_offers','?')} offers, "
        f"avg {fmt_lpa(t.get('avg_salary_lpa'))}, highest {fmt_lpa(t.get('highest_salary_lpa'))}"
        for yr, t in totals.items()
    ]
    co_lines = [f"  {yr}: {cnt} companies" for yr, cnt in companies.items()]

    if total_lines:
        prompt = (
            f"College: {college}\n\nThree-year placement totals:\n"
            + "\n".join(total_lines) + "\n\nCompanies per year:\n"
            + "\n".join(co_lines) + "\n\n"
            "Write 300-400 words summarising 3-year overall placement performance. "
            "Discuss year-on-year trends in offers, packages, and company count."
        )
        text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
        if text:
            parent_id = str(uuid.uuid4())
            pc, pn, pe = _bc(text, "placements", "placement_3yr_summary/overall",
                             {"is_parent": True, "totals": totals, "companies": companies},
                             ["placement", "overall", "offers", "salary",
                              "companies", "LPA", "3-year", "trend"],
                             college, cs, year)
            if pc:
                pc["id"] = parent_id
                all_chunks.append(pc)
                all_nodes.extend(pn); all_edges.extend(pe)

                for bd in by_branch:
                    branch = bd.get("branch", "Unknown")
                    rows = [
                        f"  {yr}: {bd.get(yr,{}).get('total_students','?')} students, "
                        f"{bd.get(yr,{}).get('students_placed_offers','?')} offers, "
                        f"avg {fmt_lpa(bd.get(yr,{}).get('avg_salary_lpa'))}, "
                        f"highest {fmt_lpa(bd.get(yr,{}).get('highest_salary_lpa'))}"
                        for yr in ["2023-24", "2024-25", "2025-26"]
                        if bd.get(yr)
                    ]
                    if not rows:
                        continue
                    text_b = ollama_generate(
                        f"College: {college}  |  Branch: {branch}\n\n"
                        "3-year placement:\n" + "\n".join(rows) + "\n\n"
                        "Write 280-360 words. Discuss trend in avg and highest packages, "
                        "offer count changes, and industry demand signals.",
                        _MODEL, system=SYSTEM_WRITER
                    )
                    if not text_b:
                        continue
                    c, n, e = _bc(text_b, "placements",
                                  f"placement_3yr_summary/{branch}",
                                  {"branch": branch},
                                  ["placement", branch, "package", "LPA", "offers",
                                   "salary", "three year"],
                                  college, cs, year, parent_id=parent_id)
                    if c:
                        all_chunks.append(c)
                        all_nodes.extend(n); all_edges.extend(e)

    return all_chunks, all_nodes, all_edges


def describe_companies(companies_sheet: dict, college: str, cs: str, year: str):
    all_chunks, all_nodes, all_edges = [], [], []
    for yr, companies in companies_sheet.items():
        if not companies:
            continue
        it_cos   = [c["company_name"] for c in companies if "IT" in str(c.get("industry_vertical","")).upper()]
        core_cos = [c["company_name"] for c in companies if "Core" in str(c.get("industry_vertical",""))]
        all_names = [c["company_name"] for c in companies]
        dist_str  = ", ".join(f"{v}:{cnt}" for v, cnt in Counter(c.get("industry_vertical","Other") for c in companies).most_common())
        prompt = (
            f"College: {college}  |  Year: {yr}\n\n"
            f"Total companies: {len(companies)}\nIndustry mix: {dist_str}\n"
            f"IT companies: {', '.join(it_cos[:15]) or 'none'}\n"
            f"Core/Mfg: {', '.join(core_cos[:15]) or 'none'}\n"
            f"Notable: {', '.join(all_names[:40])}\n\n"
            "Write 300-380 words about campus recruitment. Cover total count, "
            "industry mix, well-known recruiters, and what it says about the college's industry connections."
        )
        text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
        if not text:
            continue
        c, n, e = _bc(text, "company_visits", f"company_visited_lists/{yr}",
                      {"year": yr, "total_companies": len(companies),
                       "it_count": len(it_cos), "core_count": len(core_cos),
                       "companies": all_names},
                      ["companies", yr, "campus recruitment", "IT", "core",
                       "placement drive", "recruiter"],
                      college, cs, year)
        if c:
            all_chunks.append(c)
            all_nodes.extend(n); all_edges.extend(e)
    return all_chunks, all_nodes, all_edges


def describe_placement_detail(detail_sheet: dict, college: str, cs: str, year: str):
    all_chunks, all_nodes, all_edges = [], [], []
    for yr in ["2025-26", "2024-25", "2023-24"]:
        branch_list = detail_sheet.get(yr, [])
        summary     = detail_sheet.get(f"{yr}_summary", {})
        if not branch_list:
            continue
        rows = "\n".join(
            f"  {b.get('branch','?')}: {b.get('total_students','?')} students, "
            f"{b.get('students_placed_offers','?')} offers, "
            f"avg {fmt_lpa(b.get('avg_salary_lpa'))}, highest {fmt_lpa(b.get('highest_salary_lpa'))}"
            for b in branch_list
        )
        summ = ""
        if summary:
            ot = summary.get("overall_total", {})
            summ = f"\nTotal offers: {ot.get('students_placed_offers','?')}, Companies: {summary.get('companies_visited','?')}"
        prompt = (
            f"College: {college}  |  Year: {yr}\n\nBranch-wise placements:\n{rows}{summ}\n\n"
            "Write 350-450 words. Cover every branch, compare by package and offer count, "
            "highlight top branch and standout packages."
        )
        text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
        if not text:
            continue
        c, n, e = _bc(text, "placements", f"placement_by_year_detail/{yr}",
                      {"year": yr, "branches": [b.get("branch") for b in branch_list],
                       "companies_visited": summary.get("companies_visited") if summary else None},
                      ["placement", yr, "branch", "package", "LPA", "offers", "detailed"],
                      college, cs, year)
        if c:
            all_chunks.append(c)
            all_nodes.extend(n); all_edges.extend(e)
    return all_chunks, all_nodes, all_edges


def describe_admission_docs(docs_sheet: dict, college: str, cs: str, year: str):
    all_chunks, all_nodes, all_edges = [], [], []
    for cat_label, doc_list in docs_sheet.items():
        if not isinstance(doc_list, list) or not doc_list:
            continue
        docs = [d.get("document","") for d in doc_list if d.get("document")]
        prompt = (
            f"College: {college}  |  Year: {year}\n\n"
            f"Admission type: {cat_label}\nRequired documents: {', '.join(docs)}\n\n"
            "Write 220-300 words listing and explaining each document. Describe why "
            "each is needed and whether requirements differ by reservation category."
        )
        text = ollama_generate(prompt, _MODEL, system=SYSTEM_WRITER)
        if not text:
            continue
        c, n, e = _bc(text, "admission_documents", f"admission_documents/{cat_label[:50]}",
                      {"admission_type": cat_label, "documents": docs, "doc_count": len(docs)},
                      ["documents", "admission", "scholarship", "marksheet", "certificate"],
                      college, cs, year)
        if c:
            all_chunks.append(c)
            all_nodes.extend(n); all_edges.extend(e)
    return all_chunks, all_nodes, all_edges


# =============================================================================
# Cross-chunk relationship builder  (graph traversal links)
# =============================================================================

def build_cross_chunk_relations(chunks: list[dict]) -> None:
    """
    Injects `related_section_ids` into each chunk so a graph traversal can
    hop from (e.g.) 'CSE cutoffs' → 'CSE placement' → 'companies 2025-26'
    without re-querying the vector index.

    Strategy: two chunks are "related" if they share the same branch entity
    OR if they are in parent-child relationship OR if they are adjacent
    sections in the same category.
    """
    # Build branch → chunk_ids mapping
    branch_map: dict[str, list[str]] = {}
    for chunk in chunks:
        for branch in chunk.get("entities", {}).get("branches", []):
            branch_map.setdefault(branch, []).append(chunk["id"])

    # Build parent → children mapping
    parent_map: dict[str, list[str]] = {}
    for chunk in chunks:
        pid = chunk.get("parent_id")
        if pid:
            parent_map.setdefault(pid, []).append(chunk["id"])

    for chunk in chunks:
        related: set[str] = set()

        # Same branch
        for branch in chunk.get("entities", {}).get("branches", []):
            for cid in branch_map.get(branch, []):
                if cid != chunk["id"]:
                    related.add(cid)

        # Parent ↔ children
        pid = chunk.get("parent_id")
        if pid:
            related.add(pid)
            for sibling_id in parent_map.get(pid, []):
                if sibling_id != chunk["id"]:
                    related.add(sibling_id)

        # Children of this chunk
        for child_id in parent_map.get(chunk["id"], []):
            related.add(child_id)

        chunk["related_section_ids"] = list(related)[:10]


# =============================================================================
# Q&A generation  (4 types per chunk)
# =============================================================================

def generate_qa_pairs(chunks: list[dict], college: str, model: str) -> list[dict]:
    """
    4 types per chunk: factual / comparative / advice / eligibility.
    Plus colloquial_q: the same answer, reachable via broken-English question.
    """
    qa_pairs: list[dict] = []
    log.info("Generating Q&A pairs from %d chunks...", len(chunks))

    QA_SYS = (
        "You are an expert at creating diverse RAG evaluation questions for a college "
        "information chatbot. Questions must be realistic and answerable solely from "
        "the provided knowledge text. Return ONLY valid JSON."
    )

    for chunk in tqdm(chunks, desc="QA generation", unit="chunk"):
        text     = chunk["text"]
        category = chunk["category"]

        prompt = (
            f"College: {college} | Category: {category}\n\n"
            f"Knowledge:\n{text[:900]}\n\n"
            "Generate 5 question-answer pairs:\n"
            '  {"type":"factual",     "question":"...","answer":"..."}  // specific number/name\n'
            '  {"type":"comparative", "question":"...","answer":"..."}  // compares 2 items\n'
            '  {"type":"advice",      "question":"...","answer":"..."}  // what should I do\n'
            '  {"type":"eligibility", "question":"...","answer":"..."}  // do I qualify\n'
            '  {"type":"colloquial",  "question":"...","answer":"..."}  // broken/informal English\n\n'
            "Each answer: 1-3 sentences, answerable from the text only.\n"
            "Output as JSON array of 5 objects."
        )
        raw = ollama_generate(prompt, model, system=QA_SYS, temperature=0.4, max_tokens=800)
        if not raw:
            continue
        pairs = safe_json_parse(raw)
        if not isinstance(pairs, list):
            continue
        for pair in pairs[:5]:
            if not isinstance(pair, dict):
                continue
            q = str(pair.get("question","")).strip()
            a = str(pair.get("answer","")).strip()
            if not q or not a:
                continue
            qa_pairs.append({
                "id":              str(uuid.uuid4()),
                "source_chunk_id": chunk["id"],
                "category":        category,
                "source_section":  chunk["source_section"],
                "question_type":   pair.get("type", "factual"),
                "question":        q,
                "answer":          a,
                "keywords":        chunk["keywords"],
                "college":         college,
            })
    return qa_pairs


# =============================================================================
# Output writers
# =============================================================================

def write_jsonl(records: list[dict], path: Path, label: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %-28s → %s  (%d records)", label, path, len(records))


def write_json(data: Any, path: Path, label: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("Saved %-28s → %s", label, path)


def write_query_surfaces(chunks: list[dict], path: Path) -> None:
    """
    Separate lightweight index of colloquial variant queries.
    Each record: { id, parent_chunk_id, variant_text, category, intent_tags }
    Embed `variant_text` into a parallel 'query_surfaces' collection.
    When a user query matches here, retrieve the parent chunk from knowledge_base.
    """
    records = []
    for chunk in chunks:
        for variant in chunk.get("colloquial_variants", []):
            if not variant.strip():
                continue
            records.append({
                "id":              str(uuid.uuid4()),
                "parent_chunk_id": chunk["id"],
                "variant_text":    variant,
                "category":        chunk["category"],
                "intent_tags":     chunk["intent_tags"],
                "source_section":  chunk["source_section"],
            })
        for variant in chunk.get("question_variants", []):
            if not variant.strip():
                continue
            records.append({
                "id":              str(uuid.uuid4()),
                "parent_chunk_id": chunk["id"],
                "variant_text":    variant,
                "category":        chunk["category"],
                "intent_tags":     chunk["intent_tags"],
                "source_section":  chunk["source_section"],
            })
    write_jsonl(records, path, "query_surfaces.jsonl")


def write_propositions(chunks: list[dict], path: Path) -> None:
    records = []
    for chunk in chunks:
        for prop in chunk.get("propositions", []):
            if not prop.strip():
                continue
            records.append({
                "id":              str(uuid.uuid4()),
                "parent_chunk_id": chunk["id"],
                "category":        chunk["category"],
                "source_section":  chunk["source_section"],
                "text":            prop,
                "keywords":        chunk["keywords"],
                "metadata":        chunk["metadata"],
            })
    write_jsonl(records, path, "propositions.jsonl")


def write_summary_index(chunks: list[dict], path: Path) -> None:
    index = [{
        "chunk_id":       c["id"],
        "parent_id":      c.get("parent_id"),
        "category":       c["category"],
        "intent_tags":    c.get("intent_tags", []),
        "source_section": c["source_section"],
        "summary":        c.get("one_line_summary") or c["text"][:120] + "...",
        "keywords":       c["keywords"][:8],
        "quality_score":  c.get("quality_score", 0),
        "low_quality":    c.get("low_quality_flag", False),
    } for c in chunks]
    write_json(index, path, "summary_index.json")


# =============================================================================
# Main pipeline
# =============================================================================

def build_knowledge_base(input_path: str, model: str, skip_qa: bool = False) -> None:
    global _MODEL
    _MODEL = model

    log.info("Loading JSON: %s", input_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    college = data.get("college", "College")
    year    = data.get("academic_year", "2025-26")
    sheets  = data.get("sheets", {})

    # Derive short name (e.g. "ADCET" from "Annasaheb Dange College of Engineering and Technology")
    words = college.split()
    college_short = "".join(w[0].upper() for w in words if w[0].isupper()) or college[:8]

    log.info("College       : %s  (%s)", college, college_short)
    log.info("Academic Year : %s", year)
    log.info("Sheets found  : %s", list(sheets.keys()))

    check_ollama(model)
    log.info("Ollama model '%s' confirmed.", model)

    all_chunks: list[dict]     = []
    all_graph_nodes: list[dict] = []
    all_graph_edges: list[dict] = []

    cutoff_sheet = sheets.get("fy27_cutoff_bus_hostel_fees", {})

    section_processors = [
        ("Programs offered",      lambda: describe_programs(
            sheets.get("programs_offered", {}), college, college_short, year)),
        ("Tuition fees",          lambda: describe_tuition_fees(
            cutoff_sheet.get("tuition_fees_2025_26", []), college, college_short, year)),
        ("Hostel fees",           lambda: describe_hostels(
            cutoff_sheet.get("hostels", []), college, college_short, year)),
        ("Bus routes",            lambda: describe_bus_routes(
            cutoff_sheet.get("bus_routes", {}), college, college_short, year)),
        ("Admission cutoffs",     lambda: describe_cutoffs(
            cutoff_sheet.get("cutoff_data", []), college, college_short, year)),
        ("Qualifying criteria",   lambda: describe_qualifying_criteria(
            cutoff_sheet.get("qualifying_criteria_12th"), college, college_short, year)),
        ("Placement 3yr summary", lambda: describe_placement_summary(
            sheets.get("placement_3yr_summary", {}), college, college_short, year)),
        ("Companies visited",     lambda: describe_companies(
            sheets.get("company_visited_lists", {}), college, college_short, year)),
        ("Placement year detail", lambda: describe_placement_detail(
            sheets.get("placement_by_year_detail", {}), college, college_short, year)),
        ("Admission documents",   lambda: describe_admission_docs(
            sheets.get("admission_documents", {}), college, college_short, year)),
    ]

    for section_name, processor in tqdm(section_processors, desc="Sections", unit="section"):
        log.info("Processing: %s", section_name)
        try:
            chunks, nodes, edges = processor()
            all_chunks.extend(chunks)
            all_graph_nodes.extend(nodes)
            all_graph_edges.extend(edges)
            lq = sum(1 for c in chunks if c.get("low_quality_flag"))
            log.info("  → %d chunks  (low-quality: %d)  %d KG nodes  %d KG edges",
                     len(chunks), lq, len(nodes), len(edges))
        except Exception as exc:
            log.warning("  Section '%s' failed: %s", section_name, exc)

    log.info("Building cross-chunk relationships...")
    build_cross_chunk_relations(all_chunks)

    # Deduplicate KG nodes by id (keep first occurrence)
    seen_nids: set[str] = set()
    unique_nodes = [n for n in all_graph_nodes if not (n["id"] in seen_nids or seen_nids.add(n["id"]))]

    log.info("Total: %d chunks  |  %d KG nodes  |  %d KG edges",
             len(all_chunks), len(unique_nodes), len(all_graph_edges))

    # ── Write all outputs ──────────────────────────────────────────────────
    write_jsonl(all_chunks,   Path("knowledge_base.jsonl"),  "knowledge_base.jsonl")
    write_propositions(all_chunks, Path("propositions.jsonl"))
    write_query_surfaces(all_chunks, Path("query_surfaces.jsonl"))
    write_summary_index(all_chunks, Path("summary_index.json"))
    write_json(unique_nodes,    Path("graph_nodes.json"),    "graph_nodes.json")
    write_json(all_graph_edges, Path("graph_edges.json"),    "graph_edges.json")

    if not skip_qa:
        qa_pairs = generate_qa_pairs(all_chunks, college, model)
        write_json(qa_pairs, Path("qa_pairs.json"), "qa_pairs.json")
    else:
        log.info("Q&A generation skipped (--skip-qa)")
        qa_pairs = []

    # ── Summary stats ──────────────────────────────────────────────────────
    cat_counts: dict[str, int] = {}
    prop_total = coll_total = qv_total = lq_total = 0
    for c in all_chunks:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
        prop_total  += len(c.get("propositions", []))
        coll_total  += len(c.get("colloquial_variants", []))
        qv_total    += len(c.get("question_variants", []))
        lq_total    += int(c.get("low_quality_flag", False))

    avg_q = sum(c.get("quality_score", 0) for c in all_chunks) / max(len(all_chunks), 1)

    print("\n" + "═" * 68)
    print(f"  Knowledge Base · {college}  [{year}]")
    print("═" * 68)
    print(f"  {'Category':<36} {'Chunks':>6}")
    print("  " + "─" * 44)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat:<36} {cnt:>6}")
    print("  " + "─" * 44)
    print(f"  {'TOTAL chunks':<36} {len(all_chunks):>6}")
    print(f"  {'Propositions (factoid index)':<36} {prop_total:>6}")
    print(f"  {'Colloquial query variants':<36} {coll_total:>6}")
    print(f"  {'Formal question variants':<36} {qv_total:>6}")
    print(f"  {'KG nodes':<36} {len(unique_nodes):>6}")
    print(f"  {'KG edges':<36} {len(all_graph_edges):>6}")
    print(f"  {'Q&A pairs':<36} {len(qa_pairs):>6}")
    print(f"  {'Low-quality chunks flagged':<36} {lq_total:>6}")
    print(f"  {'Average quality score':<36} {avg_q:>6.3f}")
    print("═" * 68)
    print("  Output files:")
    print("    knowledge_base.jsonl   primary chunks  → embed 'contextualised_text'")
    print("    propositions.jsonl     atomic facts    → embed 'text'  (factoid index)")
    print("    query_surfaces.jsonl   colloquial+formal variants → embed 'variant_text'")
    print("    graph_nodes.json       KG nodes        → load into Neo4j / NetworkX")
    print("    graph_edges.json       KG edges        → load into Neo4j / NetworkX")
    print("    summary_index.json     summaries       → two-stage retrieval Stage 1")
    print("    qa_pairs.json          Q&A pairs       → eval / fine-tuning")
    print("═" * 68)
    print()
    print("  EMBEDDING GUIDE (check '_embed_fields' key in each chunk):")
    print("  1. Embed 'contextualised_text'       → primary vector store")
    print("  2. Embed each 'propositions' item    → factoid vector store")
    print("  3. Embed each 'colloquial_variants'  → query_surfaces collection")
    print("  4. Embed each 'question_variants'    → query_surfaces collection")
    print("  5. Embed each 'hypothetical_questions' → HyDE aux index (optional)")
    print()
    print("  GRAPH USAGE:")
    print("  Load graph_nodes.json + graph_edges.json into Neo4j:")
    print("    MERGE (n:Node {id: row.id}) SET n += row.properties")
    print("  Or use NetworkX for local traversal.")
    print()


# =============================================================================
# Entry point
# =============================================================================

_MODEL = DEFAULT_MODEL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert college JSON → robustness-first RAG knowledge base (v3)"
    )
    parser.add_argument("--input",    default="output.json",  help="Input JSON path")
    parser.add_argument("--model",    default=DEFAULT_MODEL,  help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--skip-qa",  action="store_true",    help="Skip Q&A generation (~40%% faster)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    build_knowledge_base(args.input, args.model, skip_qa=args.skip_qa)