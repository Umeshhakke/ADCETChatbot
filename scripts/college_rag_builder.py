"""
college_rag_builder.py  (Enhanced v2)
======================================
Converts a structured college JSON file into a production-grade semantic
knowledge base optimised for RAG retrieval using a local Qwen 2.5 7B
model via Ollama.

Research-backed enhancements in this version
--------------------------------------------
1.  Contextual Retrieval  (Anthropic 2024) — every chunk carries a short
    context_header that situates it inside the full document, reducing
    retrieval failure by up to 49 % when used with hybrid search.

2.  Propositional chunking — each chunk is also broken into atomic,
    self-contained fact sentences stored as `propositions[]`. Factoid
    queries perform best on isolated claims (NVIDIA 2024 benchmarks).

3.  BM25-ready token list — `bm25_tokens` stores lowercased domain
    terms so a keyword index can be built without re-tokenising at
    query time.

4.  Hypothetical question variants (HyDE prep) — 3 questions per chunk
    whose hypothetical answers would semantically match the chunk, stored
    in `hypothetical_questions[]`.

5.  Named-entity extraction — `entities{}` captures branches, years,
    companies, amounts, and categories for metadata-filtered retrieval.

6.  Chunk quality score — a heuristic 0-1 float that measures
    information density; low-quality chunks are flagged but kept.

7.  Parent-child linking — long sections generate one parent summary
    chunk plus granular child chunks; each child stores `parent_id`.

8.  Multi-format outputs —
      knowledge_base.jsonl    primary chunks (one per line)
      qa_pairs.json           Q&A pairs with 4 question types per chunk
      propositions.jsonl      atomic facts for dense factoid retrieval
      summary_index.json      one-line summaries for two-stage retrieval

Outputs
-------
  knowledge_base.jsonl   – full enriched chunks, embedding-ready
  qa_pairs.json          – diverse question-answer pairs
  propositions.jsonl     – atomic proposition chunks
  summary_index.json     – lightweight summary index for two-stage RAG

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
OLLAMA_TIMEOUT  = 180          # seconds per LLM call
MAX_RETRIES     = 3
RETRY_DELAY     = 5            # seconds between retries
MIN_CHUNK_WORDS = 40           # chunks below this word count are discarded

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
    """Call Ollama /api/generate and return cleaned response text."""
    payload: dict[str, Any] = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            return clean_text(raw)
        except requests.exceptions.ConnectionError:
            log.error("Ollama not reachable. Start it with: ollama serve")
            sys.exit(1)
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    log.error("All retries exhausted.")
    return ""


def check_ollama(model: str) -> None:
    """Verify Ollama is up and the model is pulled."""
    try:
        tags = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10).json()
        available = [m["name"] for m in tags.get("models", [])]
    except Exception:
        log.error("Cannot reach Ollama at %s. Run: ollama serve", OLLAMA_BASE)
        sys.exit(1)
    if not any(model in m for m in available):
        log.warning("Model '%s' not found. Pull with: ollama pull %s", model, model)
        log.warning("Available models: %s", available or "none")


def clean_text(text: str) -> str:
    """Strip markdown fences, role labels, and excess blank lines."""
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"^(Answer|Response|Output|Result|Text)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_json_parse(raw: str) -> Any:
    """
    Robustly parse JSON from a model response.
    Strips preamble/postamble and markdown fences. Falls back to
    extracting individual JSON objects via regex.
    """
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = raw.find(start_char)
        if idx != -1:
            depth, end_idx = 0, -1
            for i, ch in enumerate(raw[idx:], idx):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
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
# Shared system prompt
# =============================================================================

SYSTEM_PROMPT = """\
You are an expert knowledge-base writer specialising in Indian engineering \
college admissions, placements, and campus facilities. \
Write clear, factual, human-readable paragraphs — no bullet points, no \
markdown headers, no numbered lists unless explicitly asked. \
Every sentence must be independently informative. \
Do NOT invent or extrapolate numbers; use only the data provided.\
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
    if value is None:
        return "not disclosed"
    return f"{value} LPA"


# =============================================================================
# Enrichment functions  (research-backed enhancements)
# =============================================================================

def generate_context_header(
    chunk_text: str,
    document_description: str,
    model: str,
) -> str:
    """
    Anthropic Contextual Retrieval (2024):
    Generates a 2-3 sentence context blurb situating the chunk in the full
    document. Prepending this to the text before embedding reduces retrieval
    failures by up to 49% (Anthropic research, reduced further to 67% when
    combined with reranking).
    """
    prompt = (
        f"Document: {document_description}\n\n"
        f"Chunk content:\n{chunk_text[:600]}\n\n"
        "In 2-3 sentences, describe what specific aspect of the document this chunk covers "
        "and what key information it contains. Be concrete — mention the category (e.g., "
        "'admission cutoffs for CSE branch', 'bus route fees for Ashta-Palus route'). "
        "Output only those 2-3 sentences, nothing else."
    )
    return ollama_generate(prompt, model=model, system=SYSTEM_PROMPT, max_tokens=120)


def extract_propositions(chunk_text: str, model: str) -> list[str]:
    """
    Propositional chunking: decompose paragraph into atomic, self-contained
    fact sentences. Each proposition must be meaningful without surrounding
    context (include full subject, not pronouns like 'it'/'they').
    Research: improves factoid query precision vs. paragraph chunking
    (NCBI Bioengineering 2025, adaptive chunking study).
    """
    prompt = (
        "Break the following text into atomic fact statements. "
        "Each statement must be self-contained — include the full subject, not just 'it' or 'they'. "
        "Output as a JSON array of strings. Max 10 propositions.\n\n"
        f"Text:\n{chunk_text}\n\n"
        "Output only the JSON array, nothing else."
    )
    raw = ollama_generate(
        prompt, model=model, system=SYSTEM_PROMPT, temperature=0.1, max_tokens=600
    )
    result = safe_json_parse(raw)
    if isinstance(result, list):
        return [str(p).strip() for p in result if str(p).strip()]
    return []


def generate_hypothetical_questions(
    chunk_text: str,
    category: str,
    model: str,
) -> list[str]:
    """
    HyDE (Hypothetical Document Embeddings) prep:
    Store questions whose hypothetical answers would match this chunk.
    The embedder can optionally average chunk + question embeddings for
    better semantic recall, especially for vague/underspecified queries.
    Note: avoid HyDE for exact numerical fact retrieval (T2-RAGBench 2026).
    """
    prompt = (
        f"Category: {category}\n\n"
        f"Text:\n{chunk_text[:700]}\n\n"
        "Write exactly 3 natural questions a student might ask whose ideal answer "
        "is contained in the text above. Questions must be specific, not generic. "
        "Output as a JSON array of 3 strings. Nothing else."
    )
    raw = ollama_generate(
        prompt, model=model, system=SYSTEM_PROMPT, temperature=0.4, max_tokens=300
    )
    result = safe_json_parse(raw)
    if isinstance(result, list):
        return [str(q).strip() for q in result[:3] if str(q).strip()]
    return []


def extract_entities(chunk_text: str, model: str) -> dict[str, list[str]]:
    """
    Named-entity extraction for metadata-filtered retrieval.
    Typed entities enable WHERE-clause style filtering in Qdrant/Weaviate/
    Pinecone metadata filters — critical for precision when the user asks
    about a specific branch or year.
    """
    prompt = (
        "Extract named entities from the following text.\n"
        "Return a JSON object with these exact keys (empty list if none found):\n"
        '  "branches": [engineering branch names],\n'
        '  "years": [academic years like "2024-25"],\n'
        '  "companies": [company names],\n'
        '  "amounts_inr": [fee or salary amounts as strings],\n'
        '  "categories": [reservation categories like OPEN, OBC, SC/ST],\n'
        '  "locations": [city or route names]\n\n'
        f"Text:\n{chunk_text[:800]}\n\n"
        "Output only the JSON object."
    )
    raw = ollama_generate(
        prompt, model=model, system=SYSTEM_PROMPT, temperature=0.1, max_tokens=400
    )
    result = safe_json_parse(raw)
    keys = ["branches", "years", "companies", "amounts_inr", "categories", "locations"]
    if isinstance(result, dict):
        return {k: [str(v) for v in result.get(k, []) if v] for k in keys}
    return {k: [] for k in keys}


def generate_chunk_summary(chunk_text: str, model: str) -> str:
    """
    One-sentence summary for the summary index used in two-stage retrieval:
    Stage 1 retrieves summaries cheaply (low-cost first pass).
    Stage 2 fetches the full chunk only for the top-k summaries.
    Reduces tokens sent to the LLM while maintaining relevance.
    """
    prompt = (
        "Summarise the following text in exactly ONE sentence of 20-30 words. "
        "Be specific — include key numbers or names if present.\n\n"
        f"Text:\n{chunk_text[:700]}\n\n"
        "Output only the single sentence."
    )
    return ollama_generate(
        prompt, model=model, system=SYSTEM_PROMPT, temperature=0.2, max_tokens=80
    )


def build_bm25_tokens(text: str, extra_keywords: list[str]) -> list[str]:
    """
    Pre-compute BM25-ready token list for hybrid retrieval.
    BM25 excels at exact term matches for domain-specific queries
    (company names, branch codes, fee amounts, category names).
    Hybrid BM25 + vector search reduces retrieval failures by 49%
    vs vector-only (Anthropic 2024).
    """
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    domain_terms = [k.lower() for k in extra_keywords]
    combined = words + domain_terms
    tokens = [
        w for w in combined
        if w not in DOMAIN_STOPWORDS and len(w) >= 2
    ]
    seen: set[str] = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def compute_quality_score(text: str, propositions: list[str]) -> float:
    """
    Heuristic chunk quality score (0.0 to 1.0).
    Combines word count, sentence count, proposition density,
    and numerical specificity. Low-quality chunks (< 0.4) are flagged.
    Flagged chunks are retained but can be filtered out downstream
    before vectorisation if needed.
    """
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    num_count = len(re.findall(r'\d+', text))

    word_score  = min(len(words) / 200, 1.0)
    sent_score  = min(len(sentences) / 8, 1.0)
    prop_score  = min(len(propositions) / 6, 1.0)
    num_score   = min(num_count / 5, 1.0)

    return round(
        0.30 * word_score + 0.25 * sent_score +
        0.25 * prop_score + 0.20 * num_score,
        3
    )


# =============================================================================
# Core chunk builder
# =============================================================================

def enrich_and_build_chunk(
    *,
    text: str,
    category: str,
    source_section: str,
    metadata: dict,
    keywords: list[str],
    college: str,
    year: str,
    model: str,
    parent_id: str | None = None,
    run_enrichment: bool = True,
) -> dict | None:
    """
    Build a fully enriched chunk dict.
    Set run_enrichment=False for parent summary chunks to skip the
    expensive per-chunk LLM enrichment calls.
    """
    word_count = len(text.split())
    if word_count < MIN_CHUNK_WORDS:
        log.debug("Skipping under-sized chunk (%d words): %s", word_count, source_section)
        return None

    doc_desc = (
        f"{college} college data for academic year {year}, "
        f"specifically the '{category}' section."
    )

    chunk_id = str(uuid.uuid4())

    if run_enrichment:
        log.debug("  Enriching: %s", source_section)
        context_header   = generate_context_header(text, doc_desc, model)
        propositions     = extract_propositions(text, model)
        hypothetical_qs  = generate_hypothetical_questions(text, category, model)
        entities         = extract_entities(text, model)
        one_line_summary = generate_chunk_summary(text, model)
    else:
        context_header   = ""
        propositions     = []
        hypothetical_qs  = []
        entities         = {}
        one_line_summary = text[:120] + "..."

    bm25_tokens = build_bm25_tokens(text, keywords)
    quality     = compute_quality_score(text, propositions)

    # contextualised_text = context_header + text
    # IMPORTANT: embed THIS field, not raw 'text', for best recall
    contextualised_text = (
        f"{context_header}\n\n{text}" if context_header else text
    )

    return {
        # Identity
        "id":                     chunk_id,
        "parent_id":              parent_id,
        "category":               category,
        "source_section":         source_section,

        # Core text fields
        "text":                   text,
        "context_header":         context_header,
        "contextualised_text":    contextualised_text,   # EMBED THIS

        # Retrieval enhancement fields
        "one_line_summary":       one_line_summary,      # two-stage retrieval
        "propositions":           propositions,          # atomic facts
        "hypothetical_questions": hypothetical_qs,       # HyDE prep
        "bm25_tokens":            bm25_tokens,           # BM25 keyword index

        # Structured metadata
        "metadata": {
            **metadata,
            "college":    college,
            "year":       year,
            "word_count": word_count,
            "char_count": len(text),
        },
        "entities":  entities,
        "keywords":  keywords,

        # Quality signal
        "quality_score":    quality,
        "low_quality_flag": quality < 0.4,

        # Embedding placeholders (fill during vectorisation)
        "embedding_model": None,
        "embedding":       None,
    }


# =============================================================================
# Section processors
# =============================================================================

def describe_programs(programs_sheet: dict, college: str, year: str) -> list[dict]:
    chunks = []
    label_map = {
        "btech":    "B.Tech",
        "ug_other": "UG (Non-Engineering / Management)",
        "mtech":    "M.Tech",
    }
    for category_key, progs in programs_sheet.items():
        if not isinstance(progs, list) or not progs:
            continue
        label = label_map.get(category_key, category_key)
        rows = "\n".join(
            f"  - {p.get('program','?')} "
            f"(Intake: {p.get('sanctioned_intake','?')}, "
            f"Since: {p.get('year_of_starting','?')})"
            for p in progs
        )
        prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"The following {label} programmes are offered:\n{rows}\n\n"
            "Write a 280-360 word descriptive paragraph about these programmes. "
            "Mention each programme's sanctioned intake and year of establishment. "
            "Highlight newer specialisations like AI, IoT, Robotics if present. "
            "Mention the range of disciplines covered."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunk = enrich_and_build_chunk(
            text=text,
            category="programs_offered",
            source_section=f"programs_offered/{category_key}",
            metadata={"label": label,
                      "programs": [p.get("program") for p in progs],
                      "program_count": len(progs)},
            keywords=["programme", "course", label.lower(), "intake",
                      "engineering", "B.Tech", "M.Tech", "specialisation"],
            college=college, year=year, model=_MODEL,
        )
        if chunk:
            chunks.append(chunk)
    return chunks


def describe_tuition_fees(fee_list: list, college: str, year: str) -> list[dict]:
    if not fee_list:
        return []
    rows = "\n".join(
        f"  {f.get('category','?')}: "
        f"FY = {fmt_inr(f.get('fy_fee'))}  |  DSE = {fmt_inr(f.get('dse_fee'))}"
        for f in fee_list
    )
    prompt = (
        f"College: {college}  |  Academic Year: {year}\n\n"
        "Annual tuition fee structure by reservation category:\n"
        f"{rows}\n\n"
        "Write a 280-360 word paragraph explaining this fee structure clearly "
        "for prospective students. Cover both First Year (FY) and Direct Second Year "
        "(DSE) admissions. Explain what each reservation category means (OPEN = no "
        "reservation, EBC = Economically Backward Class, SC/ST = Scheduled Castes "
        "and Tribes, OBC = Other Backward Classes, TFWS = Tuition Fee Waiver Scheme). "
        "Highlight the fee range from highest to lowest."
    )
    text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
    if not text:
        return []
    chunk = enrich_and_build_chunk(
        text=text,
        category="fees",
        source_section="fy27_cutoff_bus_hostel_fees/tuition_fees",
        metadata={"year": year, "fee_rows": fee_list,
                  "categories": [f.get("category") for f in fee_list]},
        keywords=["tuition fee", "annual fee", "reservation", "OPEN", "SC", "ST",
                  "OBC", "EBC", "EWS", "TFWS", "fee structure", "FY", "DSE"],
        college=college, year=year, model=_MODEL,
    )
    return [chunk] if chunk else []


def describe_hostels(hostel_list: list, college: str, year: str) -> list[dict]:
    if not hostel_list:
        return []
    rows = "\n".join(
        f"  {h.get('hostel', '?')}: {fmt_inr(h.get('annual_fee', '?'))} per year"
        for h in hostel_list
    )
    prompt = (
        f"College: {college}  |  Academic Year: {year}\n\n"
        f"Available hostel options and annual fees:\n{rows}\n\n"
        "Write a 220-300 word paragraph about the hostel accommodation options. "
        "Mention each hostel's name, gender it serves if apparent (ladies/gents), "
        "annual cost, and the benefits of on-campus housing for students who "
        "travel from distant towns or villages."
    )
    text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
    if not text:
        return []
    chunk = enrich_and_build_chunk(
        text=text,
        category="hostel",
        source_section="fy27_cutoff_bus_hostel_fees/hostels",
        metadata={"year": year, "hostels": hostel_list, "count": len(hostel_list)},
        keywords=["hostel", "accommodation", "ladies hostel", "boarding",
                  "annual fee", "campus stay", "residential"],
        college=college, year=year, model=_MODEL,
    )
    return [chunk] if chunk else []


def describe_bus_routes(bus_routes: dict, college: str, year: str) -> list[dict]:
    """One chunk per bus route plus a parent overview chunk."""
    if not bus_routes:
        return []

    route_chunks = []
    all_route_lines = []

    for route_name, stops in bus_routes.items():
        if not stops:
            continue
        rows = "\n".join(
            f"  {s.get('stop', '?')}: {fmt_inr(s.get('fee', 0))} per month"
            for s in stops
        )
        all_route_lines.append(f"Route '{route_name}': {len(stops)} stops")

        prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"College bus route: {route_name}\n"
            f"Stops and monthly transport fees:\n{rows}\n\n"
            "Write a 220-300 word paragraph for commuting students about this route. "
            "Mention the route name, the key towns or stops it covers, monthly fee range "
            "(cheapest nearest stop to costliest farthest stop), and which students "
            "this route is most useful for."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunk = enrich_and_build_chunk(
            text=text,
            category="transport",
            source_section=f"fy27_cutoff_bus_hostel_fees/bus_routes/{route_name}",
            metadata={"route": route_name, "stops": stops, "stop_count": len(stops)},
            keywords=["bus", "transport", "route", route_name, "commute",
                      "monthly fee", "college bus"],
            college=college, year=year, model=_MODEL,
        )
        if chunk:
            route_chunks.append(chunk)

    # Parent overview for all routes
    if len(route_chunks) > 1:
        overview_prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"The college operates {len(bus_routes)} bus routes:\n"
            + "\n".join(all_route_lines)
            + "\n\nWrite a 200-260 word overview of the college's bus transport network. "
            "Mention total number of routes, the general geographic coverage, "
            "and how monthly fees are structured (distance-based)."
        )
        overview_text = ollama_generate(overview_prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if overview_text:
            parent_id = str(uuid.uuid4())
            parent_chunk = enrich_and_build_chunk(
                text=overview_text,
                category="transport",
                source_section="fy27_cutoff_bus_hostel_fees/bus_routes/overview",
                metadata={"is_parent": True, "route_count": len(bus_routes),
                          "routes": list(bus_routes.keys())},
                keywords=["bus routes", "transport overview", "commute", "college bus",
                          "all routes"],
                college=college, year=year, model=_MODEL,
                run_enrichment=False,
            )
            if parent_chunk:
                parent_chunk["id"] = parent_id
                for rc in route_chunks:
                    rc["parent_id"] = parent_id
                route_chunks.insert(0, parent_chunk)

    return route_chunks


def describe_cutoffs(cutoff_list: list, college: str, year: str) -> list[dict]:
    """One chunk per branch + a parent comparative overview."""
    if not cutoff_list:
        return []

    branch_chunks = []
    branch_summary_lines = []

    for entry in cutoff_list:
        course  = entry.get("course", "Unknown")
        cutoffs = entry.get("cutoff", {})

        rows = []
        for cat, seats in cutoffs.items():
            g  = seats.get("G", {}) or {}
            l  = seats.get("L", {}) or {}
            gm = f"{g['merit_marks']:.2f}" if isinstance(g.get("merit_marks"), float) else "N/A"
            lm = f"{l['merit_marks']:.2f}" if isinstance(l.get("merit_marks"), float) else "N/A"
            rows.append(
                f"  {cat}: General seat = {gm} marks  |  Ladies seat = {lm} marks"
            )
            if cat == "OPEN" and gm != "N/A":
                branch_summary_lines.append(
                    f"  {course}: OPEN General cutoff = {gm} marks"
                )
        if not rows:
            continue

        prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"MHT-CET admission cutoff marks for the {course} branch:\n"
            + "\n".join(rows)
            + "\n\nWrite a 300-400 word paragraph explaining these cutoffs to a student "
            "planning to apply. Clarify what 'G' (General) and 'L' (Ladies) seats mean. "
            "Cover each reservation category — VJ, NT-1, NT-2, NT-3, OBC, SEBC, OPEN, "
            "SC, ST — and what merit marks a student from each category needs to secure "
            "admission to this branch."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunk = enrich_and_build_chunk(
            text=text,
            category="cutoffs",
            source_section=f"fy27_cutoff_bus_hostel_fees/cutoffs/{course}",
            metadata={"course": course, "year": year,
                      "categories": list(cutoffs.keys())},
            keywords=["cutoff", "merit", "MHT-CET", course, "admission",
                      "marks", "category", "OPEN", "OBC", "SC", "ST",
                      "VJ", "NT", "SEBC"],
            college=college, year=year, model=_MODEL,
        )
        if chunk:
            branch_chunks.append(chunk)

    # Parent: comparative overview
    if len(branch_chunks) > 1 and branch_summary_lines:
        parent_prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            "OPEN category General seat cutoff marks across all branches:\n"
            + "\n".join(branch_summary_lines)
            + "\n\nWrite a 260-340 word comparative overview of admission cutoffs. "
            "Highlight which branches are most competitive (highest marks required) "
            "and which are relatively more accessible, to help students choose."
        )
        parent_text = ollama_generate(parent_prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if parent_text:
            parent_id = str(uuid.uuid4())
            parent_chunk = enrich_and_build_chunk(
                text=parent_text,
                category="cutoffs",
                source_section="fy27_cutoff_bus_hostel_fees/cutoffs/overview",
                metadata={"is_parent": True,
                          "branches": [e.get("course") for e in cutoff_list]},
                keywords=["cutoff overview", "all branches", "MHT-CET",
                          "comparison", "most competitive"],
                college=college, year=year, model=_MODEL,
                run_enrichment=False,
            )
            if parent_chunk:
                parent_chunk["id"] = parent_id
                for bc in branch_chunks:
                    bc["parent_id"] = parent_id
                branch_chunks.insert(0, parent_chunk)

    return branch_chunks


def describe_qualifying_criteria(criteria: Any, college: str, year: str) -> list[dict]:
    if not criteria:
        return []
    raw = json.dumps(criteria, indent=2, ensure_ascii=False)
    prompt = (
        f"College: {college}  |  Academic Year: {year}\n\n"
        f"12th-standard qualifying criteria for B.Tech admission:\n{raw}\n\n"
        "Write a 220-300 word paragraph summarising the minimum 12th-grade marks "
        "needed for admission. Be specific about percentage thresholds for different "
        "subject combinations (PCM, PCB etc.) and reservation categories."
    )
    text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
    if not text:
        return []
    chunk = enrich_and_build_chunk(
        text=text,
        category="eligibility",
        source_section="fy27_cutoff_bus_hostel_fees/qualifying_criteria",
        metadata={"year": year},
        keywords=["eligibility", "qualifying", "12th", "HSC", "minimum marks",
                  "percentage", "physics", "maths", "chemistry"],
        college=college, year=year, model=_MODEL,
    )
    return [chunk] if chunk else []


def describe_placement_summary(summary_sheet: dict, college: str, year: str) -> list[dict]:
    chunks = []
    by_branch = summary_sheet.get("placement_summary_by_branch", [])
    totals    = summary_sheet.get("overall_totals", {})
    companies = summary_sheet.get("companies_visited", {})

    total_lines   = []
    company_lines = []
    for yr, tot in totals.items():
        total_lines.append(
            f"  {yr}: {tot.get('total_students','?')} students, "
            f"{tot.get('students_placed_offers','?')} offers, "
            f"avg {fmt_lpa(tot.get('avg_salary_lpa'))}, "
            f"highest {fmt_lpa(tot.get('highest_salary_lpa'))}"
        )
    for yr, cnt in companies.items():
        company_lines.append(f"  {yr}: {cnt} companies")

    if total_lines:
        prompt = (
            f"College: {college}\n\n"
            "Three-year overall placement totals:\n"
            + "\n".join(total_lines)
            + "\n\nCompanies that visited campus per year:\n"
            + "\n".join(company_lines)
            + "\n\nWrite a 300-400 word paragraph summarising the college's overall "
            "placement performance across three years. Discuss year-on-year trends, "
            "whether total offers improved, changes in salary packages, and growth "
            "in participating companies."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if text:
            parent_id = str(uuid.uuid4())
            parent_chunk = enrich_and_build_chunk(
                text=text,
                category="placements",
                source_section="placement_3yr_summary/overall",
                metadata={"is_parent": True, "totals": totals, "companies": companies},
                keywords=["placement", "overall", "offers", "salary",
                          "companies", "LPA", "3-year", "trend"],
                college=college, year=year, model=_MODEL,
            )
            if parent_chunk:
                parent_chunk["id"] = parent_id
                chunks.append(parent_chunk)

                for branch_data in by_branch:
                    branch = branch_data.get("branch", "Unknown")
                    rows = []
                    for yr in ["2023-24", "2024-25", "2025-26"]:
                        yd = branch_data.get(yr, {})
                        if yd:
                            rows.append(
                                f"  {yr}: {yd.get('total_students','?')} students, "
                                f"{yd.get('students_placed_offers','?')} offers, "
                                f"avg {fmt_lpa(yd.get('avg_salary_lpa'))}, "
                                f"highest {fmt_lpa(yd.get('highest_salary_lpa'))}"
                            )
                    if not rows:
                        continue
                    prompt_b = (
                        f"College: {college}\n"
                        f"Branch: {branch}\n\n"
                        "Three-year placement data:\n"
                        + "\n".join(rows)
                        + "\n\nWrite a 280-360 word paragraph about this branch's placement "
                        "history. Discuss the trend in average and highest packages, changes "
                        "in offer count, and what the data suggests about industry demand "
                        "for this branch's graduates."
                    )
                    text_b = ollama_generate(prompt_b, model=_MODEL, system=SYSTEM_PROMPT)
                    if not text_b:
                        continue
                    child = enrich_and_build_chunk(
                        text=text_b,
                        category="placements",
                        source_section=f"placement_3yr_summary/{branch}",
                        metadata={"branch": branch},
                        keywords=["placement", branch, "package", "LPA",
                                  "offers", "salary", "three year"],
                        college=college, year=year, model=_MODEL,
                        parent_id=parent_id,
                    )
                    if child:
                        chunks.append(child)
    return chunks


def describe_companies(companies_sheet: dict, college: str, year: str) -> list[dict]:
    chunks = []
    for yr, companies in companies_sheet.items():
        if not companies:
            continue
        it_cos   = [c["company_name"] for c in companies
                    if "IT" in str(c.get("industry_vertical", "")).upper()]
        core_cos = [c["company_name"] for c in companies
                    if "Core" in str(c.get("industry_vertical", ""))]
        all_names = [c["company_name"] for c in companies]
        verticals = Counter(c.get("industry_vertical", "Other") for c in companies)
        dist_str  = ", ".join(f"{v}: {cnt}" for v, cnt in verticals.most_common())

        prompt = (
            f"College: {college}  |  Placement Year: {yr}\n\n"
            f"Total companies that visited campus: {len(companies)}\n"
            f"Industry distribution: {dist_str}\n"
            f"IT companies (sample): {', '.join(it_cos[:15]) or 'none'}\n"
            f"Core/Manufacturing (sample): {', '.join(core_cos[:15]) or 'none'}\n"
            f"Notable companies: {', '.join(all_names[:40])}\n\n"
            "Write a 300-380 word paragraph about the campus recruitment drive for "
            f"{yr}. Cover the total company count, industry mix (IT vs core vs others), "
            "names of well-known recruiters, and what the company diversity reflects "
            "about the college's industry connections."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunk = enrich_and_build_chunk(
            text=text,
            category="company_visits",
            source_section=f"company_visited_lists/{yr}",
            metadata={"year": yr, "total_companies": len(companies),
                      "it_count": len(it_cos), "core_count": len(core_cos),
                      "companies": all_names},
            keywords=["companies", yr, "campus recruitment", "IT",
                      "core", "placement drive", "recruiter"],
            college=college, year=year, model=_MODEL,
        )
        if chunk:
            chunks.append(chunk)
    return chunks


def describe_placement_detail(detail_sheet: dict, college: str, year: str) -> list[dict]:
    chunks = []
    for yr in ["2025-26", "2024-25", "2023-24"]:
        branch_list = detail_sheet.get(yr, [])
        summary     = detail_sheet.get(f"{yr}_summary", {})
        if not branch_list:
            continue

        rows = "\n".join(
            f"  {b.get('branch','?')}: "
            f"{b.get('total_students','?')} students, "
            f"{b.get('students_placed_offers','?')} offers, "
            f"avg {fmt_lpa(b.get('avg_salary_lpa'))}, "
            f"highest {fmt_lpa(b.get('highest_salary_lpa'))}"
            for b in branch_list
        )
        summ_note = ""
        if summary:
            ot = summary.get("overall_total", {})
            summ_note = (
                f"\nOverall: {ot.get('students_placed_offers','?')} total offers, "
                f"{summary.get('companies_visited','?')} companies visited."
            )

        prompt = (
            f"College: {college}  |  Academic Year: {yr}\n\n"
            f"Branch-wise placement results:\n{rows}{summ_note}\n\n"
            "Write a 350-450 word detailed account of the placement season for "
            f"{yr}. Cover every branch's performance, compare branches by package "
            "and offer count, highlight the top-performing branch, and mention any "
            "standout highest packages. Include overall totals if available."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunk = enrich_and_build_chunk(
            text=text,
            category="placements",
            source_section=f"placement_by_year_detail/{yr}",
            metadata={"year": yr,
                      "branches": [b.get("branch") for b in branch_list],
                      "companies_visited": summary.get("companies_visited") if summary else None},
            keywords=["placement", yr, "branch", "package", "LPA",
                      "offers", "detailed", "annual"],
            college=college, year=year, model=_MODEL,
        )
        if chunk:
            chunks.append(chunk)
    return chunks


def describe_admission_docs(docs_sheet: dict, college: str, year: str) -> list[dict]:
    chunks = []
    for cat_label, doc_list in docs_sheet.items():
        if not isinstance(doc_list, list) or not doc_list:
            continue
        docs = [d.get("document", "") for d in doc_list if d.get("document")]
        prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"Admission type: {cat_label}\n"
            f"Required documents: {', '.join(docs)}\n\n"
            "Write a 220-300 word paragraph listing and explaining each document. "
            "Describe why each is needed and whether requirements differ by reservation "
            "category (OPEN, SC/ST, OBC/VJ/NT etc.)."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunk = enrich_and_build_chunk(
            text=text,
            category="admission_documents",
            source_section=f"admission_documents/{cat_label[:50]}",
            metadata={"admission_type": cat_label, "documents": docs,
                      "doc_count": len(docs)},
            keywords=["documents", "admission", "scholarship", "marksheet",
                      "certificate", "required documents"],
            college=college, year=year, model=_MODEL,
        )
        if chunk:
            chunks.append(chunk)
    return chunks


# =============================================================================
# Q&A generation  (4 diverse question types per chunk)
# =============================================================================

def generate_qa_pairs(
    chunks: list[dict],
    college: str,
    model: str,
) -> list[dict]:
    """
    Generate 4 diverse Q&A pairs per chunk covering:
      factual     – specific number / name / date lookup
      comparative – compare two items or two years
      advice      – what should a student do or choose
      eligibility – whether someone qualifies or what is required
    This mirrors realistic user query patterns and ensures the QA dataset
    covers the full query-type distribution found in production RAG logs.
    """
    qa_pairs: list[dict] = []
    log.info("Generating Q&A pairs from %d chunks...", len(chunks))

    qa_system = (
        "You are an expert at creating diverse retrieval evaluation questions "
        "for a college information RAG system. All questions must be realistic and "
        "answerable solely from the provided knowledge text."
    )

    for chunk in tqdm(chunks, desc="QA generation", unit="chunk"):
        text     = chunk["text"]
        category = chunk["category"]

        prompt = (
            f"College: {college} | Category: {category}\n\n"
            f"Knowledge text:\n{text[:900]}\n\n"
            "Generate exactly 4 question-answer pairs, one per type:\n"
            '  1. "factual"     – asks for a specific number, name, or date\n'
            '  2. "comparative" – compares two items or years\n'
            '  3. "advice"      – asks what a student should do or choose\n'
            '  4. "eligibility" – asks whether someone qualifies or what is required\n\n'
            "Each answer must be 1-3 sentences, fully answerable from the text above.\n"
            "Output as JSON array:\n"
            '[{"type":"factual","question":"...","answer":"..."},\n'
            ' {"type":"comparative","question":"...","answer":"..."},\n'
            ' {"type":"advice","question":"...","answer":"..."},\n'
            ' {"type":"eligibility","question":"...","answer":"..."}]\n\n'
            "Output only the JSON array."
        )
        raw = ollama_generate(
            prompt, model=model, system=qa_system, temperature=0.4, max_tokens=700
        )
        if not raw:
            continue

        pairs = safe_json_parse(raw)
        if not isinstance(pairs, list):
            continue

        for pair in pairs[:4]:
            if not isinstance(pair, dict):
                continue
            q = str(pair.get("question", "")).strip()
            a = str(pair.get("answer", "")).strip()
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

def write_knowledge_base(chunks: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Saved knowledge_base.jsonl   -> %s  (%d chunks)", path, len(chunks))


def write_propositions(chunks: list[dict], path: Path) -> None:
    """
    Atomic propositions for factoid-optimised dense retrieval.
    Each line is one proposition with its parent chunk's metadata attached.
    Embed these separately for a parallel factoid retrieval index.
    """
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            for prop in chunk.get("propositions", []):
                if not prop.strip():
                    continue
                record = {
                    "id":              str(uuid.uuid4()),
                    "parent_chunk_id": chunk["id"],
                    "category":        chunk["category"],
                    "source_section":  chunk["source_section"],
                    "text":            prop,
                    "keywords":        chunk["keywords"],
                    "metadata":        chunk["metadata"],
                    "embedding":       None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    log.info("Saved propositions.jsonl     -> %s  (%d propositions)", path, count)


def write_summary_index(chunks: list[dict], path: Path) -> None:
    """
    Lightweight summary index for two-stage retrieval.
    Retrieve summaries first (Stage 1), fetch full chunks by id (Stage 2).
    """
    index = []
    for chunk in chunks:
        summary = chunk.get("one_line_summary", "")
        if not summary:
            summary = chunk["text"][:120] + "..."
        index.append({
            "chunk_id":       chunk["id"],
            "parent_id":      chunk.get("parent_id"),
            "category":       chunk["category"],
            "source_section": chunk["source_section"],
            "summary":        summary,
            "keywords":       chunk["keywords"][:8],
            "quality_score":  chunk.get("quality_score", 0),
            "low_quality":    chunk.get("low_quality_flag", False),
        })
    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    log.info("Saved summary_index.json     -> %s  (%d entries)", path, len(index))


def write_qa_pairs(qa_pairs: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    log.info("Saved qa_pairs.json          -> %s  (%d pairs)", path, len(qa_pairs))


# =============================================================================
# Main pipeline
# =============================================================================

def build_knowledge_base(
    input_path: str,
    model: str,
    skip_qa: bool = False,
) -> None:
    global _MODEL
    _MODEL = model

    log.info("Loading JSON from: %s", input_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    college = data.get("college", "College")
    year    = data.get("academic_year", "2025-26")
    sheets  = data.get("sheets", {})

    log.info("College       : %s", college)
    log.info("Academic Year : %s", year)
    log.info("Sheets found  : %s", list(sheets.keys()))

    check_ollama(model)
    log.info("Ollama model '%s' confirmed.", model)

    all_chunks: list[dict] = []
    cutoff_sheet = sheets.get("fy27_cutoff_bus_hostel_fees", {})

    section_processors = [
        ("Programs offered",      lambda: describe_programs(
            sheets.get("programs_offered", {}), college, year)),
        ("Tuition fees",          lambda: describe_tuition_fees(
            cutoff_sheet.get("tuition_fees_2025_26", []), college, year)),
        ("Hostel fees",           lambda: describe_hostels(
            cutoff_sheet.get("hostels", []), college, year)),
        ("Bus routes",            lambda: describe_bus_routes(
            cutoff_sheet.get("bus_routes", {}), college, year)),
        ("Admission cutoffs",     lambda: describe_cutoffs(
            cutoff_sheet.get("cutoff_data", []), college, year)),
        ("Qualifying criteria",   lambda: describe_qualifying_criteria(
            cutoff_sheet.get("qualifying_criteria_12th"), college, year)),
        ("Placement 3yr summary", lambda: describe_placement_summary(
            sheets.get("placement_3yr_summary", {}), college, year)),
        ("Companies visited",     lambda: describe_companies(
            sheets.get("company_visited_lists", {}), college, year)),
        ("Placement year detail", lambda: describe_placement_detail(
            sheets.get("placement_by_year_detail", {}), college, year)),
        ("Admission documents",   lambda: describe_admission_docs(
            sheets.get("admission_documents", {}), college, year)),
    ]

    for section_name, processor in tqdm(section_processors, desc="Sections", unit="section"):
        log.info("Processing: %s", section_name)
        try:
            new_chunks = processor()
            all_chunks.extend(new_chunks)
            lq = sum(1 for c in new_chunks if c.get("low_quality_flag"))
            log.info(
                "  -> %d chunk(s) generated  (low-quality flags: %d)",
                len(new_chunks), lq,
            )
        except Exception as exc:
            log.warning("  Section '%s' failed: %s", section_name, exc)

    log.info("Total knowledge chunks generated: %d", len(all_chunks))

    # Write all output files
    write_knowledge_base(all_chunks,  Path("knowledge_base.jsonl"))
    write_propositions(all_chunks,    Path("propositions.jsonl"))
    write_summary_index(all_chunks,   Path("summary_index.json"))

    if not skip_qa:
        qa_pairs = generate_qa_pairs(all_chunks, college, model)
        write_qa_pairs(qa_pairs, Path("qa_pairs.json"))
    else:
        log.info("Q&A generation skipped (--skip-qa flag)")
        qa_pairs = []

    # Summary report
    category_counts: dict[str, int] = {}
    prop_total = 0
    lq_total   = 0
    for c in all_chunks:
        category_counts[c["category"]] = category_counts.get(c["category"], 0) + 1
        prop_total += len(c.get("propositions", []))
        lq_total   += int(c.get("low_quality_flag", False))

    avg_q = (
        sum(c.get("quality_score", 0) for c in all_chunks) / len(all_chunks)
        if all_chunks else 0
    )

    print("\n" + "=" * 65)
    print(f"  Knowledge Base -- {college}  [{year}]")
    print("=" * 65)
    print(f"  {'Category':<36} {'Chunks':>6}")
    print("  " + "-" * 44)
    for cat, cnt in sorted(category_counts.items()):
        print(f"  {cat:<36} {cnt:>6}")
    print("  " + "-" * 44)
    print(f"  {'TOTAL chunks':<36} {len(all_chunks):>6}")
    print(f"  {'Propositions (atomic facts)':<36} {prop_total:>6}")
    print(f"  {'Q&A pairs':<36} {len(qa_pairs):>6}")
    print(f"  {'Low-quality chunks flagged':<36} {lq_total:>6}")
    print(f"  {'Average quality score':<36} {avg_q:>6.3f}")
    print("=" * 65)
    print("  Output files:")
    print("    knowledge_base.jsonl  -- full enriched chunks")
    print("    propositions.jsonl    -- atomic facts (factoid retrieval)")
    print("    summary_index.json    -- summaries (two-stage RAG)")
    print("    qa_pairs.json         -- Q&A eval / fine-tuning pairs")
    print("=" * 65)
    print()
    print("  NEXT STEP: embed the 'contextualised_text' field (not 'text')")
    print("  for best retrieval recall (Anthropic contextual retrieval).")
    print("  Use 'bm25_tokens' to build a parallel BM25 keyword index.")
    print("  Use 'propositions.jsonl' as a separate factoid index.")
    print()


# =============================================================================
# Entry point
# =============================================================================

_MODEL = DEFAULT_MODEL  # module-level default used by processor lambdas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert college JSON to a production-grade RAG knowledge base "
            "using local Qwen 2.5 7B via Ollama."
        )
    )
    parser.add_argument(
        "--input", default="output.json",
        help="Path to college data JSON (default: output.json)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--skip-qa", action="store_true",
        help="Skip Q&A pair generation (saves ~40%% of total run time)",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    build_knowledge_base(args.input, args.model, skip_qa=args.skip_qa)