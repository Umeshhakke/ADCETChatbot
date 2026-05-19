"""
college_rag_builder.py
======================
Converts a structured college JSON file into a high-quality semantic
knowledge base optimised for RAG retrieval using a local Qwen 2.5 7B
model via Ollama.

Outputs
-------
  knowledge_base.jsonl  – one embedding-ready chunk per line (JSONL)
  qa_pairs.json         – curated question-answer pairs for retrieval eval

Usage
-----
  python college_rag_builder.py                          # uses output.json
  python college_rag_builder.py --input my_file.json
  python college_rag_builder.py --input output.json --model qwen2.5:7b-instruct
"""

import argparse
import json
import logging
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = "qwen2.5:7b-instruct"
OLLAMA_BASE    = "http://localhost:11434"
OLLAMA_TIMEOUT = 180          # seconds per request
MAX_RETRIES    = 3
RETRY_DELAY    = 5            # seconds between retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------------------------

def ollama_generate(prompt: str, model: str, system: str = "") -> str:
    """Call the Ollama /api/generate endpoint and return the response text."""
    payload: dict[str, Any] = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 900,
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
            return clean_generation(raw)
        except requests.exceptions.ConnectionError:
            log.error("Ollama not reachable. Is `ollama serve` running?")
            sys.exit(1)
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    log.error("All retries exhausted for model call.")
    return ""


def clean_generation(text: str) -> str:
    """Remove markdown fences, leading labels, and stray artifacts."""
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"^(Answer|Response|Output|Result)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def check_ollama(model: str) -> None:
    """Verify Ollama is running and the requested model is available."""
    try:
        tags = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10).json()
        available = [m["name"] for m in tags.get("models", [])]
    except Exception:
        log.error("Cannot reach Ollama at %s. Start it with: ollama serve", OLLAMA_BASE)
        sys.exit(1)

    if not any(model in m for m in available):
        log.warning(
            "Model '%s' not found locally. Pull it with: ollama pull %s", model, model
        )
        log.warning("Available models: %s", available or "none")


# ---------------------------------------------------------------------------
# System prompt (shared across all generation calls)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert knowledge-base writer specialising in Indian engineering \
college admissions, placements, and campus facilities. \
Write clear, factual, human-readable paragraphs — no bullet points, no \
markdown headers, no lists. Every sentence should be informative on its own. \
Do NOT invent numbers; use only the data provided."""


# ---------------------------------------------------------------------------
# Per-section raw-data → natural-language converters
# ---------------------------------------------------------------------------

def fmt_inr(value: Any) -> str:
    """Format a rupee amount nicely."""
    if value is None:
        return "not specified"
    try:
        v = float(str(value).replace(",", "").replace("/-", "").strip())
        return f"₹{v:,.0f}"
    except ValueError:
        return str(value)


def fmt_lpa(value: Any) -> str:
    if value is None:
        return "not disclosed"
    return f"{value} LPA"


# ── Programs offered ──────────────────────────────────────────────────────

def describe_programs(programs_sheet: dict, college: str, year: str) -> list[dict]:
    chunks = []
    for category, progs in programs_sheet.items():
        if not isinstance(progs, list) or not progs:
            continue
        label = {"btech": "B.Tech", "ug_other": "UG (Non-Engineering)", "mtech": "M.Tech"}.get(category, category)
        rows = "\n".join(
            f"  - {p.get('program','?')} | Intake: {p.get('sanctioned_intake','?')} | Since: {p.get('year_of_starting','?')}"
            for p in progs
        )
        prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"The following {label} programmes are offered:\n{rows}\n\n"
            "Write a 250-350 word descriptive paragraph about these programmes, "
            "mentioning intake size and the year each started where notable."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunks.append(build_chunk(
            text=text,
            category="programs_offered",
            source_section=f"programs_offered/{category}",
            metadata={"label": label, "programs": [p.get("program") for p in progs]},
            keywords=["programme", "course", label.lower(), "intake", "engineering"],
        ))
    return chunks


# ── Tuition fees ─────────────────────────────────────────────────────────

def describe_tuition_fees(fee_list: list, college: str, year: str) -> list[dict]:
    if not fee_list:
        return []
    rows = "\n".join(
        f"  {f.get('category','?')}: FY = {fmt_inr(f.get('fy_fee'))}  |  DSE = {fmt_inr(f.get('dse_fee'))}"
        for f in fee_list
    )
    prompt = (
        f"College: {college}  |  Academic Year: {year}\n\n"
        "Tuition fee structure (annual) by reservation category:\n"
        f"{rows}\n\n"
        "Write a 250-350 word paragraph explaining the fee structure clearly "
        "for prospective students, covering both First Year (FY) and Direct Second Year (DSE) admissions."
    )
    text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
    if not text:
        return []
    return [build_chunk(
        text=text,
        category="fees",
        source_section="fy27_cutoff_bus_hostel_fees/tuition_fees",
        metadata={"year": year, "fee_rows": fee_list},
        keywords=["tuition fee", "annual fee", "reservation", "OPEN", "SC/ST", "OBC", "fee structure"],
    )]


# ── Hostel fees ──────────────────────────────────────────────────────────

def describe_hostels(hostel_list: list, college: str, year: str) -> list[dict]:
    if not hostel_list:
        return []
    rows = "\n".join(
        f"  {h.get('hostel','?')}: {fmt_inr(h.get('annual_fee','?'))}"
        for h in hostel_list
    )
    prompt = (
        f"College: {college}  |  Academic Year: {year}\n\n"
        f"Hostel facilities and annual fees:\n{rows}\n\n"
        "Write a 200-300 word paragraph describing the hostel options, "
        "their names, and annual costs in a way that helps students plan accommodation."
    )
    text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
    if not text:
        return []
    return [build_chunk(
        text=text,
        category="hostel",
        source_section="fy27_cutoff_bus_hostel_fees/hostels",
        metadata={"year": year, "hostels": hostel_list},
        keywords=["hostel", "accommodation", "ladies hostel", "boarding", "annual fee"],
    )]


# ── Bus routes ───────────────────────────────────────────────────────────

def describe_bus_routes(bus_routes: dict, college: str, year: str) -> list[dict]:
    chunks = []
    for route_name, stops in bus_routes.items():
        if not stops:
            continue
        rows = "\n".join(
            f"  {s.get('stop','?')}: {fmt_inr(s.get('fee', 0))} per month"
            for s in stops
        )
        prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"College bus route: {route_name}\n"
            f"Stops and monthly fees:\n{rows}\n\n"
            "Write a 200-280 word paragraph describing this bus route for commuting students, "
            "mentioning the route name, key stops, and typical monthly transport costs."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunks.append(build_chunk(
            text=text,
            category="transport",
            source_section=f"fy27_cutoff_bus_hostel_fees/bus_routes/{route_name}",
            metadata={"route": route_name, "stops": stops},
            keywords=["bus", "transport", "route", route_name, "commute", "monthly fee"],
        ))
    return chunks


# ── Admission cutoffs ────────────────────────────────────────────────────

def describe_cutoffs(cutoff_list: list, college: str, year: str) -> list[dict]:
    """One chunk per branch."""
    chunks = []
    for entry in cutoff_list:
        course = entry.get("course", "Unknown")
        cutoffs = entry.get("cutoff", {})
        rows = []
        for cat, seats in cutoffs.items():
            g = seats.get("G", {}) or {}
            l = seats.get("L", {}) or {}
            g_marks = f"{g.get('merit_marks', 'N/A'):.2f}" if isinstance(g.get("merit_marks"), float) else "N/A"
            l_marks = f"{l.get('merit_marks', 'N/A'):.2f}" if isinstance(l.get("merit_marks"), float) else "N/A"
            rows.append(f"  {cat}: General seat cutoff = {g_marks} marks  |  Ladies seat cutoff = {l_marks} marks")
        if not rows:
            continue
        rows_str = "\n".join(rows)
        prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"Admission cutoff data for {course} branch (MHT-CET merit marks):\n{rows_str}\n\n"
            "Write a 280-360 word paragraph explaining the admission cutoff for this branch. "
            "Mention different reservation categories (OPEN, OBC, SC, ST, VJ, NT etc.) "
            "and what merit marks a student needs to secure admission."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunks.append(build_chunk(
            text=text,
            category="cutoffs",
            source_section=f"fy27_cutoff_bus_hostel_fees/cutoffs/{course}",
            metadata={"course": course, "year": year, "raw_cutoffs": cutoffs},
            keywords=["cutoff", "merit", course, "MHT-CET", "admission", "marks", "category"],
        ))
    return chunks


# ── Qualifying criteria ──────────────────────────────────────────────────

def describe_qualifying_criteria(criteria: Any, college: str, year: str) -> list[dict]:
    if not criteria:
        return []
    raw = json.dumps(criteria, indent=2)
    prompt = (
        f"College: {college}  |  Academic Year: {year}\n\n"
        f"12th-standard qualifying criteria for admission:\n{raw}\n\n"
        "Write a 200-280 word paragraph summarising the minimum 12th-grade eligibility "
        "requirements for admission to this college."
    )
    text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
    if not text:
        return []
    return [build_chunk(
        text=text,
        category="eligibility",
        source_section="fy27_cutoff_bus_hostel_fees/qualifying_criteria",
        metadata={"year": year},
        keywords=["eligibility", "qualifying", "12th", "HSC", "marks", "minimum"],
    )]


# ── Placement 3-year summary ─────────────────────────────────────────────

def describe_placement_summary(summary_sheet: dict, college: str) -> list[dict]:
    chunks = []
    by_branch = summary_sheet.get("placement_summary_by_branch", [])
    totals    = summary_sheet.get("overall_totals", {})
    companies = summary_sheet.get("companies_visited", {})

    # Overall summary chunk
    total_text_parts = []
    for yr, tot in totals.items():
        total_text_parts.append(
            f"  {yr}: {tot.get('total_students','?')} students, "
            f"{tot.get('students_placed_offers','?')} offers, "
            f"avg {fmt_lpa(tot.get('avg_salary_lpa'))}, "
            f"highest {fmt_lpa(tot.get('highest_salary_lpa'))}"
        )
    company_parts = []
    for yr, cnt in companies.items():
        company_parts.append(f"  {yr}: {cnt} companies")

    if total_text_parts:
        prompt = (
            f"College: {college}\n\n"
            "Three-year overall placement totals:\n" + "\n".join(total_text_parts) + "\n\n"
            "Companies visited per year:\n" + "\n".join(company_parts) + "\n\n"
            "Write a 280-380 word paragraph summarising the college's overall placement "
            "performance across the last three academic years, highlighting trends."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if text:
            chunks.append(build_chunk(
                text=text,
                category="placements",
                source_section="placement_3yr_summary/overall",
                metadata={"totals": totals, "companies": companies},
                keywords=["placement", "overall", "offers", "salary", "companies", "LPA"],
            ))

    # Per-branch chunks
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
        prompt = (
            f"College: {college}\n\n"
            f"Branch: {branch}\nThree-year placement data:\n" + "\n".join(rows) + "\n\n"
            "Write a 280-360 word paragraph describing the placement performance of this "
            "branch over three years, highlighting year-on-year trends, average packages, "
            "and highest packages achieved."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunks.append(build_chunk(
            text=text,
            category="placements",
            source_section=f"placement_3yr_summary/{branch}",
            metadata={"branch": branch, "years": ["2023-24", "2024-25", "2025-26"]},
            keywords=["placement", branch, "package", "LPA", "offers", "salary"],
        ))
    return chunks


# ── Companies visited ────────────────────────────────────────────────────

def describe_companies(companies_sheet: dict, college: str) -> list[dict]:
    chunks = []
    for yr, companies in companies_sheet.items():
        if not companies:
            continue
        it_cos  = [c["company_name"] for c in companies if "IT" in str(c.get("industry_vertical", "")).upper()]
        core_cos = [c["company_name"] for c in companies if "Core" in str(c.get("industry_vertical", ""))]
        all_names = [c["company_name"] for c in companies]
        prompt = (
            f"College: {college}  |  Placement Year: {yr}\n\n"
            f"Total companies that visited: {len(companies)}\n"
            f"IT sector companies: {', '.join(it_cos[:20]) or 'none listed'}\n"
            f"Core/Manufacturing companies: {', '.join(core_cos[:20]) or 'none listed'}\n"
            f"All companies (first 40): {', '.join(all_names[:40])}\n\n"
            "Write a 280-360 word paragraph describing the campus recruitment drive for this "
            "year. Mention notable companies, the mix of IT vs core industries, "
            "and the variety of eligible branches."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunks.append(build_chunk(
            text=text,
            category="company_visits",
            source_section=f"company_visited_lists/{yr}",
            metadata={"year": yr, "total_companies": len(companies), "companies": all_names},
            keywords=["companies", yr, "campus recruitment", "IT", "core", "placement drive"],
        ))
    return chunks


# ── Year-wise placement detail ───────────────────────────────────────────

def describe_placement_detail(detail_sheet: dict, college: str) -> list[dict]:
    chunks = []
    years = ["2025-26", "2024-25", "2023-24"]
    for yr in years:
        branch_list = detail_sheet.get(yr, [])
        summary     = detail_sheet.get(f"{yr}_summary", {})
        if not branch_list:
            continue
        rows = "\n".join(
            f"  {b.get('branch','?')}: {b.get('total_students','?')} students, "
            f"{b.get('students_placed_offers','?')} offers, "
            f"avg {fmt_lpa(b.get('avg_salary_lpa'))}, "
            f"highest {fmt_lpa(b.get('highest_salary_lpa'))}"
            for b in branch_list
        )
        summ_str = (
            f"Overall offers: {summary.get('overall_total',{}).get('students_placed_offers','?')}, "
            f"Companies visited: {summary.get('companies_visited','?')}"
            if summary else ""
        )
        prompt = (
            f"College: {college}  |  Academic Year: {yr}\n\n"
            f"Branch-wise placement statistics:\n{rows}\n"
            + (f"\nSummary: {summ_str}" if summ_str else "") + "\n\n"
            "Write a 320-420 word paragraph giving a detailed account of the placement season "
            f"for {yr}. Mention each branch's performance, notable packages, and the overall "
            "campus placement atmosphere."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunks.append(build_chunk(
            text=text,
            category="placements",
            source_section=f"placement_by_year_detail/{yr}",
            metadata={"year": yr, "branches": [b.get("branch") for b in branch_list]},
            keywords=["placement", yr, "branch", "package", "LPA", "offers"],
        ))
    return chunks


# ── Admission documents ──────────────────────────────────────────────────

def describe_admission_docs(docs_sheet: dict, college: str, year: str) -> list[dict]:
    chunks = []
    for category_label, doc_list in docs_sheet.items():
        if not isinstance(doc_list, list) or not doc_list:
            continue
        docs = [d.get("document", "") for d in doc_list if d.get("document")]
        prompt = (
            f"College: {college}  |  Academic Year: {year}\n\n"
            f"Admission category: {category_label}\n"
            f"Required documents: {', '.join(docs)}\n\n"
            "Write a 220-300 word paragraph listing and explaining the documents a student "
            "must submit during admission, highlighting any caste-category-specific requirements."
        )
        text = ollama_generate(prompt, model=_MODEL, system=SYSTEM_PROMPT)
        if not text:
            continue
        chunks.append(build_chunk(
            text=text,
            category="admission_documents",
            source_section=f"admission_documents/{category_label[:40]}",
            metadata={"admission_type": category_label, "documents": docs},
            keywords=["documents", "admission", "scholarship", "marksheet", "certificate"],
        ))
    return chunks


# ---------------------------------------------------------------------------
# Chunk builder
# ---------------------------------------------------------------------------

def build_chunk(
    text: str,
    category: str,
    source_section: str,
    metadata: dict,
    keywords: list[str],
) -> dict:
    return {
        "id":             str(uuid.uuid4()),
        "category":       category,
        "source_section": source_section,
        "text":           text,
        "metadata":       metadata,
        "keywords":       keywords,
        # Embedding-ready fields (vectors NOT populated here)
        "embedding_model": None,
        "embedding":       None,
    }


# ---------------------------------------------------------------------------
# QA pair generation
# ---------------------------------------------------------------------------

QA_TEMPLATES = {
    "placements": [
        "What is the highest placement package for {branch} in {year}?",
        "How many students from {branch} were placed in {year}?",
        "What is the average salary package offered in {branch}?",
    ],
    "fees": [
        "What is the tuition fee for OPEN category students?",
        "How much do SC/ST students pay as annual tuition fee?",
        "What is the fee difference between FY and DSE admissions?",
    ],
    "hostel": [
        "What are the hostel options available for girls?",
        "What is the annual hostel fee at {college}?",
    ],
    "transport": [
        "What is the monthly bus fee from {stop} on the {route} route?",
        "Which bus routes does {college} operate?",
    ],
    "cutoffs": [
        "What is the MHT-CET cutoff for {course} in the OPEN category?",
        "What merit marks are needed for admission to {course} branch?",
    ],
    "company_visits": [
        "Which companies visited {college} for campus placement in {year}?",
        "How many IT companies visited the college in {year}?",
    ],
    "programs_offered": [
        "What B.Tech programmes does {college} offer?",
        "When did {college} start the {program} programme?",
    ],
}


def generate_qa_pairs(chunks: list[dict], college: str, model: str) -> list[dict]:
    """Generate Q&A pairs from knowledge chunks using Qwen."""
    qa_pairs = []
    log.info("Generating Q&A pairs from %d chunks…", len(chunks))

    for chunk in tqdm(chunks, desc="QA generation", unit="chunk"):
        text     = chunk["text"]
        category = chunk["category"]

        prompt = (
            f"Based on the following knowledge about {college}, generate exactly 3 "
            "high-quality question-answer pairs that would be useful for a student "
            "looking up information. Format your output strictly as JSON array:\n"
            '[{"question": "...", "answer": "..."}, ...]\n\n'
            f"Knowledge:\n{text}\n\n"
            "Output only the JSON array, nothing else."
        )
        raw = ollama_generate(prompt, model=model, system=SYSTEM_PROMPT)
        if not raw:
            continue

        # Strip any accidental markdown or preamble
        raw = re.sub(r"^[^[\[]*", "", raw, count=1)
        raw = re.sub(r"[^\]]*$", "", raw[::-1], count=1)[::-1]
        try:
            pairs = json.loads(raw)
            if not isinstance(pairs, list):
                raise ValueError("Not a list")
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract individual JSON objects
            pairs = []
            for m in re.finditer(r'\{[^{}]+\}', raw, re.DOTALL):
                try:
                    obj = json.loads(m.group())
                    if "question" in obj and "answer" in obj:
                        pairs.append(obj)
                except json.JSONDecodeError:
                    pass

        for pair in pairs[:3]:
            if not pair.get("question") or not pair.get("answer"):
                continue
            qa_pairs.append({
                "id":             str(uuid.uuid4()),
                "source_chunk_id": chunk["id"],
                "category":       category,
                "source_section": chunk["source_section"],
                "question":       pair["question"].strip(),
                "answer":         pair["answer"].strip(),
                "keywords":       chunk["keywords"],
            })

    return qa_pairs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_knowledge_base(input_path: str, model: str) -> None:
    global _MODEL
    _MODEL = model

    log.info("Loading JSON from %s", input_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    college = data.get("college", "College")
    year    = data.get("academic_year", "2025-26")
    sheets  = data.get("sheets", {})

    log.info("College : %s", college)
    log.info("Year    : %s", year)
    log.info("Sheets  : %s", list(sheets.keys()))

    check_ollama(model)
    log.info("Ollama model '%s' confirmed.", model)

    all_chunks: list[dict] = []

    # ── Process each section ──────────────────────────────────────────────
    section_processors = [
        ("Programs offered",      lambda: describe_programs(
            sheets.get("programs_offered", {}), college, year)),
        ("Tuition fees",          lambda: describe_tuition_fees(
            sheets.get("fy27_cutoff_bus_hostel_fees", {}).get("tuition_fees_2025_26", []), college, year)),
        ("Hostel fees",           lambda: describe_hostels(
            sheets.get("fy27_cutoff_bus_hostel_fees", {}).get("hostels", []), college, year)),
        ("Bus routes",            lambda: describe_bus_routes(
            sheets.get("fy27_cutoff_bus_hostel_fees", {}).get("bus_routes", {}), college, year)),
        ("Admission cutoffs",     lambda: describe_cutoffs(
            sheets.get("fy27_cutoff_bus_hostel_fees", {}).get("cutoff_data", []), college, year)),
        ("Qualifying criteria",   lambda: describe_qualifying_criteria(
            sheets.get("fy27_cutoff_bus_hostel_fees", {}).get("qualifying_criteria_12th"), college, year)),
        ("Placement 3yr summary", lambda: describe_placement_summary(
            sheets.get("placement_3yr_summary", {}), college)),
        ("Companies visited",     lambda: describe_companies(
            sheets.get("company_visited_lists", {}), college)),
        ("Placement year detail", lambda: describe_placement_detail(
            sheets.get("placement_by_year_detail", {}), college)),
        ("Admission documents",   lambda: describe_admission_docs(
            sheets.get("admission_documents", {}), college, year)),
    ]

    for section_name, processor in tqdm(section_processors, desc="Sections", unit="section"):
        log.info("Processing: %s", section_name)
        try:
            new_chunks = processor()
            all_chunks.extend(new_chunks)
            log.info("  → %d chunk(s) generated", len(new_chunks))
        except Exception as exc:
            log.warning("  Section '%s' failed: %s", section_name, exc)

    log.info("Total knowledge chunks: %d", len(all_chunks))

    # ── Write knowledge_base.jsonl ────────────────────────────────────────
    kb_path = Path("knowledge_base.jsonl")
    with kb_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Saved knowledge base → %s  (%d chunks)", kb_path, len(all_chunks))

    # ── Generate Q&A pairs ────────────────────────────────────────────────
    qa_pairs = generate_qa_pairs(all_chunks, college, model)
    log.info("Total Q&A pairs: %d", len(qa_pairs))

    qa_path = Path("qa_pairs.json")
    with qa_path.open("w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    log.info("Saved Q&A pairs   → %s  (%d pairs)", qa_path, len(qa_pairs))

    # ── Print summary ─────────────────────────────────────────────────────
    category_counts: dict[str, int] = {}
    for c in all_chunks:
        category_counts[c["category"]] = category_counts.get(c["category"], 0) + 1

    print("\n" + "="*60)
    print(f"  Knowledge Base Summary  —  {college}")
    print("="*60)
    for cat, cnt in sorted(category_counts.items()):
        print(f"  {cat:<30} {cnt:>4} chunk(s)")
    print(f"  {'TOTAL':<30} {len(all_chunks):>4} chunk(s)")
    print(f"  {'Q&A pairs':<30} {len(qa_pairs):>4}")
    print("="*60)
    print(f"  Outputs: {kb_path}   {qa_path}")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_MODEL = DEFAULT_MODEL   # module-level for closures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert college JSON → RAG knowledge base using local Qwen 2.5 7B"
    )
    parser.add_argument(
        "--input", default="output.json",
        help="Path to the college data JSON file (default: output.json)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    build_knowledge_base(args.input, args.model)
