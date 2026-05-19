"""
Run the retrieval + answer-generation step for the ADCET college chatbot.

Pipeline:
  1. Embed the user query with the same model used for vectorization.
  2. Search propositions for exact facts.
  3. Search main_chunks for broader context.
  4. Merge and rerank results with semantic score + keyword overlap.
  5. Build a grounded prompt and optionally ask an Ollama model to answer.

Usage:
  python rag_query.py "What is the hostel fee?"
  python rag_query.py "What is the cutoff for CSE OBC?"
  python rag_query.py "Which companies visited in 2024-25?" --show-context
  python rag_query.py "What B.Tech programs are offered?" --no-generate
  python rag_query.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VECTOR_DIR = SCRIPT_DIR / "vector_db"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_DATA_DIR = SCRIPT_DIR

warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)

QUERY_REPLACEMENTS = {
    "companes": "companies",
    "companys": "companies",
    "placemnt": "placement",
    "placemnts": "placements",
    "departmnt": "department",
    "departments": "branches programs courses departments",
    "department": "branch program course department",
    "hsotel": "hostel",
    "hostal": "hostel",
    "anser": "answer",
    "knowladge": "knowledge",
    "previus": "previous",
    "privious": "previous",
    "student tie sit": "students eligible sit",
    "tie sit": "eligible sit",
    "allow student": "eligible branches students",
    "allow students": "eligible branches students",
    "aids": "AIDS Artificial Intelligence Data Science AI DS",
    "iot": "IoT Internet of Things Cyber Security Block Chain CSE IOT",
    "cse": "CSE Computer Science Engineering",
    "rai": "RAI Robotics Artificial Intelligence",
    "robotics": "Robotics Artificial Intelligence RAI",
    "boys": "boys male gents",
    "girls": "girls ladies female",
}

INTENT_CATEGORY_HINTS = {
    "programs": {"intake", "offer", "offers", "department", "departments", "branch", "branches", "program", "programs", "course", "courses", "started", "start", "robotics", "iot", "aids"},
    "placements_summary": {"placement", "placements", "placed", "package", "salary", "highest", "average", "offers", "students", "last", "previous"},
    "placements_company_visits": {"company", "companies", "recruiter", "recruiters", "eligible", "placement"},
    "fees": {"fee", "fees", "tuition"},
    "hostel": {"hostel", "boys", "girls", "ladies"},
    "cutoffs": {"cutoff", "cutoffs", "marks", "merit", "obc", "open", "sc", "st", "sebc", "tfws"},
    "transport": {"bus", "route", "transport", "stop"},
    "admission_documents": {"document", "documents", "certificate", "admission"},
}


def load_manifest(vector_dir: Path) -> dict[str, Any]:
    path = vector_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Vector manifest not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> set[str]:
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "in", "of", "for",
        "to", "and", "or", "at", "by", "on", "with", "from", "this", "that",
        "what", "which", "who", "when", "where", "how", "does", "do", "tell",
        "me", "about", "college", "adcet",
    }
    return {
        word
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 1 and word not in stopwords
    }


def expand_query(query: str) -> str:
    expanded = query
    lowered = f" {query.lower()} "
    additions = []
    for source, replacement in QUERY_REPLACEMENTS.items():
        if source in lowered:
            additions.append(replacement)

    terms = tokenize(query)
    if "previous" in terms or "last" in terms:
        additions.extend(["2024-25", "previous year", "last year"])
    if "intake" in terms:
        additions.extend(["sanctioned intake", "programs offered"])
    if "started" in terms or "start" in terms:
        additions.extend(["year of starting", "started in"])
    if "placed" in terms:
        additions.extend(["students placed offers placement summary"])
    if additions:
        expanded = f"{query} {' '.join(additions)}"
    return expanded


def detect_intents(query: str) -> set[str]:
    terms = tokenize(expand_query(query))
    intents = set()
    for category, hints in INTENT_CATEGORY_HINTS.items():
        if terms & hints:
            intents.add(category)
    return intents


def primary_intent(query: str) -> str | None:
    terms = tokenize(expand_query(query))
    priority = [
        ("hostel", {"hostel", "boys", "girls", "ladies"}),
        ("transport", {"bus", "route", "transport", "stop"}),
        ("cutoffs", {"cutoff", "cutoffs", "marks", "merit"}),
        ("placements_company_visits", {"company", "companies", "recruiter", "recruiters", "eligible"}),
        ("placements_summary", {"placement", "placements", "placed", "package", "salary", "highest", "average"}),
        ("admission_documents", {"document", "documents", "certificate", "admission"}),
        ("programs", {"intake", "department", "departments", "branch", "branches", "program", "programs", "course", "courses", "started"}),
        ("fees", {"fee", "fees", "tuition"}),
    ]
    for category, hints in priority:
        if terms & hints:
            return category
    return None


def is_company_eligibility_query(query: str) -> bool:
    terms = tokenize(expand_query(query))
    return bool(
        terms & {"company", "companies", "recruiter", "recruiters"}
        and terms & {"eligible", "allow", "sit", "branch", "branches", "department", "mechanical", "civil", "electrical", "cse"}
    )


def parse_json_metadata(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def chroma_query(
    *,
    client: Any,
    collection_name: str,
    query_embedding: list[float],
    top_k: int,
    dataset: str,
) -> list[dict[str, Any]]:
    collection = client.get_collection(collection_name)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results = []
    for item_id, doc, meta, distance in zip(ids, docs, metas, distances):
        metadata = {key: parse_json_metadata(value) for key, value in (meta or {}).items()}
        score = 1.0 - float(distance)
        results.append({
            "id": item_id,
            "dataset": dataset,
            "text": doc,
            "metadata": metadata,
            "semantic_score": score,
            "distance": float(distance),
        })
    return results


def rerank_results(query: str, results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    expanded_terms = tokenize(expand_query(query))
    intents = detect_intents(query)
    primary_category = primary_intent(query)
    reranked = []

    for result in results:
        text_terms = tokenize(result["text"])
        metadata = result.get("metadata", {})
        focused_metadata = {
            "category": metadata.get("category", ""),
            "source_section": metadata.get("source_section", ""),
            "source_table": metadata.get("source_table", ""),
            "entities": metadata.get("entities", ""),
            "keywords": metadata.get("keywords", ""),
            "company": metadata.get("company", ""),
            "eligible_branches": metadata.get("eligible_branches", ""),
        }
        metadata_terms = tokenize(json.dumps(focused_metadata, ensure_ascii=False))
        all_terms = text_terms | metadata_terms
        overlap = len(expanded_terms & all_terms) / max(len(query_terms), 1)

        dataset_boost = 0.08 if result["dataset"] == "propositions" else 0.0
        source_boost = 0.04 if result["metadata"].get("source_sheet") else 0.0
        category = result["metadata"].get("category") or result.get("category") or ""
        intent_boost = 0.18 if category in intents else 0.0
        primary_boost = 0.32 if primary_category and category == primary_category else 0.0
        off_intent_penalty = -0.12 if primary_category and category != primary_category else 0.0
        structured_boost = 0.12 if result["dataset"] == "structured_facts" else 0.0
        if is_company_eligibility_query(query) and result["dataset"] == "structured_facts":
            structured_boost += 0.75
        rerank_score = (
            (0.64 * result["semantic_score"])
            + (0.24 * overlap)
            + dataset_boost
            + source_boost
            + intent_boost
            + primary_boost
            + off_intent_penalty
            + structured_boost
        )

        result = dict(result)
        result["keyword_overlap"] = round(overlap, 4)
        result["rerank_score"] = round(rerank_score, 6)
        reranked.append(result)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    selected = []
    seen_texts: set[str] = set()
    for item in reranked:
        signature = re.sub(r"\s+", " ", item["text"][:220].lower())
        if signature in seen_texts:
            continue
        seen_texts.add(signature)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def load_local_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    kb_path = data_dir / "knowledge_base.jsonl"
    if kb_path.exists():
        with kb_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                records.append({
                    "id": row.get("id", ""),
                    "dataset": "local_main_chunks",
                    "text": row.get("contextualised_text") or row.get("text", ""),
                    "metadata": row.get("metadata", {}) | {
                        "category": row.get("category", ""),
                        "source_section": row.get("source_section", ""),
                    },
                    "semantic_score": 0.0,
                    "distance": 1.0,
                })

    prop_path = data_dir / "propositions.jsonl"
    if prop_path.exists():
        with prop_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                records.append({
                    "id": row.get("id", ""),
                    "dataset": "local_propositions",
                    "text": row.get("text", ""),
                    "metadata": row.get("metadata", {}) | {
                        "category": row.get("category", ""),
                        "source_section": row.get("source_section", ""),
                    },
                    "semantic_score": 0.0,
                    "distance": 1.0,
                })

    output_path = data_dir / "output.json"
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            output = json.load(f)
        company_lists = output.get("sheets", {}).get("company_visited_lists", {})
        branch_filters = {
            "mechanical": ["mech", "mechanical", "all branch"],
            "civil": ["civil", "all branch"],
            "electrical": ["ele", "elec", "electrical", "all branch"],
            "cse": ["cse", "computer", "it", "all branch"],
            "food": ["food", "all branch"],
            "aeronautical": ["aero", "aeronautical", "all branch"],
        }
        branch_matches: dict[str, dict[str, list[str]]] = {
            branch: {} for branch in branch_filters
        }
        for year, companies in company_lists.items():
            for company in companies:
                name = company.get("company_name")
                branches = company.get("eligible_branches")
                vertical = company.get("industry_vertical")
                if not name:
                    continue
                branch_text = str(branches or "").lower()
                for branch_name, aliases in branch_filters.items():
                    if any(alias in branch_text for alias in aliases):
                        branch_matches[branch_name].setdefault(year, []).append(str(name))
                text = (
                    f"In {year}, {name} visited campus for placements. "
                    f"Eligible branches: {branches or 'not specified'}. "
                    f"Industry vertical: {vertical or 'not specified'}."
                )
                records.append({
                    "id": f"company_eligibility:{year}:{name}",
                    "dataset": "structured_facts",
                    "text": text,
                    "metadata": {
                        "category": "placements_company_visits",
                        "source_section": f"company_visited_lists/{year}",
                        "source_file": output.get("source_file", ""),
                        "source_sheet": "Sheet2",
                        "source_table": f"company_visited_lists/{year}",
                        "year": year,
                        "company": name,
                        "eligible_branches": branches or "",
                        "industry_vertical": vertical or "",
                    },
                    "semantic_score": 0.0,
                    "distance": 1.0,
                })
        for branch_name, year_map in branch_matches.items():
            for year, names in year_map.items():
                if not names:
                    continue
                listed = ", ".join(names[:60])
                more = f" and {len(names) - 60} more" if len(names) > 60 else ""
                text = (
                    f"In {year}, these companies allowed or included {branch_name} students "
                    f"for placements based on eligible branch data: {listed}{more}."
                )
                records.append({
                    "id": f"company_eligibility_summary:{branch_name}:{year}",
                    "dataset": "structured_facts",
                    "text": text,
                    "metadata": {
                        "category": "placements_company_visits",
                        "source_section": f"company_visited_lists/{year}",
                        "source_file": output.get("source_file", ""),
                        "source_sheet": "Sheet2",
                        "source_table": f"company_visited_lists/{year}",
                        "year": year,
                        "eligible_branches": branch_name,
                        "company_count": len(names),
                    },
                    "semantic_score": 0.0,
                    "distance": 1.0,
                })

    return records


def lexical_search(query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_terms = tokenize(expand_query(query))
    intents = detect_intents(query)
    results = []
    for record in records:
        metadata = record.get("metadata", {})
        focused_metadata = {
            "category": metadata.get("category", ""),
            "source_section": metadata.get("source_section", ""),
            "source_table": metadata.get("source_table", ""),
            "company": metadata.get("company", ""),
            "eligible_branches": metadata.get("eligible_branches", ""),
            "industry_vertical": metadata.get("industry_vertical", ""),
        }
        text = f"{record.get('text', '')} {json.dumps(focused_metadata, ensure_ascii=False)}"
        text_terms = tokenize(text)
        if not text_terms:
            continue
        overlap_count = len(query_terms & text_terms)
        if overlap_count == 0:
            continue
        category = metadata.get("category", "")
        category_boost = 0.25 if category in intents else 0.0
        structured_boost = 0.6 if is_company_eligibility_query(query) and record["dataset"] == "structured_facts" else 0.0
        primary_boost = 0.2 if primary_intent(query) == category else 0.0
        score = min(overlap_count / max(len(query_terms), 1), 0.8) + category_boost + primary_boost + structured_boost
        candidate = dict(record)
        candidate["semantic_score"] = max(candidate.get("semantic_score", 0.0), score)
        candidate["keyword_overlap"] = score
        results.append(candidate)
    results.sort(key=lambda item: item["semantic_score"], reverse=True)
    return results[:top_k]


def build_context(results: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    used_chars = 0

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source = " / ".join(
            str(value)
            for value in [
                metadata.get("source_file"),
                metadata.get("source_sheet"),
                metadata.get("source_table") or metadata.get("source_section"),
            ]
            if value
        )
        text = re.sub(r"\s+", " ", result["text"]).strip()
        block = (
            f"[{index}] dataset={result['dataset']} "
            f"score={result['rerank_score']:.4f} source={source}\n{text}"
        )
        if used_chars + len(block) > max_chars:
            break
        blocks.append(block)
        used_chars += len(block)

    return "\n\n".join(blocks)


def generate_answer(
    *,
    query: str,
    context: str,
    ollama_base: str,
    ollama_model: str,
    timeout: int,
) -> str:
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
Mention the source bracket numbers when useful.

Retrieved context:
{context}

User question: {query}

Answer:"""

    response = requests.post(
        f"{ollama_base.rstrip('/')}/api/generate",
        json={
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 350,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def print_results(results: list[dict[str, Any]]) -> None:
    print("\nRetrieved context:")
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        print(
            f"\n[{index}] {result['dataset']} "
            f"score={result['rerank_score']:.4f} "
            f"semantic={result['semantic_score']:.4f} "
            f"overlap={result['keyword_overlap']:.2f}"
        )
        print(f"source: {metadata.get('source_sheet', '')} / {metadata.get('source_table', metadata.get('source_section', ''))}")
        print(re.sub(r"\s+", " ", result["text"]).strip()[:900])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve ADCET RAG context and generate a grounded answer.")
    parser.add_argument("query", nargs="?", help="Student/admission question to answer. Omit it to start interactive mode.")
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--embedding-model", help="Override embedding model. Defaults to manifest model.")
    parser.add_argument("--online-model-checks", action="store_true", help="Allow Hugging Face network checks while loading the embedding model.")
    parser.add_argument("--prop-k", type=int, default=8, help="Initial proposition results to retrieve.")
    parser.add_argument("--chunk-k", type=int, default=10, help="Initial main chunk results to retrieve.")
    parser.add_argument("--lexical-k", type=int, default=12, help="Local keyword fallback results to retrieve.")
    parser.add_argument("--final-k", type=int, default=8, help="Final merged context count.")
    parser.add_argument("--max-context-chars", type=int, default=7000)
    parser.add_argument("--no-generate", action="store_true", help="Only retrieve/rerank; do not call Ollama.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context before the answer.")
    parser.add_argument("--ollama-base", default=DEFAULT_OLLAMA_BASE)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-timeout", type=int, default=120)
    return parser.parse_args()


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not args.online_model_checks:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    vector_dir = args.vector_dir.resolve()
    manifest = load_manifest(vector_dir)
    embedding_model_name = args.embedding_model or manifest["embedding_model"]
    print(f"Loading query embedder: {embedding_model_name}")

    try:
        embedder = SentenceTransformer(
            embedding_model_name,
            local_files_only=not args.online_model_checks,
        )
    except TypeError:
        embedder = SentenceTransformer(embedding_model_name)

    client = chromadb.PersistentClient(
        path=str(vector_dir / "chroma"),
        settings=Settings(anonymized_telemetry=False),
    )
    local_records = load_local_records(args.data_dir.resolve())
    return embedder, client, manifest, manifest["datasets"], local_records


def answer_query(
    *,
    query: str,
    args: argparse.Namespace,
    embedder: Any,
    client: Any,
    datasets: dict[str, Any],
    local_records: list[dict[str, Any]],
) -> None:
    query = query.strip()
    if not query:
        return
    expanded_query = expand_query(query)

    query_vector = embedder.encode(
        [expanded_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32").tolist()

    proposition_results = chroma_query(
        client=client,
        collection_name=datasets["propositions"]["collection"],
        query_embedding=query_vector,
        top_k=args.prop_k,
        dataset="propositions",
    )
    chunk_results = chroma_query(
        client=client,
        collection_name=datasets["main_chunks"]["collection"],
        query_embedding=query_vector,
        top_k=args.chunk_k,
        dataset="main_chunks",
    )

    lexical_results = lexical_search(query, local_records, args.lexical_k)
    merged = rerank_results(query, proposition_results + chunk_results + lexical_results, args.final_k)
    context = build_context(merged, args.max_context_chars)

    if args.show_context or args.no_generate:
        print_results(merged)

    if args.no_generate:
        return

    try:
        answer = generate_answer(
            query=query,
            context=context,
            ollama_base=args.ollama_base,
            ollama_model=args.ollama_model,
            timeout=args.ollama_timeout,
        )
    except requests.exceptions.ConnectionError:
        print("\nOllama is not reachable. Start it with `ollama serve`, or rerun with --no-generate to test retrieval only.")
        return
    except requests.HTTPError as exc:
        print(f"\nOllama returned an HTTP error: {exc}")
        return

    print("\nAnswer:")
    print(answer)

    if args.show_context:
        print("\nPrompt context:")
        print(context)


def interactive_loop(
    args: argparse.Namespace,
    embedder: Any,
    client: Any,
    datasets: dict[str, Any],
    local_records: list[dict[str, Any]],
) -> None:
    print("\nADCET RAG interactive mode")
    print("Type a question and press Enter. Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        try:
            query = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.lower() in {"exit", "quit", "q", ":q"}:
            print("Exiting.")
            break

        answer_query(
            query=query,
            args=args,
            embedder=embedder,
            client=client,
            datasets=datasets,
            local_records=local_records,
        )
        print()


def main() -> None:
    args = parse_args()
    embedder, client, _manifest, datasets, local_records = load_runtime(args)

    if args.query:
        answer_query(
            query=args.query,
            args=args,
            embedder=embedder,
            client=client,
            datasets=datasets,
            local_records=local_records,
        )
    else:
        interactive_loop(args, embedder, client, datasets, local_records)


if __name__ == "__main__":
    main()
