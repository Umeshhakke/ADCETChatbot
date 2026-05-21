import hashlib
import json
import re
from pathlib import Path
from typing import Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from knowledge_utils import detect_category, normalize_text
from settings import runtime_config


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def humanize_key(key: str) -> str:
    key = re.sub(r"[_\-]+", " ", str(key))
    key = re.sub(r"\s+", " ", key).strip()
    return key or "item"


def resolve_data_folder() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    configured = Path(runtime_config.knowledge_dir)
    candidates = []

    if configured.is_absolute():
        candidates.append(configured)
    else:
        candidates.extend([
            Path.cwd() / configured,
            project_root / configured,
            project_root / configured.name,
        ])

    candidates.extend([
        project_root / "data_file",
    ])

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Knowledge directory not found. Tried: {', '.join(str(c) for c in candidates)}"
    )


def scalar_to_text(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def structured_to_text(value: Any, indent: int = 0) -> str:
    prefix = "  " * indent

    if isinstance(value, dict):
        lines = []
        for key, child in value.items():
            label = humanize_key(key)
            if isinstance(child, (dict, list)):
                nested = structured_to_text(child, indent + 1)
                lines.append(f"{prefix}{label}:")
                if nested:
                    lines.append(nested)
            else:
                lines.append(f"{prefix}{label}: {scalar_to_text(child)}")
        return "\n".join(lines)

    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                nested = structured_to_text(item, indent + 1)
                if nested:
                    lines.append(nested)
            else:
                lines.append(f"{prefix}- {scalar_to_text(item)}")
        return "\n".join(lines)

    return f"{prefix}{scalar_to_text(value)}"


def compact_list(values: Any) -> str:
    if not isinstance(values, list):
        return scalar_to_text(values)
    return ", ".join(scalar_to_text(value) for value in values)


def add_record(
    docs: list[dict],
    *,
    title: str,
    content: str,
    source: str,
    doc_type: str = "json",
    metadata: dict[str, Any] | None = None,
) -> None:
    cleaned_content = clean_text(content)
    if not cleaned_content:
        return

    cleaned_title = clean_text(title)
    docs.append({
        "title": cleaned_title,
        "content": cleaned_content,
        "url": source,
        "type": doc_type,
        "metadata": metadata or {},
    })


def document_required_label(document: dict) -> str:
    name = scalar_to_text(document.get("document_name", "")).strip()
    required = document.get("required")
    condition = scalar_to_text(document.get("condition", "")).strip()

    if required is True:
        return f"- {name}: required"
    if condition:
        return f"- {name}: may be required ({condition})"
    if required is False:
        return f"- {name}: may be required"
    return f"- {name}"


def extract_document_records(data: dict, source: str) -> list[dict]:
    docs: list[dict] = []
    for admission in data.get("admissions", []):
        admission_type = scalar_to_text(admission.get("admission_type", "")).strip()
        for category in admission.get("categories", []):
            category_name = scalar_to_text(category.get("category_name", "")).strip()
            documents = category.get("required_documents", [])
            lines = [
                f"Admission type: {admission_type}",
                f"Category name: {category_name}",
                "Required documents:",
                *(document_required_label(document) for document in documents),
            ]
            add_record(
                docs,
                title=f"Admission documents - {admission_type} - {category_name}",
                content="\n".join(lines),
                source=source,
                metadata={
                    "admission_type": admission_type,
                    "category_name": category_name,
                    "record_kind": "admission_documents",
                },
            )
    return docs


def extract_cutoff_records(data: dict, source: str) -> list[dict]:
    docs: list[dict] = []
    for group, courses in data.items():
        if not isinstance(courses, dict):
            continue
        for course, categories in courses.items():
            if not isinstance(categories, dict):
                continue
            course_lines = [f"Group: {group}", f"Course: {course}", "Cutoff records:"]
            for category, record in categories.items():
                if not isinstance(record, dict):
                    continue
                merit_no = scalar_to_text(record.get("merit_no", "Data not found"))
                merit_marks = scalar_to_text(record.get("merit_marks", "Data not found"))
                course_lines.append(
                    f"- Category {category}: merit marks {merit_marks}; merit number {merit_no}"
                )
                add_record(
                    docs,
                    title=f"Cutoff - {group} - {course} - {category}",
                    content="\n".join([
                        f"Group: {group}",
                        f"Course: {course}",
                        f"Category: {category}",
                        f"Merit marks: {merit_marks}",
                        f"Merit number: {merit_no}",
                    ]),
                    source=source,
                    metadata={
                        "group": group,
                        "course": course,
                        "category_name": category,
                        "record_kind": "cutoff",
                    },
                )
            add_record(
                docs,
                title=f"Cutoff summary - {group} - {course}",
                content="\n".join(course_lines),
                source=source,
                metadata={"group": group, "course": course, "record_kind": "cutoff_summary"},
            )
    return docs


def extract_bus_records(data: dict, source: str) -> list[dict]:
    docs: list[dict] = []
    for route in data.get("bus_routes", []):
        route_name = scalar_to_text(route.get("route_name", "")).strip()
        stops = route.get("stops", [])
        summary = [f"Bus route: {route_name}", "Stops and monthly fees:"]
        for stop in stops:
            stop_name = scalar_to_text(stop.get("stop_name", "")).strip()
            monthly_fee = scalar_to_text(stop.get("monthly_fee", "")).strip()
            summary.append(f"- {stop_name}: monthly fee {monthly_fee}")
            add_record(
                docs,
                title=f"Bus fee - {route_name} - {stop_name}",
                content="\n".join([
                    f"Bus route: {route_name}",
                    f"Stop name: {stop_name}",
                    f"Monthly fee: {monthly_fee}",
                ]),
                source=source,
                metadata={"route_name": route_name, "stop_name": stop_name, "record_kind": "bus_fee"},
            )
        add_record(
            docs,
            title=f"Bus route summary - {route_name}",
            content="\n".join(summary),
            source=source,
            metadata={"route_name": route_name, "record_kind": "bus_route"},
        )
    return docs


def extract_college_fee_records(data: dict, source: str) -> list[dict]:
    docs: list[dict] = []
    root = data.get("college_fees", {})
    year = scalar_to_text(root.get("academic_year", "")).strip()
    for program in root.get("programs", []):
        category = scalar_to_text(program.get("category", "")).strip()
        fees = program.get("fees", {})
        add_record(
            docs,
            title=f"College fees - {category}",
            content="\n".join([
                f"Academic year: {year}",
                f"Category: {category}",
                f"FY fee: {scalar_to_text(fees.get('FY', 'Not available'))}",
                f"DSE fee: {scalar_to_text(fees.get('DSE', 'Not available'))}",
            ]),
            source=source,
            metadata={"academic_year": year, "category_name": category, "record_kind": "college_fee"},
        )
    return docs


def extract_hostel_records(data: dict, source: str) -> list[dict]:
    docs: list[dict] = []
    root = data.get("hostel_details", {})
    year = scalar_to_text(root.get("academic_year", "")).strip()
    for hostel in root.get("hostels", []):
        hostel_name = scalar_to_text(hostel.get("hostel_name", "")).strip()
        hostel_type = scalar_to_text(hostel.get("hostel_type", "")).strip()
        add_record(
            docs,
            title=f"Hostel fee - {hostel_name}",
            content="\n".join([
                f"Academic year: {year}",
                f"Hostel name: {hostel_name}",
                f"Hostel type: {hostel_type}",
                f"Annual fees: {scalar_to_text(hostel.get('annual_fees', 'Not available'))}",
            ]),
            source=source,
            metadata={
                "academic_year": year,
                "hostel_name": hostel_name,
                "hostel_type": hostel_type,
                "record_kind": "hostel_fee",
            },
        )
    return docs


def extract_program_records(data: dict, source: str) -> list[dict]:
    docs: list[dict] = []
    root = data.get("academic_programs", {})
    for level_key in ["undergraduate_programs", "postgraduate_programs"]:
        level = humanize_key(level_key)
        programs = root.get(level_key, [])
        summary_lines = [f"Program level: {level}", f"Total programs: {len(programs)}", "Programs:"]
        programs_by_degree: dict[str, list[dict]] = {}
        for program in root.get(level_key, []):
            program_name = scalar_to_text(program.get("program_name", "")).strip()
            degree = scalar_to_text(program.get("degree", "")).strip()
            intake = scalar_to_text(program.get("sanctioned_intake", "Not available"))
            summary_lines.append(f"- {degree} {program_name}: sanctioned intake {intake}")
            programs_by_degree.setdefault(degree, []).append(program)
            add_record(
                docs,
                title=f"Academic program - {degree} - {program_name}",
                content="\n".join([
                    f"Program level: {level}",
                    f"Program name: {program_name}",
                    f"Degree: {degree}",
                    f"Program code: {scalar_to_text(program.get('program_code', 'Not available'))}",
                    f"Sanctioned intake: {scalar_to_text(program.get('sanctioned_intake', 'Not available'))}",
                    f"Year started: {scalar_to_text(program.get('year_started', 'Not available'))}",
                ]),
                source=source,
                metadata={
                    "program_name": program_name,
                    "degree": degree,
                    "program_level": level,
                    "record_kind": "academic_program",
                },
            )
        add_record(
            docs,
            title=f"Academic program summary - {level}",
            content="\n".join(summary_lines),
            source=source,
            metadata={"program_level": level, "record_kind": "academic_program_summary"},
        )
        for degree, degree_programs in programs_by_degree.items():
            degree_lines = [
                f"Program level: {level}",
                f"Degree: {degree}",
                f"Total {degree} programs: {len(degree_programs)}",
                "Programs:",
            ]
            for program in degree_programs:
                program_name = scalar_to_text(program.get("program_name", "")).strip()
                intake = scalar_to_text(program.get("sanctioned_intake", "Not available"))
                degree_lines.append(f"- {program_name}: sanctioned intake {intake}")
            add_record(
                docs,
                title=f"Academic program summary - {degree}",
                content="\n".join(degree_lines),
                source=source,
                metadata={
                    "program_level": level,
                    "degree": degree,
                    "record_kind": "academic_program_degree_summary",
                },
            )
    return docs


def extract_eligibility_records(data: dict, source: str) -> list[dict]:
    docs: list[dict] = []
    root = data.get("engineering_admission_eligibility", {})
    subjects = compact_list(root.get("academic_requirement", {}).get("required_subjects", []))
    alternative = scalar_to_text(root.get("academic_requirement", {}).get("alternative_combination", "")).strip()
    program = scalar_to_text(root.get("program", "")).strip()
    for category in root.get("categories", []):
        category_name = scalar_to_text(category.get("category", "")).strip()
        add_record(
            docs,
            title=f"Admission eligibility - {program} - {category_name}",
            content="\n".join([
                f"Program: {program}",
                f"Category: {category_name}",
                f"Required subjects: {subjects}",
                f"Alternative combination: {alternative}",
                f"Minimum marks: {scalar_to_text(category.get('minimum_marks', 'Not available'))}",
            ]),
            source=source,
            metadata={"program": program, "category_name": category_name, "record_kind": "admission_eligibility"},
        )
    return docs


def extract_placement_summary_records(data: dict, source: str) -> list[dict]:
    docs: list[dict] = []
    root = data.get("placement_3yr_summary", {})
    for branch_record in root.get("placement_summary_by_branch", []):
        branch = scalar_to_text(branch_record.get("branch", "")).strip()
        lines = [f"Branch: {branch}", "Placement summary by academic year:"]
        for year, stats in branch_record.items():
            if year == "branch" or not isinstance(stats, dict):
                continue
            lines.append(
                f"- {year}: total students {scalar_to_text(stats.get('total_students', 'Not available'))}; "
                f"students placed/offers {scalar_to_text(stats.get('students_placed_offers', 'Not available'))}; "
                f"average salary LPA {scalar_to_text(stats.get('avg_salary_lpa', 'Not available'))}; "
                f"highest salary LPA {scalar_to_text(stats.get('highest_salary_lpa', 'Not available'))}"
            )
        add_record(
            docs,
            title=f"Placement summary - {branch}",
            content="\n".join(lines),
            source=source,
            metadata={"branch": branch, "record_kind": "placement_summary"},
        )

    totals = root.get("overall_totals", {})
    if isinstance(totals, dict):
        add_record(
            docs,
            title="Placement overall totals",
            content="Overall placement totals:\n" + structured_to_text(totals),
            source=source,
            metadata={"record_kind": "placement_totals"},
        )
    return docs


def extract_company_visit_records(data: dict, source: str) -> list[dict]:
    docs: list[dict] = []
    for year, companies in data.get("placement_company_visits", {}).items():
        if not isinstance(companies, list):
            continue
        year_lines = [f"Academic year: {year}", "Companies visited:"]
        branch_index: dict[str, list[str]] = {}
        for company in companies:
            company_name = scalar_to_text(company.get("company", "")).strip()
            sector = scalar_to_text(company.get("sector", "")).strip()
            branches = compact_list(company.get("eligible_branches", []))
            year_lines.append(f"- {company_name} ({sector}); eligible branches: {branches}")
            for branch in company.get("eligible_branches", []):
                branch_name = scalar_to_text(branch).strip()
                branch_index.setdefault(branch_name, []).append(f"{company_name} ({sector})")
            add_record(
                docs,
                title=f"Company visit - {year} - {company_name}",
                content="\n".join([
                    f"Academic year: {year}",
                    f"Company: {company_name}",
                    f"Sector: {sector}",
                    f"Eligible branches: {branches}",
                ]),
                source=source,
                metadata={
                    "academic_year": year,
                    "company": company_name,
                    "sector": sector,
                    "eligible_branches": branches,
                    "record_kind": "company_visit",
                },
            )
        add_record(
            docs,
            title=f"Company visits summary - {year}",
            content="\n".join(year_lines),
            source=source,
            metadata={"academic_year": year, "record_kind": "company_visit_summary"},
        )
        for branch_name, company_names in branch_index.items():
            add_record(
                docs,
                title=f"Company visits - {year} - {branch_name}",
                content="\n".join([
                    f"Academic year: {year}",
                    f"Eligible branch: {branch_name}",
                    "Companies:",
                    *(f"- {company_name}" for company_name in company_names),
                ]),
                source=source,
                metadata={
                    "academic_year": year,
                    "eligible_branches": branch_name,
                    "record_kind": "company_visit_branch_summary",
                },
            )
    return docs


def extract_source_documents(data: Any, json_file: Path) -> list[dict]:
    source = json_file.name
    extractors = {
        "document.json": extract_document_records,
        "cutoff.json": extract_cutoff_records,
        "bus.json": extract_bus_records,
        "college_fees.json": extract_college_fee_records,
        "hostel.json": extract_hostel_records,
        "program_offered.json": extract_program_records,
        "elligibility.json": extract_eligibility_records,
        "3yr.json": extract_placement_summary_records,
        "company_visited.json": extract_company_visit_records,
    }
    extractor = extractors.get(source)
    if extractor and isinstance(data, dict):
        return extractor(data, source)
    return iter_json_documents(data, humanize_key(json_file.stem), source)


def iter_json_documents(value: Any, title: str, source: str, depth: int = 0) -> list[dict]:
    docs: list[dict] = []

    if isinstance(value, dict):
        explicit_content = value.get("content") or value.get("text") or value.get("answer")
        explicit_title = value.get("title") or value.get("question") or title

        if explicit_content:
            docs.append({
                "title": clean_text(str(explicit_title)),
                "content": clean_text(str(explicit_content)),
                "url": source,
                "type": "json",
            })
            return docs

        scalar_only = all(not isinstance(child, (dict, list)) for child in value.values())
        has_scalar_context = any(not isinstance(child, (dict, list)) for child in value.values())
        has_nested_context = any(isinstance(child, (dict, list)) for child in value.values())

        if scalar_only:
            docs.append({
                "title": clean_text(title),
                "content": clean_text(structured_to_text(value)),
                "url": source,
                "type": "json",
            })
            return docs

        if depth > 0 and (has_scalar_context or has_nested_context):
            aggregate_content = clean_text(structured_to_text(value))
            if aggregate_content:
                docs.append({
                    "title": clean_text(title),
                    "content": aggregate_content,
                    "url": source,
                    "type": "json",
                })

        for key, child in value.items():
            child_title = f"{title} - {humanize_key(key)}"
            docs.extend(iter_json_documents(child, child_title, source, depth + 1))
        return docs

    if isinstance(value, list):
        for index, item in enumerate(value, 1):
            if isinstance(item, (dict, list)):
                docs.extend(iter_json_documents(item, f"{title} - item {index}", source, depth + 1))
            else:
                docs.append({
                    "title": clean_text(title),
                    "content": clean_text(scalar_to_text(item)),
                    "url": source,
                    "type": "json",
                })
        return docs

    docs.append({
        "title": clean_text(title),
        "content": clean_text(scalar_to_text(value)),
        "url": source,
        "type": "json",
    })
    return docs


def load_documents() -> list[dict]:
    data_folder = resolve_data_folder()

    all_documents = []

    for json_file in data_folder.glob("*.json"):
        print(f"Loading: {json_file.name}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_documents.extend(extract_source_documents(data, json_file))

    return all_documents

def build_chunks(documents: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=runtime_config.chunk_size,
        chunk_overlap=runtime_config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []
    seen_hashes: set[str] = set()

    for doc_index, document in tqdm(enumerate(documents), total=len(documents), desc="Chunking"):
        content = clean_text(document.get("content", ""))

        if len(content) < min(runtime_config.min_chunk_chars, 25):
            continue

        title = clean_text(document.get("title", ""))
        source = document.get("url", "").strip()
        doc_type = document.get("type", "webpage").strip()
        category = detect_category(source, title, content)
        extra_metadata = document.get("metadata", {}) if isinstance(document.get("metadata"), dict) else {}

        for chunk_index, chunk in enumerate(splitter.split_text(content)):
            chunk = clean_text(chunk)

            if len(chunk) < min(runtime_config.min_chunk_chars, 25):
                continue

            stored_chunk = f"{title}\n{chunk}" if title and title.lower() not in chunk[:200].lower() else chunk
            chunk_hash = hashlib.md5(stored_chunk.encode("utf-8")).hexdigest()

            if chunk_hash in seen_hashes:
                continue

            seen_hashes.add(chunk_hash)

            all_ids.append(f"doc_{doc_index}_chunk_{chunk_index}")

            all_chunks.append(stored_chunk)

            all_metadatas.append(
                {
                    "source": source,
                    "title": title[:200],
                    "type": doc_type,
                    "category": category,
                    "chunk_index": chunk_index,
                    **{
                        key: scalar_to_text(value)[:500]
                        for key, value in extra_metadata.items()
                        if value is not None and scalar_to_text(value).strip()
                    },
                }
            )

    return all_chunks, all_metadatas, all_ids


def store_embeddings(chunks: list[str], metadatas: list[dict], ids: list[str]) -> None:
    print(f"Loading embedding model: {runtime_config.embedding_model}")

    embedder = SentenceTransformer(
        runtime_config.embedding_model,
        local_files_only=runtime_config.hf_local_files_only,
    )

    print("Creating embeddings...")

    embeddings = embedder.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    print(f"Writing Chroma collection to {runtime_config.chroma_path}")

    client = chromadb.PersistentClient(path=runtime_config.chroma_path)

    try:
        client.delete_collection(runtime_config.collection_name)
        print(f"Deleted existing collection: {runtime_config.collection_name}")
    except Exception:
        pass

    collection = client.create_collection(name=runtime_config.collection_name)

    batch_size = 500

    for start in tqdm(range(0, len(chunks), batch_size), desc="Saving"):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


def main() -> None:
    print(f"Loading knowledge folder: {runtime_config.knowledge_dir}")

    documents = load_documents()

    print(f"Loaded {len(documents)} source documents")

    chunks, metadatas, ids = build_chunks(documents)

    if not chunks:
        raise ValueError("No valid chunks were generated from JSON files.")

    print(f"Prepared {len(chunks)} chunks")

    store_embeddings(chunks, metadatas, ids)

    print("\nKnowledge base build complete")
    print(f"Collection: {runtime_config.collection_name}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding model: {runtime_config.embedding_model}")


if __name__ == "__main__":
    main()
