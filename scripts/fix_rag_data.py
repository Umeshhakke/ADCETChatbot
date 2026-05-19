"""
Repair and regenerate RAG-ready data artifacts from output.json.

This script removes unsupported LLM prose from the prepared artifacts and
rebuilds deterministic, source-grounded records for vectorization.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "output.json"
KB_PATH = ROOT / "knowledge_base.jsonl"
PROP_PATH = ROOT / "propositions.jsonl"
SUMMARY_PATH = ROOT / "summary_index.json"
QA_PATH = ROOT / "qa_pairs.json"

SOURCE_SHEETS = {
    "programs": "Sheet1",
    "programs_offered": "Sheet1",
    "fees": "FY 27",
    "hostel": "FY 27",
    "transport": "FY 27",
    "cutoffs": "FY 27",
    "eligibility": "FY 27",
    "placements_summary": "TPO Data",
    "placements_company_visits": "Sheet2",
    "admission_documents": "Doc",
}

BRANCH_ALIASES = {
    "Mech": ["Mechanical Engineering", "Mechanical", "Mech"],
    "AERO": ["Aeronautical Engineering", "Aeronautical", "AERO"],
    "CSE": ["Computer Science & Engineering", "Computer Science Engineering", "CSE"],
    "AIDS": ["Artificial Intelligence and Data Science", "AI and Data Science", "AIDS"],
    "RAI": ["Robotics And Artificial Intelligence", "Robotics AI", "RAI"],
    "CIVIL": ["Civil Engineering", "Civil", "CIVIL"],
    "ELECT": ["Electrical Engineering", "Electrical", "ELECT"],
    "Food": ["Food Technology", "Food"],
    "CSE IOT": [
        "Computer Science & Engineering IoT Cyber Security Blockchain",
        "CSE IoT",
        "CSE IOT",
    ],
}

CUTOFF_CATEGORIES = ["VJ", "NT-1", "NT-2", "NT-3", "OBC", "SEBC", "OPEN", "SC", "ST"]


def load_data() -> dict[str, Any]:
    with INPUT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def fmt_inr(value: Any) -> str:
    if value is None:
        return "not specified"
    try:
        number = float(str(value).replace(",", "").replace("/-", "").replace("/", "").strip())
        return f"Rs.{number:,.0f}"
    except ValueError:
        return str(value)


def fmt_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "not specified"
    return f"{value}{suffix}"


def dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def fix_program_classification(data: dict[str, Any]) -> None:
    programs = data.get("sheets", {}).get("programs_offered", {})
    btech = programs.get("btech", [])
    ug_other = programs.get("ug_other", [])
    fixed_btech = []

    for item in btech:
        name = str(item.get("program", ""))
        if "Bachelor of Business Administration" in name or "Bachelor of Computer Application" in name:
            ug_other.append(item)
        else:
            fixed_btech.append(item)

    programs["btech"] = fixed_btech
    programs["ug_other"] = ug_other


def context_for(category: str, source_section: str, metadata: dict[str, Any], year: str) -> str:
    sheet = metadata.get("source_sheet", SOURCE_SHEETS.get(category, "unknown"))
    table = metadata.get("source_table", source_section)
    return f"Source-grounded {category} data for academic year {year}; source sheet: {sheet}; source table: {table}."


def make_chunk(
    *,
    data: dict[str, Any],
    category: str,
    source_section: str,
    text: str,
    propositions: list[str],
    metadata: dict[str, Any],
    entities: dict[str, list[str]],
    keywords: list[str],
    summary: str,
    parent_id: str | None = None,
) -> dict[str, Any]:
    college = data["college"]
    year = data["academic_year"]
    chunk_id = str(uuid.uuid4())
    metadata = {
        "source_file": data["source_file"],
        "source_sheet": SOURCE_SHEETS.get(category, metadata.get("source_sheet")),
        "source_table": source_section,
        **metadata,
        "college": college,
        "year": year,
        "word_count": len(text.split()),
        "char_count": len(text),
    }
    for key in ["branches", "years", "companies", "amounts_inr", "categories", "locations"]:
        entities.setdefault(key, [])
        entities[key] = dedupe(entities[key])

    header = context_for(category, source_section, metadata, year)
    token_source = f"{text} {' '.join(keywords)} {' '.join(sum(entities.values(), []))}"
    return {
        "id": chunk_id,
        "parent_id": parent_id,
        "category": category,
        "source_section": source_section,
        "text": text,
        "context_header": header,
        "contextualised_text": f"{header}\n\n{text}",
        "one_line_summary": summary,
        "propositions": propositions,
        "hypothetical_questions": make_questions(category, metadata, entities),
        "bm25_tokens": dedupe(words(token_source)),
        "metadata": metadata,
        "entities": entities,
        "keywords": dedupe(keywords),
        "quality_score": quality_score(text, propositions),
        "low_quality_flag": False,
        "embedding_model": None,
        "embedding": None,
    }


def quality_score(text: str, propositions: list[str]) -> float:
    numeric = len(re.findall(r"\d+", text))
    score = 0.35 + min(len(text.split()) / 180, 0.25) + min(len(propositions) / 10, 0.25) + min(numeric / 12, 0.15)
    return round(min(score, 1.0), 3)


def make_questions(category: str, metadata: dict[str, Any], entities: dict[str, list[str]]) -> list[str]:
    if category == "fees":
        return ["What are the FY and DSE fees by category?", "What is the OPEN category fee?", "Which fee categories are listed?"]
    if category == "hostel":
        return ["What are the hostel fees?", "Which hostel has what annual fee?", "What hostel options are available?"]
    if category == "transport":
        route = metadata.get("route", "this route")
        return [f"What are the bus fees for {route}?", f"Which stops are on {route}?", "What monthly transport fees are listed?"]
    if category == "cutoffs":
        branch = metadata.get("course", "this branch")
        return [f"What are the cutoffs for {branch}?", f"What is the OPEN cutoff for {branch}?", f"What are the category-wise marks for {branch}?"]
    if category == "programs":
        label = metadata.get("label", "programs")
        return [f"Which {label} programs are offered?", "What is the sanctioned intake?", "When did the programs start?"]
    if category.startswith("placements"):
        return ["What are the placement details?", "Which companies visited?", "What salary packages are listed?"]
    if category == "admission_documents":
        return ["Which documents are required for admission?", "What documents are needed by category?", "What should students bring for admission?"]
    return ["What information is available in this section?"]


def build_program_chunks(data: dict[str, Any]) -> list[dict[str, Any]]:
    programs = data["sheets"].get("programs_offered", {})
    label_map = {"btech": "B.Tech", "ug_other": "UG non-engineering", "mtech": "M.Tech"}
    chunks = []
    for key, rows in programs.items():
        if not rows:
            continue
        lines = [
            f"{item['program']} has sanctioned intake {item.get('sanctioned_intake')} and started in {item.get('year_of_starting')}."
            for item in rows
        ]
        props = list(lines)
        label = label_map.get(key, key)
        text = f"{label} programs offered at {data['college']}: " + " ".join(lines)
        branches = [item.get("program") for item in rows]
        chunks.append(make_chunk(
            data=data,
            category="programs",
            source_section=f"programs_offered/{key}",
            text=text,
            propositions=props,
            metadata={
                "label": label,
                "programs": branches,
                "program_count": len(rows),
                "raw_rows": rows,
            },
            entities={"branches": branches, "years": [item.get("year_of_starting") for item in rows]},
            keywords=["programs", "course", label, "intake", "sanctioned intake"],
            summary=f"{label} programs with sanctioned intake and year of starting.",
        ))
    return chunks


def build_fee_chunks(data: dict[str, Any], sheet: dict[str, Any]) -> list[dict[str, Any]]:
    fees = sheet.get("tuition_fees_2025_26", [])
    props = []
    amounts = []
    for row in fees:
        props.append(f"For {row['category']}, the FY fee is {fmt_inr(row.get('fy_fee'))} and the DSE fee is {fmt_inr(row.get('dse_fee'))}.")
        amounts.extend([fmt_inr(row.get("fy_fee")), fmt_inr(row.get("dse_fee"))])
    text = "Annual tuition fee structure for 2025-26: " + " ".join(props)
    return [make_chunk(
        data=data,
        category="fees",
        source_section="fy27_cutoff_bus_hostel_fees/tuition_fees_2025_26",
        text=text,
        propositions=props,
        metadata={"fee_rows": fees, "categories": [f.get("category") for f in fees]},
        entities={"amounts_inr": amounts, "categories": [f.get("category") for f in fees]},
        keywords=["fees", "tuition", "FY", "DSE", "OPEN", "EBC", "EWS", "OBC", "TFWS", "SC", "ST"],
        summary="FY and DSE tuition fees by reservation category for 2025-26.",
    )] if fees else []


def build_hostel_chunks(data: dict[str, Any], sheet: dict[str, Any]) -> list[dict[str, Any]]:
    hostels = sheet.get("hostels", [])
    props = [f"{h['hostel']} annual fee is {fmt_inr(h.get('annual_fee'))}." for h in hostels]
    return [make_chunk(
        data=data,
        category="hostel",
        source_section="fy27_cutoff_bus_hostel_fees/hostels",
        text="Hostel fee details for 2025-26: " + " ".join(props),
        propositions=props,
        metadata={"hostels": hostels, "count": len(hostels)},
        entities={"amounts_inr": [fmt_inr(h.get("annual_fee")) for h in hostels]},
        keywords=["hostel", "annual fee", "ladies hostel", "boys hostel"],
        summary="Annual hostel fees for ladies and boys hostels.",
    )] if hostels else []


def build_transport_chunks(data: dict[str, Any], sheet: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = []
    for route, stops in sheet.get("bus_routes", {}).items():
        props = [f"On the {route} bus route, {stop['stop']} has a monthly fee of {fmt_inr(stop.get('fee'))}." for stop in stops]
        text = f"Bus route {route} monthly fees: " + " ".join(props)
        chunks.append(make_chunk(
            data=data,
            category="transport",
            source_section=f"fy27_cutoff_bus_hostel_fees/bus_routes/{route}",
            text=text,
            propositions=props,
            metadata={"route": route, "stops": stops, "stop_count": len(stops)},
            entities={"amounts_inr": [fmt_inr(s.get("fee")) for s in stops], "locations": [route] + [s.get("stop") for s in stops]},
            keywords=["bus", "transport", "route", route, "monthly fee"],
            summary=f"Monthly bus fees for the {route} route.",
        ))
    return chunks


def build_cutoff_chunks(data: dict[str, Any], sheet: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = []
    for entry in sheet.get("cutoff_data", []):
        course = entry.get("course")
        cutoffs = entry.get("cutoff", {})
        props = []
        records = []
        for cat in CUTOFF_CATEGORIES:
            seats = cutoffs.get(cat, {})
            for seat_type in ["G", "L"]:
                values = seats.get(seat_type, {}) if isinstance(seats, dict) else {}
                merit_no = values.get("merit_no")
                marks = values.get("merit_marks")
                records.append({"course": course, "category": cat, "seat_type": seat_type, "merit_no": merit_no, "merit_marks": marks})
                if marks is None:
                    props.append(f"For {course}, {cat} {seat_type} seat cutoff is not specified.")
                else:
                    props.append(f"For {course}, {cat} {seat_type} seat cutoff is merit number {merit_no} with {marks:.2f} marks.")
        aliases = BRANCH_ALIASES.get(course, [course])
        chunks.append(make_chunk(
            data=data,
            category="cutoffs",
            source_section=f"fy27_cutoff_bus_hostel_fees/cutoffs/{course}",
            text=f"MHT-CET cutoff data for {course}: " + " ".join(props),
            propositions=props,
            metadata={"course": course, "branch_aliases": aliases, "cutoff_records": records, "categories": CUTOFF_CATEGORIES},
            entities={"branches": aliases, "categories": CUTOFF_CATEGORIES},
            keywords=["cutoff", "MHT-CET", "merit marks", "merit number", course] + CUTOFF_CATEGORIES + aliases,
            summary=f"Category-wise G and L seat cutoff marks for {course}.",
        ))
    return chunks


def build_eligibility_chunks(data: dict[str, Any], sheet: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = sheet.get("qualifying_criteria_12th")
    if not criteria:
        return []
    raw = json.dumps(criteria, ensure_ascii=False)
    text = f"12th qualifying criteria for admission are recorded as: {raw}"
    return [make_chunk(
        data=data,
        category="eligibility",
        source_section="fy27_cutoff_bus_hostel_fees/qualifying_criteria_12th",
        text=text,
        propositions=[text],
        metadata={"criteria": criteria},
        entities={"categories": ["OPEN", "reserved"]},
        keywords=["eligibility", "12th", "qualifying criteria", "PCM", "PCB"],
        summary="12th qualifying criteria for admission.",
    )]


def build_placement_chunks(data: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = []
    summary = data["sheets"].get("placement_3yr_summary", {})
    by_branch = summary.get("placement_summary_by_branch", [])
    totals = summary.get("overall_totals", {})
    companies = summary.get("companies_visited", {})
    if totals or companies:
        props = []
        years = []
        for yr, row in totals.items():
            years.append(yr)
            props.append(f"In {yr}, overall placement offers were {fmt_value(row.get('students_placed_offers'))} and total students were {fmt_value(row.get('total_students'))}.")
        for yr, count in companies.items():
            years.append(yr)
            props.append(f"In {yr}, companies visited count was {fmt_value(count)}.")
        chunks.append(make_chunk(
            data=data,
            category="placements_summary",
            source_section="placement_3yr_summary/overall",
            text="Overall placement summary: " + " ".join(props),
            propositions=props,
            metadata={"totals": totals, "companies_visited": companies},
            entities={"years": years},
            keywords=["placements", "offers", "companies visited", "salary", "LPA"],
            summary="Overall placement offers and company visit counts by year.",
        ))
    for branch in by_branch:
        name = branch.get("branch")
        props = []
        years = []
        for yr in ["2023-24", "2024-25", "2025-26"]:
            row = branch.get(yr, {})
            if not row:
                continue
            years.append(yr)
            props.append(
                f"For {name} in {yr}, total students were {fmt_value(row.get('total_students'))}, offers were {fmt_value(row.get('students_placed_offers'))}, average salary was {fmt_value(row.get('avg_salary_lpa'), ' LPA')}, and highest salary was {fmt_value(row.get('highest_salary_lpa'), ' LPA')}."
            )
        chunks.append(make_chunk(
            data=data,
            category="placements_summary",
            source_section=f"placement_3yr_summary/{name}",
            text=f"Branch placement summary for {name}: " + " ".join(props),
            propositions=props,
            metadata={"branch": name, "year_rows": {yr: branch.get(yr) for yr in years}},
            entities={"branches": [name], "years": years},
            keywords=["placements", name, "offers", "average salary", "highest salary", "LPA"],
            summary=f"Three-year placement summary for {name}.",
        ))
    for yr, companies_list in data["sheets"].get("company_visited_lists", {}).items():
        names = [c.get("company_name") for c in companies_list if c.get("company_name")]
        props = [f"In {yr}, {name} visited campus for placements." for name in names]
        chunks.append(make_chunk(
            data=data,
            category="placements_company_visits",
            source_section=f"company_visited_lists/{yr}",
            text=f"Company visits for {yr}: " + " ".join(props),
            propositions=props,
            metadata={"placement_year": yr, "total_companies": len(names), "companies": companies_list},
            entities={"years": [yr], "companies": names},
            keywords=["companies", "campus recruitment", "placement drive", yr] + names[:20],
            summary=f"{len(names)} companies listed as campus visitors in {yr}.",
        ))
    return chunks


def build_document_chunks(data: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = []
    for label, docs in data["sheets"].get("admission_documents", {}).items():
        names = [d.get("document") for d in docs if d.get("document")]
        props = [f"For {label}, document required or listed: {name}." for name in names]
        chunks.append(make_chunk(
            data=data,
            category="admission_documents",
            source_section=f"admission_documents/{label[:50]}",
            text=f"Admission documents for {label}: " + " ".join(props),
            propositions=props,
            metadata={"admission_type": label, "documents": docs, "doc_count": len(names)},
            entities={"categories": ["OPEN", "EBC", "TFWS", "EWS", "SC", "ST", "VJ", "VJNT", "NT1", "NT2", "NT3", "OBC", "SBC", "SEBC"]},
            keywords=["documents", "admission", "certificate", "marksheet", "scholarship"],
            summary=f"Required admission documents for {label}.",
        ))
    return chunks


def build_qa(chunks: list[dict[str, Any]], college: str) -> list[dict[str, Any]]:
    pairs = []
    for chunk in chunks:
        propositions = chunk.get("propositions", [])
        if not propositions:
            continue
        question = chunk["hypothetical_questions"][0] if chunk.get("hypothetical_questions") else f"What does {chunk['source_section']} contain?"
        answer = " ".join(propositions[:3])
        pairs.append({
            "id": str(uuid.uuid4()),
            "source_chunk_id": chunk["id"],
            "category": chunk["category"],
            "source_section": chunk["source_section"],
            "question_type": "factual",
            "question": question,
            "answer": answer,
            "keywords": chunk["keywords"],
            "college": college,
            "needs_manual_review": False,
        })
    return pairs


def build_all(data: dict[str, Any]) -> list[dict[str, Any]]:
    sheet = data["sheets"].get("fy27_cutoff_bus_hostel_fees", {})
    chunks = []
    chunks.extend(build_program_chunks(data))
    chunks.extend(build_fee_chunks(data, sheet))
    chunks.extend(build_hostel_chunks(data, sheet))
    chunks.extend(build_transport_chunks(data, sheet))
    chunks.extend(build_cutoff_chunks(data, sheet))
    chunks.extend(build_eligibility_chunks(data, sheet))
    chunks.extend(build_placement_chunks(data))
    chunks.extend(build_document_chunks(data))
    return chunks


def main() -> None:
    data = load_data()
    fix_program_classification(data)
    write_json(INPUT_PATH, data)

    chunks = build_all(data)
    write_jsonl(KB_PATH, chunks)

    propositions = []
    for chunk in chunks:
        for prop in chunk.get("propositions", []):
            propositions.append({
                "id": str(uuid.uuid4()),
                "parent_chunk_id": chunk["id"],
                "category": chunk["category"],
                "source_section": chunk["source_section"],
                "text": prop,
                "keywords": chunk["keywords"],
                "metadata": chunk["metadata"],
                "entities": chunk["entities"],
                "embedding": None,
            })
    write_jsonl(PROP_PATH, propositions)

    summaries = [{
        "chunk_id": chunk["id"],
        "parent_id": chunk.get("parent_id"),
        "category": chunk["category"],
        "source_section": chunk["source_section"],
        "summary": chunk["one_line_summary"],
        "keywords": chunk["keywords"][:8],
        "quality_score": chunk["quality_score"],
        "low_quality": chunk["low_quality_flag"],
        "metadata": {
            "source_file": chunk["metadata"].get("source_file"),
            "source_sheet": chunk["metadata"].get("source_sheet"),
            "source_table": chunk["metadata"].get("source_table"),
        },
    } for chunk in chunks]
    write_json(SUMMARY_PATH, summaries)
    write_json(QA_PATH, build_qa(chunks, data["college"]))

    print(f"Fixed {INPUT_PATH.name}")
    print(f"Wrote {len(chunks)} chunks to {KB_PATH.name}")
    print(f"Wrote {len(propositions)} propositions to {PROP_PATH.name}")
    print(f"Wrote {len(summaries)} summaries to {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
