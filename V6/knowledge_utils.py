from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "can",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "me",
    "of",
    "on",
    "or",
    "please",
    "tell",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "will",
}


CATEGORY_KEYWORDS = {
    "admission": [
        "admission",
        "admissions",
        "apply",
        "application",
        "eligibility",
        "management quota",
        "jee",
        "mht cet",
        "cap round",
        "seat matrix",
        "documents",
        "document",
    ],
    "fees": [
        "fee",
        "fees",
        "fee structure",
        "tuition",
        "development fees",
        "payment",
    ],
    "scholarship": [
        "scholarship",
        "scholarships",
        "freeship",
        "government scholarship",
    ],
    "cutoff": [
        "cutoff",
        "cutoffs",
        "cut off",
        "percentile",
        "merit",
        "merit no",
        "merit number",
        "rank",
        "closing rank",
        "admission score",
        "admission marks",
        "minimum marks",
        "minimum percentile",
        "required percentile",
        "required marks",
        "eligibility score",
        "safe percentile",
        "chance of admission",
        "marks needed",
        "score needed",
        "get admission",
        "admission chance",
    ],
    "hostel": [
        "hostel",
        "accommodation",
        "mess",
        "boys hostel",
        "girls hostel",
    ],
    "placement": [
        "placement",
        "placements",
        "recruiter",
        "salary",
        "package",
        "campus placement",
        "company visited",
    ],
    "results": [
        "result",
        "results",
        "marks",
        "marksheet",
        "student results",
        "exam result",
    ],
    "syllabus": [
        "syllabus",
        "curriculum",
        "subject",
        "subjects",
        "course structure",
    ],
    "department": [
        "department",
        "faculty",
        "laboratory",
        "library",
        "mechanical engineering",
        "civil engineering",
        "electrical engineering",
        "computer science",
        "artificial intelligence",
        "food technology",
        "aeronautical engineering",
    ],
}


QUERY_EXPANSIONS = {
    "cse": ["computer science", "computer science engineering"],
    "cs": ["computer science", "computer science engineering"],
    "ai ds": ["artificial intelligence and data science", "aids"],
    "aids": ["artificial intelligence and data science", "ai ds"],
    "iot": ["internet of things", "cyber security", "cybersecurity"],
    "fees": ["fee structure", "tuition fees"],
    "hostel": ["hostel accommodation", "mess facility"],
    "admission": ["eligibility", "management quota", "application"],
    "placement": ["placement records", "offers", "average salary", "highest salary"],
    "placements": ["placement", "placement records", "offers", "salary package"],
    "result": ["results", "marks"],
    "statistics": ["records", "overall total", "average salary", "highest salary"],
    "cutoff": ["cut off", "percentile", "cap round"],
    "cutoffs": ["cutoff", "cut off", "merit marks", "merit number", "percentile", "cap round"],
    "cut off": ["cutoff", "percentile", "cap round", "merit marks"],
    "marks needed": ["cutoff", "cut off", "merit marks", "required percentile", "admission chance"],
    "score needed": ["cutoff", "cut off", "merit marks", "required percentile", "admission chance"],
    "get admission": ["cutoff", "cut off", "merit marks", "merit number"],
}


COMMON_QUERY_FIXES = {
    "abot": "about",
    "admisson": "admission",
    "admision": "admission",
    "admisison": "admission",
    "admsn": "admission",
    "brnach": "branch",
    "collage": "college",
    "colleg": "college",
    "criteriya": "criteria",
    "cutof": "cutoff",
    "cuttof": "cutoff",
    "cuttoff": "cutoff",
    "depet": "department",
    "dept": "department",
    "depts": "departments",
    "depertment": "department",
    "depertments": "departments",
    "departmnt": "department",
    "eligiblity": "eligibility",
    "elligibility": "eligibility",
    "makrs": "marks",
    "markes": "marks",
    "meritno": "merit number",
    "numeber": "number",
    "numbr": "number",
    "pecentile": "percentile",
    "percentil": "percentile",
    "percetile": "percentile",
    "retrival": "retrieval",
    "teh": "the",
}

DOMAIN_TERMS = {
    "adcet", "admission", "admissions", "aeronautical", "ai", "aids",
    "artificial", "bca", "btech", "cap", "category", "cet", "civil",
    "college", "computer", "criteria", "cutoff", "cutoffs", "cyber",
    "data", "def", "department", "documents", "engineering", "ews",
    "fee", "fees", "food", "general", "hostel", "iot", "jee", "ladies",
    "marks", "mechanical", "merit", "mht", "nt1", "nt2", "nt3", "obc",
    "open", "percentile", "placement", "placements", "rank", "robotics",
    "science", "score", "sebc", "st", "tech", "technology", "tfws", "vj",
}

PROTECTED_TOKENS = {
    "ai", "cs", "cse", "ds", "ews", "iot", "jee", "mht", "nt", "nt1",
    "nt2", "nt3", "obc", "sc", "st", "tfws", "vj",
}

CUTOFF_INTENT_PATTERNS = [
    r"\b(cut\s*off|cutoff|cutoffs|merit|percentile|closing\s+rank|rank)\b",
    r"\b(chance|safe|need|needed|required|require|score|marks|mark)\b.*\b(admission|admit|seat)\b",
    r"\b(admission|admit|seat)\b.*\b(chance|safe|need|needed|required|require|score|marks|mark)\b",
    r"\b(how\s+much|how\s+many|minimum|required)\b.*\b(marks|score|percentile)\b",
]

QUALIFYING_CRITERIA_PATTERNS = [
    r"\b(eligibility|eligible|criteria|qualification|qualifying)\b",
    r"\b(12th|pcm|physics|chemistry|mathematics|technical\s+subject)\b",
]


@dataclass(frozen=True)
class ProcessedQuery:
    original: str
    corrected: str
    expanded: str
    category: str


def normalize_text(text: str) -> str:
    text = text.lower().replace("&", " and ").replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_spelling_vocabulary() -> list[str]:
    category_terms = {
        token
        for keywords in CATEGORY_KEYWORDS.values()
        for phrase in keywords
        for token in normalize_text(phrase).split()
    }
    expansion_terms = {
        token
        for phrase in QUERY_EXPANSIONS
        for token in normalize_text(phrase).split()
    }
    return sorted(DOMAIN_TERMS | STOPWORDS | set(COMMON_QUERY_FIXES.values()) | category_terms | expansion_terms)


SPELLING_VOCABULARY = build_spelling_vocabulary()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def significant_terms(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1]


def detect_category(url: str, title: str, content: str) -> str:
    combined = normalize_text(f"{url} {title} {content[:4000]}")
    token_set = set(combined.split())

    best_category = "general"
    best_score = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)
            if not normalized_keyword:
                continue

            if " " in normalized_keyword:
                if normalized_keyword in combined:
                    score += 3
            elif normalized_keyword in token_set:
                score += 1

        if score > best_score:
            best_category = category
            best_score = score

    return best_category


def detect_query_category(query: str) -> str:
    normalized = normalize_text(query)

    explicit_cutoff = bool(re.search(r"\b(cut\s*off|cutoff|cutoffs|merit|rank|percentile)\b", normalized))
    cutoff_score = sum(1 for pattern in CUTOFF_INTENT_PATTERNS if re.search(pattern, normalized))
    criteria_score = sum(1 for pattern in QUALIFYING_CRITERIA_PATTERNS if re.search(pattern, normalized))

    if criteria_score and not explicit_cutoff:
        return "admission"

    if cutoff_score:
        return "cutoff"

    return detect_category("", query, query)


def correct_spelling(query: str) -> str:
    """Correct common user typos before retrieval without changing domain acronyms."""
    normalized = normalize_text(query)
    if not normalized:
        return ""

    for wrong, right in COMMON_QUERY_FIXES.items():
        normalized = re.sub(rf"\b{re.escape(wrong)}\b", right, normalized)

    corrected_tokens: list[str] = []
    for token in normalized.split():
        if token in PROTECTED_TOKENS or token in SPELLING_VOCABULARY or token.isdigit() or len(token) <= 2:
            corrected_tokens.append(token)
            continue

        match = process.extractOne(token, SPELLING_VOCABULARY, scorer=fuzz.ratio)
        if match and match[1] >= 88:
            corrected_tokens.append(match[0])
        else:
            corrected_tokens.append(token)

    return " ".join(corrected_tokens)


def expand_query(query: str, disabled_aliases: set[str] | None = None) -> str:
    expanded_parts = [query.strip()]
    normalized_query = normalize_text(query)
    disabled_aliases = disabled_aliases or set()

    for alias, expansions in QUERY_EXPANSIONS.items():
        normalized_alias = normalize_text(alias)
        if not normalized_alias or normalized_alias in disabled_aliases:
            continue

        matches = (
            normalized_alias in normalized_query.split()
            if " " not in normalized_alias
            else normalized_alias in normalized_query
        )

        if matches:
            expanded_parts.extend(expansions)

    deduped_parts: list[str] = []
    seen: set[str] = set()
    for part in expanded_parts:
        normalized_part = normalize_text(part)
        if not normalized_part or normalized_part in seen:
            continue
        seen.add(normalized_part)
        deduped_parts.append(part.strip())

    return " | ".join(deduped_parts)


def preprocess_query(query: str) -> ProcessedQuery:
    corrected = correct_spelling(query)
    category = detect_query_category(corrected)
    expanded = expand_query(
        corrected,
        disabled_aliases={"admission"} if category == "cutoff" else None,
    )

    if category == "cutoff":
        expanded = (
            f"{expanded} | cutoff | cut off | merit marks | merit number | "
            "required marks | required percentile | admission chance | cap round"
        )

    return ProcessedQuery(
        original=query,
        corrected=corrected,
        expanded=expanded,
        category=category,
    )
