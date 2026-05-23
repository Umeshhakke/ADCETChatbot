import csv
import json
import re
from pathlib import Path
from typing import Iterable
from threading import Lock

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from knowledge_utils import (
    detect_category,
    normalize_text,
    preprocess_query,
    significant_terms,
    tokenize,
)
from llm_provider import create_llm_provider
from settings import runtime_config

_chatbot_instance = None
_chatbot_lock = Lock()

GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "who are you", "what are you"
}

UNAVAILABLE_RESPONSE = "Information not available in adcet_data.json."

BLOCKED_RESPONSES = {
    "connect with a staff",
    "contact staff",
    "data unavailable",
    "information unavailable",
}

class ADCETRAGChatbot:
    def __init__(self) -> None:
        print(f"Loading LLM provider: {runtime_config.llm_provider}")
        try:
            self.llm = create_llm_provider(runtime_config)
        except Exception as exc:
            print(f"LLM unavailable -> fallback mode: {exc}")
            self.llm = None

        print(f"Loading embedding model: {runtime_config.embedding_model}")
        try:
            self.embedder = SentenceTransformer(
                runtime_config.embedding_model,
                local_files_only=runtime_config.hf_local_files_only,
            )
        except Exception as exc:
            print(f"Embedding fallback (BM25 only): {exc}")
            self.embedder = None

        print(f"Loading reranker: {runtime_config.reranker_model}")
        try:
            self.reranker = CrossEncoder(
                runtime_config.reranker_model,
                local_files_only=runtime_config.hf_local_files_only,
            )
        except Exception as exc:
            print(f"No reranker -> using score ranking: {exc}")
            self.reranker = None

        print(f"Loading ChromaDB: {runtime_config.chroma_path}")
        client = chromadb.PersistentClient(path=runtime_config.chroma_path)
        self.collection = client.get_collection(runtime_config.collection_name)

        print("Building BM25 index...")
        all_data = self.collection.get()
        docs, metas, ids = all_data["documents"], all_data["metadatas"], all_data["ids"]

        self.records = []
        for doc_id, doc, meta in zip(ids, docs, metas):
            meta = self.normalize_metadata(doc, meta or {})
            self.records.append({
                "id": doc_id,
                "doc": doc,
                "meta": meta,
                "search_text": self.compose_search_text(doc, meta),
            })

        self.docs = [r["doc"] for r in self.records]
        self.metas = [r["meta"] for r in self.records]
        self.ids = [r["id"] for r in self.records]

        self.bm25 = BM25Okapi([tokenize(t) for t in [r["search_text"] for r in self.records]])
        self.cutoff_data = self.load_cutoff_data()
        self.cutoff_rows = self.load_cutoff_rows()
        self.program_data = self.load_program_data()
        self.last_topic: str | None = None

    # ------------------ METADATA ------------------

    @staticmethod
    def normalize_metadata(doc: str, meta: dict) -> dict:
        meta = dict(meta)
        meta["source"] = str(meta.get("source", "")).strip()
        meta["title"] = str(meta.get("title", "")).strip()
        meta["type"] = str(meta.get("type", "webpage")).strip()
        meta["category"] = detect_category(meta["source"], meta["title"], doc)
        return meta

    @staticmethod
    def compose_search_text(doc: str, meta: dict) -> str:
        return f"{meta.get('title','')}\n{meta.get('category','')}\n{meta.get('source','')}\n{doc}"

    @staticmethod
    def compose_rerank_text(doc: str, meta: dict) -> str:
        return f"Title: {meta.get('title','')}\nCategory: {meta.get('category','')}\nSource: {meta.get('source','')}\nContent:\n{doc}"

    # ------------------ STRUCTURED PROGRAMS ------------------

    @staticmethod
    def load_program_data() -> dict:
        program_path = Path(__file__).resolve().parent.parent / "data_file" / "program_offered.json"
        if not program_path.exists():
            return {}

        try:
            with program_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            print(f"Program lookup disabled: {exc}")
            return {}

    def engineering_departments(self) -> list[dict]:
        undergraduate_programs = (
            self.program_data
            .get("academic_programs", {})
            .get("undergraduate_programs", [])
        )
        departments = []
        for program in undergraduate_programs:
            degree = str(program.get("degree", "")).strip().lower()
            name = str(program.get("program_name", "")).strip()
            if degree == "b.tech" and name:
                departments.append(program)
        return departments

    @staticmethod
    def is_department_query(query: str, last_topic: str | None = None) -> bool:
        normalized = normalize_text(query)
        if re.search(r"\b(department|departments|branch|branches|programs offered|courses offered)\b", normalized):
            return True
        if last_topic == "departments" and re.search(r"\b(which are they|what are they|list them|name them|show them|they)\b", normalized):
            return True
        return False

    def answer_department_query(self, question: str) -> str | None:
        if not self.is_department_query(question, self.last_topic):
            return None

        departments = self.engineering_departments()
        if not departments:
            return None

        self.last_topic = "departments"
        normalized = normalize_text(question)
        names = [program["program_name"] for program in departments]
        wants_count_only = bool(re.search(r"\b(how many|number|count|total)\b", normalized)) and not re.search(
            r"\b(which|list|name|show)\b", normalized
        )

        if wants_count_only:
            return f"ADCET has {len(names)} B.Tech engineering departments."

        lines = [
            f"ADCET has {len(names)} B.Tech engineering departments:",
            *[f"{index}. {name}" for index, name in enumerate(names, 1)],
        ]
        return "\n".join(lines)

    # ------------------ STRUCTURED CUTOFF ------------------

    @staticmethod
    def load_cutoff_data() -> dict:
        cutoff_path = Path(__file__).resolve().parent.parent / "data_file" / "cutoff.json"
        if not cutoff_path.exists():
            return {}

        try:
            with cutoff_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            print(f"Cutoff lookup disabled: {exc}")
            return {}

    @staticmethod
    def load_cutoff_rows() -> list[dict]:
        cutoff_path = Path(__file__).resolve().parent.parent / "knowledge" / "cutoff_cleaned.csv"
        if not cutoff_path.exists():
            return []

        try:
            with cutoff_path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception as exc:
            print(f"Cutoff CSV lookup disabled: {exc}")
            return []

    @staticmethod
    def detect_cutoff_category(query: str) -> str | None:
        normalized = normalize_text(query)
        category_aliases = {
            "OPEN": ["open", "general category", "general quota"],
            "OBC": ["obc"],
            "SC": ["sc"],
            "ST": ["st"],
            "SEBC": ["sebc"],
            "EWS": ["ews"],
            "TFWS": ["tfws", "tuition fee waiver"],
            "DEF": ["def", "defence", "defense"],
            "VJ": ["vj", "vjnt"],
            "NT-1": ["nt1", "nt 1", "nt-1"],
            "NT-2": ["nt2", "nt 2", "nt-2"],
            "NT-3": ["nt3", "nt 3", "nt-3"],
        }

        for category, aliases in category_aliases.items():
            if any(re.search(rf"\b{re.escape(normalize_text(alias))}\b", normalized) for alias in aliases):
                return category
        return None

    def detect_cutoff_course(self, query: str) -> str | None:
        normalized = normalize_text(query)
        course_aliases = {
            "Computer Science & Engineering": [
                "computer science engineering", "computer science", "cse", "cs",
            ],
            "Computer Science (IoT & Cyber Security)": [
                "iot cyber security", "cyber security", "cybersecurity", "cse iot", "iot",
            ],
            "AI & Data Science": [
                "ai and data science", "artificial intelligence and data science", "ai ds", "aids",
            ],
            "Robotics and Artificial Intelligence": [
                "robotics and artificial intelligence", "robotics artificial intelligence", "robotics", "rai",
            ],
            "Mechanical Engineering": ["mechanical engineering", "mechanical", "mech"],
            "Electrical Engineering": ["electrical engineering", "electrical"],
            "Civil Engineering": ["civil engineering", "civil"],
            "Aeronautical Engineering": ["aeronautical engineering", "aeronautical", "aero"],
            "Food Technology": ["food technology", "food tech"],
        }

        for course, aliases in course_aliases.items():
            if any(re.search(rf"\b{re.escape(normalize_text(alias))}\b", normalized) for alias in aliases):
                return course

        available_courses = {
            course
            for group_data in self.cutoff_data.values()
            if isinstance(group_data, dict)
            for course in group_data
        }
        available_courses.update(row["Course"] for row in self.cutoff_rows if row.get("Course"))
        for course in available_courses:
            if normalize_text(course) in normalized:
                return course
        return None

    def find_cutoff_record(self, group: str, course: str, category: str) -> tuple[str, dict] | None:
        record = self.cutoff_data.get(group, {}).get(course, {}).get(category)
        if record:
            return group, record

        if group == "Ladies":
            record = self.cutoff_data.get("General", {}).get(course, {}).get(category)
            if record:
                return "General", record

        csv_course_aliases = {
            "Computer Science & Engineering": "CSE",
            "Computer Science (IoT & Cyber Security)": "CS IOT",
            "AI & Data Science": "AIDS",
            "Robotics and Artificial Intelligence": "RAI",
            "Mechanical Engineering": "Mech",
            "Electrical Engineering": "ELEC",
            "Civil Engineering": "CIVIL",
            "Aeronautical Engineering": "Aero",
            "Food Technology": "Food",
        }
        accepted_course_names = {normalize_text(course), normalize_text(csv_course_aliases.get(course, ""))}

        for target_group in [group, "General"] if group == "Ladies" else [group]:
            for row in self.cutoff_rows:
                row_group = normalize_text(row.get("Group", ""))
                if target_group == "Ladies":
                    group_matches = any(term in row_group for term in ["ladies", "ledies", "female"]) or row_group.startswith("l ")
                else:
                    group_matches = "general" in row_group or row_group.startswith("g ")
                if not group_matches:
                    continue
                if normalize_text(row.get("Course", "")) not in accepted_course_names:
                    continue

                merit_no = row.get(f"{category}_Merit No", "").strip() or "Data not found"
                merit_marks = row.get(f"{category}_Merit Marks", "").strip() or "Data not found"
                return target_group, {"merit_no": merit_no, "merit_marks": merit_marks}

        return None

    def answer_cutoff_query(self, processed_query) -> str | None:
        if processed_query.category != "cutoff" or not (self.cutoff_data or self.cutoff_rows):
            return None

        query = processed_query.corrected
        group = "Ladies" if re.search(r"\b(ladies|girls|female|woman|women)\b", normalize_text(query)) else "General"
        course = self.detect_cutoff_course(query)
        category = self.detect_cutoff_category(query)

        if not course:
            return (
                "Cutoff marks depend on the branch and reservation category. "
                "Please mention the branch, for example CSE, AI & DS, Mechanical, Civil, Electrical, IoT, Robotics, Aeronautical, or Food Technology."
            )

        if not category:
            category = "OPEN"
            category_note = "No category was specified, so I am using OPEN category."
        else:
            category_note = f"Using {category} category."

        lookup_result = self.find_cutoff_record(group, course, category)
        if not lookup_result:
            return None

        group, record = lookup_result

        merit_no = record.get("merit_no", "Data not found")
        merit_marks = record.get("merit_marks", "Data not found")

        if merit_no == "Data not found" and merit_marks == "Data not found":
            return f"Cutoff data is not available for {group} {course} under {category} category."

        if isinstance(merit_marks, str):
            try:
                merit_marks = float(merit_marks)
            except ValueError:
                pass

        if isinstance(merit_marks, (int, float)):
            merit_marks = round(float(merit_marks), 2)

        return (
            f"{category_note} For {group} {course}, the cutoff merit marks are {merit_marks} "
            f"and the cutoff merit number is {merit_no}."
        )

    # ------------------ SEARCH ------------------

    @staticmethod
    def category_adjustment(query_category: str, doc: str, meta: dict) -> float:
        doc_category = meta.get("category", "general")
        text = normalize_text(f"{meta.get('title','')} {meta.get('source','')} {doc[:1200]}")
        record_kind = normalize_text(str(meta.get("record_kind", "")))

        if query_category == "cutoff":
            adjustment = 0.0
            if doc_category == "cutoff":
                adjustment += 0.45
            if "cutoff" in record_kind:
                adjustment += 0.30
            if any(term in text for term in ["cut off", "cutoff", "merit marks", "merit number", "percentile"]):
                adjustment += 0.20
            if doc_category in {"admission", "results"} and any(
                term in text for term in ["minimum marks", "marksheet", "eligibility", "criteria"]
            ):
                adjustment -= 0.35
            return adjustment

        if query_category == "admission":
            adjustment = 0.0
            if doc_category == "admission":
                adjustment += 0.25
            if record_kind in {"admission documents", "admission eligibility"}:
                adjustment += 0.25
            if any(term in text for term in [
                "document", "documents", "certificate", "certificates",
                "marksheet", "required", "domicile", "aadhar", "aadhaar",
            ]):
                adjustment += 0.25
            if doc_category == "cutoff" and any(term in text for term in [
                "cutoff", "cut off", "merit marks", "merit number", "percentile",
            ]):
                adjustment -= 0.35
            return adjustment

        if query_category == "hostel":
            adjustment = 0.0
            if doc_category == "hostel":
                adjustment += 0.45
            if "hostel" in record_kind:
                adjustment += 0.30
            if "hostel" in text or "accommodation" in text:
                adjustment += 0.25
            if "bus route" in text or "bus fees" in text:
                adjustment -= 0.45
            return adjustment

        if query_category == "transport":
            adjustment = 0.0
            if "bus" in record_kind:
                adjustment += 0.45
            if any(term in text for term in ["bus route", "stop name", "monthly fee", "transport"]):
                adjustment += 0.25
            if "hostel" in text or "college fees" in text:
                adjustment -= 0.20
            return adjustment

        if query_category == "fees":
            adjustment = 0.0
            if "college fee" in record_kind:
                adjustment += 0.35
            if "bus" in record_kind and "bus" not in text[:100]:
                adjustment -= 0.10
            return adjustment

        if query_category == "placement":
            adjustment = 0.0
            if "placement" in record_kind or "company visit" in record_kind:
                adjustment += 0.30
            return adjustment

        if query_category == "department":
            adjustment = 0.0
            if "academic program" in record_kind:
                adjustment += 0.35
            return adjustment

        if query_category != "general" and doc_category == query_category:
            return 0.25

        return 0.0

    @staticmethod
    def metadata_adjustment(query: str, meta: dict) -> float:
        normalized_query = normalize_text(query)
        query_terms = set(normalized_query.split())
        score = 0.0

        for key, raw_value in meta.items():
            if key in {"source", "title", "type", "category", "chunk_index"}:
                continue
            value = normalize_text(str(raw_value))
            if not value:
                continue

            value_terms = {term for term in value.split() if len(term) > 1}
            overlap = query_terms & value_terms
            if value and value in normalized_query:
                score += 0.25
            elif overlap:
                score += min(0.18, 0.06 * len(overlap))

        return min(score, 0.35)

    def hybrid_search(self, processed_query):
        expanded = processed_query.expanded
        query_terms = set(significant_terms(expanded) or tokenize(expanded))
        candidates = {}

        # VECTOR SEARCH
        if self.embedder:
            q_emb = self.embedder.encode([expanded], normalize_embeddings=True).tolist()

            res = self.collection.query(
                query_embeddings=q_emb,
                n_results=runtime_config.retrieval_count,
                include=["documents", "metadatas", "distances"],
            )

            for doc_id, doc, meta, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ):
                candidates[doc_id] = {
                    "doc": doc,
                    "meta": self.normalize_metadata(doc, meta or {}),
                    "vector": 1 / (1 + float(dist)),
                    "bm25": 0,
                    "lexical": 0,
                }

        # BM25
        bm25_terms = list(query_terms) or tokenize(expanded)
        scores = self.bm25.get_scores(bm25_terms)
        max_bm25 = float(np.max(scores)) if len(scores) else 0.0

        for i in np.argsort(scores)[::-1][: runtime_config.retrieval_count * 2]:
            doc_id = self.ids[i]
            candidates.setdefault(doc_id, {
                "doc": self.docs[i],
                "meta": self.metas[i],
                "vector": 0,
                "bm25": 0,
                "lexical": 0,
            })
            candidates[doc_id]["bm25"] = float(scores[i]) / max_bm25 if max_bm25 > 0 else 0.0

        # FINAL SCORE
        ranked = []
        for c in candidates.values():
            doc_terms = set(significant_terms(self.compose_search_text(c["doc"], c["meta"])))
            c["lexical"] = len(query_terms & doc_terms) / max(len(query_terms), 1)
            score = (
                0.40 * c["vector"]
                + 0.30 * c["bm25"]
                + 0.20 * c["lexical"]
                + self.category_adjustment(processed_query.category, c["doc"], c["meta"])
                + self.metadata_adjustment(processed_query.expanded, c["meta"])
            )
            ranked.append((c["doc"], c["meta"], score))

        return sorted(ranked, key=lambda x: x[2], reverse=True)[:runtime_config.retrieval_count]

    # ------------------ RERANK ------------------

    def rerank(self, processed_query, docs):
        docs = list(docs)
        if not docs:
            return []

        if not self.reranker:
            return [((d, m), s) for d, m, s in docs]

        pairs = [[processed_query.expanded, self.compose_rerank_text(d, m)] for d, m, _ in docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            (
                (
                    ((d, m), s),
                    float(model_score)
                    + self.category_adjustment(processed_query.category, d, m)
                    + self.metadata_adjustment(processed_query.expanded, m),
                )
                for (d, m, s), model_score in zip(docs, scores)
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        return [x[0] for x in ranked[:runtime_config.top_k]]

    # ------------------ CONTEXT ------------------

    def build_context(self, docs):
        blocks = []
        for i, ((doc, meta), score) in enumerate(docs, 1):
            facets = {
                key: value
                for key, value in meta.items()
                if key not in {"source", "title", "type", "category", "chunk_index"}
                and str(value).strip()
            }
            facet_text = "\n".join(f"{key}: {value}" for key, value in sorted(facets.items()))
            blocks.append(
                f"[CHUNK {i}]\nTITLE: {meta.get('title','')}\n"
                f"CATEGORY: {meta.get('category','')}\nSOURCE: {meta.get('source','')}\n"
                f"RELEVANCE: {score:.3f}\n"
                f"METADATA:\n{facet_text or 'none'}\nCONTENT:\n{doc}"
            )
        return "\n\n".join(blocks)

    def build_prompt(self, q, ctx):
        return f"""
You are ADCET assistant.

Use only context. Answer the user's exact question.
Prefer the chunks whose metadata and title match the user's branch, category,
year, route, stop, company, hostel, admission type, or program.
Do not mix values from different chunks unless the question asks for a
comparison or summary.

If the question asks for admission documents, certificates, or marksheets,
answer from document/admission context and do not answer with cutoff marks.
If the retrieved context has separate admission categories and the user did
not specify a category/caste, ask the user to mention the category instead of
combining all category-wise document lists.

If the question asks how many marks, score, rank, percentile, or chance is
needed for admission, answer from cutoff/merit-mark data, not qualifying
eligibility criteria.

If not found -> {UNAVAILABLE_RESPONSE}

Question:
{q}

Context:
{ctx}

Answer:
""".strip()

    # ------------------ FALLBACK ------------------

    @staticmethod
    def is_blocked_or_unhelpful(response: str) -> bool:
        normalized = normalize_text(response)
        if not normalized:
            return True
        if UNAVAILABLE_RESPONSE.lower() in response.lower():
            return True
        return any(blocked in normalized for blocked in BLOCKED_RESPONSES)

    def fallback_answer_from_ranked(self, ranked) -> str:
        snippets = []
        if not ranked:
            return UNAVAILABLE_RESPONSE

        top_score = ranked[0][1]
        selected = []
        for item in ranked:
            (_doc_meta, score) = item
            if len(selected) >= runtime_config.answer_source_count:
                break
            if score >= top_score - 0.20:
                selected.append(item)

        if not selected:
            selected = ranked[:1]

        for (doc, _meta), _score in selected:
            snippet = re.sub(r"\s+", " ", doc).strip()
            if snippet:
                snippets.append(snippet)
        return "\n".join(snippets) if snippets else UNAVAILABLE_RESPONSE

    @staticmethod
    def metadata_focus_fields(query: str, top_meta: dict) -> dict[str, str]:
        normalized_query = normalize_text(query)
        query_terms = set(normalized_query.split())
        focus_fields = [
            "category_name",
            "course",
            "group",
            "branch",
            "eligible_branches",
            "academic_year",
            "route_name",
            "stop_name",
            "hostel_type",
            "degree",
            "program_name",
        ]
        matches: dict[str, str] = {}
        compact_query = normalized_query.replace(" ", "")
        for field in focus_fields:
            value = str(top_meta.get(field, "")).strip()
            normalized_value = normalize_text(value)
            if not normalized_value:
                continue
            value_terms = {term for term in normalized_value.split() if len(term) > 1}
            compact_value = normalized_value.replace(" ", "")
            if normalized_value in normalized_query or compact_value in compact_query or query_terms & value_terms:
                matches[field] = normalized_value
        return matches

    def focus_ranked_context(self, processed_query, ranked):
        if not ranked:
            return ranked

        top_meta = ranked[0][0][1]
        focus = self.metadata_focus_fields(processed_query.expanded, top_meta)
        if not focus:
            return ranked

        top_record_kind = normalize_text(str(top_meta.get("record_kind", "")))
        focused = []
        for item in ranked:
            (_doc, meta), _score = item
            record_kind = normalize_text(str(meta.get("record_kind", "")))
            if top_record_kind and record_kind != top_record_kind:
                continue
            if all(normalize_text(str(meta.get(field, ""))) == value for field, value in focus.items()):
                focused.append(item)

        return focused or ranked

    @staticmethod
    def context_category_names(ranked) -> list[str]:
        categories: list[str] = []
        seen: set[str] = set()
        for (doc, _meta), _score in ranked:
            for match in re.finditer(r"category name:\s*([^\n\r]+)", doc, flags=re.IGNORECASE):
                category = re.sub(r"\s+", " ", match.group(1)).strip()
                normalized = normalize_text(category)
                if category and normalized not in seen:
                    seen.add(normalized)
                    categories.append(category)
        return categories

    @staticmethod
    def query_mentions_context_category(query: str, categories: list[str]) -> bool:
        normalized_query = normalize_text(query)
        query_terms = set(normalized_query.split())
        for category in categories:
            category_terms = [term for term in normalize_text(category).split() if len(term) > 1]
            if any(term in query_terms for term in category_terms):
                return True
        return False

    def category_clarification(self, question: str, ranked) -> str | None:
        normalized = normalize_text(question)
        if not re.search(r"\b(documents?|certificates?|marksheets?)\b", normalized):
            return None

        categories = self.context_category_names(ranked)
        if len(categories) <= 1 or self.query_mentions_context_category(question, categories):
            return None

        return (
            "The retrieved admission document data is category-wise. "
            f"Please mention the category/caste, for example: {', '.join(categories[:6])}."
        )

    # ------------------ ANSWER ------------------

    def answer_query(self, question: str) -> str:
        q = question.lower().strip()

        if q in GREETINGS:
            return "Hello! Ask about admissions, fees, placements, hostel, syllabus, etc."

        processed_query = preprocess_query(question)

        retrieved = self.hybrid_search(processed_query)
        ranked = self.rerank(processed_query, retrieved)

        if not ranked:
            return UNAVAILABLE_RESPONSE

        sources = list({m.get("source","") for (d,m),_ in ranked if m.get("source")})

        clarification = self.category_clarification(processed_query.corrected, ranked)
        if clarification:
            return clarification

        ranked = self.focus_ranked_context(processed_query, ranked)

        if self.llm:
            try:
                ctx = self.build_context(ranked)
                resp = self.llm.generate(self.build_prompt(processed_query.corrected, ctx))
                if not self.is_blocked_or_unhelpful(resp):
                    return resp
            except Exception:
                pass

        return self.fallback_answer_from_ranked(ranked)


# ------------------ RUN ------------------

# chatbot = ADCETRAGChatbot()

# def answer_query(question: str) -> str:
#     return chatbot.answer_query(question)
def get_chatbot() -> ADCETRAGChatbot:
    global _chatbot_instance
    if _chatbot_instance is None:
        with _chatbot_lock:
            if _chatbot_instance is None:
                print("Initializing ADCETRAGChatbot (first request)...")
                _chatbot_instance = ADCETRAGChatbot()
                print("Chatbot ready.")
    return _chatbot_instance

def answer_query(question: str) -> str:
    return get_chatbot().answer_query(question)


def main():
    print("ADCET RAG Ready")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Bot:", answer_query(q))


if __name__ == "__main__":
    main()
