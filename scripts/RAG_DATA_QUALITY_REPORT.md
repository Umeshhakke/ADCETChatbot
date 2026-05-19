# RAG Data Quality Report

## Scope

This report reviews the prepared data inside the `scripts` folder for use in a future RAG chatbot. It does not evaluate a completed RAG pipeline, because vectorization, vector database indexing, retrieval, reranking, and final LLM answer generation have not been built yet.

The current goal is to decide whether the prepared knowledge data is good enough to move into vectorization and what should be improved before building the full RAG system.

## Files Reviewed

- `FY_CUT_OFF__Placement_data__Fee__Hostel_2025_-_26.xlsx`
- `excel_to_json.py`
- `output.json`
- `college_rag_builder.py`
- `knowledge_base.jsonl`
- `propositions.jsonl`
- `summary_index.json`
- `qa_pairs.json`
- `requirement.txt`

## Current Data Preparation Status

The data preparation stage is mostly complete for a first RAG prototype.

Generated artifacts:

| Artifact | Purpose | Current Status |
|---|---|---|
| `output.json` | Structured JSON extracted from Excel | Available |
| `knowledge_base.jsonl` | Main chunk-level knowledge base | Available |
| `propositions.jsonl` | Atomic fact-level knowledge base | Available |
| `summary_index.json` | Short summaries for two-stage retrieval | Available |
| `qa_pairs.json` | Generated question-answer pairs for testing/evaluation | Available |

Current counts:

| Item | Count |
|---|---:|
| Main knowledge chunks | 40 |
| Atomic propositions | 378 |
| QA pairs | 160 |
| Summary index entries | 40 |
| Low-quality chunks flagged | 0 |
| Average chunk quality score | Approximately 0.952 |

The chunk coverage includes:

| Category | Chunk Count |
|---|---:|
| Placements | 12 |
| Cutoffs | 10 |
| Transport | 8 |
| Company visits | 3 |
| Programs offered | 2 |
| Admission documents | 2 |
| Fees | 1 |
| Hostel | 1 |
| Eligibility | 1 |

## Overall Data Quality Verdict

The prepared data is sufficient to start a prototype RAG pipeline.

However, it should not be treated as production-ready student-facing knowledge yet. The data is well structured and has useful RAG-specific fields, but some generated text needs factual validation, metadata extraction needs improvement, and some source-to-chunk transformations should be made stricter before vectorization.

Recommended decision:

Proceed to vectorization only after fixing the critical data quality issues listed below, especially hallucinated wording, empty entity metadata, and a few parsing/classification issues.

## What Has Been Achieved

### 1. Excel Data Was Converted Into Structured JSON

The `excel_to_json.py` script parses the source Excel file and writes `output.json`.

The extracted structured data includes:

- FY/DSE tuition fees
- Hostel fees
- Bus routes and stop-wise fees
- Admission cutoffs
- 12th qualifying criteria
- Placement summaries
- Company visited lists
- Placement year-wise details
- Admission documents
- Programs offered

This is a strong foundation because RAG performs better when source data is structured before chunking.

### 2. Main Knowledge Chunks Were Generated

The `knowledge_base.jsonl` file contains 40 enriched chunks. Each chunk includes:

- `id`
- `parent_id`
- `category`
- `source_section`
- `text`
- `context_header`
- `contextualised_text`
- `one_line_summary`
- `propositions`
- `hypothetical_questions`
- `bm25_tokens`
- `metadata`
- `entities`
- `keywords`
- `quality_score`
- `low_quality_flag`
- `embedding_model`
- `embedding`

This is a good RAG-ready schema. The most important field for vectorization is:

```json
"contextualised_text"
```

That should be embedded instead of only embedding the raw `text` field.

### 3. Atomic Facts Were Generated

The `propositions.jsonl` file contains 378 atomic fact records.

This is useful for fact-based questions such as:

- "What is the hostel fee?"
- "What is the CSE intake?"
- "What is the cutoff for OBC?"
- "Which companies visited in 2024-25?"

For the future RAG system, these propositions can be indexed separately from paragraph chunks.

Recommended future retrieval strategy:

- Use `knowledge_base.jsonl` for broad explanatory answers.
- Use `propositions.jsonl` for exact factual answers.

### 4. Summary Index Was Generated

The `summary_index.json` file contains one summary per chunk.

This can support two-stage retrieval:

1. Search summaries first.
2. Fetch the full chunk by `chunk_id`.

This is useful when the final knowledge base becomes larger.

### 5. QA Pairs Were Generated

The `qa_pairs.json` file contains 160 QA pairs.

Question types include:

- factual
- comparative
- advice
- eligibility

These QA pairs are useful for retrieval testing, but they should not be blindly trusted as ground truth because they were generated from LLM-produced chunks. They should be manually reviewed before being used as an evaluation benchmark.

## Critical Data Quality Issues To Fix Before RAG

### 1. Embeddings Are Not Present Yet

This is expected because the RAG pipeline has not been built yet.

Current fields:

```json
"embedding_model": null,
"embedding": null
```

This is not a data quality bug, but it means the data is not yet searchable by semantic retrieval.

Action before RAG:

- Generate embeddings for `contextualised_text`.
- Save the embedding model name in `embedding_model`.
- Store vectors in FAISS, Chroma, Qdrant, Pinecone, or another vector database.

### 2. Entity Metadata Is Empty

The `entities` object exists in the chunks, but the extracted values appear empty across the generated knowledge base.

Example structure:

```json
"entities": {
  "branches": [],
  "years": [],
  "companies": [],
  "amounts_inr": [],
  "categories": [],
  "locations": []
}
```

This is a significant limitation for college RAG.

Why it matters:

- Users will ask branch-specific questions.
- Users will ask category-specific cutoff questions.
- Users will ask year-specific placement questions.
- Users will ask company-specific placement questions.
- Users will ask fee/location/route-specific questions.

If entity metadata is empty, metadata filtering will not work.

Recommended fix:

Use deterministic extraction from structured JSON wherever possible instead of relying only on LLM extraction.

Examples:

- For cutoff chunks, set `entities.branches` from course/branch name.
- For fee chunks, set `entities.amounts_inr` from fee fields.
- For company chunks, set `entities.companies` from company names.
- For placement chunks, set `entities.years` from academic year keys.
- For bus chunks, set `entities.locations` from route and stop names.
- For reservation categories, set `entities.categories` from `OPEN`, `OBC`, `SC`, `ST`, `EWS`, `TFWS`, etc.

This should be fixed before building metadata-aware retrieval.

### 3. Some Generated Text Adds Interpretation Beyond Source Data

Several chunks are written in polished natural language. That helps readability, but it introduces risk because the language model sometimes adds assumptions, explanations, or interpretations not directly present in the source data.

Examples of risky wording patterns:

- "likely to attract"
- "significant draw"
- "safe and secure environment"
- "students can focus solely on their studies"
- "implied by the forward-looking nature"
- "highly sought after"
- "more accessible"

These may sound natural, but they are not always source-grounded facts.

Why this matters:

RAG should retrieve trustworthy source-grounded information. If the chunk itself contains unsupported generated claims, the final chatbot may answer confidently with information that was never in the original Excel data.

Recommended fix:

Change chunk generation style from descriptive/marketing language to strict factual language.

Preferred style:

```text
The New boys hostel fee is Rs.30,000 per year. The Old boys hostel fee is Rs.20,000 per year. The New ladies hostel fee is Rs.35,000 per year. The Subhadra ladies hostel fee is Rs.30,000 per year.
```

Avoid:

```text
Living in a college hostel provides a safe and secure environment where students can focus solely on their studies.
```

### 4. Some Generated Context May Contradict Source Data

One sampled context header stated that Civil Engineering had an intake of 120 students, while the structured source shows Civil Engineering intake as 60.

This type of mismatch is critical because `contextualised_text` includes the context header. If the context header is embedded and retrieved, the chatbot may answer with the wrong number.

Recommended fix:

- Validate every generated `context_header` against the structured source.
- For numeric fields, avoid LLM-generated summaries when possible.
- Generate context headers deterministically from JSON fields.
- Add automated checks for important numeric facts like intake, fees, cutoffs, placement counts, and salary values.

### 5. BBA/BCA Are Classified Under `btech`

In `output.json`, BBA and BCA appear inside the `programs_offered.btech` list, while `programs_offered.ug_other` is empty.

Why this matters:

If a user asks "Which B.Tech programs are offered?", the chatbot may include BBA and BCA incorrectly unless the retrieval/generation layer handles this carefully.

Recommended fix:

- Separate B.Tech programs from BBA/BCA programs.
- Put BBA/BCA into `ug_other`.
- Ensure the B.Tech chunk only includes engineering B.Tech programs.

### 6. QA Pairs Need Review Before Evaluation Use

The 160 QA pairs are useful, but they are generated from the processed chunks. If chunks contain unsupported claims or mistakes, QA pairs may also contain those mistakes.

Recommended use:

- Use `qa_pairs.json` as a starting point only.
- Manually review the QA pairs before using them as an evaluation set.
- Add manually written questions from real student/admission scenarios.

## Important Enhancements Before Vectorization

### 1. Preserve Raw Structured Facts Alongside Generated Text

For high-risk data such as fees, cutoffs, intake, documents, and placements, keep raw structured fields in metadata.

Example for fees:

```json
"metadata": {
  "category": "OPEN",
  "fy_fee": 134026,
  "dse_fee": 134026,
  "year": "2025-26"
}
```

Example for cutoffs:

```json
"metadata": {
  "branch": "Computer Science & Engineering",
  "category": "OBC",
  "seat_type": "G",
  "merit_no": "...",
  "merit_marks": "..."
}
```

This allows your final RAG answer to rely on exact fields instead of only generated paragraphs.

### 2. Create Smaller Fact Chunks For Numeric Data

Some chunks are long and contain many numbers. This can reduce retrieval precision for questions about a single cutoff or fee.

Recommended:

- Keep the current paragraph chunks.
- Also create smaller row-level records for numeric facts.

Useful row-level chunk types:

- One chunk per branch and reservation category cutoff.
- One chunk per fee category.
- One chunk per hostel.
- One chunk per bus route stop.
- One chunk per company/year.
- One chunk per admission document/category.

This will improve exact retrieval.

### 3. Add Source Traceability

Each chunk should clearly identify where the data came from.

Recommended metadata fields:

```json
"source_file": "FY_CUT_OFF__Placement_data__Fee__Hostel_2025_-_26.xlsx",
"source_sheet": "FY 27",
"source_table": "tuition_fees_2025_26",
"source_row": null
```

If row numbers are hard to track, at least include source sheet and source table.

This will help debugging and citation.

### 4. Normalize Common Terms

The same concepts may appear in multiple forms.

Examples:

- `CSE`
- `Computer Science & Engineering`
- `Computer Science and Engineering`
- `CSE IOT`
- `Computer Science & Engineering (Internet of Things and Cyber Security Including Block Chain Technology)`

Recommended:

Create normalized aliases:

```json
"branch_aliases": ["CSE", "Computer Science", "Computer Science & Engineering"]
```

This improves retrieval when users ask short informal questions.

### 5. Improve Category Labels

The current categories are useful but can be made more retrieval-friendly.

Recommended categories:

- `programs`
- `fees`
- `hostel`
- `transport`
- `cutoffs`
- `eligibility`
- `placements_summary`
- `placements_company_visits`
- `admission_documents`

This allows better filtering and routing.

### 6. Add Answer Priority Rules

Some questions should be answered from structured facts, not from generated descriptive chunks.

Recommended priority:

| Question Type | Preferred Source |
|---|---|
| Fees | Structured metadata / proposition |
| Cutoffs | Row-level cutoff records |
| Hostel | Hostel fee records |
| Bus route fees | Route-stop records |
| Documents | Admission document chunks |
| Placement summary | Placement summary chunks |
| Company list | Company visit records |
| General explanation | Main knowledge chunks |

This can be implemented later in the RAG retriever/router.

## Recommended Data Readiness Checklist

Before vectorization, complete these checks:

- [ ] Fix BBA/BCA classification under `programs_offered`.
- [ ] Regenerate or manually correct context headers with numeric mismatches.
- [ ] Remove unsupported marketing/advice-style claims from chunk text.
- [ ] Fill `entities` deterministically from structured JSON.
- [ ] Add source sheet/table metadata to each chunk.
- [ ] Create smaller fact-level records for cutoffs, fees, hostel, bus stops, and companies.
- [ ] Manually review high-risk QA pairs.
- [ ] Add branch aliases and normalized category names.
- [ ] Validate all important numbers against `output.json`.

## Recommended Vectorization Strategy

Once the above data issues are fixed, vectorization can proceed.

Recommended indexes:

### 1. Main Chunk Vector Index

Input file:

```text
knowledge_base.jsonl
```

Embed field:

```text
contextualised_text
```

Use for:

- General questions
- Explanatory answers
- Multi-sentence responses

### 2. Proposition Vector Index

Input file:

```text
propositions.jsonl
```

Embed field:

```text
text
```

Use for:

- Exact facts
- Fees
- Cutoffs
- Intake
- Companies
- Documents

### 3. Optional Summary Index

Input file:

```text
summary_index.json
```

Embed field:

```text
summary
```

Use for:

- Two-stage retrieval
- Larger future knowledge bases

### 4. Keyword/BM25 Index

Use:

```text
bm25_tokens
```

Use for:

- Exact branch names
- Company names
- Reservation categories
- Fee amounts
- Route names
- Academic years

Best retrieval design:

```text
Hybrid retrieval = vector search + BM25 keyword search + metadata filtering
```

## Suggested RAG Pipeline After Data Fixes

Recommended flow:

1. User asks a question.
2. Detect question type: fee, cutoff, placement, document, program, transport, hostel, general.
3. Apply metadata filter if possible.
4. Retrieve from main chunk index and proposition index.
5. Combine vector results with BM25 keyword results.
6. Optionally rerank retrieved chunks.
7. Generate answer using only retrieved context.
8. Return answer with source section/category.
9. If no relevant context is found, say the information is not available in the current knowledge base.

## Final Recommendation

The prepared data is a strong first version and is enough to begin a prototype RAG implementation.

For reliable use, make data corrections before vectorization. The most important improvements are:

1. Remove unsupported generated claims.
2. Fix numeric/context inconsistencies.
3. Populate entity metadata.
4. Separate BBA/BCA from B.Tech programs.
5. Add smaller fact-level records for exact lookup.

After these changes, the data will be much more suitable for a vector database and LLM-based RAG chatbot.
