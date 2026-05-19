# RAG Pipeline Production Gap Report

Date: 2026-05-19  
Project: ADCET college chatbot RAG  
Scope reviewed: `scripts/rag_query.py`, `scripts/vectorize_rag_data.py`, `scripts/fix_rag_data.py`, generated JSON/JSONL artifacts, current Chroma/FAISS vector DB outputs, parser scripts, and supporting requirements.

## Executive Summary

The current RAG system has useful cleaned data and a working vector database, but the query pipeline is not production-ready. It is still a single-script retrieval prototype with manual query replacements, weak query understanding, no graph/workflow orchestration, no true hybrid search, no robust reranking, no answer validation, and no automated evaluation.

The user-reported failures are valid:

- Misspelled or casual questions often fail even when the answer exists.
- Indirect wording such as "departments", "last year", "student sit for placement", "iot branch", or "aids intake" is not reliably mapped to the correct facts.
- The current workaround in `rag_query.py` uses hardcoded `QUERY_REPLACEMENTS`, which is not scalable or reliable for real users.
- Retrieval is partly semantic but still depends too much on character/keyword matching and ad hoc boosts.
- The answer-generation model says "information not available" when relevant context was not retrieved, not necessarily because the knowledge base lacks the answer.

Recommended direction: rebuild the query side as a proper retrieval graph with query normalization, intent/entity extraction, multi-query expansion, hybrid dense+sparse retrieval, metadata-aware routing, reranking, answer synthesis, and evaluation.

## Current Architecture Observed

### Data Preparation

The cleaned artifacts are in good shape for a prototype:

- `scripts/output.json`
- `scripts/knowledge_base.jsonl`
- `scripts/propositions.jsonl`
- `scripts/summary_index.json`
- `scripts/qa_pairs.json`

The cleaned data includes factual chunks, propositions, metadata, entities, and source traceability.

### Vectorization

`scripts/vectorize_rag_data.py` builds:

- Chroma collections:
  - `adcet_main_chunks`
  - `adcet_propositions`
  - `adcet_summaries`
- FAISS indexes:
  - `main_chunks.index`
  - `propositions.index`
  - `summaries.index`

The current embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model is lightweight and fast, but it is not ideal for noisy Indian college-admission queries with abbreviations, spelling mistakes, short acronyms, and domain-specific branch names.

### Query Runtime

`scripts/rag_query.py` currently performs:

1. Query embedding.
2. Chroma search over propositions.
3. Chroma search over main chunks.
4. Local lexical fallback over JSONL files.
5. Manual reranking using semantic score, keyword overlap, and boosts.
6. Ollama answer generation.
7. Interactive loop.

This is useful for testing but not a production architecture.

## User-Reported Failure Cases

The following examples were reported:

```text
which companes allow student tie sit for placemnt in mechanical department
```

Expected intent:

- Find companies whose eligible branches include Mechanical or all branches.

Observed earlier behavior:

- The system answered that information was not available.

Root causes:

- Misspellings: `companes`, `placemnt`.
- Broken phrase: `tie sit`.
- "department" should map to branch/program.
- Company eligibility data exists in structured `output.json`, but the vectorized propositions did not preserve eligible-branch logic in a retrieval-friendly way.

```text
what is the highest package of cse previous year
```

Expected intent:

- Previous year likely means `2024-25`.
- CSE means Computer Science Engineering.
- Answer should use branch-wise placement summary.

Root causes:

- Relative time expression not normalized.
- CSE abbreviation not semantically expanded.
- Dense vector retrieval alone may retrieve CSE IoT or unrelated placement records.

```text
what is the intake of aids
```

Expected answer:

- Artificial Intelligence and Data Science intake is 60.

Root causes:

- `AIDS` is a domain acronym.
- The embedding model may not associate `aids` with "Artificial Intelligence and Data Science".
- Program facts are stored in a shared B.Tech chunk, so exact acronym lookup needs alias support.

```text
when iot branch started
```

Expected answer:

- CSE IoT branch started in 2021.

Root causes:

- `iot` must map to "Computer Science & Engineering (Internet of Things and Cyber Security Including Block Chain Technology)".
- "branch started" must map to `year_of_starting`.

```text
what are the departments in this college
```

Expected answer:

- List offered programs/branches.

Root causes:

- The system treats "departments" as an exact term rather than a synonym for branches/programs/courses.

```text
do college offer robotics
```

Expected answer:

- Yes, Robotics And Artificial Intelligence is offered, intake 60, started in 2025.

Root causes:

- The query is syntactically imperfect.
- Robotics is a program name, but semantic retrieval returned unrelated company records before the manual fix.

## Current Bugs And Insufficiencies

### 1. Hardcoded Query Replacement Is Not Production-Ready

Current issue in `rag_query.py`:

```python
QUERY_REPLACEMENTS = {
    "companes": "companies",
    "placemnt": "placement",
    "aids": "AIDS Artificial Intelligence Data Science AI DS",
    ...
}
```

Problems:

- It only works for spelling mistakes already known by the developer.
- Every new user typo requires a code change.
- It mixes spelling correction, acronym expansion, synonym expansion, and intent routing in one dictionary.
- It can introduce false positives. Example: a medical meaning of "AIDS" is irrelevant here, but the system blindly rewrites it.
- It is not explainable or configurable outside the code.

Recommended replacement:

- Use a real spell-correction layer plus a domain lexicon generated from the knowledge base.
- Keep domain aliases in data/config, not code.
- Use query expansion and reranking instead of direct string replacement.

Relevant libraries:

- `symspellpy`: Python port of SymSpell; designed for fast spelling correction with low memory use. Its API supports edit-distance based lookup and configurable dictionary thresholds. Source: https://symspellpy.readthedocs.io/ and https://symspellpy.readthedocs.io/en/latest/api/symspellpy.html
- `pyspellchecker`: pure Python spell checker using edit distance and word frequency. It supports setting Levenshtein distance up to 2 and custom dictionaries. Source: https://pypi.org/project/pyspellchecker/ and https://pyspellchecker.readthedocs.io/en/master/code.html
- `rapidfuzz`: recommended for fuzzy matching domain terms such as branch names, company names, categories, and aliases. It should be used with a domain dictionary, not as a global answer mechanism.

### 2. No Dynamic Domain Lexicon

The system already has structured facts, but it does not build a reusable lexicon from them.

Needed dynamic lexicon sources:

- Branch names from `programs_offered`.
- Branch aliases from cutoff data: `CSE`, `AIDS`, `RAI`, `CSE IOT`, `Mech`, `ELECT`, etc.
- Company names from `company_visited_lists`.
- Reservation categories: `OPEN`, `OBC`, `SC`, `ST`, `SEBC`, `TFWS`, `EWS`, etc.
- Facilities: hostel names, bus route names, stop names.
- Common user terms: department, branch, course, program, placement, package, intake, fee.

Current problem:

- Some aliases exist inside `fix_rag_data.py`, but they are not exported as a separate search-time artifact.
- Query-time correction is therefore hardcoded instead of data-driven.

Recommended artifact:

```text
scripts/retrieval_lexicon.json
```

Suggested structure:

```json
{
  "branches": {
    "aids": "Artificial Intelligence (AI) and Data Science",
    "iot": "Computer Science & Engineering (Internet of Things and Cyber Security Including Block Chain Technology)",
    "cse": "Computer Science & Engineering",
    "rai": "Robotics And Artificial Intelligence"
  },
  "synonyms": {
    "department": ["branch", "program", "course"],
    "package": ["salary", "highest salary", "LPA"],
    "placed": ["offers", "students placed"]
  },
  "companies": ["TCS", "Capgemini", "Infosys"],
  "categories": ["OPEN", "OBC", "SC", "ST", "SEBC", "TFWS"]
}
```

### 3. Retrieval Is Not True Hybrid Search

Current behavior:

- Dense vector search is done in Chroma.
- A custom local lexical fallback scans JSONL files.
- Results are merged manually with custom scoring.

Problems:

- The lexical fallback is outside the vector DB and outside a clean retriever abstraction.
- There is no BM25 index, despite `bm25_tokens` being generated.
- There is no sparse vector index.
- There is no Reciprocal Rank Fusion or standard fusion strategy.
- Chroma metadata stores complex fields as JSON strings, reducing reliable metadata filtering.

Recommended approach:

- Use true hybrid retrieval: dense vector + sparse/BM25 retrieval + metadata filtering + fusion.
- Either keep Chroma for simple local prototyping and add a proper BM25 retriever, or move to Qdrant for native dense+sparse hybrid retrieval.

Relevant libraries and docs:

- LangChain BM25Retriever provides a standard BM25 retriever interface. Source: https://docs.langchain.com/oss/python/integrations/retrievers/bm25/
- Qdrant supports hybrid search with dense and sparse vectors; its documentation describes combining semantic dense vectors with precise sparse signals. Source: https://qdrant.tech/documentation/concepts/hybrid-queries/ and https://qdrant.tech/course/essentials/day-3/hybrid-search/
- Qdrant hybrid search with FastEmbed includes dense model, sparse model, query processing, and fusion concepts. Source: https://qdrant.tech/documentation/tutorials-search-engineering/hybrid-search-fastembed/

### 4. No Reranker Model

Current reranking:

```text
rerank_score = semantic_score + keyword_overlap + manual boosts
```

Problems:

- Keyword overlap cannot understand phrasing like "previous year package".
- Manual boosts can rank wrong data highly.
- There is no query-document cross-attention.
- It cannot reliably distinguish:
  - CSE vs CSE IoT
  - programs vs placements
  - company visit records vs branch placement summaries

Recommended rerankers:

- `sentence-transformers` CrossEncoder reranker.
- `BAAI/bge-reranker-base`, `BAAI/bge-reranker-v2-m3`, or another locally available reranker.
- `FlashRank` for CPU-friendly reranking.

Relevant sources:

- Sentence Transformers CrossEncoder docs describe pairwise query-document scoring and reranking. Source: https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html
- Sentence Transformers reranking evaluator computes ranking metrics such as MRR, NDCG, and MAP over retrieved documents. Source: https://sbert.net/docs/package_reference/cross_encoder/evaluation.html
- FlashRank is a lightweight Python reranking library designed to rerank existing search/retrieval results before sending them to an LLM. Source: https://pypi.org/project/flashrank/

### 5. No Query Understanding Layer

Current script directly embeds the raw or manually expanded query.

Missing steps:

- Spell correction.
- Acronym expansion.
- Intent classification.
- Entity extraction.
- Time normalization.
- Metadata filter construction.
- Query decomposition.
- Multi-query generation.

Example:

```text
what is the highest package of cse previous year
```

Should be transformed into structured query state:

```json
{
  "intent": "placement_salary",
  "branch": "Computer Science Engineering",
  "metric": "highest_salary_lpa",
  "year": "2024-25"
}
```

The retriever can then apply filters:

```text
category = placements_summary
branch ~= Computer Science Engineering
year = 2024-25
```

Relevant framework options:

- LlamaIndex RouterRetriever can select among candidate retrievers using metadata and query information. Source: https://docs.llamaindex.ai/en/stable/api_reference/retrievers/router/
- LlamaIndex router modules are designed for choosing data sources, deciding summary vs semantic search, and multi-routing. Source: https://docs.llamaindex.ai/en/stable/module_guides/querying/router/
- LangChain MultiQueryRetriever can generate alternate phrasings for the same query to improve recall. Source: https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever

### 6. No Graph-Based RAG Architecture

Current pipeline is one Python file:

```text
rag_query.py
```

It contains:

- CLI parsing.
- Query normalization.
- Chroma access.
- Local JSONL fallback.
- Reranking.
- Prompt construction.
- Ollama generation.
- Interactive loop.

This violates separation of concerns and makes testing difficult.

Recommended graph nodes:

```text
User Query
  -> NormalizeQueryNode
  -> SpellCorrectionNode
  -> IntentAndEntityExtractionNode
  -> QueryExpansionNode
  -> RetrieverRouterNode
       -> StructuredFactRetriever
       -> PropositionRetriever
       -> MainChunkRetriever
       -> SummaryRetriever
       -> BM25/SparseRetriever
  -> FusionNode
  -> RerankerNode
  -> ContextValidationNode
  -> AnswerGenerationNode
  -> GroundingCheckNode
  -> ResponseFormatterNode
```

Recommended framework:

- LangGraph. Its documentation describes building stateful LLM applications as graphs and includes checkpointing/persistence options. Source: https://reference.langchain.com/python/langgraph/overview

Why graph architecture matters here:

- Each stage can be tested independently.
- Retrieval failure can loop back into query expansion.
- Low-confidence answers can trigger fallback retrieval.
- State can store normalized query, detected intent, retrieved documents, confidence score, answer, and citations.

### 7. No Structured Retriever For Tabular Facts

The data contains structured tables:

- Program intake and year of starting.
- Fee rows.
- Hostel fees.
- Cutoff rows.
- Placement yearly metrics.
- Company eligibility by branch.
- Bus route stop-wise fees.

Current issue:

- These are embedded as text and then searched semantically.
- For exact facts, semantic retrieval is often the wrong primary method.

Recommended approach:

- Build a structured fact store alongside vector search.
- Use deterministic lookups for:
  - intake
  - year started
  - fees
  - cutoffs
  - highest/average salary
  - students placed/offers
  - company eligibility

Possible implementation:

```text
SQLite or DuckDB structured_fact_store
```

Tables:

```text
programs(program, aliases, level, intake, year_started)
fees(category, fy_fee, dse_fee)
hostels(hostel_name, gender, annual_fee)
cutoffs(branch, category, seat_type, merit_no, merit_marks)
placements(branch, year, total_students, offers, avg_lpa, highest_lpa)
company_eligibility(year, company, industry, eligible_branches)
transport(route, stop, monthly_fee)
documents(admission_type, document, applicable_categories)
```

The query graph should route exact numeric queries to this structured store before semantic search.

### 8. Chroma Metadata Is Not Used Enough

`vectorize_rag_data.py` stores metadata, but complex values are serialized as JSON strings because Chroma accepts scalar metadata.

Problems:

- Filtering by branch/category/company is awkward.
- Nested entities are not first-class searchable metadata.
- Chroma is fine for local prototypes but limited for production-grade hybrid retrieval and structured metadata filtering.

Recommendation:

- Keep Chroma only for local development.
- For production, consider Qdrant or a DB setup with proper metadata filters and hybrid search.
- If staying with Chroma, flatten important metadata into scalar fields:
  - `branch`
  - `branch_aliases_text`
  - `year`
  - `category_label`
  - `company`
  - `route`
  - `fee_amount`

### 9. Embedding Model Is Too Weak For The Query Distribution

Current model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This is fast, but the observed failures suggest it is not strong enough for:

- misspellings
- short domain acronyms
- noisy student language
- tabular fact retrieval
- branch/company name ambiguity

Recommended candidates:

- `BAAI/bge-small-en-v1.5` or `BAAI/bge-base-en-v1.5` for stronger English retrieval.
- `intfloat/e5-base-v2` or related E5 models.
- `BAAI/bge-m3` if multilingual/hybrid sparse+dense support is desired.
- Keep `all-MiniLM-L6-v2` only if speed and CPU constraints are more important than recall.

Important: changing embedding models requires rebuilding vector indexes.

### 10. No Evaluation Harness

The project has `qa_pairs.json`, but no automated retrieval/answer evaluation script.

Needed evaluations:

- Retrieval recall@k.
- MRR@k.
- NDCG@k.
- Faithfulness.
- Answer correctness.
- Context precision.
- Context recall.
- Robustness to typos and paraphrases.

Relevant library:

- Ragas provides metrics for evaluating RAG pipelines, including context and answer quality metrics. Sources: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/ and https://docs.ragas.io/en/latest/references/evaluate/

Recommended test set:

Create `scripts/rag_eval_cases.json` with examples like:

```json
[
  {
    "query": "what is the intake of aids",
    "expected_answer_contains": ["60"],
    "expected_source_section": "programs_offered/btech"
  },
  {
    "query": "when iot branch started",
    "expected_answer_contains": ["2021"],
    "expected_source_section": "programs_offered/btech"
  },
  {
    "query": "what is the highest package of cse previous year",
    "expected_answer_contains": ["8 LPA", "2024-25"],
    "expected_source_section": "placement_3yr_summary/Computer Science Engineering"
  }
]
```

### 11. No Confidence Gate Or Retrieval Failure Recovery

Current behavior:

- If top retrieved context is weak or off-topic, the answer model may say information is unavailable.
- There is no automated retry with alternative retrieval strategy.

Production behavior should be:

1. Run initial retrieval.
2. If top score is low or categories conflict with intent, run query expansion.
3. If still low, try structured retriever.
4. If still low, ask a clarification question.
5. Only say unavailable after all relevant retrievers fail.

### 12. No Citation Contract

The answer prompt asks for bracket numbers "when useful", but it does not enforce source-grounded citations.

Recommendation:

- Every answer should include at least one source section unless no answer is found.
- The answer generator should output structured JSON:

```json
{
  "answer": "...",
  "citations": [
    {"source_section": "...", "source_sheet": "..."}
  ],
  "confidence": "high|medium|low",
  "used_context_ids": ["..."]
}
```

### 13. Data Modeling Still Needs More Fact-Level Records

The system improved data quality, but vectorized propositions still do not cover every useful retrieval path.

Missing/weak records:

- One record per program alias.
- One record per company + eligible branch + year.
- One record per branch placement metric + year + metric name.
- One record per cutoff category/seat type.
- One record per hostel option.
- One record per bus stop.

This matters because users ask single-fact questions. A fact-level record should directly contain the answer.

Example desired record:

```json
{
  "fact_type": "program_intake",
  "program": "Artificial Intelligence (AI) and Data Science",
  "aliases": ["AIDS", "AI DS", "Artificial Intelligence Data Science"],
  "intake": 60,
  "year_started": 2021,
  "text": "Artificial Intelligence (AI) and Data Science, also called AIDS or AI DS, has sanctioned intake 60 and started in 2021."
}
```

## Recommended Production Architecture

### Layer 1: Data Build

Scripts:

- `excel_to_json.py`
- `fix_rag_data.py`
- new `build_fact_store.py`
- new `build_retrieval_lexicon.py`

Outputs:

- `output.json`
- `knowledge_base.jsonl`
- `propositions.jsonl`
- `summary_index.json`
- `fact_store.sqlite` or `fact_store.duckdb`
- `retrieval_lexicon.json`

### Layer 2: Index Build

Recommended indexes:

- Dense vector index for broad semantic chunks.
- Dense vector index for propositions.
- Sparse/BM25 index for exact terms.
- Structured SQL/DuckDB index for tabular exact facts.

Recommended vector DB options:

- Local prototype: Chroma + BM25Retriever + SQLite.
- Production local/server: Qdrant with dense + sparse hybrid search.

### Layer 3: Query Graph

Recommended with LangGraph:

```text
QueryInput
  -> QueryNormalizer
  -> SpellCorrector
  -> EntityExtractor
  -> IntentClassifier
  -> QueryExpander
  -> RetrieverRouter
  -> HybridRetriever
  -> StructuredRetriever
  -> Reranker
  -> ContextVerifier
  -> AnswerGenerator
  -> CitationFormatter
```

State object:

```json
{
  "raw_query": "...",
  "normalized_query": "...",
  "corrected_query": "...",
  "intent": "...",
  "entities": {},
  "candidate_contexts": [],
  "reranked_contexts": [],
  "answer": "",
  "confidence": "",
  "citations": []
}
```

### Layer 4: Evaluation

Recommended:

- `rag_eval_cases.json`
- `evaluate_retrieval.py`
- `evaluate_answers.py`
- Ragas metrics for answer/context quality.
- CrossEncoder reranking evaluator for retrieval quality.

## Recommended Libraries To Add

### Query Correction

| Need | Library | Why |
|---|---|---|
| Fast spelling correction | `symspellpy` | Fast edit-distance correction with dictionary support. |
| Simple pure Python spell checking | `pyspellchecker` | Easy to integrate; supports custom dictionaries. |
| Fuzzy domain alias matching | `rapidfuzz` | Good for branch/company/category matching against a controlled lexicon. |

### Retrieval

| Need | Library | Why |
|---|---|---|
| BM25 sparse retrieval | `rank-bm25` or LangChain `BM25Retriever` | Handles exact terms, typos after correction, acronyms, company names. |
| Dense vector retrieval | Chroma or Qdrant | Current Chroma is okay locally; Qdrant is stronger for hybrid production. |
| Hybrid dense+sparse retrieval | Qdrant | Supports hybrid retrieval with dense and sparse vectors. |
| Query routing | LlamaIndex RouterRetriever or LangGraph custom router | Routes fee/cutoff/program/placement queries to the right retriever. |

### Reranking

| Need | Library | Why |
|---|---|---|
| Accurate query-document reranking | `sentence-transformers` CrossEncoder | Pairwise query-doc scoring improves final context selection. |
| CPU-friendly reranking | `flashrank` | Lightweight reranker for retrieval pipelines. |

### Orchestration

| Need | Library | Why |
|---|---|---|
| Stateful graph workflow | `langgraph` | Production-style graph nodes, state, retries, checkpointers. |
| Alternative high-level RAG framework | `llama-index` | Router retrievers, query engines, query transforms. |
| Pipeline-style RAG | Haystack | Mature pipeline abstractions for retrievers/rankers/generators. |

### Evaluation

| Need | Library | Why |
|---|---|---|
| RAG answer/context metrics | `ragas` | Measures faithfulness, answer relevance, context precision/recall, etc. |
| Retrieval ranking metrics | Sentence Transformers evaluators | Measures reranking quality using MRR/NDCG/MAP. |

## Proposed New File Structure

```text
scripts/
  data_build/
    excel_to_json.py
    fix_rag_data.py
    build_fact_store.py
    build_retrieval_lexicon.py

  indexing/
    vectorize_dense.py
    build_sparse_index.py
    build_qdrant_index.py

  rag/
    state.py
    graph.py
    nodes/
      normalize_query.py
      spell_correct.py
      extract_entities.py
      classify_intent.py
      expand_query.py
      route_retriever.py
      retrieve_structured.py
      retrieve_dense.py
      retrieve_sparse.py
      fuse_results.py
      rerank.py
      generate_answer.py
      validate_answer.py
    prompts/
      answer_prompt.txt
      entity_extraction_prompt.txt

  evaluation/
    rag_eval_cases.json
    evaluate_retrieval.py
    evaluate_answers.py
```

## Priority Fix List

### P0: Remove Hardcoded Query Replacement From Runtime

Replace:

```text
QUERY_REPLACEMENTS
INTENT_CATEGORY_HINTS embedded directly in rag_query.py
```

With:

- `retrieval_lexicon.json`
- `symspellpy` or `pyspellchecker`
- `rapidfuzz` domain alias resolver
- LLM or classifier-based intent extraction

### P1: Add Structured Fact Store

Build SQLite/DuckDB tables for exact facts.

This will fix:

- intake queries
- fee queries
- hostel queries
- cutoff queries
- placement metrics
- company eligibility
- bus stop fees

### P2: Add True Hybrid Retrieval

Either:

- Chroma + LangChain BM25Retriever + RRF fusion, or
- Qdrant dense+sparse hybrid search.

### P3: Add Reranker

Use:

- `FlashRank` for CPU-friendly reranking, or
- `sentence-transformers` CrossEncoder for stronger accuracy.

### P4: Build LangGraph Workflow

Move from one script to a graph with recoverable nodes.

### P5: Add Evaluation

Build a test set from real failure examples and run it after every change.

## Acceptance Criteria For The Next Production Iteration

The system should answer these without hardcoded query replacement:

| Query | Expected Behavior |
|---|---|
| `what is the intake of aids` | Returns AI/Data Science intake 60. |
| `when iot branch started` | Returns CSE IoT started in 2021. |
| `do college offer robotics` | Says yes, Robotics And Artificial Intelligence is offered, intake 60, started 2025. |
| `what are the departments in this college` | Lists B.Tech branches and optionally UG/M.Tech programs. |
| `what is highest package of cse previous year` | Interprets previous year as 2024-25 and returns CSE highest salary 8 LPA. |
| `which companes allow student tie sit for placemnt in mechanical department` | Corrects spelling/intent and lists Mechanical-eligible companies from company eligibility data. |
| `fees for boys hsotel` | Corrects hostel spelling and returns boys hostel fees. |

Minimum retrieval metrics:

- Recall@5 >= 0.90 on curated factual test cases.
- MRR@5 >= 0.80 for exact fact questions.
- No unsupported answer when relevant retrieved context exists.
- No "not available" answer when expected source section appears in top-k context.

## Final Recommendation

Do not keep expanding `QUERY_REPLACEMENTS`. It is a debugging patch, not a production solution.

The next implementation should be a graph-based RAG pipeline with:

1. Dynamic spelling correction using a domain dictionary.
2. Dynamic alias resolution from the data.
3. Intent/entity extraction.
4. Structured fact lookup for numeric/table questions.
5. Hybrid dense+sparse retrieval.
6. Reranking with CrossEncoder or FlashRank.
7. Answer grounding and citation validation.
8. RAG evaluation using a fixed regression test set.

This will address the core problem: answers exist in the knowledge base, but retrieval fails because the system lacks robust query understanding and production-grade retrieval orchestration.

## Reference Links

- LangGraph Python API reference: https://reference.langchain.com/python/langgraph/overview
- LlamaIndex RouterRetriever: https://docs.llamaindex.ai/en/stable/api_reference/retrievers/router/
- LlamaIndex routing guide: https://docs.llamaindex.ai/en/stable/module_guides/querying/router/
- LangChain BM25Retriever: https://docs.langchain.com/oss/python/integrations/retrievers/bm25/
- LangChain MultiQueryRetriever reference: https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever
- Qdrant hybrid queries: https://qdrant.tech/documentation/concepts/hybrid-queries/
- Qdrant hybrid search course: https://qdrant.tech/course/essentials/day-3/hybrid-search/
- Qdrant hybrid search with FastEmbed: https://qdrant.tech/documentation/tutorials-search-engineering/hybrid-search-fastembed/
- SymSpellPy docs: https://symspellpy.readthedocs.io/
- SymSpellPy API: https://symspellpy.readthedocs.io/en/latest/api/symspellpy.html
- pyspellchecker PyPI: https://pypi.org/project/pyspellchecker/
- pyspellchecker API docs: https://pyspellchecker.readthedocs.io/en/master/code.html
- Sentence Transformers CrossEncoder docs: https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html
- Sentence Transformers reranking evaluator: https://sbert.net/docs/package_reference/cross_encoder/evaluation.html
- FlashRank PyPI: https://pypi.org/project/flashrank/
- Ragas metrics: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
- Ragas evaluate reference: https://docs.ragas.io/en/latest/references/evaluate/
