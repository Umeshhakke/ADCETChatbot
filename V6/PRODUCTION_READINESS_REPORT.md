# ADCET V6 RAG Production Readiness Report

## Completed Improvements

### 1. Query correction before retrieval

Added a preprocessing layer in `knowledge_utils.py` that runs before vector search and BM25 search.

What it now handles:

- Common spelling mistakes such as `makrs -> marks`, `admisson -> admission`, `cuttof -> cutoff`, `abot -> about`, and `collage -> college`.
- Domain-safe correction so important short terms and category codes are not damaged, for example `CSE`, `AI`, `IoT`, `OBC`, `SC`, `ST`, `TFWS`, `EWS`, `NT1`, `NT2`, `NT3`.
- Query normalization before retrieval, so the retriever works on cleaned text instead of raw user typos.

Example:

```text
Input:  how much makrs need to get admisson in cse open
Clean:  how much marks need to get admission in cse open
Intent: cutoff
```

### 2. Cutoff intent routing

The previous system confused these two different meanings:

- Cutoff question: "How many marks are needed to get admission?"
- Eligibility question: "What minimum PCM marks are required for eligibility?"

This caused the chatbot to retrieve qualifying criteria instead of cutoff data.

The new logic explicitly classifies admission-mark, score, percentile, rank, and chance questions as `cutoff` intent unless the user is clearly asking about academic eligibility or PCM criteria.

Examples now routed to cutoff:

- "how much marks need to get admission"
- "required marks for CSE"
- "safe percentile for admission"
- "cutoff for OBC CSE"
- "chance of admission in AI DS"

Examples still routed to eligibility/admission:

- "what is eligibility criteria for engineering"
- "minimum marks for PCM"
- "what subjects are required for admission"

### 3. Category-aware retrieval ranking

The hybrid retriever in `rag_chatbot.py` now uses:

- Vector score when the embedding model is available.
- Normalized BM25 score.
- Lexical overlap score.
- Category boost for matching intent.
- Penalty for eligibility/document chunks when the user is asking about cutoff marks.

This prevents `minimum_marks` eligibility chunks and marksheet/document chunks from outranking `merit_marks` cutoff chunks.

### 4. Deterministic cutoff answer path

Added direct cutoff lookup from:

- `data_file/cutoff.json`
- `knowledge/cutoff_cleaned.csv`

This is important for production because cutoff answers should be exact and should not depend only on LLM generation.

Supported aliases include:

- `CSE`, `CS`, `computer science`
- `AI DS`, `AIDS`, `artificial intelligence and data science`
- `IoT`, `cyber security`
- `RAI`, `robotics`
- `mech`, `civil`, `electrical`, `aero`, `food tech`
- `OPEN`, `OBC`, `SC`, `ST`, `SEBC`, `EWS`, `TFWS`, `DEF`, `VJ`, `NT1`, `NT2`, `NT3`

If the user asks a cutoff question without branch/category, the bot now asks for the missing branch instead of returning unrelated criteria.

### 5. Windows console fallback fix

The app previously used Unicode arrows in fallback logging. On Windows CP1252 consoles, this could crash the chatbot when embedding/reranker loading failed.

Changed these logs to ASCII-safe text:

```text
LLM unavailable -> fallback mode
No reranker -> using score ranking
```

### 6. Prompt guardrail

The LLM prompt now explicitly says:

```text
If the question asks how many marks, score, rank, percentile, or chance is needed for admission, answer from cutoff/merit-mark data, not qualifying eligibility criteria.
```

This reduces hallucination and keeps generated answers aligned with retrieved cutoff context.

### 7. Department query handling

Added a deterministic department/program answer path from `data_file/program_offered.json`.

This fixes the issue from `V6/prompt.txt` where:

- "how many departments are there in adcet" returned unavailable.
- "numeber of departments" returned `5`.
- "departments" returned only four departments.
- "which are they" lost the previous context.

The bot now answers B.Tech engineering department questions directly from structured program data and returns all 9 departments:

1. Mechanical Engineering
2. Computer Science & Engineering
3. Electrical Engineering
4. Civil Engineering
5. Aeronautical Engineering
6. Food Technology
7. Artificial Intelligence (AI) and Data Science
8. Computer Science & Engineering (Internet of Things and Cyber Security Including Block Chain Technology)
9. Robotics and Artificial Intelligence

The chatbot instance also remembers the last department topic, so a follow-up like "which are they" after asking the department count returns the full department list.

## Validation Performed

### Compile check

Passed:

```powershell
python -m py_compile V6\knowledge_utils.py V6\rag_chatbot.py V6\build_rag.py
```

### Query preprocessing checks

Validated:

```text
how much makrs need to get admisson in this collage
=> corrected to: how much marks need to get admission in this college
=> category: cutoff
```

```text
what is eligibility criteria for engineering
=> category: admission
```

```text
minimum marks for pcm
=> category: admission
```

```text
numeber of departments
=> corrected to: number of departments
=> structured department answer
```

### Cutoff answer checks

Validated:

```text
Q: how much makrs need to get admisson in cse open
A: Using OPEN category. For General Computer Science & Engineering, the cutoff merit marks are 87.14 and the cutoff merit number is 43336.
```

```text
Q: cutoff for food tech open ladies
A: Using OPEN category. For Ladies Food Technology, the cutoff merit marks are 93.28 and the cutoff merit number is 24721.
```

```text
Q: how much marks need to get admission in this college
A: Cutoff marks depend on the branch and reservation category. Please mention the branch...
```

### Department answer checks

Validated:

```text
Q: how many departments are there in adcet
A: ADCET has 9 B.Tech engineering departments.
```

```text
Q: which are they
A: ADCET has 9 B.Tech engineering departments:
1. Mechanical Engineering
2. Computer Science & Engineering
3. Electrical Engineering
4. Civil Engineering
5. Aeronautical Engineering
6. Food Technology
7. Artificial Intelligence (AI) and Data Science
8. Computer Science & Engineering (Internet of Things and Cyber Security Including Block Chain Technology)
9. Robotics and Artificial Intelligence
```

## Remaining Production Work

### 1. Model deployment must be made deterministic

Current risk:

- The embedding model and reranker may try to access Hugging Face if not cached locally.
- In restricted deployment environments, this can fail.
- On low-memory machines, model loading can fail with paging-file or memory errors.

Production requirement:

- Pre-download and package the embedding model and reranker.
- Set model paths explicitly through environment variables.
- Run deployment with `HF_HUB_OFFLINE=1` after models are cached.
- Add a startup health check that reports whether vector search, reranker, and LLM are actually available.

### 2. Chroma DB rebuild should be part of deployment

Current risk:

- Code can improve retrieval scoring, but stale or poorly chunked Chroma data can still reduce answer quality.
- Some cutoff data appears in large JSON-like chunks.

Production requirement:

- Rebuild Chroma after final data cleanup.
- Prefer atomic fact chunks for cutoff data, one course/category/group fact per chunk.
- Store stronger metadata such as `category=cutoff`, `course`, `quota_category`, and `group`.

### 3. Add automated evaluation tests

Production should not rely only on manual testing.

Recommended test set:

- 30 cutoff questions with spelling mistakes.
- 20 eligibility questions.
- 20 document/admission-process questions.
- 20 branch alias questions.
- 20 ambiguous questions where the bot should ask for branch/category clarification.

Each test should verify:

- Detected intent.
- Top retrieved category.
- Final answer contains the expected fact or clarification.

### 4. Add API-level validation

Before deployment, test the actual chatbot API endpoint, not only direct Python calls.

Required checks:

- Concurrent requests.
- Empty input.
- Very long input.
- Non-English/mixed-language input.
- Repeated user questions.
- LLM unavailable fallback.
- Chroma unavailable startup failure.

### 5. Improve observability

Add structured logs for:

- Original query.
- Corrected query.
- Detected intent.
- Top retrieved document categories.
- Whether deterministic cutoff lookup was used.
- LLM provider status.
- Response latency.

Do not log personal user data in production unless required and explicitly approved.

### 6. Data quality cleanup

Known data issues:

- `knowledge/cutoff_cleaned.csv` contains group text like `L(Ledies)`.
- Some course names are abbreviations in CSV and full names in JSON.
- Eligibility data uses `minimum_marks`, which can conflict semantically with cutoff `merit_marks`.

Recommended cleanup:

- Standardize course names across all files.
- Standardize group names to `General` and `Ladies`.
- Standardize category names to one format: `OPEN`, `OBC`, `SC`, `ST`, etc.
- Keep cutoff and eligibility data in separate structured sources with explicit metadata.

## Production Status

The V6 chatbot is improved for the reported cutoff/marks problem and spelling mistakes before retrieval. The main remaining blocker for production is deployment reliability: models, Chroma DB, environment variables, and API health checks must be made deterministic and tested under the same environment where the chatbot will be hosted.
