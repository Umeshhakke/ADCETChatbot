# V6 RAG Pipeline Changes

## What Was Corrected

The chatbot was answering some queries from shortcut logic before using Chroma retrieval. That made questions like `what are the documents required for admission` behave incorrectly and could mix category-wise document data.

The V6 flow now retrieves from the trained Chroma collection first. The content used for answers comes from `data_file`, including the category-wise admission documents in `data_file/document.json`.

## Source Data

The active RAG source is:

```text
D:\NEWChatbotADCET\data_file
```

Configured in `.env`:

```text
ADCET_KNOWLEDGE_DIR=./data_file
ADCET_CHUNK_SIZE=1800
ADCET_CHUNK_OVERLAP=180
ADCET_HF_LOCAL_FILES_ONLY=true
```

## Code Changes

1. Removed direct answer shortcuts from the main chat flow.
   - No direct structured cutoff answer before retrieval.
   - No direct department answer before retrieval.
   - No hardcoded document-list formatter.
   - No hardcoded fee/hostel formatter.

2. Improved JSON-to-RAG chunking in `V6/build_rag.py`.
   - `data_file` JSON is converted into clean retrieval records, not answer templates.
   - Admission documents are stored one record per admission type and category/caste.
   - Cutoffs are stored one record per group, course, and category.
   - Bus fees are stored one record per route and stop.
   - College fees are stored one record per fee category.
   - Hostel fees are stored one record per hostel.
   - Programs include per-program records plus degree summaries, so B.Tech count/list queries retrieve the full B.Tech context.
   - Placement company data includes company records plus branch/year summaries.
   - Placement statistics are stored by branch.

3. Improved query intent handling in `V6/knowledge_utils.py`.
   - Document/certificate/marksheet queries route to admission retrieval.
   - Cutoff routing is stricter and requires marks/rank/percentile/cutoff style wording.
   - Common typo `admissio` is corrected to `admission`.

4. Improved retrieval behavior in `V6/rag_chatbot.py`.
   - The chatbot retrieves from Chroma and builds the LLM prompt from retrieved chunks.
   - Metadata such as category, course, group, branch, year, degree, route, stop, hostel type, and record kind is added to context.
   - Retrieval scoring boosts chunks whose metadata matches the query.
   - After reranking, context is focused to matching metadata when a specific category/branch/year/degree/route is present.
   - If document context contains multiple admission categories and the user did not specify category/caste, it asks for the category instead of merging all lists.
   - If the LLM is unavailable, fallback output is retrieved text from Chroma, not a fixed answer.

5. Made paths deterministic in `V6/settings.py`.
   - Relative paths resolve from the project root.
   - This avoids different Chroma databases being loaded depending on the current working directory.

## Rebuild Command

Run this after changing files in `data_file`:

```powershell
python V6\build_rag.py
```

Current rebuilt collection:

```text
Path: D:\NEWChatbotADCET\chroma_db
Collection: adcet_college
Chunks: 286
Source: D:\NEWChatbotADCET\data_file
```

## Verified Behavior

```text
what documents required for admission
```

Asks the user to mention category/caste because the retrieved document data is category-wise.

```text
what documents required for obc admission
```

Retrieves the OBC / SBC / SEBC document record from `document.json` and answers from that context.

```text
what documents required for sc admission
```

Retrieves the SC / ST document record from `document.json` and answers from that context.

```text
documents for open first year admission
```

Retrieves the OPEN / EBC / TFWS / EWS First Year document record.

```text
documents for dsy open admission
```

Retrieves the DSY OPEN / EBC / TFWS / EWS document record.

```text
college fees for ews category
```

Retrieves only the EBC / EWS / OBC fee record.

```text
how many btech programs are offered
```

Retrieves the B.Tech degree summary and returns 9.

```text
which companies visited in 2024-25 for mechanical
```

Retrieves the 2024-25 Mechanical Engineering company summary.
