import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ============================================================
# CONFIG
# ============================================================

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "qwen2.5:7b"

VECTOR_DB_PATH = "./vectordb"

TOP_K = 5

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

knowledge_collection = client.get_collection("knowledge_base")
factoid_collection = client.get_collection("factoids")
query_collection = client.get_collection("query_surfaces")

print("System ready.\n")


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_collection(collection, query, top_k=5):
    query_embedding = embed_model.encode(query).tolist()

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
    except Exception as e:
        print(f"Search failed: {e}")
        return []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output = []

    for d, m, dist in zip(docs, metas, distances):
        output.append({
            "text": d,
            "metadata": m,
            "distance": dist
        })

    return output


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve(query):
    kb_results = search_collection(
        knowledge_collection,
        query,
        top_k=TOP_K
    )

    fact_results = search_collection(
        factoid_collection,
        query,
        top_k=3
    )

    query_results = search_collection(
        query_collection,
        query,
        top_k=2
    )

    combined = kb_results + fact_results + query_results

    # sort by similarity
    combined = sorted(combined, key=lambda x: x["distance"])

    return combined[:8]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):
    context_parts = []

    seen = set()

    for r in results:
        text = r["text"]

        if text in seen:
            continue

        seen.add(text)

        context_parts.append(text)

    return "\n\n".join(context_parts)


# ============================================================
# ASK LLM
# ============================================================

SYSTEM_PROMPT = """
You are an expert college admission assistant.

Answer ONLY from the provided context.

Rules:
- If information is unavailable, say so clearly.
- Keep answers accurate.
- Use bullet points when useful.
- Mention cutoff marks, fees, placements, hostel, etc precisely.
- Do not hallucinate.
"""


def ask_llm(query, context):

    prompt = f"""
CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    query = input("\nQuestion: ").strip()

    if query.lower() in ["exit", "quit"]:
        break

    print("\nSearching knowledge base...")

    results = retrieve(query)

    context = build_context(results)

    print("\nGenerating answer...\n")

    answer = ask_llm(query, context)

    print("=" * 60)
    print(answer)
    print("=" * 60)