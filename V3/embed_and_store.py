# embed_and_store.py
# ============================================================
# Embed all generated RAG files and store into ChromaDB
# ============================================================

import json
import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

CHROMA_DIR = "./vectordb"

KB_FILE = "knowledge_base.jsonl"
PROP_FILE = "propositions.jsonl"
QUERY_FILE = "query_surfaces.jsonl"

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("=" * 60)
print("Loading embedding model...")
print("=" * 60)

model = SentenceTransformer(EMBED_MODEL)

print(f"Loaded: {EMBED_MODEL}")

# ============================================================
# CHROMA CLIENT
# ============================================================

client = chromadb.PersistentClient(path=CHROMA_DIR)

kb_collection = client.get_or_create_collection(
    name="knowledge_base"
)

prop_collection = client.get_or_create_collection(
    name="factoids"
)

query_collection = client.get_or_create_collection(
    name="query_surfaces"
)

print("\nCollections ready.")

# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def safe_str(x):
    if x is None:
        return ""
    return str(x)


# ============================================================
# KNOWLEDGE BASE EMBEDDING
# ============================================================

def embed_knowledge_base():
    print("\n" + "=" * 60)
    print("Embedding KNOWLEDGE BASE")
    print("=" * 60)

    records = load_jsonl(KB_FILE)

    docs = []
    ids = []
    metas = []

    for r in tqdm(records):

        text = (
            r.get("contextualised_text")
            or r.get("text")
            or ""
        ).strip()

        if not text:
            continue

        docs.append(text)

        ids.append(r.get("id", str(uuid.uuid4())))

        metas.append({
            "category": safe_str(r.get("category")),
            "source_section": safe_str(r.get("source_section")),
            "college": safe_str(r.get("college")),
            "parent_id": safe_str(r.get("parent_id")),
        })

    print(f"Documents: {len(docs)}")

    embeddings = model.encode(
        docs,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    kb_collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metas,
    )

    print("Knowledge base stored.")


# ============================================================
# PROPOSITIONS / FACTOIDS
# ============================================================

def embed_factoids():
    print("\n" + "=" * 60)
    print("Embedding FACTOIDS")
    print("=" * 60)

    records = load_jsonl(PROP_FILE)

    docs = []
    ids = []
    metas = []

    for r in tqdm(records):

        text = r.get("text", "").strip()

        if not text:
            continue

        docs.append(text)

        ids.append(r.get("id", str(uuid.uuid4())))

        metas.append({
            "category": safe_str(r.get("category")),
            "source_section": safe_str(r.get("source_section")),
            "parent_chunk_id": safe_str(r.get("parent_chunk_id")),
        })

    print(f"Factoids: {len(docs)}")

    embeddings = model.encode(
        docs,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    prop_collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metas,
    )

    print("Factoids stored.")


# ============================================================
# QUERY SURFACES
# ============================================================

def embed_query_surfaces():
    print("\n" + "=" * 60)
    print("Embedding QUERY SURFACES")
    print("=" * 60)

    records = load_jsonl(QUERY_FILE)

    docs = []
    ids = []
    metas = []

    for r in tqdm(records):

        text = r.get("variant_text", "").strip()

        if not text:
            continue

        docs.append(text)

        ids.append(r.get("id", str(uuid.uuid4())))

        metas.append({
            "category": safe_str(r.get("category")),
            "source_section": safe_str(r.get("source_section")),
            "parent_chunk_id": safe_str(r.get("parent_chunk_id")),
        })

    print(f"Query variants: {len(docs)}")

    embeddings = model.encode(
        docs,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    query_collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metas,
    )

    print("Query surfaces stored.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nStarting embedding pipeline...\n")

    embed_knowledge_base()

    if Path(PROP_FILE).exists():
        embed_factoids()

    if Path(QUERY_FILE).exists():
        embed_query_surfaces()

    print("\n" + "=" * 60)
    print("ALL DATA EMBEDDED SUCCESSFULLY")
    print("=" * 60)

    print("\nVector DB Location:")
    print(CHROMA_DIR)

    print("\nCollections:")
    print(" - knowledge_base")
    print(" - factoids")
    print(" - query_surfaces")