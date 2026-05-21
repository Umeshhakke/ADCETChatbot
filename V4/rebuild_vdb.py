"""
Rebuild vector store (nomic-embed-text) + BM25 index.
"""
import json, os, re, time, pickle, shutil
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize

# Config
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"   # stable & fast
CHROMA_DIR = "./chroma_db_hybrid"
JSON_DIR = "../data_to_trained"
BATCH_SIZE = 50

# ------------------------------------------------------------
# 1. Load all JSON files
# ------------------------------------------------------------
def load_all_json_files(json_dir: str) -> List[Document]:
    docs = []
    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(json_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    docs.append(Document(page_content=item, metadata={"source": filename}))
                elif isinstance(item, dict):
                    docs.append(Document(page_content=json.dumps(item, ensure_ascii=False), metadata={"source": filename}))
        elif isinstance(data, dict):
            for key, sentences in data.items():
                if isinstance(sentences, list):
                    for sent in sentences:
                        docs.append(Document(page_content=sent, metadata={"source": filename, "group": key}))
                else:
                    docs.append(Document(page_content=json.dumps(sentences, ensure_ascii=False), metadata={"source": filename, "group": key}))
    return docs

print("📂 Loading JSON files...")
documents = load_all_json_files(JSON_DIR)
print(f"✅ Loaded {len(documents)} documents.\n")

# ------------------------------------------------------------
# 2. Build BM25 index
# ------------------------------------------------------------
print("📝 Building BM25 index...")
tokenized_corpus = [word_tokenize(doc.page_content.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
with open("bm25_index.pkl", "wb") as f:
    pickle.dump((tokenized_corpus, documents), f)
print("✅ BM25 index saved.\n")

# ------------------------------------------------------------
# 3. Create Chroma vector store (nomic-embed-text)
# ------------------------------------------------------------
print(f"🧠 Embedding with {EMBED_MODEL}...")
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBED_MODEL)
_ = embeddings.embed_query("test")
print("   ✅ Connection ok.\n")

# Clear old DB
shutil.rmtree(CHROMA_DIR, ignore_errors=True)

print("💾 Building Chroma vector database (batch progress)...")
first_batch = documents[:BATCH_SIZE]
vectordb = Chroma.from_texts(
    texts=[doc.page_content for doc in first_batch],
    embedding=embeddings,
    metadatas=[doc.metadata for doc in first_batch],
    persist_directory=CHROMA_DIR,
    collection_name="college_knowledge"
)

remaining = documents[BATCH_SIZE:]
start_time = time.time()
for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="   Embedding", unit="batch"):
    batch = remaining[i:i+BATCH_SIZE]
    vectordb.add_texts(
        texts=[doc.page_content for doc in batch],
        metadatas=[doc.metadata for doc in batch]
    )

elapsed = time.time() - start_time
print(f"   ✅ Vector store saved to {CHROMA_DIR}")
print(f"   Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")

print("🎉 Rebuild complete. You can now run the hybrid chatbot.")