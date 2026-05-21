"""
RAG Backbone Builder – Fixed for langchain-chroma (no .persist())
"""

import json, os, re, time
from typing import List, Tuple
import networkx as nx
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Config
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "./chroma_db"
JSON_DIR = "../data_to_trained"
GRAPH_FILE = "knowledge_graph.json"
BATCH_SIZE = 50

# ------------------------------------------------------------
# 1. Load JSON files
# ------------------------------------------------------------
def load_all_json_files(json_dir: str) -> List[Document]:
    docs = []
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"JSON directory '{json_dir}' not found.")
    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(json_dir, filename)
        print(f"  Loading {filename}...")
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
print(f"✅ Loaded {len(documents)} knowledge snippets.\n")

# ------------------------------------------------------------
# 2. Build Knowledge Graph
# ------------------------------------------------------------
print("🕸️ Building knowledge graph...")
G = nx.Graph()

def extract_entities(text: str) -> List[Tuple[str, str]]:
    entities = []
    course_patterns = [
        "Mechanical Engineering", "Computer Science & Engineering",
        "Electrical Engineering", "Civil Engineering", "Aeronautical Engineering",
        "Food Technology", "AI & Data Science", "IoT & Cyber Security",
        "Robotics and Artificial Intelligence", "Computer Science (IoT & Cyber Security)"
    ]
    for course in course_patterns:
        if course.lower() in text.lower():
            entities.append(("COURSE", course))
    cat_list = ["OPEN","OBC","SC","ST","VJ","NT-1","NT-2","NT-3","SEBC","DEF","TFWS","EWS","General","Ladies","EBC"]
    for cat in cat_list:
        if re.search(r'\b' + re.escape(cat) + r'\b', text, re.IGNORECASE):
            entities.append(("CATEGORY", cat.upper()))
    if "bus route" in text.lower():
        match = re.search(r"bus route ([A-Za-z\-]+)", text, re.IGNORECASE)
        if match:
            entities.append(("BUS_ROUTE", match.group(1)))
    if "visited the campus" in text.lower():
        match = re.search(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
        if match:
            entities.append(("COMPANY", match.group(1)))
    return entities

for doc in documents:
    ents = extract_entities(doc.page_content)
    for etype, ename in ents:
        if not G.has_node(ename):
            G.add_node(ename, type=etype)
    for i in range(len(ents)):
        for j in range(i+1, len(ents)):
            G.add_edge(ents[i][1], ents[j][1], weight=1.0)

print(f"   Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
graph_data = nx.node_link_data(G, edges="links")
with open(GRAPH_FILE, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2, ensure_ascii=False)
print(f"   Graph saved to {GRAPH_FILE}\n")

# ------------------------------------------------------------
# 3. Embed & store in Chroma (batch, no .persist())
# ------------------------------------------------------------
print("🧠 Creating embeddings with Ollama...")
print(f"   Model: {EMBED_MODEL}")

try:
    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)
    _ = embeddings.embed_query("test")
    print("   ✅ Ollama embedding connection successful.")
except Exception as e:
    print(f"   ❌ Failed to connect to Ollama: {e}")
    exit(1)

# Clear old DB (optional)
# import shutil; shutil.rmtree(CHROMA_DIR, ignore_errors=True)

print("💾 Building Chroma vector database in batches...")
print(f"   Batch size: {BATCH_SIZE}, total documents: {len(documents)}")

# First batch creates the collection
first_batch = documents[:BATCH_SIZE]
vectordb = Chroma.from_texts(
    texts=[doc.page_content for doc in first_batch],
    embedding=embeddings,
    metadatas=[doc.metadata for doc in first_batch],
    persist_directory=CHROMA_DIR,
    collection_name="college_knowledge"
)

# Remaining batches with progress bar
remaining = documents[BATCH_SIZE:]
start_time = time.time()
for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="   Embedding", unit="batch"):
    batch = remaining[i:i+BATCH_SIZE]
    vectordb.add_texts(
        texts=[doc.page_content for doc in batch],
        metadatas=[doc.metadata for doc in batch]
    )

elapsed = time.time() - start_time
print(f"   ✅ Vector store persisted to {CHROMA_DIR}")
print(f"   Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

print("\n🎉 RAG backbone is ready!")
print(f"   • {len(documents)} documents embedded")
print(f"   • Knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"   • Vector store : {CHROMA_DIR}")
print(f"   • Graph file   : {GRAPH_FILE}")