import json, pickle, re
from typing import List, Tuple
import networkx as nx
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Config
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"
CHROMA_DIR = "./chroma_db_hybrid"
GRAPH_FILE = "knowledge_graph.json"
BM25_FILE = "bm25_index.pkl"
TOP_K_DENSE = 10
TOP_K_BM25 = 10
TOP_K_GRAPH = 5

# ------------------------------------------------------------
# 1. Load components
# ------------------------------------------------------------
print("📂 Loading vector store, BM25, and graph...")
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBED_MODEL)
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, collection_name="college_knowledge")

with open(BM25_FILE, "rb") as f:
    tokenized_corpus, all_docs = pickle.load(f)
bm25 = BM25Okapi(tokenized_corpus)

with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    graph_data = json.load(f)
G = nx.node_link_graph(graph_data, edges="links")

print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# ------------------------------------------------------------
# 2. Entity extraction
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 3. Hybrid Retriever: Dense + BM25 + Graph
# ------------------------------------------------------------
class HybridGraphRetriever:
    def __init__(self, vectordb, bm25, docs_list, graph, top_dense=10, top_bm25=10, graph_k=5):
        self.vectordb = vectordb
        self.bm25 = bm25
        self.docs_list = docs_list
        self.graph = graph
        self.top_dense = top_dense
        self.top_bm25 = top_bm25
        self.graph_k = graph_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Dense search
        dense_docs = self.vectordb.similarity_search(query, k=self.top_dense)

        # BM25 search
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.top_bm25]
        bm25_docs = [self.docs_list[i] for i in top_indices]

        # Fusion (deduplicate)
        seen = set()
        combined = []
        for doc in dense_docs + bm25_docs:
            if doc.page_content not in seen:
                combined.append(doc)
                seen.add(doc.page_content)

        # Graph augmentation
        query_entities = extract_entities(query)
        doc_entities = set()
        for d in combined:
            for _, ename in extract_entities(d.page_content):
                doc_entities.add(ename)

        neighbours = set()
        for ent in list(query_entities) + list(doc_entities):
            if self.graph.has_node(ent):
                neighbours.update(self.graph.neighbors(ent))
        neighbours = list(neighbours)[:self.graph_k]

        for nb in neighbours:
            extra = self.vectordb.similarity_search(nb, k=2)
            for ed in extra:
                if ed.page_content not in seen:
                    combined.append(ed)
                    seen.add(ed.page_content)

        return combined

retriever = HybridGraphRetriever(vectordb, bm25, all_docs, G, top_dense=TOP_K_DENSE, top_bm25=TOP_K_BM25, graph_k=TOP_K_GRAPH)

# ------------------------------------------------------------
# 4. LLM and Prompt
# ------------------------------------------------------------
llm = ChatOllama(base_url=OLLAMA_URL, model=LLM_MODEL, temperature=0)

template = """You are a precise college information assistant. Use ONLY the provided context to answer.
If the context contains a list of items (e.g., programs, branches, documents), list ALL of them completely.
If you don't know the answer, say so.

Conversation History:
{history}

Context:
{context}

Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

# ------------------------------------------------------------
# 5. Manual memory
# ------------------------------------------------------------
history: List[Tuple[str, str]] = []
print("🤖 Hybrid RAG Chatbot ready. Type 'exit' to stop.\n")

while True:
    try:
        user_input = input("You: ")
    except (EOFError, KeyboardInterrupt):
        break
    if user_input.lower() in ["exit", "quit"]:
        break

    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs])

    history_str = ""
    for q, a in history[-5:]:
        history_str += f"User: {q}\nAssistant: {a}\n"

    filled_prompt = prompt.format(history=history_str, context=context, question=user_input)
    response = llm.invoke(filled_prompt)
    answer = response.content

    history.append((user_input, answer))
    print(f"Bot: {answer}\n")