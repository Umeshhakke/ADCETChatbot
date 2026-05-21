import json
import os
import re
from typing import List, Tuple

import networkx as nx
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# =============================================================================
# Configuration
# =============================================================================
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"
CHROMA_DIR = "./chroma_db"
GRAPH_FILE = "knowledge_graph.json"
TOP_K_VECTOR = 5          # initial vector search results
TOP_K_GRAPH = 3           # additional docs via graph neighbours

# =============================================================================
# 1. Load persisted vector store and graph
# =============================================================================
print("📂 Loading vector store and graph...")

embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBED_MODEL)
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="college_knowledge"
)

# Load knowledge graph
with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    graph_data = json.load(f)
G = nx.node_link_graph(graph_data, edges="links")
print(f"   Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# =============================================================================
# 2. Entity extraction (same as during build)
# =============================================================================
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

# =============================================================================
# 3. Custom Graph-Augmented Retriever
# =============================================================================
class GraphAugmentedRetriever:
    def __init__(self, vectordb, graph, top_k=5, graph_top_k=3):
        self.vectordb = vectordb
        self.graph = graph
        self.top_k = top_k
        self.graph_top_k = graph_top_k

    def get_relevant_documents(self, query: str) -> List:
        # Step 1: Vector search
        docs = self.vectordb.similarity_search(query, k=self.top_k)

        # Step 2: Extract entities from query and retrieved docs
        query_entities = extract_entities(query)
        doc_entities = set()
        for d in docs:
            for _, ename in extract_entities(d.page_content):
                doc_entities.add(ename)

        # Step 3: Find graph neighbours of all involved entities
        neighbours = set()
        for entity in list(query_entities) + list(doc_entities):
            if self.graph.has_node(entity):
                neighbours.update(self.graph.neighbors(entity))
        neighbours = list(neighbours)[:self.graph_top_k]

        # Step 4: Retrieve additional documents that mention these neighbours
        extra_docs = []
        seen = {d.page_content for d in docs}
        for neighbour in neighbours:
            # Chroma doesn't support text contains filter easily;
            # we'll just do a vector search for the neighbour as a query
            temp_docs = self.vectordb.similarity_search(neighbour, k=2)
            for td in temp_docs:
                if td.page_content not in seen:
                    extra_docs.append(td)
                    seen.add(td.page_content)

        return docs + extra_docs[:self.graph_top_k]

# Instantiate retriever
retriever = GraphAugmentedRetriever(vectordb, G, top_k=TOP_K_VECTOR, graph_top_k=TOP_K_GRAPH)

# =============================================================================
# 4. LLM and Prompt
# =============================================================================
llm = ChatOllama(base_url=OLLAMA_URL, model=LLM_MODEL, temperature=0)

template = """You are a helpful college information assistant. Use the following conversation history and retrieved context to answer the user's question. If you don't know the answer, say so.

Conversation History:
{history}

Context:
{context}

Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

# =============================================================================
# 5. Manual memory (last 5 turns)
# =============================================================================
history: List[Tuple[str, str]] = []

print("\n🤖 Hybrid RAG Chatbot ready. Type 'exit' to stop.\n")

while True:
    try:
        user_input = input("You: ")
    except (EOFError, KeyboardInterrupt):
        break
    if user_input.lower() in ["exit", "quit"]:
        break

    # Retrieve context
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs])

    # Build history string
    history_str = ""
    for q, a in history[-5:]:
        history_str += f"User: {q}\nAssistant: {a}\n"

    # Generate answer
    filled_prompt = prompt.format(history=history_str, context=context, question=user_input)
    response = llm.invoke(filled_prompt)
    answer = response.content

    # Save to memory
    history.append((user_input, answer))

    print(f"Bot: {answer}\n")