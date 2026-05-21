import os
import json

from langchain.schema import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA

from langchain_community.llms import Ollama


# ==========================================
# PATH CONFIG
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DOCS_FOLDER = os.path.join(
    BASE_DIR,
    "..",
    "data_to_trained"
)

CHROMA_DB_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ==========================================
# SETTINGS
# ==========================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

LOCAL_MODEL = "qwen2.5:7b"


# ==========================================
# LOAD DOCUMENTS
# ==========================================

def load_documents():

    docs = []

    print("📄 Loading documents...\n")

    for filename in os.listdir(DOCS_FOLDER):

        filepath = os.path.join(
            DOCS_FOLDER,
            filename
        )

        ext = os.path.splitext(
            filename
        )[1].lower()

        try:

            # ==================================
            # PDF FILES
            # ==================================

            if ext == ".pdf":

                loader = PyPDFLoader(
                    filepath
                )

                loaded_docs = loader.load()

                docs.extend(loaded_docs)

            # ==================================
            # TEXT FILES
            # ==================================

            elif ext == ".txt":

                loader = TextLoader(
                    filepath,
                    encoding="utf-8"
                )

                loaded_docs = loader.load()

                docs.extend(loaded_docs)

            # ==================================
            # JSON FILES
            # ==================================

            elif ext == ".json":

                with open(
                    filepath,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                # Convert entire JSON into text
                text = json.dumps(
                    data,
                    indent=2,
                    ensure_ascii=False
                )

                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename
                    }
                )

                docs.append(doc)

            else:

                print(f"⚠️ Skipped: {filename}")

                continue

            print(f"✅ Loaded: {filename}")

        except Exception as e:

            print(f"❌ Failed loading: {filename}")

            print(e)

    # ==================================
    # FINAL CHECK
    # ==================================

    if len(docs) == 0:

        raise ValueError(
            f"\n❌ No documents loaded from:\n{DOCS_FOLDER}"
        )

    print(f"\n📚 Total documents loaded: {len(docs)}")

    return docs


# ==========================================
# BUILD / LOAD VECTOR STORE
# ==========================================

def build_or_load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ==================================
    # LOAD EXISTING DB
    # ==================================

    if os.path.exists(CHROMA_DB_DIR):

        if len(os.listdir(CHROMA_DB_DIR)) > 0:

            print("\n🔁 Loading existing vector database...\n")

            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings
            )

            return vectorstore

    # ==================================
    # BUILD NEW DB
    # ==================================

    docs = load_documents()

    print("\n✂️ Splitting documents...\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(
        docs
    )

    print(f"✅ {len(chunks)} chunks created")

    if len(chunks) == 0:

        raise ValueError(
            "❌ No chunks created."
        )

    print("\n🧠 Creating embeddings + vector DB...\n")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    vectorstore.persist()

    print("✅ Vector database saved")

    return vectorstore


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\n🚀 Starting RAGsmith...\n")

    # ==================================
    # CHECK DOCS FOLDER
    # ==================================

    if not os.path.exists(DOCS_FOLDER):

        os.makedirs(DOCS_FOLDER)

        print(f"📁 Created folder:\n{DOCS_FOLDER}")

        print("\n➡️ Add files and run again.")

        exit()

    # ==================================
    # VECTOR STORE
    # ==================================

    vectorstore = build_or_load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    # ==================================
    # OLLAMA MODEL
    # ==================================

    llm = Ollama(
        model=LOCAL_MODEL
    )

    # ==================================
    # QA CHAIN
    # ==================================

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    print("\n✅ RAGsmith Ready!")
    print("Type 'exit' to quit.\n")

    # ==================================
    # CHAT LOOP
    # ==================================

    while True:

        query = input("❓ Question: ")

        if query.lower() in [
            "exit",
            "quit"
        ]:
            break

        try:

            result = qa.invoke({
                "query": query
            })

            print("\n🤖 Answer:\n")

            print(result["result"])

            print("\n📚 Sources:")

            for doc in result["source_documents"]:

                print(
                    f" - {doc.metadata.get('source', 'Unknown')}"
                )

            print("\n" + "=" * 60 + "\n")

        except Exception as e:

            print(f"\n❌ Error:\n{e}\n")