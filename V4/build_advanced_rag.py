import os
import json
import torch
from tqdm import tqdm

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# =====================================================
# CONFIG
# =====================================================

DATA_FOLDER = "../data_to_trained"
VECTOR_DB_PATH = "./vector_store"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing Device: {DEVICE}")


# =====================================================
# LOAD DOCUMENTS
# =====================================================

documents = []

for file in os.listdir(DATA_FOLDER):

    if not file.endswith(".json"):
        continue

    path = os.path.join(DATA_FOLDER, file)

    print(f"\nLoading: {file}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):

        for item in data:

            # ----------------------------------------
            # STRING FORMAT
            # ----------------------------------------
            if isinstance(item, str):

                text = item.strip()

            # ----------------------------------------
            # DICT FORMAT
            # ----------------------------------------
            elif isinstance(item, dict):

                lines = []

                for k, v in item.items():

                    if v is None or str(v).strip() == "-":
                        v = "Data Not Found"

                    lines.append(f"{k}: {v}")

                text = "\n".join(lines)

            else:
                continue

            # ----------------------------------------
            # NORMALIZATION
            # ----------------------------------------

            text = text.replace("Cse", "Computer Science Engineering")
            text = text.replace("ELE", "Electrical Engineering")
            text = text.replace("&", " and ")

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file
                    }
                )
            )

print(f"\nLoaded Docs: {len(documents)}")


# =====================================================
# ADVANCED CHUNKING
# =====================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
    separators=[
        "\n\n",
        "\n",
        ". ",
        ", ",
        " "
    ]
)

chunks = splitter.split_documents(documents)

print(f"Generated Chunks: {len(chunks)}")


# =====================================================
# EMBEDDING MODEL
# =====================================================

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={
        "device": DEVICE
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32
    }
)


# =====================================================
# CREATE VECTOR DATABASE
# =====================================================

print("\nGenerating embeddings and FAISS index...")

db = FAISS.from_documents(
    chunks,
    embedding_model
)

db.save_local(VECTOR_DB_PATH)

print("\nAdvanced RAG Vector DB Created Successfully!")