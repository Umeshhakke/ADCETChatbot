import hashlib
import json
import re
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from knowledge_utils import detect_category
from settings import runtime_config


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def load_documents() -> list[dict]:
    data_folder = Path(__file__).resolve().parent.parent / "data_file"

    all_documents = []

    for json_file in data_folder.glob("*.json"):
        print(f"Loading: {json_file.name}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # CASE 1: already list of docs
        if isinstance(data, list):
            all_documents.extend(data)

        # CASE 2: dict format (your current problem)
        elif isinstance(data, dict):
            for key, value in data.items():

                # if value is list of strings
                if isinstance(value, list):
                    for item in value:
                        all_documents.append({
                            "title": key,
                            "content": str(item),
                            "url": "",
                            "type": "json"
                        })

                # if value is string
                else:
                    all_documents.append({
                        "title": key,
                        "content": str(value),
                        "url": "",
                        "type": "json"
                    })

        else:
            print(f"Skipping unsupported format: {json_file.name}")

    return all_documents

def build_chunks(documents: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=runtime_config.chunk_size,
        chunk_overlap=runtime_config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []
    seen_hashes: set[str] = set()

    for doc_index, document in tqdm(enumerate(documents), total=len(documents), desc="Chunking"):
        content = clean_text(document.get("content", ""))

        if len(content) < runtime_config.min_chunk_chars:
            continue

        title = clean_text(document.get("title", ""))
        source = document.get("url", "").strip()
        doc_type = document.get("type", "webpage").strip()
        category = detect_category(source, title, content)

        for chunk_index, chunk in enumerate(splitter.split_text(content)):
            chunk = clean_text(chunk)

            if len(chunk) < runtime_config.min_chunk_chars:
                continue

            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()

            if chunk_hash in seen_hashes:
                continue

            seen_hashes.add(chunk_hash)

            all_ids.append(f"doc_{doc_index}_chunk_{chunk_index}")

            all_chunks.append(chunk)

            all_metadatas.append(
                {
                    "source": source,
                    "title": title[:200],
                    "type": doc_type,
                    "category": category,
                    "chunk_index": chunk_index,
                }
            )

    return all_chunks, all_metadatas, all_ids


def store_embeddings(chunks: list[str], metadatas: list[dict], ids: list[str]) -> None:
    print(f"Loading embedding model: {runtime_config.embedding_model}")

    embedder = SentenceTransformer(runtime_config.embedding_model)

    print("Creating embeddings...")

    embeddings = embedder.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    print(f"Writing Chroma collection to {runtime_config.chroma_path}")

    client = chromadb.PersistentClient(path=runtime_config.chroma_path)

    try:
        client.delete_collection(runtime_config.collection_name)
        print(f"Deleted existing collection: {runtime_config.collection_name}")
    except Exception:
        pass

    collection = client.create_collection(name=runtime_config.collection_name)

    batch_size = 500

    for start in tqdm(range(0, len(chunks), batch_size), desc="Saving"):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


def main() -> None:
    print(f"Loading knowledge folder: {runtime_config.knowledge_json_path}")

    documents = load_documents()

    print(f"Loaded {len(documents)} source documents")

    chunks, metadatas, ids = build_chunks(documents)

    if not chunks:
        raise ValueError("No valid chunks were generated from JSON files.")

    print(f"Prepared {len(chunks)} chunks")

    store_embeddings(chunks, metadatas, ids)

    print("\nKnowledge base build complete")
    print(f"Collection: {runtime_config.collection_name}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding model: {runtime_config.embedding_model}")


if __name__ == "__main__":
    main()