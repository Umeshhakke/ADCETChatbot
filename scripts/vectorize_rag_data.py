"""
Build embeddings and local vector indexes for the ADCET RAG data.

Inputs:
  - knowledge_base.jsonl    main paragraph chunks; embeds contextualised_text
  - propositions.jsonl      atomic facts; embeds text
  - summary_index.json      summaries; embeds summary

Outputs, by default:
  - vector_db/chroma/       persistent Chroma collections
  - vector_db/faiss/        FAISS cosine-similarity indexes + metadata JSON
  - vector_db/manifest.json build details

Usage:
  python vectorize_rag_data.py
  python vectorize_rag_data.py --query "What is the hostel fee?"
  python vectorize_rag_data.py --backend chroma
  python vectorize_rag_data.py --model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "vector_db"

DATASETS = {
    "main_chunks": {
        "input": "knowledge_base.jsonl",
        "collection": "adcet_main_chunks",
        "text_field": "contextualised_text",
        "file_type": "jsonl",
    },
    "propositions": {
        "input": "propositions.jsonl",
        "collection": "adcet_propositions",
        "text_field": "text",
        "file_type": "jsonl",
    },
    "summaries": {
        "input": "summary_index.json",
        "collection": "adcet_summaries",
        "text_field": "summary",
        "file_type": "json",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def read_records(path: Path, file_type: str) -> list[dict[str, Any]]:
    if file_type == "jsonl":
        return read_jsonl(path)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def stable_id(dataset_name: str, record: dict[str, Any], index: int) -> str:
    for key in ("id", "chunk_id", "parent_chunk_id"):
        value = record.get(key)
        if value:
            return f"{dataset_name}:{value}"
    return f"{dataset_name}:row-{index}"


def flatten_for_metadata(value: Any) -> Any:
    """Chroma accepts only scalar metadata values, so complex values become JSON."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_metadata(dataset_name: str, record: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "dataset": dataset_name,
        "category": record.get("category", ""),
        "source_section": record.get("source_section", ""),
    }

    if dataset_name == "summaries":
        source_meta = record.get("metadata", {})
        metadata.update({
            "parent_id": record.get("parent_id", ""),
            "quality_score": record.get("quality_score", 0),
            "low_quality": record.get("low_quality", False),
            "source_file": source_meta.get("source_file", ""),
            "source_sheet": source_meta.get("source_sheet", ""),
            "source_table": source_meta.get("source_table", ""),
        })
    else:
        source_meta = record.get("metadata", {})
        metadata.update({
            "parent_id": record.get("parent_id") or record.get("parent_chunk_id", ""),
            "college": source_meta.get("college", ""),
            "year": source_meta.get("year", ""),
            "source_file": source_meta.get("source_file", ""),
            "source_sheet": source_meta.get("source_sheet", ""),
            "source_table": source_meta.get("source_table", ""),
            "entities": record.get("entities", {}),
            "keywords": record.get("keywords", []),
        })

    return {key: flatten_for_metadata(value) for key, value in metadata.items()}


def prepare_dataset(input_dir: Path, dataset_name: str) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    spec = DATASETS[dataset_name]
    input_path = input_dir / spec["input"]
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    raw_records = read_records(input_path, spec["file_type"])
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    clean_records: list[dict[str, Any]] = []

    for index, record in enumerate(raw_records):
        text = str(record.get(spec["text_field"], "")).strip()
        if not text:
            continue
        ids.append(stable_id(dataset_name, record, index))
        texts.append(text)
        metadatas.append(build_metadata(dataset_name, record))
        clean_records.append(record)

    if not texts:
        raise ValueError(f"No embeddable records found in {input_path}")
    return ids, texts, metadatas, clean_records


def embed_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32")


def build_faiss_index(
    *,
    dataset_name: str,
    output_dir: Path,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
    vectors: np.ndarray,
) -> None:
    import faiss

    faiss_dir = output_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(faiss_dir / f"{dataset_name}.index"))
    metadata_payload = {
        "dataset": dataset_name,
        "metric": "cosine_similarity_via_normalized_inner_product",
        "dimension": int(vectors.shape[1]),
        "records": [
            {
                "id": ids[i],
                "text": texts[i],
                "metadata": metadatas[i],
            }
            for i in range(len(ids))
        ],
    }
    with (faiss_dir / f"{dataset_name}.metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_chroma_collection(
    *,
    dataset_name: str,
    output_dir: Path,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
    vectors: np.ndarray,
    reset_collection: bool,
) -> None:
    import chromadb

    chroma_dir = output_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection_name = DATASETS[dataset_name]["collection"]

    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=vectors.tolist(),
    )


def search_faiss(
    *,
    output_dir: Path,
    dataset_name: str,
    query_vector: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    import faiss

    faiss_dir = output_dir / "faiss"
    index = faiss.read_index(str(faiss_dir / f"{dataset_name}.index"))
    with (faiss_dir / f"{dataset_name}.metadata.json").open(encoding="utf-8") as f:
        metadata_payload = json.load(f)

    scores, positions = index.search(query_vector, top_k)
    records = metadata_payload["records"]
    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos < 0:
            continue
        record = records[pos]
        results.append({
            "score": float(score),
            "id": record["id"],
            "category": record["metadata"].get("category", ""),
            "source_section": record["metadata"].get("source_section", ""),
            "text": record["text"][:500],
        })
    return results


def write_manifest(
    *,
    output_dir: Path,
    model_name: str,
    backend: str,
    counts: dict[str, int],
    dimension: int,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "backend": backend,
        "datasets": {
            name: {
                "input": DATASETS[name]["input"],
                "collection": DATASETS[name]["collection"],
                "text_field": DATASETS[name]["text_field"],
                "record_count": counts[name],
            }
            for name in counts
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed ADCET RAG data and build local vector indexes.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--backend", choices=["faiss", "chroma", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query", help="Optional smoke-test query after building indexes.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-reset", action="store_true", help="Do not clear existing output directory/collections first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    backend = args.backend

    if output_dir.exists() and not args.no_reset:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {args.model}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)

    counts: dict[str, int] = {}
    dimension = 0

    for dataset_name in tqdm(DATASETS.keys(), desc="Datasets", unit="dataset"):
        ids, texts, metadatas, _records = prepare_dataset(input_dir, dataset_name)
        print(f"\nEmbedding {dataset_name}: {len(texts)} records")
        vectors = embed_texts(model, texts, args.batch_size)
        dimension = int(vectors.shape[1])
        counts[dataset_name] = len(texts)

        if backend in {"faiss", "both"}:
            build_faiss_index(
                dataset_name=dataset_name,
                output_dir=output_dir,
                ids=ids,
                texts=texts,
                metadatas=metadatas,
                vectors=vectors,
            )

        if backend in {"chroma", "both"}:
            build_chroma_collection(
                dataset_name=dataset_name,
                output_dir=output_dir,
                ids=ids,
                texts=texts,
                metadatas=metadatas,
                vectors=vectors,
                reset_collection=not args.no_reset,
            )

    write_manifest(
        output_dir=output_dir,
        model_name=args.model,
        backend=backend,
        counts=counts,
        dimension=dimension,
    )

    print("\nVectorization complete.")
    print(f"Output directory: {output_dir}")
    print(f"Record counts: {counts}")

    if args.query:
        query_vector = embed_texts(model, [args.query], args.batch_size)
        print(f"\nFAISS smoke test for query: {args.query}")
        if backend not in {"faiss", "both"}:
            print("Skipping FAISS smoke test because --backend did not build FAISS.")
            return
        for dataset_name in DATASETS:
            print(f"\n[{dataset_name}]")
            for result in search_faiss(
                output_dir=output_dir,
                dataset_name=dataset_name,
                query_vector=query_vector,
                top_k=args.top_k,
            ):
                print(f"- score={result['score']:.4f} {result['category']} {result['source_section']}")
                print(f"  {result['text']}")


if __name__ == "__main__":
    main()
