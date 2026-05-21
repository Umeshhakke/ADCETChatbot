import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_value: str) -> str:
    path = Path(path_value.strip())
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


@dataclass(frozen=True)
class RuntimeConfig:
    llm_provider: str = os.getenv("ADCET_LLM_PROVIDER", "ollama").strip().lower()
    llm_model: str = os.getenv("ADCET_LLM_MODEL", "qwen2.5:7b").strip()
    llm_temperature: float = float(os.getenv("ADCET_LLM_TEMPERATURE", "0.0"))
    llm_max_tokens: int = int(os.getenv("ADCET_LLM_MAX_TOKENS", "220"))
    llm_top_k: int = int(os.getenv("ADCET_LLM_TOP_K", "40"))
    llm_top_p: float = float(os.getenv("ADCET_LLM_TOP_P", "0.9"))
    llm_repeat_penalty: float = float(os.getenv("ADCET_LLM_REPEAT_PENALTY", "1.1"))
    llm_context_window: int = int(os.getenv("ADCET_LLM_CONTEXT_WINDOW", "8192"))

    gpt4all_model_path: str = os.getenv(
        "ADCET_GPT4ALL_MODEL_PATH",
        "",
    ).strip()
    gpt4all_device: str = os.getenv("ADCET_GPT4ALL_DEVICE", "cpu").strip()
    gpt4all_allow_download: bool = get_bool("ADCET_GPT4ALL_ALLOW_DOWNLOAD", False)

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url: str = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    ).rstrip("/")

    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).rstrip("/")

    embedding_model: str = os.getenv(
        "ADCET_EMBEDDING_MODEL",
        "BAAI/bge-base-en-v1.5",
    ).strip()
    reranker_model: str = os.getenv(
        "ADCET_RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ).strip()
    hf_local_files_only: bool = get_bool("ADCET_HF_LOCAL_FILES_ONLY", False)
    chroma_path: str = resolve_project_path(os.getenv("ADCET_CHROMA_PATH", "./chroma_db"))
    collection_name: str = os.getenv("ADCET_COLLECTION_NAME", "adcet_college").strip()
    knowledge_json_path: str = resolve_project_path(os.getenv("ADCET_KNOWLEDGE_JSON", "./adcet_data.json"))
    knowledge_dir: str = resolve_project_path(os.getenv("ADCET_KNOWLEDGE_DIR", "./data_file"))

    top_k: int = int(os.getenv("ADCET_TOP_K", "5"))
    retrieval_count: int = int(os.getenv("ADCET_RETRIEVAL_COUNT", "14"))
    chunk_size: int = int(os.getenv("ADCET_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("ADCET_CHUNK_OVERLAP", "80"))
    min_chunk_chars: int = int(os.getenv("ADCET_MIN_CHUNK_CHARS", "120"))
    answer_source_count: int = int(os.getenv("ADCET_ANSWER_SOURCE_COUNT", "3"))


runtime_config = RuntimeConfig()
