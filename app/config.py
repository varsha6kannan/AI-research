from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load .env from project root so OPENAI_API_KEY and other secrets
# can be stored there instead of the shell environment.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Paths
    datasets_dir: str = "datasets"
    chroma_persist_dir: str = "chroma_db"
    chroma_collection: str = "medical_chunks"

    # Retrieval
    top_k: int = 10
    rerank_top_n: int = 3

    # Models
    medcpt_query_encoder: str = "ncbi/MedCPT-Query-Encoder"
    medcpt_article_encoder: str = "ncbi/MedCPT-Article-Encoder"
    medcpt_cross_encoder: str = "ncbi/MedCPT-Cross-Encoder"

    # LLM / Ollama (OpenAI-compatible)
    # For OpenAI, set OPENAI_MODEL and OPENAI_API_KEY.
    # For Ollama, set:
    #   OPENAI_MODEL=ollama/phi3:latest
    #   OPENAI_BASE_URL=http://localhost:11434/v1
    #   OPENAI_API_KEY=sk-proj-1234  (or any non-empty placeholder)
    openai_model: str = "gpt-4o"
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # Optional outbound webhook
    webhook_url: str | None = None

    # Chunking
    chunk_max_chars: int = 1800
    chunk_min_chars: int = 400
    chunk_similarity_threshold: float = 0.35
    sentence_embed_batch_size: int = 32
    embed_batch_size: int = 32

    # Semantic query disambiguation
    disambiguation_tau_high: float = 0.75
    disambiguation_tau_low: float = 0.55
    disambiguation_tau_diversity: float = 0.5
    intent_index_path: str = "runtime/intent_index.json"
    intent_cluster_k: int = 5
    # Confidence-based disambiguation (retrieval scores)
    low_conf_threshold: float = 0.70
    ambiguity_margin: float = 0.08

    # Guardrails: validate LLM output (and optionally input) before returning.
    guardrails_output_enabled: bool = True
    guardrails_input_enabled: bool = True


settings = Settings()

