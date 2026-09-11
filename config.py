"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv("BWC_APP_TITLE", "BWC Sales Intelligence")
    research_provider: str = os.getenv("BWC_RESEARCH_PROVIDER", "live").lower()
    request_timeout: int = int(os.getenv("BWC_REQUEST_TIMEOUT", "12"))
    free_provider_timeout: float = float(os.getenv("BWC_FREE_PROVIDER_TIMEOUT", "3"))
    provider_retry_attempts: int = int(os.getenv("BWC_PROVIDER_RETRY_ATTEMPTS", "1"))
    research_concurrency: int = int(os.getenv("BWC_RESEARCH_CONCURRENCY", "8"))
    research_cache_ttl: int = int(os.getenv("BWC_RESEARCH_CACHE_TTL", "900"))
    research_depth: str = os.getenv("BWC_RESEARCH_DEPTH", "normal").lower()
    brave_api_key: str = os.getenv("BRAVE_SEARCH_API_KEY", "")
    searxng_enabled: bool = os.getenv("SEARXNG_ENABLED", "false").lower() in {"1", "true", "yes"}
    searxng_url: str = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")
    jina_enabled: bool = os.getenv("JINA_ENABLED", "true").lower() in {"1", "true", "yes"}
    jina_reader_url: str = os.getenv("JINA_READER_URL", "https://r.jina.ai").rstrip("/")
    jina_api_key: str = os.getenv("JINA_API_KEY", "")
    direct_reader_enabled: bool = os.getenv("DIRECT_READER_ENABLED", "true").lower() in {"1", "true", "yes"}
    max_research_rounds: int = int(os.getenv("MAX_RESEARCH_ROUNDS", "3"))
    max_search_queries_per_round: int = int(os.getenv("MAX_SEARCH_QUERIES_PER_ROUND", "10"))
    max_results_per_query: int = int(os.getenv("MAX_RESULTS_PER_QUERY", "10"))
    max_pages_to_read: int = int(os.getenv("MAX_PAGES_TO_READ", "15"))
    max_followup_queries: int = int(os.getenv("MAX_FOLLOWUP_QUERIES", "10"))
    research_debug: bool = os.getenv("RESEARCH_DEBUG", "false").lower() in {"1", "true", "yes"}
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    ai_mode: str = os.getenv("BWC_AI_MODE", "none").lower()
    ollama_url: str = os.getenv("BWC_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model: str = os.getenv("BWC_OLLAMA_MODEL", "llama3.2:1b")
    ai_timeout: int = int(os.getenv("BWC_AI_TIMEOUT", "60"))
    ai_max_input_chars: int = int(os.getenv("BWC_AI_MAX_INPUT_CHARS", "30000"))


settings = Settings()