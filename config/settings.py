"""Configuration settings for cocode MCP server."""

import os
import logging
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


from typing import TypeVar

T = TypeVar('T', int, float)

def _get_numeric_env(key: str, default: T, min_val: T = 0, conv_func=None) -> T:
    """Safely get numeric value from environment variable with validation."""
    if conv_func is None:
        conv_func = type(default)

    value = os.getenv(key, str(default))
    try:
        result = conv_func(value)
        if result < min_val:
            logger.warning(f"{key}={value} below minimum {min_val}, using {min_val}")
            return min_val
        return result
    except ValueError:
        logger.error(f"Invalid {key}={value}, using default {default}")
        return default

def _get_int_env(key: str, default: int, min_val: int = 0) -> int:
    """Safely get integer from environment variable with validation."""
    return _get_numeric_env(key, default, min_val, int)

def _get_float_env(key: str, default: float, min_val: float = 0.0) -> float:
    """Safely get float from environment variable with validation."""
    return _get_numeric_env(key, default, min_val, float)


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "COCOINDEX_DATABASE_URL", "postgresql://localhost:5432/cocode"
        )
    )

    # OpenAI
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    embedding_dimensions: int = field(
        default_factory=lambda: _get_int_env("EMBEDDING_DIMENSIONS", 1024, min_val=1)
    )

    # Jina (for late chunking - preferred if available)
    jina_api_key: str = field(
        default_factory=lambda: os.getenv("JINA_API_KEY", "")
    )
    jina_model: str = field(
        default_factory=lambda: os.getenv("JINA_MODEL", "jina-embeddings-v3")
    )
    use_late_chunking: bool = field(
        default_factory=lambda: os.getenv("USE_LATE_CHUNKING", "true").lower() == "true"
    )

    # Cohere (optional, for reranking)
    cohere_api_key: str = field(
        default_factory=lambda: os.getenv("COHERE_API_KEY", "")
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "rerank-v3.5")
    )

    # Indexing - larger chunks for coherent code blocks
    # Research suggests 32-64 lines (~2000-4000 chars) for best retrieval
    chunk_size: int = field(
        default_factory=lambda: _get_int_env("CHUNK_SIZE", 2000, min_val=100)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _get_int_env("CHUNK_OVERLAP", 400, min_val=0)
    )

    # Search - tighter results
    default_top_k: int = field(
        default_factory=lambda: _get_int_env("DEFAULT_TOP_K", 10, min_val=1)
    )
    rerank_candidates: int = field(
        default_factory=lambda: _get_int_env("RERANK_CANDIDATES", 30, min_val=1)
    )

    # BM25 parameters
    bm25_k1: float = field(
        default_factory=lambda: _get_float_env("BM25_K1", 1.2, min_val=0.1)
    )
    bm25_b: float = field(
        default_factory=lambda: _get_float_env("BM25_B", 0.75, min_val=0.0)
    )

    # Hybrid search weights
    vector_weight: float = field(
        default_factory=lambda: _get_float_env("VECTOR_WEIGHT", 0.6, min_val=0.0)
    )
    bm25_weight: float = field(
        default_factory=lambda: _get_float_env("BM25_WEIGHT", 0.4, min_val=0.0)
    )

    # File category weights for search ranking
    # Implementation files get highest priority, tests/docs deprioritized
    implementation_weight: float = field(
        default_factory=lambda: _get_float_env("IMPLEMENTATION_WEIGHT", 1.0, min_val=0.0)
    )
    documentation_weight: float = field(
        default_factory=lambda: _get_float_env("DOCUMENTATION_WEIGHT", 0.7, min_val=0.0)
    )
    test_weight: float = field(
        default_factory=lambda: _get_float_env("TEST_WEIGHT", 0.3, min_val=0.0)
    )
    config_weight: float = field(
        default_factory=lambda: _get_float_env("CONFIG_WEIGHT", 0.6, min_val=0.0)
    )

    # Result diversity (MMR algorithm)
    diversity_lambda: float = field(
        default_factory=lambda: _get_float_env("DIVERSITY_LAMBDA", 0.6, min_val=0.0)
    )

    # Centrality boosting (graph-based importance)
    # 1.0 = full effect, 0 = disabled, >1 = amplified
    centrality_weight: float = field(
        default_factory=lambda: _get_float_env("CENTRALITY_WEIGHT", 1.0, min_val=0.0)
    )

    # File patterns - code + docs
    included_extensions: list[str] = field(
        default_factory=lambda: [
            ".py", ".rs", ".ts", ".tsx", ".js", ".jsx",
            ".go", ".java", ".cpp", ".c", ".h", ".hpp",
            ".rb", ".php", ".swift", ".kt", ".scala",
            ".md", ".mdx",
        ]
    )
    excluded_patterns: list[str] = field(
        default_factory=lambda: [
            ".*", "__pycache__", "node_modules", "target",
            "dist", "build", ".git", "venv", ".venv"
        ]
    )


settings = Settings()
