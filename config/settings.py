"""Configuration settings for cocode MCP server."""

import logging
import os
from dataclasses import dataclass, field
from typing import TypeVar

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar('T', int, float)


def _get_numeric_env(key: str, default: T, min_val: T = 0, conv_func=None) -> T:
    """Get numeric value from environment with validation."""
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
    return _get_numeric_env(key, default, min_val, int)


def _get_float_env(key: str, default: float, min_val: float = 0.0) -> float:
    return _get_numeric_env(key, default, min_val, float)


@dataclass
class Settings:
    """Application settings from environment variables."""

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv("COCOINDEX_DATABASE_URL", "postgresql://localhost:5432/cocode")
    )

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
    embedding_dimensions: int = field(default_factory=lambda: _get_int_env("EMBEDDING_DIMENSIONS", 1024, min_val=1))

    # Jina (late chunking)
    jina_api_key: str = field(default_factory=lambda: os.getenv("JINA_API_KEY", ""))
    jina_model: str = field(default_factory=lambda: os.getenv("JINA_MODEL", "jina-embeddings-v3"))
    use_late_chunking: bool = field(default_factory=lambda: os.getenv("USE_LATE_CHUNKING", "true").lower() == "true")

    # Mistral (Codestral Embed)
    mistral_api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))
    mistral_embed_model: str = field(default_factory=lambda: os.getenv("MISTRAL_EMBED_MODEL", "codestral-embed"))

    # Embedding provider selection: "jina", "mistral", or "openai"
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "jina"))

    # Cohere (reranking)
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "rerank-v3.5"))
    enable_reranker: bool = field(default_factory=lambda: os.getenv("ENABLE_RERANKER", "true").lower() == "true")

    # Indexing
    chunk_size: int = field(default_factory=lambda: _get_int_env("CHUNK_SIZE", 2000, min_val=100))
    chunk_overlap: int = field(default_factory=lambda: _get_int_env("CHUNK_OVERLAP", 400, min_val=0))

    # Search
    default_top_k: int = field(default_factory=lambda: _get_int_env("DEFAULT_TOP_K", 10, min_val=1))
    rerank_candidates: int = field(default_factory=lambda: _get_int_env("RERANK_CANDIDATES", 30, min_val=1))

    # BM25
    bm25_k1: float = field(default_factory=lambda: _get_float_env("BM25_K1", 1.2, min_val=0.1))
    bm25_b: float = field(default_factory=lambda: _get_float_env("BM25_B", 0.75, min_val=0.0))

    # Hybrid search weights
    vector_weight: float = field(default_factory=lambda: _get_float_env("VECTOR_WEIGHT", 0.6, min_val=0.0))
    bm25_weight: float = field(default_factory=lambda: _get_float_env("BM25_WEIGHT", 0.4, min_val=0.0))

    # File category weights
    implementation_weight: float = field(default_factory=lambda: _get_float_env("IMPLEMENTATION_WEIGHT", 1.0, min_val=0.0))
    documentation_weight: float = field(default_factory=lambda: _get_float_env("DOCUMENTATION_WEIGHT", 0.7, min_val=0.0))
    test_weight: float = field(default_factory=lambda: _get_float_env("TEST_WEIGHT", 0.3, min_val=0.0))
    config_weight: float = field(default_factory=lambda: _get_float_env("CONFIG_WEIGHT", 0.6, min_val=0.0))

    # Result diversity (MMR)
    diversity_lambda: float = field(default_factory=lambda: _get_float_env("DIVERSITY_LAMBDA", 0.6, min_val=0.0))

    # Centrality boosting
    centrality_weight: float = field(default_factory=lambda: _get_float_env("CENTRALITY_WEIGHT", 1.0, min_val=0.0))

    # Graph traversal
    max_graph_hops: int = field(default_factory=lambda: _get_int_env("MAX_GRAPH_HOPS", 3, min_val=0))
    max_graph_results: int = field(default_factory=lambda: _get_int_env("MAX_GRAPH_RESULTS", 30, min_val=1))

    # Symbol indexing
    enable_symbol_indexing: bool = field(default_factory=lambda: os.getenv("ENABLE_SYMBOL_INDEXING", "true").lower() == "true")
    symbol_weight: float = field(default_factory=lambda: _get_float_env("SYMBOL_WEIGHT", 0.7, min_val=0.0))
    chunk_weight: float = field(default_factory=lambda: _get_float_env("CHUNK_WEIGHT", 0.3, min_val=0.0))

    # File patterns
    included_extensions: list[str] = field(default_factory=lambda: [
        ".py", ".rs", ".ts", ".tsx", ".js", ".jsx",
        ".go", ".java", ".cpp", ".c", ".h", ".hpp",
        ".rb", ".php", ".swift", ".kt", ".scala",
        ".md", ".mdx",
    ])
    excluded_patterns: list[str] = field(default_factory=lambda: [
        # Version control
        "**/.git/**",
        "**/.svn/**",
        "**/.hg/**",
        # Python
        "**/__pycache__/**",
        "**/*.pyc",
        "**/.venv/**",
        "**/venv/**",
        "**/*.egg-info/**",
        "**/.eggs/**",
        "**/site-packages/**",
        "**/dist-packages/**",
        "**/.tox/**",
        "**/.nox/**",
        "**/.pytest_cache/**",
        "**/.mypy_cache/**",
        "**/.ruff_cache/**",
        # JavaScript/Node
        "**/node_modules/**",
        # Build outputs
        "**/target/**",
        "**/dist/**",
        "**/build/**",
        "**/out/**",
        # IDE/Editor
        "**/.idea/**",
        "**/.vscode/**",
        "**/*.swp",
        "**/*.swo",
        "**/*~",
        # Other
        "**/.cache/**",
        "**/*.log",
    ])


settings = Settings()
