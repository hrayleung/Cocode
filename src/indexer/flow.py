"""CocoIndex flow definitions for codebase indexing."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cocoindex
from cocoindex.llm import LlmApiType
import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


def _normalize_cocoindex_globset_patterns(patterns: list[str]) -> list[str]:
    """Normalize patterns to the globset syntax used by CocoIndex sources.

    Notes:
    - CocoIndex uses Rust's globset (not gitignore).
    - Globset patterns match the *full path*, so gitignore patterns without slashes need a "**/" prefix
      to match at any directory depth.
    - Negations ("!") are ignored.
    - Leading slashes are stripped (CocoIndex paths are relative).
    """

    out: list[str] = []
    for raw in patterns:
        p = (raw or "").strip()
        if not p or p.startswith("#"):
            continue
        if p.startswith("!"):
            # globset doesn't support gitignore negation; ignore these.
            logger.warning(
                "Ignoring gitignore negation pattern %r because CocoIndex globset doesn't support '!'",
                p,
            )
            continue

        anchored = p.startswith("/")
        if anchored:
            p = p[1:]

        is_dir = p.endswith("/")
        body = p.rstrip("/")

        # Gitignore patterns without slashes match at any depth.
        if ("/" not in body) and not anchored:
            p = f"**/{body}"
        else:
            p = body

        # Directory patterns should match everything under the directory.
        if is_dir:
            p = p + "/**"

        out.append(p)
    return out


def _read_gitignore_patterns(repo_path: str) -> list[str]:
    gitignore_path = Path(repo_path) / ".gitignore"
    if not gitignore_path.exists():
        return []
    try:
        return gitignore_path.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as e:
        logger.debug(f"Could not read .gitignore: {e}")
        return []


class JinaEmbedSpec(cocoindex.op.FunctionSpec):
    """Spec for Jina embedding function."""

    model: str = "jina-embeddings-v3"
    dimensions: int = 1024
    task: str = "retrieval.passage"
    late_chunking: bool = True


@cocoindex.op.executor_class(cache=True, behavior_version=1)
class JinaEmbedExecutor:
    """Executor for Jina embeddings with late chunking.

    Late chunking preserves cross-chunk context by embedding the full
    document first, then extracting individual chunk embeddings.
    """
    spec: JinaEmbedSpec
    client: Any

    def prepare(self) -> None:
        """Initialize the HTTP client."""
        self.client = httpx.Client(timeout=120.0)
        self.api_key = settings.jina_api_key
        self.api_url = "https://api.jina.ai/v1/embeddings"

    def __call__(self, text: str) -> cocoindex.Vector[cocoindex.Float32, Literal[1024]]:
        """Embed a single text chunk.

        Returns a 1024-dimensional vector for pgvector storage.
        """
        response = self.client.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.spec.model,
                "input": [text],
                "task": self.spec.task,
                "dimensions": self.spec.dimensions,
                "normalized": True,
                "late_chunking": self.spec.late_chunking,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


class MistralEmbedSpec(cocoindex.op.FunctionSpec):
    """Spec for Mistral Codestral Embed function."""

    model: str = "codestral-embed"


@cocoindex.op.executor_class(cache=True, behavior_version=1)
class MistralEmbedExecutor:
    """Executor for Mistral Codestral Embed."""
    spec: MistralEmbedSpec
    client: Any

    def prepare(self) -> None:
        """Initialize the HTTP client."""
        self.client = httpx.Client(timeout=120.0)
        self.api_key = settings.mistral_api_key
        self.api_url = "https://api.mistral.ai/v1/embeddings"

    def __call__(self, text: str) -> cocoindex.Vector[cocoindex.Float32, Literal[1024]]:
        """Embed a single text chunk."""
        response = self.client.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.spec.model,
                "input": [text],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"][:1024]  # Truncate to 1024 dims


# Map file extensions to tree-sitter language names
EXTENSION_TO_LANGUAGE = {
    # Code files
    ".py": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    # Documentation
    ".md": "markdown",
    ".mdx": "markdown",
}

@dataclass
class CachedFlow:
    """Cached CocoIndex flow with its defining inputs."""

    flow: cocoindex.Flow
    repo_path: str
    included_patterns: tuple[str, ...]
    excluded_patterns: tuple[str, ...]


# Cache for open flows to avoid "already exists" errors.
# If the repo_path/patterns change, we close + recreate the flow.
_flow_cache: dict[str, CachedFlow] = {}


@cocoindex.op.function(behavior_version=1)
def detect_language(filename: str) -> str:
    """Detect programming language from filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "text")


@cocoindex.op.function(behavior_version=1)
def range_to_str(location: cocoindex.Range) -> str:
    """Convert Range to string."""
    return str(location) if location else ""


@cocoindex.op.function(behavior_version=1)
def add_context_header(filename: str, language: str, location: str, content: str) -> str:
    """Add contextual header to chunk for better embedding quality.

    Following Anthropic's Contextual Retrieval approach - prepending
    file/scope context improves retrieval precision by 49-67%.
    """
    header = f"# File: {filename}\n# Language: {language}\n# Location: {location}\n\n"
    return header + content


@cocoindex.transform_flow()
def text_to_embedding_openai(text: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[list[float]]:
    """Transform text to embeddings using OpenAI."""
    return text.transform(
        cocoindex.functions.EmbedText(
            api_type=LlmApiType.OPENAI,
            model=settings.embedding_model,
            output_dimension=settings.embedding_dimensions,
        )
    )


@cocoindex.transform_flow()
def text_to_embedding_jina(
    text: cocoindex.DataSlice[str]
) -> cocoindex.DataSlice[cocoindex.Vector[cocoindex.Float32, Literal[1024]]]:
    """Transform text to embeddings using Jina with late chunking.

    Late chunking preserves cross-chunk context for ~24% better retrieval.
    Returns 1024-dimensional vectors for pgvector storage.
    """
    return text.transform(
        JinaEmbedSpec(
            model=settings.jina_model,
            dimensions=settings.embedding_dimensions,
            task="retrieval.passage",
            late_chunking=True,
        )
    )


@cocoindex.transform_flow()
def text_to_embedding_mistral(
    text: cocoindex.DataSlice[str]
) -> cocoindex.DataSlice[cocoindex.Vector[cocoindex.Float32, Literal[1024]]]:
    """Transform text to embeddings using Mistral Codestral Embed."""
    return text.transform(MistralEmbedSpec(model=settings.mistral_embed_model))


def create_indexing_flow(
    repo_name: str,
    repo_path: str,
    included_patterns: list[str] | None = None,
    excluded_patterns: list[str] | None = None,
):
    """Create or retrieve a CocoIndex flow for indexing a repository.

    Caches flows by name to avoid "Flow already exists" errors.

    Args:
        repo_name: Unique name for the repository
        repo_path: Path to the repository root
        included_patterns: Glob patterns for files to include
        excluded_patterns: Glob patterns for files to exclude

    Returns:
        Configured CocoIndex flow
    """
    flow_name = f"CodeIndex_{repo_name}"

    if included_patterns is None:
        included_patterns = [f"**/*{ext}" for ext in settings.included_extensions]

    gitignore_patterns = _read_gitignore_patterns(repo_path)

    if excluded_patterns is None:
        excluded_patterns = settings.excluded_patterns

    # Combine configured patterns with .gitignore, then normalize for CocoIndex.
    excluded_patterns = _normalize_cocoindex_globset_patterns(
        list(excluded_patterns) + gitignore_patterns
    )

    included_key = tuple(included_patterns)
    excluded_key = tuple(excluded_patterns)

    # Check cache for existing flow with matching configuration
    cached = _flow_cache.get(flow_name)
    if cached:
        config_matches = (
            cached.repo_path == repo_path
            and cached.included_patterns == included_key
            and cached.excluded_patterns == excluded_key
        )
        if config_matches:
            return cached.flow

        # Configuration changed - close old flow before creating new one
        try:
            cached.flow.close()
        except Exception as e:
            logger.debug(f"Failed to close cached flow {flow_name}: {e}")
        _flow_cache.pop(flow_name, None)

    def code_indexing_flow(
        flow_builder: cocoindex.FlowBuilder,
        data_scope: cocoindex.DataScope,
    ):
        # Select embedding function based on configuration
        from src.embeddings.backend import get_selected_provider

        provider = get_selected_provider()
        if provider == "mistral":
            embed_fn = text_to_embedding_mistral
            logger.info("Using Mistral Codestral Embed")
        elif provider == "jina":
            embed_fn = text_to_embedding_jina
            logger.info("Using Jina embeddings with late chunking")
        else:
            embed_fn = text_to_embedding_openai
            logger.info("Using OpenAI embeddings")

        # Source: read files from repository
        data_scope["files"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(
                path=repo_path,
                included_patterns=included_patterns,
                excluded_patterns=excluded_patterns,
            )
        )

        # Collector for embeddings
        code_embeddings = data_scope.add_collector()

        # Process each file
        with data_scope["files"].row() as file:
            # Detect language from filename
            file["language"] = file["filename"].transform(detect_language)

            # Split code into semantic chunks (tree-sitter aware)
            file["chunks"] = file["content"].transform(
                cocoindex.functions.SplitRecursively(),
                language=file["language"],
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            # Process each chunk
            with file["chunks"].row() as chunk:
                # Convert location Range to string for context header
                chunk["location_str"] = chunk["location"].transform(range_to_str)

                # Create contextual text for embedding (Anthropic's approach)
                # Prepend file/language/location context for better retrieval
                chunk["contextual_text"] = flow_builder.transform(
                    add_context_header,
                    file["filename"],
                    file["language"],
                    chunk["location_str"],
                    chunk["text"],
                )

                # Embed the contextual text using selected provider
                # Jina: uses late chunking for ~24% better retrieval
                # OpenAI: standard text-embedding-3-large
                chunk["embedding"] = chunk["contextual_text"].call(embed_fn)

                # Collect: store original content for display, contextual embedding
                code_embeddings.collect(
                    filename=file["filename"],
                    location=chunk["location"],
                    content=chunk["text"],  # Original content for display
                    embedding=chunk["embedding"],  # Contextual embedding
                )

        # Export to PostgreSQL with vector index
        code_embeddings.export(
            f"{repo_name}_chunks",
            cocoindex.targets.Postgres(),
            primary_key_fields=["filename", "location"],
            vector_indexes=[
                cocoindex.VectorIndexDef(
                    field_name="embedding",
                    metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
                )
            ],
        )

    flow = cocoindex.open_flow(flow_name, code_indexing_flow)
    _flow_cache[flow_name] = CachedFlow(
        flow=flow,
        repo_path=repo_path,
        included_patterns=included_key,
        excluded_patterns=excluded_key,
    )
    return flow


def get_cached_flow(repo_name: str):
    """Get a cached flow if it exists.

    Args:
        repo_name: Repository name

    Returns:
        Cached flow object or None if not cached
    """
    flow_name = f"CodeIndex_{repo_name}"
    cached = _flow_cache.get(flow_name)
    return cached.flow if cached else None


def clear_flow_cache(repo_name: str | None = None) -> None:
    """Clear cached flows.

    Args:
        repo_name: If provided, clear only this repo's flow. Otherwise clear all.
    """
    if repo_name:
        flow_name = f"CodeIndex_{repo_name}"
        cached = _flow_cache.pop(flow_name, None)
        if cached:
            try:
                cached.flow.close()
            except Exception as e:
                logger.debug(f"Failed to close flow {flow_name}: {e}")
        return

    for flow_name, cached in list(_flow_cache.items()):
        try:
            cached.flow.close()
        except Exception as e:
            logger.debug(f"Failed to close flow {flow_name}: {e}")
    _flow_cache.clear()
