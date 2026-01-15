# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cocode is an MCP (Model Context Protocol) server providing semantic code search with hybrid vector + keyword search. Automatic indexing on first use with Reciprocal Rank Fusion (RRF) and optional Cohere reranking.

## Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the MCP server
cocode
python -m src.server

# Run tests
pytest                                        # All tests
pytest -v                                     # Verbose output
pytest tests/test_file_categorizer.py        # Single test file
pytest tests/test_file_categorizer.py::TestFileCategorizerDetection::test_python_test_files -v  # Single test method
pytest -k "test_python"                       # Run tests matching pattern
```

## Architecture

### Entry Point

`src/server.py` - FastMCP server exposing two tools:
- `codebase_retrieval(query, path, top_k)` - Main search tool
- `clear_index(path)` - Delete index for re-indexing

### Core Services (Singletons)

Both services use the singleton pattern via `get_indexer()` and `get_searcher()` factory functions.

**IndexerService** (`src/indexer/service.py`):
- `ensure_indexed(path)` - Transparent first-time/incremental indexing
- `resolve_repo_name(path)` - Handles name collisions via hash suffixes (e.g., `repo_a1b2c3` for duplicate folder names)
- Uses CocoIndex flows (`src/indexer/flow.py`) for chunking and embeddings
- Delegates to `RepoManager` for database operations

**SearchService** (`src/retrieval/service.py`):
- Orchestrates hybrid search pipeline via `search(repo_name, query, top_k)`
- Returns tiered results: top 3 with full code excerpts, remainder as compact references (signature + line numbers)
- Implements adaptive filtering based on score ratios

### Search Pipeline

The search pipeline applies multiple ranking strategies in sequence. **Critical**: Both centrality and category boosting happen BEFORE reranking to ensure proper prioritization.

```
Query → get_query_embedding() [Jina if available, else OpenAI]
         ↓
       [Parallel ThreadPoolExecutor]
       ├─ vector_search() (pgvector cosine similarity)
       └─ bm25_search() (PostgreSQL FTS with ts_rank_cd)
         ↓
       reciprocal_rank_fusion() [weights: vector=0.6, bm25=0.4]
         ↓
       apply_centrality_boost() [PageRank-based structural importance]
         ↓
       apply_category_boosting() [implementation=1.0, doc=0.7, config=0.6, test=0.3]
         ↓
       rerank_results() [Cohere rerank-v3.5, if COHERE_API_KEY set]
         ↓
       aggregate_by_file() [group chunks, keep top 3 per file]
         ↓
       expand_with_graph() [adds related files via import parsing]
```

Core ranking pipeline: `src/retrieval/hybrid.py:200-227` (RRF → centrality → category → rerank)
Post-processing: `src/retrieval/service.py:163-184` (aggregate → graph expansion)

### Database Tables

All tables defined in `src/storage/schema.py`:

**repos table**:
- Core fields: id, name, path, status
- Timestamps: created_at, last_indexed
- Stats: file_count, chunk_count
- error_message (for tracking failures)

**{schema}.chunks table** (per-repo):
- Content: filename, location, content
- Search indexes:
  - `embedding` (vector) + HNSW index for vector similarity
  - `content_tsv` (tsvector) + GIN index for BM25 keyword search
- Timestamps: created_at, updated_at
- Additional index on filename for efficient file lookups

**{repo}_centrality table** (optional, created on-demand):
- Per-repo PageRank scores for graph-based boosting

### Embedding Strategies

**OpenAI** (default): Chunks prepended with contextual header:
```
# File: {filename}
# Language: {language}
# Location: {location}
```

**Jina Late Chunking** (if `JINA_API_KEY` set): Full document embeddings with chunk extraction, preserving cross-chunk context (~24% retrieval improvement over standard chunking).

### Ranking Features

**File Category Weights** (`src/retrieval/file_categorizer.py`):
Applied before reranking to prioritize implementation code:
- `IMPLEMENTATION_WEIGHT=1.0` - Production source code
- `DOCUMENTATION_WEIGHT=0.7` - README, docs files
- `CONFIG_WEIGHT=0.6` - Configuration, JSON, YAML
- `TEST_WEIGHT=0.3` - Test files, specs

**Graph Centrality Boosting** (`src/retrieval/centrality.py`):
Computes PageRank scores from import relationships to identify structurally important files:
- Central files (imported by many) get boosted scores
- Peripheral files (tests, scripts) get penalized
- Configurable via `CENTRALITY_WEIGHT` (1.0=full effect, 0=disabled)
- Scores cached in per-repo `{repo}_centrality` table

**Graph Expansion** (`src/retrieval/graph_expansion.py`):
Adds related files by parsing import statements to include relevant context beyond direct matches.

## Environment Variables

Required (one of):
- `OPENAI_API_KEY` - For OpenAI embeddings
- `JINA_API_KEY` + `USE_LATE_CHUNKING=true` - For Jina late chunking

Database:
- `COCOINDEX_DATABASE_URL` - PostgreSQL with pgvector (default: `postgresql://localhost:5432/cocode`)

Optional:
- `COHERE_API_KEY` - Enables reranking with `rerank-v3.5`
- `EMBEDDING_MODEL` - Default: `text-embedding-3-large`
- `CHUNK_SIZE`, `CHUNK_OVERLAP` - Default: 2000/400 (characters)
- `VECTOR_WEIGHT`, `BM25_WEIGHT` - RRF weights (default: 0.6/0.4)
- `CENTRALITY_WEIGHT` - Graph centrality boosting strength (default: 1.0, 0=disabled)
- `IMPLEMENTATION_WEIGHT`, `DOCUMENTATION_WEIGHT`, `TEST_WEIGHT`, `CONFIG_WEIGHT` - Category boosting weights

See `.env.example` for complete configuration options.

## Key Implementation Details

**Incremental Indexing**: `IndexerService.ensure_indexed()` checks file modification times and only re-indexes changed files.

**Repo Name Collisions**: Multiple repos with the same folder name get unique suffixes (e.g., `myapp_a1b2c3`) using path hashing.

**BM25 Setup**: Full-text search indexes created lazily via `src/retrieval/fts_setup.py` on first search.

**Connection Pooling**: PostgreSQL connection pool managed in `src/storage/postgres.py` with automatic cleanup.

**Caching**: CocoIndex flows use behavior-versioned caching - increment version in `src/indexer/flow.py` to bust cache.
