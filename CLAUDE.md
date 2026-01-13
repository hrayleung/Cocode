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
pytest
pytest -v
pytest tests/test_file_categorizer.py -v     # Single test file
pytest tests/test_file_categorizer.py::test_name -v  # Single test
```

## Architecture

### Entry Point

`src/server.py` - FastMCP server exposing two tools:
- `codebase_retrieval(query, path, top_k)` - Main search tool
- `clear_index(path)` - Delete index for re-indexing

### Core Services (Singletons)

**IndexerService** (`src/indexer/service.py`):
- `ensure_indexed()` - Transparent first-time/incremental indexing
- `resolve_repo_name()` - Handles name collisions via hash suffixes
- Uses CocoIndex flows (`src/indexer/flow.py`) for chunking and embeddings

**SearchService** (`src/retrieval/service.py`):
- Orchestrates hybrid search pipeline
- Returns tiered results (top-k full excerpts, rest as compact references)

### Search Pipeline

```
Query → get_query_embedding() [Jina if available, else OpenAI]
         ↓
       [Parallel ThreadPoolExecutor]
       ├─ vector_search() (pgvector cosine similarity)
       └─ bm25_search() (PostgreSQL FTS)
         ↓
       reciprocal_rank_fusion() [weights: vector=0.6, bm25=0.4]
         ↓
       apply_category_boosting() [BEFORE reranking - critical]
         ↓
       rerank_results() [Cohere, if COHERE_API_KEY set]
         ↓
       expand_with_graph() [adds related files via import parsing]
```

### Database Tables

- `repos` - Repository registry (path, status, file_count, chunk_count)
- `codeindex_{repo}__{repo}_chunks` - Per-repo chunk data with:
  - `embedding` (pgvector) + HNSW index
  - `content_tsv` (tsvector) + GIN index

### Embedding Strategies

**OpenAI** (default): Chunks prepended with contextual header:
```
# File: {filename}
# Language: {language}
# Location: {location}
```

**Jina Late Chunking** (if `JINA_API_KEY` set): Full document embeddings with chunk extraction, preserving cross-chunk context (~24% retrieval improvement).

### File Category Weights

Applied before reranking to prioritize implementation code:
- `IMPLEMENTATION_WEIGHT=1.0`
- `DOCUMENTATION_WEIGHT=0.7`
- `CONFIG_WEIGHT=0.6`
- `TEST_WEIGHT=0.3`

## Environment Variables

Required (one of):
- `OPENAI_API_KEY` - For OpenAI embeddings
- `JINA_API_KEY` + `USE_LATE_CHUNKING=true` - For Jina late chunking

Database:
- `COCOINDEX_DATABASE_URL` - PostgreSQL with pgvector (default: `postgresql://localhost:5432/cocode`)

Optional:
- `COHERE_API_KEY` - Enables reranking with `rerank-v3.5`
- `EMBEDDING_MODEL` - Default: `text-embedding-3-large`
- `CHUNK_SIZE`, `CHUNK_OVERLAP` - Default: 2000/400
- `VECTOR_WEIGHT`, `BM25_WEIGHT` - RRF weights (default: 0.6/0.4)

See `.env.example` for complete configuration.