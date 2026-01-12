# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cocode is an MCP (Model Context Protocol) server that provides semantic code search with hybrid vector + keyword search. It automatically indexes codebases on first use and employs Reciprocal Rank Fusion (RRF) with optional Cohere reranking.

## Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the MCP server
cocode
# or
python -m src.server

# Run tests
pytest
pytest -v
```

## Architecture

### Core Services

**IndexerService** (`src/indexer/service.py`) - Singleton that manages codebase indexing:
- Transparently handles first-time and incremental indexing
- Tracks repositories in PostgreSQL via `RepoManager`
- Uses CocoIndex flow for embeddings and chunking

**SearchService** (`src/retrieval/service.py`) - Main search orchestration:
- Calls `hybrid_search()` combining vector and BM25 results
- Applies file category boosting and graph expansion
- Returns tiered results (few full code excerpts, remaining as compact references)

### Search Pipeline

```
Query → [Parallel]
       ├─ Vector Search (pgvector similarity via pgvector extension)
       └─ BM25 Search (PostgreSQL FTS, auto-detects ParadeDB/Tiger Data)
         ↓
Reciprocal Rank Fusion (RRF) with configurable weights
         ↓
Optional Cohere Reranking (if COHERE_API_KEY set)
         ↓
File Category Boosting (implementation > docs > config > test)
         ↓
Graph Expansion (adds related files via import/export parsing)
```

### Database Schema

- **`repos` table**: Tracked repositories (path, status, file_count, chunk_count)
- Per-repo chunk tables: `{repo_name}_chunks` with `embedding` (vector), `content_tsv` (tsvector)
- HNSW vector index + GIN index for FTS
- Requires pgvector extension and code-aware text search configuration

### Key Configuration Patterns

**Embeddings prepended with context** (Anthropic's Contextual Retrieval approach):
```
# File: {filename}
# Language: {language}
# Location: {location}

{content}
```

**Late Chunking** (Jina): When `JINA_API_KEY` is set, embeddings use late chunking where individual chunk embeddings are extracted from full document embedding, preserving cross-chunk context.

**File Category Weights** (configurable via env vars):
- `IMPLEMENTATION_WEIGHT=1.0` (highest)
- `DOCUMENTATION_WEIGHT=0.7`
- `CONFIG_WEIGHT=0.6`
- `TEST_WEIGHT=0.5` (lowest)

### Module Relationships

```
storage/postgres.py  ── Connection pooling (2-10 connections)
       ↓
storage/schema.py    ── DB tables & migrations
       ↓
indexer/service.py   ── IndexerService, uses CocoIndex flows
       ↓
retrieval/service.py ── SearchService, orchestrates search
       ↓
retrieval/hybrid.py  ── Combines vector_search.py + bm25_search.py via RRF
```

### Supported Languages

Code: `.py`, `.rs`, `.ts`, `.tsx`, `.js`, `.jsx`, `.go`, `.java`, `.cpp`, `.c`, `.h`, `.hpp`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`
Documentation: `.md`, `.mdx` (architecture understanding only)

Import graph expansion supports Python, TypeScript, JavaScript, Go, Rust.

## Environment Variables

Required:
- `COCOINDEX_DATABASE_URL` - PostgreSQL connection (default: `postgresql://localhost:5432/cocode`)
- `OPENAI_API_KEY` - For embeddings

Optional:
- `JINA_API_KEY` - Enables late chunking
- `COHERE_API_KEY` - Enables reranking
- `EMBEDDING_MODEL` - Default: `text-embedding-3-large`
- `CHUNK_SIZE`, `CHUNK_OVERLAP` - Default: 2000/400
- `VECTOR_WEIGHT`, `BM25_WEIGHT` - RRF weights, default both 0.5
- `*_WEIGHT` - File category boost weights

See `.env.example` for complete configuration.