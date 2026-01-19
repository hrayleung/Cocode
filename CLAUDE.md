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

`src/server.py` - FastMCP server exposing three tools:
- `codebase_retrieval(query, path, top_k)` - Main search tool (concise snippets)
- `codebase_retrieval_full(query, path, top_k, max_symbols, max_symbols_per_file, max_code_chars, include_dependencies)` - Key files + dependencies + full symbol implementations
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
Query → get_query_embedding() [selected provider]
         ↓
       [Parallel ThreadPoolExecutor - max_workers=3]
       ├─ vector_search() (pgvector cosine similarity on chunks)
       ├─ bm25_search() (PostgreSQL FTS with ts_rank_cd on chunks)
       └─ symbol_search() (vector + BM25 on functions/classes/methods)
         ↓
       reciprocal_rank_fusion() [weights: vector=0.6, bm25=0.4, symbol=0.7]
         ↓
       apply_centrality_boost() [PageRank-based structural importance]
         ↓
       apply_category_boosting() [implementation=1.0, doc=0.7, config=0.6, test=0.3]
         ↓
       rerank_results() [Cohere rerank-v3.5, if COHERE_API_KEY set]
         ↓
       aggregate_by_file() [group chunks, keep top 3 per file]
         ↓
       expand_with_graph() [adds related files via multi-hop import traversal (up to 3 hops)]
```

Core ranking pipeline: `src/retrieval/hybrid.py` (`hybrid_search`; RRF → centrality → category → rerank)
Symbol search: `src/retrieval/symbol_search.py` (vector + BM25 on symbols)
Post-processing: `src/retrieval/service.py` (aggregate → graph expansion)

### Database Tables

Tables used by cocode (PostgreSQL):

**repos table**:
- Core fields: id, name, path, status
- Timestamps: created_at, last_indexed
- Stats: file_count, chunk_count
- error_message (for tracking failures)

**CocoIndex-managed chunks table** (per-repo, public schema):
- `codeindex_{repo}__{repo}_chunks`
  - Content: filename, location, content, embedding (pgvector)
  - Indexes: HNSW on embedding; BM25 may add `content_tsv` + GIN lazily on first search
- `codeindex_{repo}__cocoindex_tracking` (CocoIndex internal)

**{schema}.symbols table** (per-repo, enabled by default):
- Symbol metadata: symbol_name, symbol_type (function/class/method), signature, docstring
- Location: filename, line_start, line_end (precise line numbers)
- Context: parent_symbol (for methods), visibility (public/private/internal), category (implementation/test/api/config)
- Search indexes:
  - `embedding` (vector) + HNSW index for semantic symbol search
  - `content_tsv` (tsvector) + GIN index for BM25 full-text search on symbols
- Unique constraint: (filename, symbol_name, line_start) for UPSERT support
- Timestamps: created_at, updated_at

**{schema}.edges table** (per-repo, for call graph):
- Stores function call relationships (caller → callee)
- Fields: source_symbol_id, target_symbol_id (nullable), edge_type ('calls', 'implements', etc.)
- Location tracking: source_file, source_line, target_file, target_line
- Confidence scores: 1.0=exact match, 0.7=partial, 0.5=unresolved
- Context: 'loop', 'conditional', 'recursive', etc.
- Indexes on source_symbol_id, target_symbol_id, edge_type, (source_file, target_file)
- CASCADE deletion when symbols are deleted

**{schema}.graph_cache table** (per-repo, for performance):
- Caches pre-computed import graphs to avoid rebuilding on every search
- Fields: filename (PK), imports (JSONB), imported_by (JSONB), symbol_count, edge_count
- Invalidated when files change during incremental indexing
- 30-50% faster graph queries via caching

**{repo}_centrality table** (optional, created on-demand, public schema):
- Per-repo PageRank scores for graph-based boosting

### Embedding Strategies

**Provider Selection** (via `EMBEDDING_PROVIDER` env var):
- `jina` (default): Jina late chunking - full document embeddings with chunk extraction, preserving cross-chunk context (~24% retrieval improvement)
- `mistral`: Codestral Embed - optimized for code embeddings
- `openai`: OpenAI text-embedding-3-large with contextual headers

All providers validate API keys on startup and fall back to OpenAI if validation fails.

**Contextual Headers** (OpenAI): Chunks prepended with:
```
# File: {filename}
# Language: {language}
# Location: {location}
```

### Symbol Extraction and AST Parsing

**Symbol Indexing** (`src/indexer/symbol_indexing.py`):
- Extracts functions, classes, and methods from code files
- Uses Tree-sitter AST parsing for accurate extraction (Python, Go, Rust, C/C++, JavaScript, TypeScript)
- Generates embeddings for symbols (signature + docstring + context)
- Stores with precise line numbers (line_start, line_end) for exact code references
- Enabled by default via `ENABLE_SYMBOL_INDEXING=true`

**AST-based Import Parsing** (`src/parser/ast_parser.py`):
- Replaces regex-based import extraction with Tree-sitter AST parsing
- Handles edge cases: dynamic imports, conditional imports, aliased imports, nested structures
- Falls back to regex if AST parsing unavailable or fails
- Supports 8 languages: Python, Go, Rust, C, C++, JavaScript, TypeScript, TSX

**Symbol Extraction** (`src/parser/symbol_extractor.py`):
- Extracts symbol metadata: name, type, signature, docstring, parent class, visibility
- Detects test symbols (functions starting with `test_`, classes ending with `Test`)
- Categorizes symbols: implementation, test, api, internal
- Line number accuracy: 1-indexed, inclusive (matches IDE conventions)
- Currently implemented for: Python and Go

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
- Multi-hop BFS traversal (up to 3 hops by default) to find transitive dependencies
- Uses AST-based import parsing for accurate dependency detection
- Tracks hop distance for each related file (1-hop, 2-hop, 3-hop)
- Configurable via `MAX_GRAPH_HOPS` (default: 3) and `MAX_GRAPH_RESULTS` (default: 30)
- Handles cycles gracefully (no infinite loops via visited set)
- Bidirectional: follows both imports (files this file uses) and imported_by (files that use this file)
- **Graph caching**: Uses cached import graphs when available (30-50% faster)

### Call Graph Analysis

**Call Extraction** (`src/parser/call_extractor.py`):
- AST-based function call detection for Python and Go
- Context-aware: detects calls in loops, conditionals, try blocks, lambdas
- Recursion detection: marks recursive calls automatically
- Method call handling: distinguishes `foo()` from `obj.method()`
- Chained call support: `obj.method1().method2()`

**Call Resolution** (`src/retrieval/call_graph.py`):
- Resolves function calls to symbols with confidence scores:
  - 1.0 = Exact match (same file or verified import)
  - 0.7 = Partial match (likely but uncertain)
  - 0.5 = Unresolved (external library, dynamic call)
- Stores call edges in edges table with context
- Query APIs: `get_callers()`, `get_callees()`, `trace_call_chain()`
- Note: `src/indexer/call_graph_indexing.py` contains call-edge extraction/indexing logic, but it is not currently wired into the main indexing flow.

## Environment Variables

Required (embeddings):
- `OPENAI_API_KEY` OR (`JINA_API_KEY` with `USE_LATE_CHUNKING=true`)

Embedding Provider (choose one):
- `JINA_API_KEY` - For Jina late chunking embeddings
- `MISTRAL_API_KEY` - For Mistral Codestral Embed
- `EMBEDDING_PROVIDER` - Select provider: `jina`, `mistral`, or `openai` (default: `jina`)
- `USE_LATE_CHUNKING` - Required for Jina selection (default: `true`)

Database:
- `COCOINDEX_DATABASE_URL` - PostgreSQL with pgvector (default: `postgresql://localhost:5432/cocode`)

Optional:
- `COHERE_API_KEY` - Enables reranking with `rerank-v3.5`
- `EMBEDDING_MODEL` - Default: `text-embedding-3-large`
- `CHUNK_SIZE`, `CHUNK_OVERLAP` - Default: 2000/400 (characters)
- `VECTOR_WEIGHT`, `BM25_WEIGHT` - RRF weights (default: 0.6/0.4)
- `CENTRALITY_WEIGHT` - Graph centrality boosting strength (default: 1.0, 0=disabled)
- `IMPLEMENTATION_WEIGHT`, `DOCUMENTATION_WEIGHT`, `TEST_WEIGHT`, `CONFIG_WEIGHT` - Category boosting weights

Symbol Indexing (enabled by default):
- `ENABLE_SYMBOL_INDEXING` - Enable function/class/method indexing (default: true)
- `SYMBOL_WEIGHT` - Weight for symbol search results (default: 0.7)
- `CHUNK_WEIGHT` - Weight for chunk search results (default: 0.3)

Graph Traversal (multi-hop dependency analysis):
- `MAX_GRAPH_HOPS` - Maximum hop distance for traversal (default: 3)
- `MAX_GRAPH_RESULTS` - Maximum related files to return (default: 30)

See `.env.example` for complete configuration options.

## Key Implementation Details

**Incremental Indexing**: `IndexerService.ensure_indexed()` checks file modification times and only re-indexes changed files. Symbol indexing runs alongside chunk indexing.

**Symbol Incremental Updates**: When files change, their symbols are deleted via UNIQUE constraint (filename, symbol_name, line_start), then re-extracted and upserted using `ON CONFLICT DO UPDATE`.

**Tree-sitter Setup**: Language parsers initialized lazily on first use. If Tree-sitter unavailable, falls back to regex-based extraction gracefully.

**Multi-hop Traversal Performance**: BFS traversal with early termination:
- Cycle prevention via visited set
- Result limiting (max 30 files by default)
- Hop distance tracking for context
- Both forward (imports) and reverse (imported_by) edges explored

**Repo Name Collisions**: Multiple repos with the same folder name get unique suffixes (e.g., `myapp_a1b2c3`) using path hashing.

**BM25 Setup**: Full-text search indexes created lazily via `src/retrieval/fts_setup.py` on first search. Symbols table gets its own FTS index.

**Connection Pooling**: PostgreSQL connection pool managed in `src/storage/postgres.py` with automatic cleanup.

**Caching**: CocoIndex flows use behavior-versioned caching - increment version in `src/indexer/flow.py` to bust cache.
