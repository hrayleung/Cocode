# Cocode

**Production-ready MCP server for semantic codebase search**, powered by [CocoIndex](https://cocoindex.io/) for real-time incremental indexing and intelligent ranking.

Search your codebase semantically without memorizing exact function names—find "HTTP error handler" and get all error handling logic, even if it doesn't contain those exact keywords.

## Built on CocoIndex

Cocode leverages [CocoIndex](https://cocoindex.io/), a data transformation framework designed for AI applications. CocoIndex provides:

- **Incremental Processing**: Only re-indexes changed files, not the entire codebase—real-time updates with minimal computation
- **Tree-sitter Integration**: Intelligent code chunking based on syntax structure (not arbitrary line breaks) for syntactically coherent embeddings
- **Automatic Synchronization**: Keeps the vector index in sync with source code changes across long time horizons
- **Production Infrastructure**: Battle-tested for powering code context in AI agents like Claude, Codex, and Gemini CLI

This combination enables fast, accurate semantic search that stays up-to-date as your code evolves.

## Key Features

### Hybrid Search Architecture
- **Vector Similarity**: pgvector cosine similarity for semantic understanding
- **BM25 Keyword Search**: PostgreSQL full-text search for exact term matching
- **Reciprocal Rank Fusion**: Intelligently combines multiple search strategies with configurable weights
- **Symbol-Level Indexing**: Functions, classes, and methods indexed separately with precise line numbers

### Intelligent Ranking
- **Graph-Based Centrality**: PageRank algorithm identifies structurally important files (e.g., core utilities imported by many modules)
- **Category Boosting**: Prioritizes implementation code over tests/docs/config (configurable weights)
- **Multi-Hop Graph Traversal**: Finds related files via import dependencies (up to 3 hops by default)
- **Optional Cohere Reranking**: rerank-v3.5 for improved relevance (when API key provided)

### Multiple Embedding Providers
- **Jina** (default): Late chunking preserves cross-chunk context (~24% retrieval improvement)
- **Mistral**: Codestral Embed optimized specifically for code
- **OpenAI**: text-embedding-3-large with contextual headers

### Real-Time Incremental Indexing
- **File-Level Change Detection**: Only re-indexes modified files based on timestamps
- **Automatic Index Updates**: First search triggers indexing; subsequent searches use cached results
- **Symbol-Level Updates**: UPSERT operations for functions/classes when files change
- **Graph Cache Invalidation**: Import graphs automatically refresh when dependencies change

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL with `pgvector` (`vector`) extension (the server will also attempt to enable `pgcrypto` for UUIDs)
- API key for at least one embedding provider

### Setup

```bash
# Clone and install
git clone <repo-url>
cd cocode
pip install -e ".[dev]"

# Create .env file
cp .env.example .env
# Edit .env with your API keys
```

### Required Environment Variables

```bash
# PostgreSQL (requires pgvector; server also attempts to CREATE EXTENSION pgcrypto)
COCOINDEX_DATABASE_URL=postgresql://localhost:5432/cocode

# Embeddings (choose one)
OPENAI_API_KEY=sk-...          # Required unless using Jina late chunking
JINA_API_KEY=jina_...          # Can run without OpenAI if USE_LATE_CHUNKING=true
MISTRAL_API_KEY=...            # Optional (requires OPENAI_API_KEY as fallback)

# Select provider: jina, mistral, or openai (default: jina)
EMBEDDING_PROVIDER=jina
USE_LATE_CHUNKING=true

# Optional: Cohere reranking
COHERE_API_KEY=...
```

## Usage with Claude Code

### Option 1: Add via CLI (Recommended)

```bash
# Add to your user config (available across all projects)
claude mcp add -s user cocode -- cocode

# Or add to current project only
claude mcp add -s project cocode -- cocode

# Optional: bake required env vars into the MCP entry
claude mcp add -s user \
  -e COCOINDEX_DATABASE_URL=postgresql://localhost:5432/cocode \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  cocode -- cocode
```

### Option 2: Project Configuration

Create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "cocode": {
      "command": "cocode",
      "args": [],
      "env": {
        "COCOINDEX_DATABASE_URL": "postgresql://localhost:5432/cocode",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "JINA_API_KEY": "${JINA_API_KEY}",
        "EMBEDDING_PROVIDER": "jina",
        "USE_LATE_CHUNKING": "true"
      }
    }
  }
}
```

### Option 3: Run Standalone

```bash
cocode
# or
python -m src.server
```

### Verify Installation

In Claude Code:
```
/mcp
```

## Usage with Codex CLI

Codex stores MCP server configs in `~/.codex/config.toml` (you can manage them via `codex mcp ...`).

### Option 1: Add via CLI (Recommended)

```bash
# Add the server (stdio transport)
codex mcp add cocode \
  --env COCOINDEX_DATABASE_URL=postgresql://localhost:5432/cocode \
  --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -- cocode

# Verify
codex mcp list
codex mcp get cocode
```

### Option 2: Edit `~/.codex/config.toml`

```toml
[mcp_servers.cocode]
command = "cocode"
args = []

[mcp_servers.cocode.env]
COCOINDEX_DATABASE_URL = "postgresql://localhost:5432/cocode"
OPENAI_API_KEY = "sk-..."
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `codebase_retrieval(query, path, top_k)` | Semantic search with automatic indexing. Returns relevant code snippets. |
| `clear_index(path)` | Delete index to force re-indexing on next search. |

### Result Format

`codebase_retrieval` returns a list of results:
- Top results include `content` + `locations`
- Lower-ranked results include `reference` + `lines` (a compact signature/preview)
- Some queries may include extra `"reference": "[Related via imports]"` entries from graph expansion

### Example Queries

```
"Where is user authentication implemented?"
"How does the payment processing work?"
"Find all database connection code"
```

## How It Works

### 1. Indexing Phase (Automatic on First Search)

When you search a codebase for the first time, Cocode:

1. **Repository Registration**: Creates metadata entry with path hash to handle duplicate folder names
2. **CocoIndex Processing**:
   - Scans repository for supported file types (`.py`, `.ts`, `.js`, `.go`, `.rs`, etc.)
   - Intelligently chunks code using Tree-sitter AST parsing (respects function/class boundaries)
   - Generates embeddings for each chunk using your selected provider
   - Stores chunks with HNSW vector index for fast similarity search
3. **Symbol Extraction** (parallel):
   - Parses functions, classes, and methods using Tree-sitter
   - Extracts signatures, docstrings, visibility, and categorization
   - Stores with precise line numbers and separate embeddings
4. **Import Graph Construction**:
   - Parses import statements across all files
   - Builds bidirectional dependency graph (imports + imported_by)
   - Computes PageRank centrality scores
   - Caches graph for fast traversal

**Subsequent searches reuse the index**, only re-indexing files that changed (via mtime comparison).

### 2. Search Phase (Real-Time)

When you query the codebase:

1. **Query Embedding**: Convert natural language query to vector using selected provider
2. **Parallel Search Execution** (ThreadPoolExecutor with 3 workers):
   - **Vector Search**: pgvector cosine similarity on chunk embeddings
   - **BM25 Search**: PostgreSQL full-text search with `ts_rank_cd` scoring
   - **Symbol Search**: Vector + BM25 on functions/classes/methods
3. **Reciprocal Rank Fusion**: Combines results with configurable weights (vector=0.6, bm25=0.4, symbol=0.7)
4. **Ranking Pipeline**:
   - **Centrality Boost**: Amplifies scores for structurally important files (PageRank)
   - **Category Boost**: Prioritizes implementation > docs > config > tests
   - **Cohere Rerank** (optional): LLM-based reranking for improved relevance
5. **Post-Processing**:
   - **File Aggregation**: Groups chunks by file, keeps top 3 per file
   - **Graph Expansion**: BFS traversal finds related files via imports (up to 3 hops)
   - **Tiered Results**: Top 3 with full code excerpts, remainder as compact signatures

### 3. Incremental Updates (Transparent)

On subsequent searches, Cocode automatically:

1. **Change Detection**: Compares file modification times against last index time
2. **Selective Re-indexing**: Only processes changed/new/deleted files
3. **Symbol Updates**: UPSERT operations via `ON CONFLICT` constraints
4. **Graph Cache Refresh**: Invalidates affected import relationships
5. **Centrality Recalculation**: Updates PageRank scores if graph topology changed

This ensures your search results always reflect the current state of the codebase with minimal latency.

## Architecture

```
config/
└── settings.py              # Environment-driven configuration

src/
├── server.py                # FastMCP entry point (2 tools)
│
├── indexer/                 # CocoIndex Integration
│   ├── service.py           # IndexerService singleton - orchestrates indexing
│   ├── flow.py              # CocoIndex flows for chunking/embedding
│   ├── repo_manager.py      # Repository metadata and status tracking
│   ├── symbol_indexing.py   # AST-based function/class extraction
│   └── call_graph_indexing.py # (Experimental) call edge extraction
│
├── retrieval/               # Hybrid Search Pipeline
│   ├── service.py           # SearchService singleton - orchestrates search
│   ├── hybrid.py            # RRF fusion + parallel execution
│   ├── vector_search.py     # pgvector cosine similarity
│   ├── bm25_search.py       # PostgreSQL FTS with ts_rank_cd
│   ├── symbol_search.py     # Symbol-level search (vector + BM25)
│   ├── file_categorizer.py  # Category-based score boosting
│   ├── centrality.py        # PageRank computation and boosting
│   ├── graph_expansion.py   # Multi-hop import traversal (BFS)
│   ├── graph_cache.py       # Import graph caching (30-50% faster)
│   ├── fts_setup.py         # Lazy FTS index creation
│   └── reranker.py          # Cohere rerank-v3.5 integration
│
├── parser/                  # AST-Based Code Analysis
│   ├── ast_parser.py        # Tree-sitter import parsing (8 languages)
│   ├── symbol_extractor.py  # Symbol metadata extraction (Python, Go)
│   └── call_extractor.py    # Function call extraction (Python, Go)
│
├── embeddings/              # Multi-Provider Support
│   ├── backend.py           # Provider selection + API key validation
│   ├── provider.py          # Base provider interface (singleton pattern)
│   ├── jina.py              # Jina late chunking implementation
│   ├── mistral.py           # Codestral Embed implementation
│   └── openai.py            # OpenAI embeddings implementation
│
└── storage/                 # Database Layer
    ├── schema.py            # PostgreSQL table definitions
    └── postgres.py          # Connection pooling + query helpers
```

### Search Pipeline

```
Query → Embedding
         ↓
       [Parallel Search - ThreadPoolExecutor]
       ├─ Vector search (pgvector cosine)
       ├─ BM25 search (PostgreSQL FTS)
       └─ Symbol search (functions/classes)
         ↓
       Reciprocal Rank Fusion (weights: vector=0.6, bm25=0.4, symbol=0.7)
         ↓
       Centrality Boost (PageRank)
         ↓
       Category Boost (impl=1.0, docs=0.7, config=0.6, tests=0.3)
         ↓
       Cohere Rerank (optional)
         ↓
       Aggregate by File (top 3 chunks per file)
         ↓
       Graph Expansion (related files via imports)
```

## Configuration

### Embedding Providers

| Variable | Description |
|----------|-------------|
| `EMBEDDING_PROVIDER` | `jina`, `mistral`, or `openai` (default: `jina`) |
| `JINA_API_KEY` | Jina late chunking - preserves cross-chunk context |
| `USE_LATE_CHUNKING` | Must be `true` for Jina selection (default: `true`) |
| `JINA_MODEL` | Jina model name (default: `jina-embeddings-v3`) |
| `MISTRAL_API_KEY` | Codestral Embed - optimized for code |
| `MISTRAL_EMBED_MODEL` | Mistral model name (default: `codestral-embed`) |
| `OPENAI_API_KEY` | OpenAI embeddings (default fallback) |
| `EMBEDDING_MODEL` | OpenAI embedding model (default: `text-embedding-3-large`) |
| `EMBEDDING_DIMENSIONS` | Embedding dimensions (default: `1024`; changing requires re-index) |

### Search Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_WEIGHT` | 0.6 | RRF weight for vector search |
| `BM25_WEIGHT` | 0.4 | RRF weight for BM25 |
| `SYMBOL_WEIGHT` | 0.7 | RRF weight for symbol search |
| `CENTRALITY_WEIGHT` | 1.0 | PageRank boost (0=disabled) |
| `MAX_GRAPH_HOPS` | 3 | Import traversal depth |
| `MAX_GRAPH_RESULTS` | 30 | Max related files |

### Indexing

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 2000 | Characters per chunk |
| `CHUNK_OVERLAP` | 400 | Overlap between chunks |
| `ENABLE_SYMBOL_INDEXING` | true | Index functions/classes |

## Database Schema

### Global Tables (public schema)

**repos** - Repository metadata
- Core: `id`, `name`, `path`, `status` (`pending`/`indexing`/`ready`/`failed`)
- Timestamps: `created_at`, `last_indexed`
- Stats: `file_count`, `chunk_count`
- Errors: `error_message`

**{repo}_centrality** - PageRank scores (created on demand)
- `filename` (PK), `score` (float)
- Updated when import graph topology changes

### Per-Repository Tables (schema = sanitized repo name)

**codeindex_{repo}__{repo}_chunks** - CocoIndex-managed chunk store
- Content: `filename`, `location` (char range), `content`, `embedding` (vector)
- Indexes: HNSW on `embedding` for vector search
- BM25: `content_tsv` (tsvector) + GIN index added lazily on first search
- Managed entirely by CocoIndex (incremental updates, caching)

**codeindex_{repo}__cocoindex_tracking** - CocoIndex internal state
- Tracks file hashes, chunk boundaries, and behavior versions for incremental processing
- Automatically maintained by CocoIndex

**{schema}.symbols** - Functions/classes/methods
- Metadata: `symbol_name`, `symbol_type` (function/class/method), `signature`, `docstring`
- Location: `filename`, `line_start`, `line_end` (1-indexed, inclusive)
- Context: `parent_symbol`, `visibility` (public/private/internal), `category` (implementation/test/api/config)
- Search: `embedding` (vector) + HNSW, `content_tsv` (tsvector) + GIN
- Unique constraint: `(filename, symbol_name, line_start)` for UPSERT support
- Timestamps: `created_at`, `updated_at`

**{schema}.edges** - Call graph relationships (experimental)
- Relationships: `source_symbol_id`, `target_symbol_id`, `edge_type` ('calls', 'implements', etc.)
- Location: `source_file`, `source_line`, `target_file`, `target_line`
- Confidence: `1.0` (exact match), `0.7` (partial), `0.5` (unresolved/external)
- Context: 'loop', 'conditional', 'recursive', etc.
- CASCADE deletion when symbols removed

**{schema}.graph_cache** - Import graph cache
- `filename` (PK), `imports` (JSONB), `imported_by` (JSONB)
- Stats: `symbol_count`, `edge_count`
- Invalidated automatically during incremental indexing
- 30-50% faster graph queries vs. on-the-fly parsing

All table names derived from repository folder name + path hash (to handle duplicates).

## Supported Languages

### AST-Based Features (Tree-sitter)

**Import Parsing + Graph Expansion**: Python, Go, Rust, C, C++, JavaScript, TypeScript, TSX
- Parses import/include statements to build dependency graph
- Handles dynamic imports, conditional imports, aliased imports
- Falls back to regex if Tree-sitter unavailable

**Symbol Extraction**: Python, Go
- Extracts functions, classes, methods with signatures and docstrings
- Detects visibility (public/private/internal)
- Categorizes as implementation/test/api/config
- Precise line number ranges (1-indexed, inclusive)

### Indexed File Types (Chunked + Searchable)

All these extensions are chunked and embedded for semantic + keyword search:

`.py` `.rs` `.ts` `.tsx` `.js` `.jsx` `.go` `.java` `.cpp` `.c` `.h` `.hpp` `.rb` `.php` `.swift` `.kt` `.scala` `.md` `.mdx`

Files not matching these extensions are ignored during indexing.

## Performance Characteristics

### Indexing Speed
- **Initial Index**: ~1000-5000 files/minute (depends on file size, embedding provider latency)
- **Incremental Updates**: Only changed files re-indexed (typically <1s for small changes)
- **Parallel Processing**: CocoIndex uses async/await for concurrent embedding requests

### Search Speed
- **Vector Search**: ~50-200ms for large codebases (pgvector HNSW index)
- **Hybrid Search**: ~100-400ms including RRF, ranking, and graph expansion
- **With Reranking**: +200-500ms (Cohere API call)
- **Graph Cache Hit**: 30-50% faster for import graph queries

### Storage Requirements
- **Embeddings**: ~4KB per chunk (1024 dimensions × 4 bytes)
- **Symbols**: ~1-2KB per function/class
- **Graph Cache**: ~500 bytes per file (JSONB compressed)
- **Example**: 10,000 file codebase ≈ 200-500MB database size

### Scalability
- **Tested**: Up to 100,000+ files per repository
- **PostgreSQL**: Connection pooling prevents resource exhaustion
- **HNSW Index**: Logarithmic search time (O(log n))

## Development

### Running Tests

```bash
# All tests
pytest

# Verbose output with details
pytest -v

# Single test file
pytest tests/test_file_categorizer.py -v

# Single test method
pytest tests/test_file_categorizer.py::TestFileCategorizerDetection::test_python_test_files -v

# Pattern matching
pytest -k "test_python"

# With coverage
pytest --cov=src --cov-report=html
```

### Project Structure Notes

- **Singleton Services**: `IndexerService` and `SearchService` use singleton pattern to prevent duplicate connections
- **CocoIndex Integration**: Chunk storage delegated to CocoIndex; symbol/graph storage managed directly
- **Thread Safety**: Uses locks for concurrent search requests
- **Error Handling**: Graceful fallbacks when search backends fail (e.g., vector-only if BM25 fails)

## Learn More

- **CocoIndex Framework**: [https://cocoindex.io/](https://cocoindex.io/)
- **CocoIndex GitHub**: [https://github.com/cocoindex-io/cocoindex](https://github.com/cocoindex-io/cocoindex)
- **Building Codebase RAG with CocoIndex**: [https://cocoindex.io/blogs/index-code-base-for-rag](https://cocoindex.io/blogs/index-code-base-for-rag)
- **Real-time Codebase Indexing**: [https://cocoindex.io/docs/examples/code_index](https://cocoindex.io/docs/examples/code_index)

## License

MIT

---

**Questions or issues?** Open an issue on GitHub.
