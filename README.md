# Cocode-Precise

MCP server for **precise code symbol retrieval** - get exact function/class implementations, not just snippets.

Built on [Cocode](https://github.com/hrayleung/Cocode) infrastructure with [CocoIndex](https://cocoindex.io/) for real-time incremental indexing.

## Key Difference from Cocode

| Cocode | Cocode-Precise |
|--------|----------------|
| Semantic search returns relevant chunks | **Exact symbol retrieval** returns complete function/class code |
| Good for "find related code" | Good for "give me this specific function" |
| `codebase_retrieval(query)` | `get_symbol(name)`, `search_symbols(query)`, `list_symbols()`, `read_file()` |

## MCP Tools

| Tool | Description |
|------|-------------|
| `get_symbol(name)` | Get complete source code of a function/class/method by name |
| `search_symbols(query)` | Semantic search for symbols, returns signatures and locations |
| `list_symbols()` | List all symbols in codebase or specific file |
| `read_file(filename)` | Read file content, optionally specific line range |
| `clear_index()` | Force re-indexing on next operation |

## Example Usage

```
# Get exact function code
get_symbol("authenticate_user")

# Search for symbols semantically  
search_symbols("error handling middleware")

# List all functions in a file
list_symbols(filename="auth.py", symbol_type="function")

# Read specific lines
read_file("src/server.py", start_line=50, end_line=100)
```

## Installation

```bash
git clone https://github.com/L0stInFades/Cocode-Precise.git
cd Cocode-Precise
pip install -e ".[dev]"
```

### Environment Variables

```bash
COCOINDEX_DATABASE_URL=postgresql://localhost:5432/cocode
OPENAI_API_KEY=sk-...  # or JINA_API_KEY with USE_LATE_CHUNKING=true
```

## Usage with Claude Code

```bash
claude mcp add -s user cocode-precise -- cocode-precise
```

Or in `.mcp.json`:

```json
{
  "mcpServers": {
    "cocode-precise": {
      "command": "cocode-precise",
      "env": {
        "COCOINDEX_DATABASE_URL": "postgresql://localhost:5432/cocode",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

## How It Works

1. **Indexing** (automatic on first use):
   - Scans codebase with Tree-sitter AST parsing
   - Extracts symbols (functions, classes, methods) with precise line numbers
   - Stores in PostgreSQL with embeddings for semantic search

2. **Retrieval**:
   - `get_symbol`: Queries symbol index → reads exact lines from source file
   - `search_symbols`: Vector + BM25 hybrid search on symbol embeddings
   - `list_symbols`: Direct database query with filters
   - `read_file`: Direct file read with optional line range

## License

MIT
