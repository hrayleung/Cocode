# Cocode-Precise

Cocode-Precise is an MCP (Model Context Protocol) server that provides precise code symbol retrieval. Unlike semantic search tools that return approximate matches, Cocode-Precise returns exact, complete source code for functions, classes, and methods.

## Overview

Cocode-Precise solves a specific problem: when you know what symbol you want, you need its complete implementation—not fragments or similar results. This tool indexes your codebase and lets you retrieve exact symbol definitions with their full source code.

### Key features

- **Exact symbol retrieval**: Get complete function/class/method code by name
- **Multi-language support**: Python, Go, JavaScript, TypeScript, Rust
- **Incremental indexing**: Only re-indexes changed files
- **Line-precise locations**: Every result includes exact line numbers
- **File reading**: Read any file or specific line ranges

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.10 or later
- PostgreSQL with the `pgvector` extension
- An API key for one of the following embedding providers:
  - OpenAI (`OPENAI_API_KEY`)
  - Jina (`JINA_API_KEY` with `USE_LATE_CHUNKING=true`)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/L0stInFades/Cocode-Precise.git
cd Cocode-Precise
```

2. Install the package:

```bash
pip install -e ".[dev]"
```

3. Create a `.env` file with your configuration:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Required environment variables

| Variable | Description |
|----------|-------------|
| `COCOINDEX_DATABASE_URL` | PostgreSQL connection string (e.g., `postgresql://localhost:5432/cocode`) |
| `OPENAI_API_KEY` | OpenAI API key (required unless using Jina) |
| `JINA_API_KEY` | Jina API key (optional, requires `USE_LATE_CHUNKING=true`) |

## Usage

### Running the server

```bash
cocode-precise
```

### Configuring with Claude Code

Add the MCP server to your Claude Code configuration:

```bash
claude mcp add -s user cocode-precise -- cocode-precise
```

Or create a `.mcp.json` file in your project root:

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

## MCP tools reference

Cocode-Precise provides five MCP tools:

### get_symbol

Retrieves the complete source code of a function, class, or method by name.

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol_name` | string | Yes | Name of the symbol to retrieve. Supports partial matching. |
| `path` | string | No | Path to the codebase. Defaults to current working directory. |
| `symbol_type` | string | No | Filter by type: `function`, `class`, `method`, or `interface`. |

**Response**

Returns a list of matching symbols, each containing:

- `name`: Symbol name
- `type`: Symbol type (function, class, method, interface)
- `filename`: Relative path to the file
- `lines`: Line range (e.g., `L10-25`)
- `signature`: Function/class signature
- `docstring`: Documentation string (if present)
- `code`: Complete source code

**Example**

```
get_symbol("authenticate_user")
```

### search_symbols

Searches for symbols by keywords in names, signatures, or docstrings.

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Keywords to search for. |
| `path` | string | No | Path to the codebase. Defaults to current working directory. |
| `symbol_type` | string | No | Filter by type: `function`, `class`, `method`, or `interface`. |
| `top_k` | integer | No | Number of results to return. Default: 10, maximum: 50. |

**Response**

Returns a list of matching symbols with signatures and locations. Use `get_symbol` to retrieve full source code.

**Example**

```
search_symbols("authentication", symbol_type="function")
```

### list_symbols

Lists all symbols in a codebase or specific file.

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | No | Path to the codebase. Defaults to current working directory. |
| `filename` | string | No | Filter to a specific file (partial match supported). |
| `symbol_type` | string | No | Filter by type: `function`, `class`, `method`, or `interface`. |
| `limit` | integer | No | Maximum results to return. Default: 50, maximum: 100. |

**Response**

Returns a list of symbols with names, types, filenames, line ranges, and signatures.

**Example**

```
list_symbols(filename="auth.py", symbol_type="function")
```

### read_file

Reads a file or specific line range from the codebase.

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filename` | string | Yes | Relative path to the file. |
| `path` | string | No | Path to the codebase. Defaults to current working directory. |
| `start_line` | integer | No | Starting line number (1-indexed). |
| `end_line` | integer | No | Ending line number (1-indexed, inclusive). |

**Response**

Returns file content with metadata:

- `filename`: File path
- `total_lines`: Total number of lines in the file
- `lines`: Line range (if specified)
- `content`: File content

**Limits**

- Maximum file size: 1 MB
- For larger files, use `start_line` and `end_line` to read portions

**Example**

```
read_file("src/auth/handler.py", start_line=50, end_line=100)
```

### clear_index

Clears the index for a codebase to force re-indexing on the next operation.

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | No | Path to the codebase. Defaults to current working directory. |

**Response**

Returns a status message confirming the index was cleared.

## Supported languages

Symbol extraction is available for the following languages:

| Language | File extensions | Symbol types |
|----------|-----------------|--------------|
| Python | `.py` | function, class, method |
| Go | `.go` | function, method, interface |
| JavaScript | `.js`, `.mjs`, `.cjs`, `.jsx` | function, class, method |
| TypeScript | `.ts`, `.mts`, `.cts`, `.tsx` | function, class, method, interface |
| Rust | `.rs` | function, method, class (struct/enum), interface (trait) |

## Architecture

Cocode-Precise uses the following components:

1. **CocoIndex**: Handles incremental file indexing and change detection
2. **Tree-sitter**: Parses source code into ASTs for accurate symbol extraction
3. **PostgreSQL + pgvector**: Stores symbols and embeddings
4. **FastMCP**: Provides the MCP server interface

### How indexing works

1. On first use, Cocode-Precise scans the codebase and extracts symbols using Tree-sitter
2. Symbols are stored in PostgreSQL with their locations and embeddings
3. On subsequent uses, only modified files are re-indexed
4. The index persists across sessions

## Troubleshooting

### "No symbols found" for a file

- Verify the file extension is supported (see [Supported languages](#supported-languages))
- Check that the file contains valid syntax
- Run `clear_index` and retry

### Database connection errors

- Verify PostgreSQL is running
- Check that `COCOINDEX_DATABASE_URL` is correct
- Ensure the `pgvector` extension is installed

### Slow initial indexing

- Initial indexing processes all files and may take several minutes for large codebases
- Subsequent operations use incremental indexing and are faster

## Security considerations

- File paths are validated to prevent directory traversal attacks
- SQL queries use parameterized statements to prevent injection
- LIKE pattern special characters are escaped
- File size is limited to 1 MB to prevent memory exhaustion

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Related projects

- [Cocode](https://github.com/hrayleung/Cocode): Semantic codebase search (the project this is based on)
- [CocoIndex](https://cocoindex.io/): Data transformation framework for AI applications
