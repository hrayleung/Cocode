# Search Relevance Improvements Design

## Goal

Improve search result quality by addressing three issues:
1. Wrong file types returned (tests/docs when expecting implementation)
2. Results not semantically relevant (keyword mismatch vs actual meaning)
3. Missing relevant code (false negatives)

## Architecture

**Enhanced search pipeline:**

```
Query → [Query Expansion] → Enhanced Query
                              ↓
                    [Parallel Vector+BM25]
                              ↓
                    [Category-Aware RRF]  ← different weights per file type
                              ↓
                    [Category Pre-Filter]
                              ↓
                    [Optional Rerank]
                              ↓
                    [Graph Expansion]
```

## Option A: Query Expansion + Hybrid Tuning

### Query Expansion

When user searches "auth", expand to:
```
Original: "auth"
Expanded: "auth OR authenticate OR authentication OR login OR signin"
```

**Component:** `src/retrieval/query_expander.py` (new)

```python
class QueryExpander:
    EXPANSIONS = {
        "auth": ["authenticate", "authentication", "login", "signin"],
        "db": ["database", "sql", "query", "repository"],
        "http": ["request", "response", "api", "endpoint"],
        # ... ~50 common code terms
    }

    def expand(query: str) -> str:
        # 1. Extract individual terms
        # 2. Look up expansions
        # 3. Return expanded query for BM25 (vector uses original)
```

### Category-Aware RRF Weights

| File Type | Vector Weight | BM25 Weight |
|-----------|---------------|-------------|
| Implementation (.py, .ts, etc.) | 0.7 | 0.3 |
| Documentation (.md) | 0.4 | 0.6 |
| Tests (test_*.py) | 0.3 | 0.7 |
| Config (.json, .yaml) | 0.2 | 0.8 |

**Implementation location:** `src/retrieval/hybrid.py` lines 192-195 (RRF fusion)

## Option C: Context-Enhanced Embeddings

### Current chunk context:
```
# File: src/auth/service.py
# Language: python
# Location: 10:50

def login(user):
    ...
```

### Enhanced chunk context:
```
# File: src/auth/service.py
# Language: python
# Location: 10:50
# Function: login
# Class: AuthService
# Imports: from models import User, from database import db
# Symbols: User, authenticate, check_password

def login(user):
    ...
```

**Implementation location:** `src/indexer/flow.py` around lines 291-301

### Symbol Extraction

Uses existing tree-sitter parsers to extract:
- Containing function name
- Containing class name
- Import statements
- Referenced symbols in the chunk

## Files to Modify

| File | Change |
|------|--------|
| `src/indexer/flow.py` | Add symbol extraction, enhance context header |
| `src/retrieval/hybrid.py` | Category-aware RRF weights |
| `src/retrieval/query_expander.py` | **NEW** - Query expansion logic |
| `src/retrieval/bm25_search.py` | Use expanded query |
| `config/settings.py` | Add expansion config, RRF weights per category |

## Error Handling

**Query Expansion:**
- Empty expansion: Return original query unchanged
- Over-expansion: Cap expanded query length
- Special characters: Escape SQL-specific chars

**Symbol Extraction:**
- Parse failures: Fall back to minimal context
- Large files: Cap extraction time, skip for files >1MB
- Binary files: Skip entirely

**Category-Aware RRF:**
- Unknown file types: Default to balanced weights (0.5/0.5)

**Re-indexing:**
- Partial failures: Track progress, resume on next run
- Version migration: Store context version in `repos` table

## Performance Impact

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| Query embedding | ~50ms | ~50ms | No change |
| BM25 search | ~30ms | ~40ms | +10ms |
| RRF fusion | ~5ms | ~5ms | No change |
| **Total search** | ~85ms | ~95ms | +12% |
| Indexing (per file) | ~200ms | ~350ms | +75% (one-time) |

**Memory:** No significant change (~5KB for expansion dict)
**Storage:** ~15% larger chunks (enhanced context)

## Re-indexing Requirement

Yes - context header changes require re-embedding. Detection via context version stored in `repos` table.

## Tech Stack

- OpenAI embeddings (existing)
- Tree-sitter (existing dependency)
- PostgreSQL BM25 (existing)
- No new external dependencies
