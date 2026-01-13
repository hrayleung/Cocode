# Graph Centrality Scoring Design

**Date:** 2026-01-13
**Status:** Approved
**Goal:** Improve retrieval accuracy by prioritizing structurally important files over peripheral ones (tests, scripts)

## Problem

Current retrieval returns test files over core implementation because:
- Tests have high semantic similarity (descriptive language matches queries)
- System lacks awareness of which files are structurally important
- File category boosting (TEST_WEIGHT=0.3) is insufficient when tests score very high semantically

## Solution

Compute PageRank-style centrality scores based on import graph. Files imported by many others are structurally central; files that only import (tests, scripts) are peripheral.

## Architecture

### Computation (Index Time)

```
Files → Parse Imports → Build Directed Graph → PageRank → Normalize to [0.5, 2.0]
```

- Run after CocoIndex flow completes in `_full_index()` and `_incremental_update()`
- Store in `{repo}_centrality` table
- Full recompute each time (fast for <10k files)

### Application (Search Time)

```
Query → Vector + BM25 → RRF Fusion → Centrality Boost → Category Boost → Rerank
```

- Apply before reranking so central files make it into candidate pool
- Formula: `final_score = base_score × centrality_boost`
- Configurable via `CENTRALITY_WEIGHT` env var

## Implementation

### New Files

**`src/retrieval/centrality.py`**
- `build_import_graph(repo_path, filenames)` - Parse imports, build adjacency list
- `compute_pagerank(graph, damping, iterations)` - Iterative PageRank
- `normalize_scores(scores, min, max)` - Scale to boost range
- `get_centrality_scores(repo_name, filenames)` - DB lookup for search

### Modified Files

| File | Changes |
|------|---------|
| `src/storage/schema.py` | Add centrality table creation |
| `src/indexer/service.py` | Call centrality computation after indexing |
| `src/retrieval/hybrid.py` | Apply centrality boost after RRF |
| `src/retrieval/graph_expansion.py` | Extract shared import parsing |
| `config/settings.py` | Add `CENTRALITY_WEIGHT` setting |

## PageRank Algorithm

```python
def compute_pagerank(graph, damping=0.85, iterations=20):
    files = list(graph.keys())
    n = len(files)
    scores = {f: 1.0 / n for f in files}

    # Build reverse lookup (who imports this file)
    imported_by = defaultdict(list)
    for f, imports in graph.items():
        for imp in imports:
            imported_by[imp].append(f)

    for _ in range(iterations):
        new_scores = {}
        for file in files:
            score = (1 - damping) / n
            for importer in imported_by[file]:
                if graph[importer]:  # avoid division by zero
                    score += damping * scores[importer] / len(graph[importer])
            new_scores[file] = score
        scores = new_scores

    return scores
```

## Error Handling

- Import parsing failures: Treat file as leaf (no imports), log at debug level
- Unsupported languages: Neutral centrality (1.0)
- Missing centrality data: Skip boosting, use base scores
- New files not yet in table: Neutral centrality (1.0)

## Configuration

```bash
# Default: 1.0 (full effect)
# Set to 0 to disable, >1 to amplify
CENTRALITY_WEIGHT=1.0
```

## Expected Impact

| File Type | Imported By | Centrality | Effect |
|-----------|-------------|------------|--------|
| Core modules | Many | ~1.5-2.0 | Boosted |
| Utilities | Some | ~1.0-1.5 | Slight boost |
| Entry points | Few | ~0.7-1.0 | Neutral/slight penalty |
| Tests | None | ~0.5 | Penalized |
