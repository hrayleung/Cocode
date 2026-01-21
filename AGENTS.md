# Repository Guidelines

## Project Structure & Module Organization

- `src/`: Python package for the MCP server and search/indexing pipeline.
  - `src/server.py`: FastMCP entry point (installed as the `cocode` CLI).
  - `src/indexer/`, `src/retrieval/`, `src/parser/`, `src/storage/`, `src/embeddings/`: core subsystems.
- `config/`: runtime settings loaded from environment/`.env` (`config/settings.py`).
- `tests/`: pytest suite (files named `tests/test_*.py`).
- `README.md`: usage + architecture notes; `flake.nix` for Nix-based dev setup.

## Build, Test, and Development Commands

```bash
pip install -e ".[dev]"     # editable install + test deps
cocode                      # run the MCP server
python -m src.server         # run via module (equivalent)
pytest                      # run all tests
pytest -k "hybrid" -v        # run a focused subset
```

For Nix users, `nix develop` bootstraps a dev shell (see `README.md`).

## Coding Style & Naming Conventions

- Python 3.10+; 4-space indentation; prefer type hints (`str | None`, `list[str]`).
- Names: `snake_case.py`, `snake_case()`; constants use `UPPER_SNAKE_CASE`.
- Keep imports grouped (stdlib / third-party / local) and avoid heavy side effects at import time (server startup matters).

## Testing Guidelines

- Frameworks: `pytest` + `pytest-asyncio` (for async APIs).
- Add focused unit tests for ranking/parsing/SQL helpers; avoid real API calls and a live Postgres dependency in unit tests.

## Commit & Pull Request Guidelines

- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:` (imperative, present tense).
- PRs should include: a clear description, steps to validate (`pytest ...`), and `README.md` updates when changing env vars, CLI behavior, or config examples.

## Security & Configuration Tips

- Never commit secrets: keep API keys in `.env` and start from `.env.example`.
- Postgres with `pgvector` is required; configure `COCOINDEX_DATABASE_URL` accordingly.

## Agent-Specific Notes

- Keep diffs small and run `pytest` before submitting changes.
