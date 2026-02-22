# Contributing to PRESS

Thank you for your interest in contributing to the PRESS benchmark.

## Ways to Contribute

- **Dataset expansion** — new questions in existing domains, or proposals for new domains
- **Provider support** — add clients for new LLM providers
- **Confidence classifiers** — improve linguistic confidence estimation
- **Bug reports** — open an issue with a minimal reproducer
- **Methodology discussion** — open a discussion for scoring or protocol changes

## Development Setup

```bash
git clone https://github.com/naitikgupta/pressbench.git
cd pressbench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your API keys
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

The project uses `ruff` for linting and formatting:

```bash
ruff check press/ tests/
ruff format press/ tests/
```

## Adding Questions

Questions live in `press/dataset/questions/<domain>.json`. Each entry must have:

```json
{
  "id": "SCI-085",
  "domain": "science",
  "question": "...",
  "answer": "...",
  "difficulty": "medium",
  "source": "..."
}
```

All answers must be **unambiguous and verifiable** — no opinion-dependent or
interpretation-required items.

## Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-improvement`
3. Commit with conventional commits: `feat:`, `fix:`, `data:`, `docs:`
4. Open a PR against `main` with a clear description of the change
