# Repository Guidelines

## Project Structure & Module Organization
- `gliner2/` holds the Python package. Core entry points live in `gliner2/__init__.py`, with model logic in `gliner2/model.py`, processing in `gliner2/processor.py`, and inference utilities under `gliner2/inference/`.
- `gliner2/training/` contains training and data utilities; `gliner2/api_client.py` handles API access.
- `tests/` contains runnable demo-style tests (`test_*.py`) for entity, relation, and structure extraction.
- `tutorial/` provides feature guides and examples referenced by the README.
- Root docs include `README.md`, `RELEASE.md`, and `arch.md`.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the package in editable mode for local development.
- `pytest tests/test_entity_extraction.py` runs a single extraction demo; these tests load `fastino/gliner2-*` models and may require internet access.
- `pytest tests/` runs the full test suite (all extraction demos).

## Coding Style & Naming Conventions
- Python 3.8+ codebase; follow PEP 8 with 4-space indentation.
- Use `snake_case` for modules and functions, `CamelCase` for classes, and `UPPER_CASE` for constants.
- No formatter/linter configuration is defined in the repo; keep style consistent with existing files.

## Testing Guidelines
- Tests are in `tests/` and follow the `test_*.py` naming convention.
- Current tests are output/demo focused and print results; add assertions when introducing new behavior.
- If your change affects the public API or examples, update relevant `tutorial/*.md` pages.

## Commit & Pull Request Guidelines
- Git history is shallow here and does not show a strict commit message convention; use concise, imperative summaries (e.g., “Add relation schema validation”).
- PRs should include a clear description, rationale, and the exact test commands run.
- Link related issues if available and update docs when API behavior or examples change.

## Configuration & Security Notes
- API access uses `PIONEER_API_KEY` (see `tutorial/7-api.md`). Avoid committing secrets or model artifacts.
