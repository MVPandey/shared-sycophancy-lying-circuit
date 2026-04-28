.PHONY: check lint format typecheck test

check: lint format typecheck test

lint:
	uv run ruff check src/ experiments/ tests/

format:
	uv run ruff format --check src/ experiments/ tests/

typecheck:
	uv run ty check --project . src/ tests/

test:
	uv run pytest tests/
