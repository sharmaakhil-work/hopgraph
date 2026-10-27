.PHONY: up down install test lint smoke

up:
	docker compose up -d --wait

down:
	docker compose down

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

# Small, budget-capped end-to-end run on 3 filings
smoke:
	uv run kgrag download --limit 3
	uv run kgrag estimate
	uv run kgrag ingest --max-usd 2
	uv run kgrag resolve
	uv run kgrag index
