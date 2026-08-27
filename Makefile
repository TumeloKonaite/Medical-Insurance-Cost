.PHONY: setup run test lint pipeline

setup:
	python -m pip install -r requirements.txt

run:
	uv run uvicorn src.main:app --reload

test:
	uv run --extra dev --extra monitoring pytest

lint:
	uv run --extra dev --extra monitoring ruff check .

pipeline:
	uv run python scripts/run_pipeline.py
