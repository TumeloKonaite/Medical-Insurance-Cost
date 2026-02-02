.PHONY: setup run test lint pipeline

setup:
	python -m pip install -r requirements.txt

run:
	uv run python main.py

test:
	python -m pytest

lint:
	python -m ruff check .

pipeline:
	uv run python scripts/run_pipeline.py
