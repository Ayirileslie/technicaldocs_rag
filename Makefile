.PHONY: install test lint ingest eval serve

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

ingest:
	techdocs-rag ingest

eval:
	techdocs-rag eval --alphas 0 0.5 1

serve:
	techdocs-rag serve