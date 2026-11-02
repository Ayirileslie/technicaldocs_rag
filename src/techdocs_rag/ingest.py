"""Ingestion pipeline: find -> load -> clean -> chunk -> embed -> store (incremental)."""
import logging
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_document
from .loading import clean_markdown, content_hash, find_markdown_files, load_file

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_pruned: int = 0
    chunks_written: int = 0


def ingest(
    root: Path, store, embedder, chunk_size: int = 1200, force: bool = False, prune: bool = True
) -> IngestReport:
    files = find_markdown_files(root)
    if not files:
        raise FileNotFoundError(f"No .md files under {root} (refusing to prune the index)")

    report = IngestReport(files_seen=len(files))
    current: set[str] = set()
    for path in files:
        source = path.relative_to(root).as_posix()
        current.add(source)
        raw = load_file(path)
        digest = content_hash(raw)
        if not force and store.get_file_hash(source) == digest:
            report.files_skipped += 1  # unchanged since last ingest
            continue
        chunks = chunk_document(clean_markdown(raw), size=chunk_size)
        vectors = embedder.embed_documents([c.text for c in chunks]) if chunks else []
        store.replace_source(source, digest, chunks, vectors)
        report.files_indexed += 1
        report.chunks_written += len(chunks)
        log.info("indexed %s (%d chunks)", source, len(chunks))

    if prune:
        for stale in store.list_sources() - current:
            store.delete_source(stale)
            report.files_pruned += 1
            log.info("pruned %s", stale)
    return report