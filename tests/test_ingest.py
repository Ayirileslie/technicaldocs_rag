import pytest

from fakes import FakeEmbedder, FakeStore
from techdocs_rag.ingest import ingest


def write(tmp_path, name, body):
    (tmp_path / name).write_text(body, encoding="utf-8")


def test_incremental_ingest_skips_unchanged_and_reindexes_changed(tmp_path):
    write(tmp_path, "a.md", "# A\nhello\n")
    write(tmp_path, "b.md", "# B\nworld\n")
    store, emb = FakeStore(), FakeEmbedder()

    r1 = ingest(tmp_path, store, emb)
    assert (r1.files_indexed, r1.files_skipped) == (2, 0)

    r2 = ingest(tmp_path, store, emb)
    assert (r2.files_indexed, r2.files_skipped) == (0, 2)

    write(tmp_path, "a.md", "# A\nchanged\n")
    r3 = ingest(tmp_path, store, emb)
    assert (r3.files_indexed, r3.files_skipped) == (1, 1)


def test_deleted_files_are_pruned(tmp_path):
    write(tmp_path, "a.md", "# A\nhello\n")
    write(tmp_path, "b.md", "# B\nworld\n")
    store, emb = FakeStore(), FakeEmbedder()
    ingest(tmp_path, store, emb)
    (tmp_path / "b.md").unlink()
    report = ingest(tmp_path, store, emb)
    assert report.files_pruned == 1 and "b.md" in store.deleted


def test_empty_directory_refuses_to_prune(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path, FakeStore(), FakeEmbedder())