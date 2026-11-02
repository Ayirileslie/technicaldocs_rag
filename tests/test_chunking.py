from techdocs_rag.chunking import chunk_document, split_sections, split_text

DOC = """# Install
Run this:

```bash
# not a heading
pip install foo
```

## Config
Set `x`.

## Errors
Something failed.
"""


def test_heading_inside_code_fence_is_not_a_heading():
    paths = [p for p, _ in split_sections(DOC)]
    assert paths == ["Install", "Install > Config", "Install > Errors"]


def test_chunks_carry_heading_path_prefix():
    chunks = chunk_document(DOC)
    assert all(c.text.startswith(c.section) for c in chunks)


def test_small_code_block_stays_intact():
    chunks = chunk_document(DOC)
    assert any("```bash\n# not a heading\npip install foo\n```" in c.text for c in chunks)


def test_size_limit_respected_and_big_code_is_rewrapped():
    big = "# Big\n\n```python\n" + "\n".join(f"line_{i} = {i}" for i in range(300)) + "\n```\n"
    chunks = chunk_document(big, size=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert all(c.text.count("```") == 2 for c in chunks)  # each part is a valid fenced block


def test_split_text_never_exceeds_size():
    text = ("Sentence about things. " * 200) + ("x" * 900)
    assert all(len(c) <= 300 for c in split_text(text, 300, overlap=50))