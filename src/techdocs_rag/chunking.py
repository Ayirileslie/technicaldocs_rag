"""Heading- and code-aware chunking, with a recursive splitter as the fallback.

Each chunk is prefixed with its heading path so both the embedding and the LLM keep context.
Fenced code blocks are never split unless a single block exceeds the chunk size.
"""
import re

from .models import Chunk

FENCE = re.compile(r"^\s*(```|~~~)")
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _recursive_split(text: str, separators: list[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep == "":
        return [text[i:i + size] for i in range(0, len(text), size)]
    if sep not in text:
        return _recursive_split(text, rest, size)
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += sep
        if not part:
            continue
        pieces.extend([part] if len(part) <= size else _recursive_split(part, rest, size))
    return pieces


def split_text(text: str, size: int, overlap: int = 0) -> list[str]:
    """Recursive character splitter: biggest natural boundary first, merged back up to `size`."""
    chunks: list[str] = []
    current = ""
    for piece in _recursive_split(text, SEPARATORS, size):
        if len(current) + len(piece) <= size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        tail = current[-overlap:] if overlap else ""
        if " " in tail:
            tail = tail.split(" ", 1)[1]
        current = tail + piece if len(tail) + len(piece) <= size else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) per section. '#' lines inside code fences are not headings."""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))
        buf.clear()

    for line in text.split("\n"):
        if FENCE.match(line):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return sections


def to_blocks(body: str) -> list[str]:
    """Paragraphs, with every fenced code block kept as one atomic block."""
    blocks: list[str] = []
    buf: list[str] = []
    in_code = False
    for line in body.split("\n"):
        if FENCE.match(line):
            if not in_code and buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            buf.append(line)
            in_code = not in_code
            if not in_code:
                blocks.append("\n".join(buf))
                buf = []
        elif in_code:
            buf.append(line)
        elif line.strip() == "":
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def _split_big(block: str, budget: int) -> list[str]:
    """Oversized block: split code by lines and re-wrap each part in fences."""
    if FENCE.match(block):
        lines = block.split("\n")
        closed = len(lines) > 1 and FENCE.match(lines[-1])
        inner = "\n".join(lines[1:-1] if closed else lines[1:])
        return [f"{lines[0]}\n{p}\n```" for p in split_text(inner, budget - 20)]
    return split_text(block, budget)


def chunk_document(text: str, size: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, body in split_sections(text):
        prefix = f"{path}\n\n" if path else ""
        budget = size - len(prefix)
        current = ""
        for block in to_blocks(body):
            pieces = [block] if len(block) <= budget else _split_big(block, budget)
            for piece in pieces:
                if len(current) + len(piece) + 2 <= budget:
                    current = f"{current}\n\n{piece}" if current else piece
                else:
                    if current:
                        chunks.append(Chunk(path, prefix + current))
                    current = piece
        if current:
            chunks.append(Chunk(path, prefix + current))
    return chunks