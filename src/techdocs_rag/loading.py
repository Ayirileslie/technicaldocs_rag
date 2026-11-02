"""Find, load and clean markdown files."""
import hashlib
import re
from pathlib import Path


def find_markdown_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_markdown(text: str) -> str:
    """Clean prose only. Fenced code is left untouched so `List<String>` etc. survives."""
    parts = re.split(r"(```.*?```|~~~.*?~~~)", text.replace("\r\n", "\n"), flags=re.DOTALL)
    for i in range(0, len(parts), 2):  # even indexes = prose, odd = code
        p = parts[i]
        p = re.sub(r"\A---\n.*?\n---\n", "", p, flags=re.DOTALL)       # front matter
        p = re.sub(r"<!--.*?-->", "", p, flags=re.DOTALL)               # comments
        p = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", p)                      # images
        p = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", p)                  # [text](url) -> text
        p = re.sub(r"</?(div|span|p|br|img|a|details|summary|center)\b[^>]*>", "", p)
        parts[i] = p
    text = "".join(parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()