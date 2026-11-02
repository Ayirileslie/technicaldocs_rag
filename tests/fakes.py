from techdocs_rag.models import Hit


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(t))] for t in texts]

    def embed_query(self, text):
        return [float(len(text))]


class FakeStore:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.hashes: dict[str, str] = {}
        self.writes: list[str] = []
        self.deleted: list[str] = []
        self.last_search = None

    def get_file_hash(self, source):
        return self.hashes.get(source)

    def replace_source(self, source, file_hash, chunks, vectors):
        assert len(chunks) == len(vectors)
        self.hashes[source] = file_hash
        self.writes.append(source)

    def list_sources(self):
        return set(self.hashes)

    def delete_source(self, source):
        self.hashes.pop(source, None)
        self.deleted.append(source)

    def search(self, query, vector, limit, alpha):
        self.last_search = {"limit": limit, "alpha": alpha}
        return self.hits[:limit]

    def close(self):
        pass


class FakeLLM:
    def __init__(self, reply="Use pip [1]."):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


def hit(source="a.md", section="Install", text="Install > pip install foo", score=1.0):
    return Hit(text=text, source=source, section=section, score=score)