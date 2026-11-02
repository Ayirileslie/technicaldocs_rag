"""Central configuration. Every field can be overridden by an UPPER_CASE environment variable."""
import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Settings:
    # services
    weaviate_url: str = ""
    weaviate_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    collection: str = "docChunk"
    # embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    # chunking / retrieval
    data_dir: str = "data/ci"
    chunk_size: int = 1200
    top_k: int = 5
    hybrid_alpha: float = 0.5  # 0 = BM25 only, 1 = vector only
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        values = {}
        for f in fields(cls):
            raw = os.environ.get(f.name.upper(), "").strip()
            if not raw:
                continue
            default_type = type(f.default)
            if default_type is bool:
                values[f.name] = raw.lower() in {"1", "true", "yes"}
            else:
                values[f.name] = default_type(raw)
        return cls(**values)

    def require(self, *names: str) -> None:
        missing = [n.upper() for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(f"Missing required settings: {', '.join(missing)} (see .env.example)")