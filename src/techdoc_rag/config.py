from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    #services

    weaviate_url: str = ""
    weaviate_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    collection: str = "DocChunk"

     # embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # chunking / retrieval
    data_dir: str = "data/sample_docs"
    chunk_size: int = 1200
    top_k: int = 5
    hybrid_alpha: float = 0.5
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )