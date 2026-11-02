"""Wiring. Heavy imports live inside functions so core modules stay import-light and testable."""
from .config import Settings


def build_embedder(s: Settings):
    from .embeddings import Embedder

    return Embedder(s.embed_model, s.query_prefix)


def build_store(s: Settings):
    from .store import WeaviateStore

    return WeaviateStore(s)


def build_pipeline(s: Settings):
    from .llm import GroqLLM
    from .rag import RagPipeline

    reranker = None
    if s.rerank:
        from .rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(s.rerank_model)
    return RagPipeline(s, build_embedder(s), build_store(s), GroqLLM(s.groq_api_key, s.groq_model), reranker)