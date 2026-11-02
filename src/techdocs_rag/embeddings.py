class Embedder:
    """Local sentence-transformers embedder. The model loads lazily on first use."""

    def __init__(self, model_name: str, query_prefix: str = ""):
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self._query_prefix + text])[0]