from .models import Hit


class CrossEncoderReranker:
    """Optional second-stage reranker: scores (question, chunk) pairs jointly."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, question: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(question, h.text) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)[:top_k]
        
        # Replaced `replace(h, score=float(s))` with Pydantic's `model_copy`
        return [h.model_copy(update={"score": float(s)}) for h, s in ranked]
