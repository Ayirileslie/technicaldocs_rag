"""FastAPI service: POST /ask, GET /health."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .bootstrap import build_pipeline
from .config import Settings

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)


class SourceOut(BaseModel):
    source: str
    section: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    timings_ms: dict[str, float]


def create_app(pipeline_factory=build_pipeline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline_factory(Settings.from_env())
        yield
        app.state.pipeline.close()

    app = FastAPI(title="techdocs-rag", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        try:
            result = request.app.state.pipeline.answer(body.question, k=body.top_k)
        except Exception:
            log.exception("ask failed")
            raise HTTPException(status_code=502, detail="Upstream retrieval or generation failed") from None
        return AskResponse(
            answer=result.text,
            sources=[
                SourceOut(source=h.source, section=h.section, score=h.score, snippet=h.text[:240])
                for h in result.sources
            ],
            timings_ms={k: round(v, 1) for k, v in result.timings_ms.items()},
        )

    return app


app = create_app()