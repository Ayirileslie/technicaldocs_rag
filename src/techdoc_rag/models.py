from pydantic import BaseModel, Field


class Chunk(BaseModel):
    section: str  # heading path, e.g. "Install > Config"
    text: str


class Hit(BaseModel):
    text: str
    source: str
    section: str
    score: float = 0.0


class Answer(BaseModel):
    text: str
    sources: list[Hit]
    timings_ms: dict[str, float] = Field(default_factory=dict)