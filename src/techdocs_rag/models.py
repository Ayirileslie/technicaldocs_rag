from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    section: str  # heading path, e.g. "Install > Config"
    text: str


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    section: str
    score: float = 0.0


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)