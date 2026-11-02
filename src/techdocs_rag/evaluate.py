"""Retrieval + answer evaluation on a labelled question set (JSONL)."""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .models import Hit


@dataclass
class Question:
    question: str
    expected_source: str
    expected_section: str = ""
    must_contain: list[str] = field(default_factory=list)


def load_questions(path: Path) -> list[Question]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [Question(**r) for r in rows]


def is_match(hit: Hit, q: Question) -> bool:
    return hit.source.endswith(q.expected_source) and q.expected_section.lower() in hit.section.lower()


def reciprocal_rank(hits: list[Hit], q: Question) -> float:
    for rank, h in enumerate(hits, 1):
        if is_match(h, q):
            return 1.0 / rank
    return 0.0


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    low = answer.lower()
    return sum(k.lower() in low for k in keywords) / len(keywords)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def evaluate(pipeline, questions: list[Question], k: int, alpha: float, with_answers: bool = False) -> dict:
    rr, latencies, recalls = [], [], []
    for q in questions:
        t = time.perf_counter()
        hits = pipeline.retrieve(q.question, k=k, alpha=alpha)
        latencies.append((time.perf_counter() - t) * 1000)
        rr.append(reciprocal_rank(hits, q))
        if with_answers:
            recalls.append(keyword_recall(pipeline.answer(q.question, k=k, alpha=alpha).text, q.must_contain))
    n = len(questions)
    return {
        "alpha": alpha,
        f"hit@{k}": sum(r > 0 for r in rr) / n,
        "mrr": sum(rr) / n,
        "answer_kw_recall": (sum(recalls) / n) if with_answers else None,
        "p50_ms": _pct(latencies, 50),
        "p95_ms": _pct(latencies, 95),
        "n": n,
    }


def to_markdown(rows: list[dict], k: int) -> str:
    head = f"| alpha | hit@{k} | MRR | answer kw recall | p50 ms | p95 ms | n |\n|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r['alpha']} | {r[f'hit@{k}']:.2f} | {r['mrr']:.2f} | "
        f"{'-' if r['answer_kw_recall'] is None else format(r['answer_kw_recall'], '.2f')} | "
        f"{r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['n']} |\n"
        for r in rows
    )
    return head + body