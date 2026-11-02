from fastapi.testclient import TestClient

from fakes import FakeEmbedder, FakeLLM, FakeStore, hit
from techdocs_rag.api import create_app
from techdocs_rag.config import Settings
from techdocs_rag.rag import RagPipeline


def factory(_settings):
    return RagPipeline(Settings(), FakeEmbedder(), FakeStore([hit()]), FakeLLM())


def test_health_and_ask():
    with TestClient(create_app(factory)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/ask", json={"question": "how to install?", "top_k": 3})
        body = r.json()
        assert r.status_code == 200
        assert body["answer"] == "Use pip [1]."
        assert body["sources"][0]["source"] == "a.md"


def test_validation_rejects_short_question():
    with TestClient(create_app(factory)) as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 422