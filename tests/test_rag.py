from fakes import FakeEmbedder, FakeLLM, FakeStore, hit
from techdocs_rag.config import Settings
from techdocs_rag.rag import NO_RESULTS, RagPipeline, build_prompt


def make(hits, llm=None, **overrides):
    s = Settings(**overrides)
    store, llm = FakeStore(hits), llm or FakeLLM()
    return RagPipeline(s, FakeEmbedder(), store, llm), store, llm


def test_answer_uses_numbered_context_and_returns_sources():
    pipe, _, llm = make([hit("a.md"), hit("b.md", "Config", "Config > set x")])
    ans = pipe.answer("how do I install?")
    assert ans.text == "Use pip [1]."
    assert [h.source for h in ans.sources] == ["a.md", "b.md"]
    _, user_prompt = llm.calls[0]
    assert "[1] a.md" in user_prompt and "[2] b.md" in user_prompt


def test_no_hits_skips_llm():
    pipe, _, llm = make([])
    ans = pipe.answer("anything")
    assert ans.text == NO_RESULTS and ans.sources == [] and llm.calls == []


def test_alpha_and_k_are_passed_to_store():
    pipe, store, _ = make([hit()] * 10, hybrid_alpha=0.5)
    pipe.retrieve("q", k=3, alpha=0.0)
    assert store.last_search == {"limit": 3, "alpha": 0.0}
    pipe.retrieve("q")
    assert store.last_search == {"limit": 5, "alpha": 0.5}


def test_prompt_contains_question():
    assert "Question: why?" in build_prompt("why?", [hit()])