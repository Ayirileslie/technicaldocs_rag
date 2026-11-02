from fakes import hit
from techdocs_rag.evaluate import Question, keyword_recall, reciprocal_rank


def test_reciprocal_rank():
    q = Question("q", expected_source="a.md", expected_section="install")
    assert reciprocal_rank([hit("b.md"), hit("a.md", "Install")], q) == 0.5
    assert reciprocal_rank([hit("b.md")], q) == 0.0


def test_section_match_is_case_insensitive_and_optional():
    assert reciprocal_rank([hit("x/a.md", "Foo > Bar")], Question("q", "a.md")) == 1.0
    assert reciprocal_rank([hit("a.md", "Foo")], Question("q", "a.md", "bar")) == 0.0


def test_keyword_recall():
    assert keyword_recall("Run PIP install", ["pip install", "3.9"]) == 0.5
    assert keyword_recall("anything", []) == 1.0