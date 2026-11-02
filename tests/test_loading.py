from techdocs_rag.loading import clean_markdown, content_hash


def test_code_blocks_are_preserved():
    text = "Intro <b>x</b>\n\n```java\nList<String> xs = new ArrayList<>();\n```\n"
    out = clean_markdown(text)
    assert "List<String> xs = new ArrayList<>();" in out


def test_front_matter_comments_images_and_links_removed():
    text = "---\ntitle: t\n---\nSee [the docs](http://x.io) ![img](a.png) <!-- hidden -->now\n"
    out = clean_markdown(text)
    assert out == "See the docs  now"


def test_hash_changes_with_content():
    assert content_hash("a") != content_hash("b")
    assert content_hash("a") == content_hash("a")