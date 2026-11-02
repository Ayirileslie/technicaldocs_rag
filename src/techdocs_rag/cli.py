import argparse
import logging
from pathlib import Path

from .config import Settings


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="techdocs-rag", description="RAG over technical markdown docs")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="index markdown files (incremental)")
    ing.add_argument("--data-dir")
    ing.add_argument("--force", action="store_true", help="re-index unchanged files")
    ing.add_argument("--no-prune", action="store_true", help="keep chunks of deleted files")

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int)

    ev = sub.add_parser("eval", help="run retrieval evaluation")
    ev.add_argument("--questions", default="eval/questions.jsonl")
    ev.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ev.add_argument("-k", type=int, default=5)
    ev.add_argument("--answers", action="store_true", help="also generate answers (uses Groq quota)")
    ev.add_argument("--out", default="eval/results.md")

    srv = sub.add_parser("serve", help="run the API")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()

    if args.cmd == "ingest":
        from .bootstrap import build_embedder, build_store
        from .ingest import ingest

        store = build_store(settings)
        try:
            report = ingest(Path(args.data_dir or settings.data_dir), store, build_embedder(settings),
                            settings.chunk_size, force=args.force, prune=not args.no_prune)
        finally:
            store.close()
        print(report)

    elif args.cmd == "ask":
        from .bootstrap import build_pipeline

        pipeline = build_pipeline(settings)
        try:
            result = pipeline.answer(args.question, k=args.k)
        finally:
            pipeline.close()
        print(result.text)
        print("\nSources:")
        for i, h in enumerate(result.sources, 1):
            print(f"  [{i}] {h.source} > {h.section}")

    elif args.cmd == "eval":
        from .bootstrap import build_pipeline
        from .evaluate import evaluate, load_questions, to_markdown

        pipeline = build_pipeline(settings)
        try:
            qs = load_questions(Path(args.questions))
            rows = [evaluate(pipeline, qs, args.k, a, args.answers) for a in args.alphas]
        finally:
            pipeline.close()
        table = to_markdown(rows, args.k)
        print(table)
        Path(args.out).write_text(table, encoding="utf-8")

    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("techdocs_rag.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()