"""Query CLI: load the store and answer questions (interactive or one-shot).

Run interactively:   python query.py
Ask one question:    python query.py --question "How are paragraphs separated?"
Tune retrieval:      python query.py -k 5 --max-distance 1.0

Thin entrypoint — the real work is in rag/pipeline.py.
"""
from __future__ import annotations
import argparse
import sys

from rag.config import INDEX_FILE, METADATA_FILE, TOP_K, MAX_DISTANCE
from rag.store import VectorStore
from rag.pipeline import RAGPipeline


def load_pipeline() -> RAGPipeline:
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("❌ Store not found. Run 'python ingest.py' first.")
        sys.exit(1)
    print("💾 Loading store...")
    store = VectorStore.load()
    return RAGPipeline(store)


def print_result(answer: str, hits) -> None:
    print("\n💡 Response:")
    print(answer)
    if hits:
        print("\n📚 Sources referenced:")
        for h in hits:
            # showing the distance helps you *see* the threshold working
            print(f" - {h.chunk.source}  (chunk {h.chunk.chunk_id}, dist={h.distance:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions against the RAG store.")
    parser.add_argument("--question", "-q", type=str, help="Ask one question and exit.")
    parser.add_argument("-k", type=int, default=TOP_K, help="How many chunks to retrieve.")
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE,
                        help="Retrieval threshold; higher = more permissive.")
    args = parser.parse_args()

    pipeline = load_pipeline()

    # One-shot mode
    if args.question:
        answer, hits = pipeline.answer(args.question, k=args.k, max_distance=args.max_distance)
        print_result(answer, hits)
        return

    # Interactive mode
    print("\n🚀 RAG v3 ready. Ctrl-C or type 'exit' to quit.")
    while True:
        try:
            q = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye.")
            break
        if q.lower() == "exit":
            break
        if not q:
            continue
        print("🔍 Searching...")
        answer, hits = pipeline.answer(q, k=args.k, max_distance=args.max_distance)
        print_result(answer, hits)


if __name__ == "__main__":
    main()
