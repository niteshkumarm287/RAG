# RAG

Experiments with Retrieval-Augmented Generation. Each version lives in its own folder
and builds on the previous one, moving step by step toward a production-grade system.

## Versions

- [`v1/`](./v1) — minimal single-file RAG: in-memory sentence-transformers embeddings
  + FAISS retrieval + Gemini generation. See [v1/README.md](./v1/README.md).
- [`v2/`](./v2) — persistent RAG: PDF/Markdown ingestion, token-based chunking, a FAISS
  index persisted to disk, and grounded answers. See [v2/README.md](./v2/README.md).
- [`v3/`](./v3) — retrieval threshold, citations, a `RAGPipeline` class, config module,
  dataclasses, and argparse CLIs. Shipped as a guided skeleton to complete.
  See [v3/README.md](./v3/README.md).

## Stack

- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`)
- Vector search: `faiss-cpu`
- LLM: Google Gemini (`gemini-2.5-flash`)
