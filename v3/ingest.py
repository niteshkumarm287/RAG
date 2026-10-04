"""Ingestion CLI: load documents -> chunk -> embed -> persist FAISS store.

Run:  python ingest.py
      python ingest.py --chunk-size 400 --chunk-overlap 40

This file is intentionally thin — the real logic lives in the `rag` package.
You implement the marked gaps.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from rag.config import DATA_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from rag.chunking import token_chunker
from rag.models import Chunk
from rag.store import VectorStore


def load_and_extract_text(file_path: str) -> str:
    """Extract raw text from a .pdf, .md, or .txt file."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}")
    return text


def run_ingestion(chunk_size: int, chunk_overlap: int) -> None:
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"📁 Created '{DATA_DIR}'. Drop PDF/MD/TXT files there and re-run.")
        return

    chunks: list[Chunk] = []

    # Walk the data dir, extract + chunk each file, and build Chunk objects.
    # TODO:
    #   - for each file under DATA_DIR:
    #       raw = load_and_extract_text(path)
    #       if not raw.strip(): continue
    #       for piece in token_chunker(raw, chunk_size, chunk_overlap):
    #           append Chunk(source=filename, text=piece, chunk_id=<running counter>)
    #   Keep a running integer id so every chunk across all files is unique.
    raise NotImplementedError  # remove once you've filled the loop above

    if not chunks:
        print("❌ No text extracted. Ingestion aborted.")
        return

    print(f"🧱 Total chunks generated: {len(chunks)}")

    # Embed all chunk texts in one batch.
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = np.array(model.encode([c.text for c in chunks])).astype("float32")

    # Build and persist the store.
    store = VectorStore.build(embeddings, chunks)
    store.save()
    print("💾 Success! Store saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    run_ingestion(args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
