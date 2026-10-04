"""Token-based chunking.

This is largely carried over from v2. It's mostly complete so you can focus on
the new concepts (threshold, store, pipeline). Read it and make sure you
understand *why* the overlap step works the way it does.
"""
from __future__ import annotations
import tiktoken

from .config import CHUNK_SIZE, CHUNK_OVERLAP, TOKENIZER_ENCODING


def token_chunker(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split `text` into overlapping chunks measured in tokens."""
    tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)
    tokens = tokenizer.encode(text)

    chunks: list[str] = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_text = tokenizer.decode(tokens[start_idx:end_idx]).strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Step forward, leaving `chunk_overlap` tokens shared with the next chunk.
        start_idx += (chunk_size - chunk_overlap)
        if start_idx >= len(tokens) or chunk_size <= chunk_overlap:
            break

    return chunks
