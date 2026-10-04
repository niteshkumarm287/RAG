"""Structured data types for the pipeline.

Using @dataclass instead of raw dicts gives you:
  - clear field names and types (your editor autocompletes them)
  - a single place that defines "what a chunk is"
  - cheap conversion to/from JSON for persistence

This is a good Python habit to build. Prefer these over passing dicts around.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    """A piece of a source document, ready to be embedded and stored."""
    source: str      # filename the chunk came from
    text: str        # the chunk's actual text
    chunk_id: int    # position of this chunk within the whole corpus

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        # TODO: build and return a Chunk from a dict loaded from metadata.json.
        # Hint: cls(source=d["source"], text=d["text"], chunk_id=d["chunk_id"])
        raise NotImplementedError


@dataclass
class RetrievedChunk:
    """A chunk returned by a search, plus how close it was to the query."""
    chunk: Chunk
    distance: float  # L2 distance from the query vector (smaller = better)
