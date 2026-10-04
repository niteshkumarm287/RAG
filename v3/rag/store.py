"""The vector store: wraps a FAISS index + its chunk metadata, with persistence.

This is where you implement the RETRIEVAL THRESHOLD. Read the TODOs carefully;
the search method is the heart of v3.
"""
from __future__ import annotations
import json

import faiss
import numpy as np

from .config import INDEX_FILE, METADATA_FILE, STORE_DIR, MAX_DISTANCE, TOP_K
from .models import Chunk, RetrievedChunk


class VectorStore:
    """Holds a FAISS index and the Chunk metadata that parallels it.

    The FAISS index stores vectors by integer position (0, 1, 2, ...).
    `self.chunks[i]` must correspond to vector `i` in the index. Keep them in sync.
    """

    def __init__(self, index: faiss.Index | None = None, chunks: list[Chunk] | None = None):
        self.index = index
        self.chunks: list[Chunk] = chunks or []

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: list[Chunk]) -> "VectorStore":
        """Create a brand-new store from embeddings + their chunks."""
        # TODO:
        #   1. dimension = embeddings.shape[1]
        #   2. index = faiss.IndexFlatL2(dimension)
        #   3. index.add(embeddings)   # embeddings must be float32
        #   4. return cls(index=index, chunks=chunks)
        raise NotImplementedError

    # ---------------------------------------------------------------- persist
    def save(self) -> None:
        """Write the index and metadata to disk."""
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls) -> "VectorStore":
        """Load a previously-saved store from disk."""
        # TODO:
        #   1. index = faiss.read_index(str(INDEX_FILE))
        #   2. read METADATA_FILE as JSON -> list of dicts
        #   3. convert each dict to a Chunk via Chunk.from_dict(...)
        #   4. return cls(index=index, chunks=[...])
        raise NotImplementedError

    # ----------------------------------------------------------------- search
    def search(
        self,
        query_vector: np.ndarray,
        k: int = TOP_K,
        max_distance: float = MAX_DISTANCE,
    ) -> list[RetrievedChunk]:
        """Return the chunks nearest to `query_vector`, filtered by threshold.

        THIS is the v3 upgrade. Steps:
          1. distances, indices = self.index.search(query_vector, k)
             (both come back shaped (1, k); use distances[0] and indices[0])
          2. Loop over zip(distances[0], indices[0]) together.
          3. Skip a hit if idx == -1  (FAISS pads with -1 when fewer than k exist).
          4. APPLY THE THRESHOLD: skip the hit if its distance > max_distance.
             This is what makes off-topic queries return nothing instead of junk.
          5. For the survivors, wrap each as RetrievedChunk(chunk=self.chunks[idx],
             distance=float(dist)) and collect them.
          6. Return the list (it may be empty — that's the point!).
        """
        # TODO: implement per the steps above.
        raise NotImplementedError
