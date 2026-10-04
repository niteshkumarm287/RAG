"""The RAGPipeline: ties the embedder, vector store, and LLM together.

Why a class? In v2 the model, index, and client were module-level globals that
ran on import. A class lets you:
  - load heavy resources ONCE in __init__
  - reuse them across many queries
  - later wrap this same object in a web server (v6) with no changes

You implement `answer()` — the retrieve-then-generate core.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

from .config import EMBED_MODEL, LLM_MODEL, TEMPERATURE, TOP_K, MAX_DISTANCE
from .store import VectorStore
from .models import RetrievedChunk


SYSTEM_PROMPT = """You are a helpful assistant answering questions based strictly \
on the provided context. If the context does not contain the answer, say \
"I don't know." Do not make up answers.

Context:
{context}"""


class RAGPipeline:
    def __init__(self, store: VectorStore):
        self.store = store
        # Heavy resources loaded once and reused.
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.llm = genai.Client()  # reads GEMINI_API_KEY from env

    # ---------------------------------------------------------------- helpers
    def embed_query(self, query: str) -> np.ndarray:
        """Turn a query string into a (1, dim) float32 array for FAISS."""
        # TODO: return np.array(self.embedder.encode([query])).astype("float32")
        raise NotImplementedError

    @staticmethod
    def build_context(hits: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a single context string with sources."""
        # TODO: join each hit as e.g.
        #   f"[Source: {h.chunk.source}]\n{h.chunk.text}"
        # separated by blank lines. Return the combined string.
        raise NotImplementedError

    # ------------------------------------------------------------------ main
    def answer(
        self,
        query: str,
        k: int = TOP_K,
        max_distance: float = MAX_DISTANCE,
    ) -> tuple[str, list[RetrievedChunk]]:
        """Retrieve relevant chunks and generate a grounded answer.

        Steps:
          1. qv = self.embed_query(query)
          2. hits = self.store.search(qv, k=k, max_distance=max_distance)
          3. If hits is empty -> return an honest "I don't know" message and [].
             (This is the threshold paying off: we refuse BEFORE calling the LLM.)
          4. context = self.build_context(hits)
          5. Call the LLM:
                 response = self.llm.models.generate_content(
                     model=LLM_MODEL,
                     contents=query,
                     config=types.GenerateContentConfig(
                         system_instruction=SYSTEM_PROMPT.format(context=context),
                         temperature=TEMPERATURE,
                     ),
                 )
          6. return response.text, hits
        """
        # TODO: implement per the steps above.
        raise NotImplementedError
