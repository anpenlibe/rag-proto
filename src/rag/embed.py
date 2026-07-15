"""Embedding — the ONLY place the embedding model lives.

bge-small-en-v1.5 is asymmetric for retrieval: the instruction prefix goes on the
QUERY side only, passages are embedded bare. We prefix explicitly (from Config) rather
than relying on library defaults, so the behaviour is visible and controlled.
"""
from __future__ import annotations

import os
from collections.abc import Iterable

from fastembed import TextEmbedding

from .config import Config

# onnxruntime otherwise grabs all cores (pegs a 16-core laptop, ~7GB RAM). Infra knob,
# NOT part of Config/config_hash — it doesn't affect results, only speed/memory.
_EMBED_THREADS = int(os.environ.get("RAG_EMBED_THREADS", "6"))


class Embedder:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._model = TextEmbedding(model_name=self.cfg.embedder_id, threads=_EMBED_THREADS)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed passages (no instruction prefix). In-process (parallel=None) — never
        spawn fastembed's multiprocessing workers (they reload the model per core)."""
        return [v.tolist() for v in self._model.embed(list(texts), parallel=None)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the bge retrieval instruction prepended."""
        prefixed = f"{self.cfg.query_instruction} {text}"
        return next(iter(self._model.embed([prefixed]))).tolist()
