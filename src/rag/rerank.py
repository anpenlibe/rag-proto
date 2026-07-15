"""Reranking — the seam between the retrieved candidate pool and the selected chunks.

Baseline is a passthrough (keeps vector-search order/score), so the trace still has a
distinct retrieve → rerank → select flow for the frontend. A cross-encoder
(`bge-reranker`) implements the same `rerank(query, candidates) -> candidates`
contract later and simply reorders + rescores.
"""
from __future__ import annotations

from .config import Config
from .retrieve import Candidate


class NoRerank:
    method = "none"
    enabled = False

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        return candidates  # passthrough: vector order/score preserved


def get_reranker(cfg: Config):
    if not cfg.rerank:
        return NoRerank(cfg)
    raise NotImplementedError("cross-encoder rerank not implemented yet")
