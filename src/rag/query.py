"""Query transformation — the seam where paraphrase-robustness levers plug in.

Baseline is identity (pass the query through). Multi-query / HyDE expansion implement
the same `transform(q) -> list[str]` contract later without touching the pipeline.
"""
from __future__ import annotations

from .config import Config


class IdentityTransform:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()

    def transform(self, query: str) -> list[str]:
        return [query]


def get_transform(cfg: Config):
    if cfg.query_mode == "identity":
        return IdentityTransform(cfg)
    raise NotImplementedError(f"query_mode={cfg.query_mode!r} not implemented yet")
