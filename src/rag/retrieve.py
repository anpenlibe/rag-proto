"""Dense retrieval over the Qdrant index (baseline).

Hybrid (dense+BM25) and cross-encoder rerank implement the same `retrieve` contract
later. Returns lightweight `Candidate`s carrying the citation/trace substrate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from .config import Config
from .embed import Embedder
from .index import get_client


@dataclass
class Candidate:
    id: str
    score: float
    url: str
    title: str
    section: str
    heading_path: list[str]
    text: str
    payload: dict = field(repr=False)


class Retriever:
    """Dense retriever. `retriever_mode` is validated here rather than ignored.

    There is no `get_retriever()` factory yet because there is only one retriever; the
    guard exists so `Config(retriever_mode="hybrid")` cannot quietly run dense and stamp
    a config_hash claiming hybrid. When hybrid lands, this becomes the factory.
    """
    mode = "dense"

    def __init__(self, cfg: Config | None = None, client: QdrantClient | None = None,
                 embedder: Embedder | None = None):
        self.cfg = cfg or Config()
        if self.cfg.retriever_mode != self.mode:
            raise NotImplementedError(
                f"retriever_mode={self.cfg.retriever_mode!r} not implemented yet "
                f"(only {self.mode!r} exists)"
            )
        self.client = client or get_client()
        self.embedder = embedder or Embedder(self.cfg)

    def retrieve(self, query: str, k: int | None = None) -> list[Candidate]:
        k = k or self.cfg.top_k
        name = self.cfg.physical_collection
        qv = self.embedder.embed_query(query)
        try:
            pts = self.client.query_points(
                name, query=qv, limit=k, with_payload=True,
            ).points
        except Exception as e:
            # The collection name carries index_hash, so a chunk/embed change with no
            # re-index surfaces here as a missing collection. Say so — the alternative
            # is querying a stale index and silently scoring the wrong thing.
            if not self.client.collection_exists(name):
                raise SystemExit(
                    f"collection {name!r} does not exist. The index for this config has "
                    f"not been built — run `python -m rag.chunk && python -m rag.index`."
                ) from e
            raise
        out = []
        for p in pts:
            pl = p.payload or {}
            out.append(Candidate(
                id=pl["id"], score=p.score, url=pl["url"], title=pl["title"],
                section=pl["section"], heading_path=pl["heading_path"],
                text=pl["text"], payload=pl,
            ))
        return out
