"""Build / load the Qdrant index over `data/chunks.jsonl`.

Local persistent path (parity upgrade to a Docker server is a one-line client change).
Chunk `id` strings are mapped to deterministic UUIDs so re-indexing is idempotent; the
original id stays in the payload. Full chunk record is stored as payload so retrieval
returns citation + trace substrate in one shot.

Run: `python -m rag.index`
"""
from __future__ import annotations

import json
import uuid

from qdrant_client import QdrantClient, models

from .config import CHUNKS_JSONL, CHUNKS_META, QDRANT_PATH, Config
from .embed import Embedder

_NS = uuid.NAMESPACE_URL


def point_uuid(chunk_id: str) -> str:
    """Stable UUID for a chunk id string (Qdrant needs int/UUID ids)."""
    return str(uuid.uuid5(_NS, chunk_id))


def get_client(cfg: Config | None = None) -> QdrantClient:
    """Open Qdrant — server if `qdrant_url` is set, else the local persistent path.

    The local path takes an EXCLUSIVE file lock: one process at a time, so a frontend
    holding it locks out every index/harness run. The server has no such limit and is
    also closer to the company's stack — set QDRANT_URL to switch (see docker-compose.yml).
    """
    cfg = cfg or Config()
    if cfg.qdrant_url:
        return QdrantClient(url=cfg.qdrant_url)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def backend(cfg: Config) -> str:
    return cfg.qdrant_url or f"local:{QDRANT_PATH.name}"


def assert_chunks_current(cfg: Config) -> dict:
    """Refuse to index chunks that a *different* config produced.

    The collection name carries `index_hash`, which catches a forgotten re-index. It
    cannot catch the opposite: edit `target_words`, skip `rag.chunk`, and re-index, and
    the stale 230-word chunks would land in a correctly-named collection — right name,
    wrong contents, and the eval would score it while the ledger claimed the new hash.
    Only the sidecar closes that edge.
    """
    try:
        meta = json.loads(CHUNKS_META.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f"no {CHUNKS_META.name} beside chunks.jsonl — run `python -m rag.chunk` "
            f"(it is written last, so a missing one may mean chunking died mid-write)"
        ) from None
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read {CHUNKS_META}: {e}") from None

    if meta.get("index_hash") != cfg.index_hash:
        raise SystemExit(
            f"chunks.jsonl was built with index_hash={meta.get('index_hash')} but the "
            f"config says index_hash={cfg.index_hash}.\nThe chunks on disk do not match "
            f"this config — run `python -m rag.chunk` before indexing."
        )
    return meta


def build_index(cfg: Config | None = None, batch: int = 256) -> int:
    cfg = cfg or Config()
    meta = assert_chunks_current(cfg)
    chunks = [json.loads(l) for l in CHUNKS_JSONL.open(encoding="utf-8") if l.strip()]
    if not chunks:
        raise SystemExit("no chunks — run `python -m rag.chunk` first")
    if len(chunks) != meta.get("n_chunks"):
        raise SystemExit(
            f"chunks.jsonl holds {len(chunks)} chunks but {CHUNKS_META.name} attests "
            f"{meta.get('n_chunks')} — the file looks truncated. Re-run `python -m rag.chunk`."
        )

    embedder = Embedder(cfg)
    client = get_client(cfg)
    name = cfg.physical_collection

    dim = len(embedder.embed([chunks[0]["embed_text"]])[0])
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    for field in ("section", "language"):
        client.create_payload_index(
            name, field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vecs = embedder.embed([c["embed_text"] for c in part])
        client.upsert(name, points=[
            models.PointStruct(id=point_uuid(c["id"]), vector=v, payload=c)
            for c, v in zip(part, vecs)
        ])

    count = client.count(name).count
    print(f"index_hash={cfg.index_hash}  collection={name}  dim={dim}")
    print(f"backend: {backend(cfg)}")
    print(f"indexed points: {count}  (chunks: {len(chunks)})")
    return count


if __name__ == "__main__":
    build_index()
