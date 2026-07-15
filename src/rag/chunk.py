"""Chunk `data/pages.jsonl` into `data/chunks.jsonl` — heading-aware, size-bounded.

Reuses the heading-split idea from `scripts/scrape.py` and adds greedy packing of
small consecutive sections (the corpus's median heading-block is only ~68 words, so
one-chunk-per-heading would produce many tiny, low-context chunks). Each chunk carries
its full provenance (citation substrate) and an `embed_text` with the heading path
prepended (retrieval context anchor).

Run: `python -m rag.chunk`
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import statistics as st

from .config import CHUNKS_JSONL, CHUNKS_META, PAGES_JSONL, Config

_H2 = re.compile(r"^##\s+(.*)")
_H3 = re.compile(r"^###\s+(.*)")


def split_by_headings(md_text: str) -> list[tuple[list[str], str]]:
    """Split markdown into (heading_path, block_text) on ## / ### headings.

    heading_path is [H2] or [H2, H3] for the section the block sits under.
    """
    h2 = h3 = None
    buf: list[str] = []
    blocks: list[tuple[list[str], str]] = []

    def flush():
        text = "\n".join(buf).strip()
        path = [h for h in (h2, h3) if h]
        return (path, text) if text else None

    for ln in md_text.splitlines():
        m2, m3 = _H2.match(ln), _H3.match(ln)
        if m2:
            b = flush()
            if b:
                blocks.append(b)
            buf, h2, h3 = [], m2.group(1).strip(), None
        elif m3:
            b = flush()
            if b:
                blocks.append(b)
            buf, h3 = [], m3.group(1).strip()
        else:
            buf.append(ln)
    b = flush()
    if b:
        blocks.append(b)
    return blocks


def common_prefix(paths: list[list[str]]) -> list[str]:
    """Deepest heading path shared by every packed block ([A,B]+[A,C] -> [A])."""
    out: list[str] = []
    for level in zip(*paths):          # zip stops at the shortest path
        if len(set(level)) != 1:
            break
        out.append(level[0])
    return out


def pack_blocks(blocks, target: int, overlap: int, min_words: int):
    """Greedy-pack consecutive small blocks up to `target` words; window-split any
    block larger than `target` (with `overlap`). Fragments < `min_words` are dropped.

    Yields `(heading_path, text, heading_paths)`. A packed chunk's `heading_path` is the
    **common prefix** of its blocks' paths, not the first block's: the path is prepended
    into `embed_text`, so claiming the first block's heading mis-anchors the vector of
    every multi-block chunk (26.6% of them) and cites merged text under a sibling
    heading it never came from. `heading_paths` keeps the full list for display.
    """
    out: list[tuple[list[str], str, list[list[str]]]] = []
    cur_text: list[str] = []
    cur_words = 0
    cur_paths: list[list[str]] = []

    def emit():
        nonlocal cur_text, cur_words, cur_paths
        if cur_text:
            out.append((common_prefix(cur_paths), "\n\n".join(cur_text).strip(), cur_paths))
        cur_text, cur_words, cur_paths = [], 0, []

    for path, text in blocks:
        n = len(text.split())
        if n > target:
            emit()  # flush whatever was packing before starting a big block
            words = text.split()
            step = max(1, target - overlap)
            for j in range(0, len(words), step):
                out.append((path, " ".join(words[j:j + target]), [path]))
                # Stop once a window reaches the end: further windows start inside it
                # and are wholly contained, costing top_k slots on duplicate text.
                if j + target >= len(words):
                    break
        else:
            if cur_text and cur_words + n > target:
                emit()
            cur_text.append(text)
            cur_words += n
            cur_paths.append(path)
    emit()
    return [(p, t, ps) for p, t, ps in out if len(t.split()) >= min_words]


def chunk_page(page: dict, cfg: Config) -> list[dict]:
    blocks = split_by_headings(page["text"]) or [([], page["text"])]
    packed = pack_blocks(blocks, cfg.target_words, cfg.overlap_words, cfg.min_chunk_words)
    section, title = page.get("section", ""), page.get("title", "")

    chunks = []
    for idx, (hpath, body, hpaths) in enumerate(packed):
        ctx = f"{section} > {title}"
        if hpath:
            ctx += " > " + " > ".join(hpath)
        chunks.append({
            "id": f"{page['id']}-{idx:03d}",   # stable => idempotent re-index
            "page_id": page["id"],
            "url": page["url"],
            "title": title,
            "section": section,
            "language": page.get("language", "en"),
            "breadcrumb": page.get("breadcrumb", []),
            "heading_path": hpath,             # common prefix — the citation anchor
            "heading_paths": hpaths,           # every block merged in (display/debug)
            "chunk_index": idx,
            "word_count": len(body.split()),
            "text": body,
            # embedding input: heading-path prepended so floating chunks keep meaning
            "embed_text": f"{ctx}\n\n{body}",
        })
    return chunks


def build_chunks(cfg: Config | None = None) -> list[dict]:
    cfg = cfg or Config()
    raw = PAGES_JSONL.read_bytes()
    pages = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    chunks = []
    for p in pages:
        chunks.extend(chunk_page(p, cfg))
    with CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Sidecar LAST: it attests that chunks.jsonl was built by this index_hash, and
    # `rag.index` refuses to index if the two disagree. Writing it after the data means
    # a crash mid-write leaves a stale/missing sidecar (loud) rather than a truncated
    # chunks.jsonl with a matching one (silent). `pages_sha` is reported provenance,
    # not hashed: a re-scrape changes the chunks but no Config field, and that should
    # be visible without re-stamping the index's identity on every scrape.
    CHUNKS_META.write_text(json.dumps({
        "index_hash": cfg.index_hash,
        "n_chunks": len(chunks),
        "n_pages": len(pages),
        "pages_sha": hashlib.sha256(raw).hexdigest()[:12],
        "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "chunker_version": cfg.chunker_version,
        "target_words": cfg.target_words,
        "overlap_words": cfg.overlap_words,
        "min_chunk_words": cfg.min_chunk_words,
        "embedder_id": cfg.embedder_id,
    }, indent=2), encoding="utf-8")
    return chunks


def main():
    cfg = Config()
    chunks = build_chunks(cfg)
    wc = [c["word_count"] for c in chunks]
    ids = [c["id"] for c in chunks]
    print(f"index_hash={cfg.index_hash}  chunker={cfg.chunker_version}  "
          f"target={cfg.target_words}w overlap={cfg.overlap_words}w (split path only)")
    print(f"chunks={len(chunks)}  unique_ids={len(set(ids))}")
    print(f"words/chunk: min={min(wc)} median={int(st.median(wc))} "
          f"mean={int(st.mean(wc))} max={max(wc)}")
    print(f"chunks > target: {sum(w > cfg.target_words for w in wc)}")
    print(f"-> {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
