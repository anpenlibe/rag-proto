"""Assemble selected candidates into a numbered, citable context block.

Each source gets an index [n] the LLM cites; `sources` maps n -> provenance so the
final answer's [n] markers resolve to real URLs.

NOTE: `sources` is keyed by **str(n)**, not int. It gets serialised into the trace
JSON, and JSON object keys are always strings — an int-keyed dict silently comes back
string-keyed, so a frontend doing `sources[citation["n"]]` would KeyError. Keying by
str here makes in-memory and on-disk shapes identical. Use `src_key(n)` to look up.
"""
from __future__ import annotations

from .retrieve import Candidate


def src_key(n: int | str) -> str:
    """Canonical source key. Use everywhere instead of raw indexing."""
    return str(n)


def build_context(candidates: list[Candidate]) -> tuple[str, dict[str, dict]]:
    lines: list[str] = []
    sources: dict[str, dict] = {}
    for n, c in enumerate(candidates, 1):
        hp = " > ".join(c.heading_path) if c.heading_path else ""
        header = f"[{n}] {c.title}" + (f" — {hp}" if hp else "")
        lines.append(f"{header}\n{c.text}")
        sources[src_key(n)] = {
            "n": n,                      # also inside the value, for list-style consumers
            "url": c.url, "title": c.title, "heading_path": c.heading_path,
            "section": c.section, "chunk_id": c.id, "score": round(c.score, 4),
        }
    return "\n\n".join(lines), sources
