"""Read a finished run's traces back off disk.

Scoring reads `runs/<run_id>/queries/*.json` and never touches Qdrant or the pipeline —
that is what lets a run be re-scored (new judge prompt, judge crashed, noise
calibration) without spending another 400k generation tokens, and lets scoring run while
another process holds the Qdrant lock.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..config import RUNS_DIR


def dedup(seq) -> list:
    """Order-preserving dedup — rank of first occurrence is what MRR needs."""
    seen, out = set(), []
    for x in seq:
        if x is not None and x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class QueryTrace:
    trace_id: str
    query_id: str | None
    query: str
    status: str                     # ok | error | retrieval_only
    selected_pages: list[str] = field(default_factory=list)
    selected_chunks: list[str] = field(default_factory=list)
    retrieved_pages: list[str] = field(default_factory=list)
    reranked_pages: list[str] = field(default_factory=list)
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    context: str = ""               # the exact numbered sources block the model saw
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0                  # whole query, end to end
    stage_latency_ms: dict = field(default_factory=dict)   # per stage, for attribution

    @property
    def has_selection(self) -> bool:
        """Retrieval metrics are valid whenever selection completed.

        The pipeline wraps only the generate span, so an *error* trace still carries the
        full retrieved/selected sets — retrieval metrics stay at full coverage even if
        every generation failed.
        """
        return bool(self.selected_pages)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _pages(stage: dict, key: str) -> list[str]:
    return dedup(c.get("page_id") for c in stage.get(key, []))


def from_record(rec: dict) -> QueryTrace:
    stages = rec.get("stages", {})
    sel = stages.get("select", {})
    return QueryTrace(
        trace_id=rec.get("trace_id", ""),
        query_id=rec.get("query_id"),
        query=rec.get("query", ""),
        status=rec.get("status", ""),
        selected_pages=_pages(sel, "selected"),
        selected_chunks=dedup(c.get("id") for c in sel.get("selected", [])),
        retrieved_pages=_pages(stages.get("retrieve", {}), "retrieved"),
        reranked_pages=_pages(stages.get("rerank", {}), "reranked"),
        answer=rec.get("answer", "") or "",
        citations=rec.get("citations", []) or [],
        sources=stages.get("assemble", {}).get("sources", {}) or {},
        context=stages.get("assemble", {}).get("context", "") or "",
        finish_reason=stages.get("generate", {}).get("finish_reason", "") or "",
        usage=rec.get("usage", {}) or {},
        latency_ms=rec.get("total_latency_ms", 0.0) or 0.0,
        stage_latency_ms={k: v.get("latency_ms", 0.0) for k, v in stages.items()},
    )


def load_run_traces(run_id: str) -> list[QueryTrace]:
    qdir = RUNS_DIR / run_id / "queries"
    if not qdir.is_dir():
        raise SystemExit(f"no traces at {qdir} — is {run_id!r} a real run id?")
    out = []
    for p in sorted(qdir.glob("*.json")):
        try:
            out.append(from_record(json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"cannot read trace {p}: {e}") from None
    if not out:
        raise SystemExit(f"run {run_id!r} has no query traces")
    return out


def by_query_id(traces: list[QueryTrace]) -> dict[str, QueryTrace]:
    return {t.query_id: t for t in traces if t.query_id}


def load_runs(run_ids: list[str]) -> list[QueryTrace]:
    """Merge several runs' traces into one scoreable population.

    The free tier caps generation at ~189 answers/day, so a fully-generated 240-query
    panel necessarily spans days — i.e. several `Run`s. Merging at *scoring* time keeps
    each day's run immutable and closed (reopening a Run would double its ledger row,
    since idempotency lives in memory, not on disk).

    Later runs win on duplicate query_ids, so re-generating a query fixes it. Callers
    must check the runs share a config_hash: merging traces from different pipelines
    would silently average two different systems.
    """
    merged: dict[str, QueryTrace] = {}
    loose: list[QueryTrace] = []
    for rid in run_ids:
        for t in load_run_traces(rid):
            if t.query_id:
                merged[t.query_id] = t
            else:
                loose.append(t)
    return list(merged.values()) + loose
