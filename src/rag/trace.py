"""Per-run, full-pipeline tracing (requirement #2, and the data source for the UI).

A **Run** owns a `runs/<run_id>/` folder and aggregates latency + token cost across its
queries; a **Tracer** records one query's every stage into that run. One CLI question =
a run of one query; a batch/eval loop = one run with many queries.

Layout per run:
    runs/<run_id>/manifest.json      run meta + aggregates (+ eval_score slot)
    runs/<run_id>/queries/<tid>.json full per-query trace (all stages)
    runs/<run_id>/queries.jsonl      one compact line per query (list view)
    runs/index.jsonl                 one line per finalized run (enumerate runs)

On close, a run also appends a row to the auto-ledger in docs/EXPERIMENTS.md.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import time
import uuid

from . import ledger
from .config import RUNS_DIR, Config


def _run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def _compact(rec: dict) -> dict:
    """One-line-per-query summary for queries.jsonl (frontend list view)."""
    sel = rec.get("stages", {}).get("select", {}).get("selected", [])
    usage = rec.get("usage", {})
    return {
        "trace_id": rec["trace_id"],
        "run_id": rec["run_id"],
        "query": rec["query"],
        "answer": rec.get("answer", ""),
        "model": rec.get("model"),
        "key_id": rec.get("key_id"),
        "n_selected": len(sel),
        "n_citations": len(rec.get("citations", [])),
        "total_tokens": usage.get("total_tokens", 0),
        "total_latency_ms": rec.get("total_latency_ms"),
    }


class Run:
    # kind distinguishes an exploratory one-off query from a measured eval pass:
    #   "adhoc" — custom single query (frontend playground): folder + index, NO ledger row
    #   "eval"  — a scored pass over a gold set: folder + index + EXPERIMENTS.md ledger row
    #   "batch" — any other multi-query batch: folder + index, no ledger
    def __init__(self, config: Config, label: str = "", kind: str = "adhoc"):
        self.config = config
        self.run_id = _run_id()
        self.label = label
        self.kind = kind
        self.created_at = _dt.datetime.now().isoformat(timespec="seconds")
        self.dir = RUNS_DIR / self.run_id
        (self.dir / "queries").mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._latencies: list[float] = []
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._closed = False
        self._write_manifest("open")

    # -- aggregates ----------------------------------------------------------------
    def _aggregates(self) -> dict:
        tot = self._prompt_tokens + self._completion_tokens
        return {
            "n_queries": self._n,
            "avg_latency_ms": round(sum(self._latencies) / self._n, 1) if self._n else None,
            "total_latency_ms": round(sum(self._latencies), 1),
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": tot,       # cost proxy (Groq bills per token)
        }

    def _write_manifest(self, status: str):
        manifest = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "label": self.label,
            "kind": self.kind,
            "status": status,
            # Three scoped hashes: config_hash is pipeline identity (and nests
            # index_hash); eval_hash records how the run was *scored*. Two runs are
            # comparable iff config_hash matches; their scores are comparable iff
            # eval_hash matches too.
            "index_hash": self.config.index_hash,
            "config_hash": self.config.config_hash,
            "eval_hash": self.config.eval_hash,
            "config": self.config.to_dict(),
            **self._aggregates(),
            "eval_score": None,   # filled by the (next-task) scoring harness
        }
        (self.dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- trace lifecycle -----------------------------------------------------------
    def new_trace(self, query: str) -> "Tracer":
        return Tracer(self, query)

    def _record(self, rec: dict):
        self._n += 1
        self._latencies.append(rec.get("total_latency_ms", 0.0))
        usage = rec.get("usage", {})
        self._prompt_tokens += usage.get("prompt_tokens", 0)
        self._completion_tokens += usage.get("completion_tokens", 0)
        (self.dir / "queries" / f"{rec['trace_id']}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        with (self.dir / "queries.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(_compact(rec), ensure_ascii=False) + "\n")
        self._write_manifest("open")   # keep counts/tokens fresh mid-run

    def close(self):
        # Idempotent: closing twice (explicit close inside a `with`, or a retry) must
        # not duplicate the index row or — worse — the EXPERIMENTS.md ledger row.
        if self._closed:
            return
        self._closed = True
        self._write_manifest("finalized")
        self._append_index()
        if self.kind == "eval":            # only measured eval passes hit the ledger
            self._append_experiments_ledger()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- run enumeration + experiment ledger ---------------------------------------
    def _append_index(self):
        agg = self._aggregates()
        line = {
            "run_id": self.run_id, "created_at": self.created_at, "label": self.label,
            "kind": self.kind, "index_hash": self.config.index_hash,
            "config_hash": self.config.config_hash, "eval_hash": self.config.eval_hash,
            "n_queries": agg["n_queries"], "total_tokens": agg["total_tokens"],
            "avg_latency_ms": agg["avg_latency_ms"],
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        with (RUNS_DIR / "index.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _append_experiments_ledger(self):
        """Append this run's ledger row. `rag.ledger` owns the row format."""
        agg = self._aggregates()
        ledger.append_row({
            "run_id": self.run_id,
            "date": self.created_at[:10],
            "label": self.label or "—",
            "config_hash": f"`{self.config.config_hash}`",
            "eval_hash": f"`{self.config.eval_hash}`",
            "n_queries": agg["n_queries"],
            "total_tokens": agg["total_tokens"],
            "avg_latency_ms": agg["avg_latency_ms"],
        })


class Tracer:
    def __init__(self, run: Run, query: str):
        self.run = run
        self.trace_id = uuid.uuid4().hex[:12]
        self.record: dict = {
            "trace_id": self.trace_id,
            "run_id": run.run_id,
            "config_hash": run.config.config_hash,
            "query": query,
            "stages": {},
            # perf_counter, not time.time: an NTP step mid-run would otherwise poison
            # total_latency_ms and the avg_latency_ms reported beside the headline.
            "_started": time.perf_counter(),
        }

    @contextlib.contextmanager
    def span(self, stage: str):
        data: dict = {}
        start = time.perf_counter()
        try:
            yield data
        finally:
            data["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            self.record["stages"][stage] = data

    def set(self, **kwargs):
        self.record.update(kwargs)

    def emit(self):
        started = self.record.pop("_started")
        self.record["total_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self.run._record(self.record)
        return self.run.dir / "queries" / f"{self.trace_id}.json"
