"""Read-model over ``runs/`` — the frontend's entire data source.

Pure functions over the filesystem: enumerate runs, load a run's compact query list,
load one full per-query trace, load a run's eval scores. **No network, no Qdrant, no API
keys** — replaying a trace works on a fresh clone. Every shape here is written by
``rag/trace.py`` (traces) and ``rag/eval/harness.py`` (eval); this module only reads —
with a **single** deliberate exception, ``delete_run`` (see below).

Two rules that matter:
  * **Enumerate the filesystem, not ``runs/index.jsonl``.** ``index.jsonl`` can lag or be
    hand-curated (some run kinds never appended to it), so globbing ``*/manifest.json`` is
    the robust source of truth.
  * **Path-safety.** ``run_id`` / ``trace_id`` arrive from URLs. Every id is validated
    against a strict pattern *and* the resolved path is asserted to live under the runs
    dir, so a crafted ``../`` can never read outside ``runs/``.

The one writer, ``delete_run``, lives here (not in the HTTP handler) precisely so it goes
through the same validated-path guard as the readers and is unit-tested the same way. It
is restricted to ``kind == "adhoc"`` runs: eval passes are referenced by hashes, ship
committed ``eval/`` artefacts, and append to ``docs/EXPERIMENTS.md`` — deleting one would
orphan measured results, so the store refuses.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path

from rag.config import EVAL_DIR
from rag.config import RUNS_DIR as _DEFAULT_RUNS_DIR

# A run_id is "YYYYMMDD-HHMMSS-<4hex>" (rag.trace._run_id); a trace_id is uuid4 hex.
# Validate the shape *and* the resolved path (below) — defence in depth against traversal.
_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{4,64}$")


class NotFound(Exception):
    """A run or trace that does not exist (server maps this to 404)."""


class NotDeletable(Exception):
    """A run that exists but may not be deleted, e.g. a non-adhoc or in-flight run
    (server maps this to 403)."""


def _runs_dir(runs_dir: Path | str | None) -> Path:
    return Path(runs_dir) if runs_dir is not None else _DEFAULT_RUNS_DIR


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_path(run_id: str, runs_dir: Path | str | None) -> Path:
    """Validated ``runs/<run_id>/`` — raises NotFound for a bad id or escape attempt."""
    if not _RUN_ID_RE.match(run_id):
        raise NotFound(f"bad run id: {run_id!r}")
    base = _runs_dir(runs_dir).resolve()
    path = (base / run_id).resolve()
    if path.parent != base:                      # no traversal outside runs/
        raise NotFound(f"run id escapes runs dir: {run_id!r}")
    if not (path / "manifest.json").exists():
        raise NotFound(f"no such run: {run_id}")
    return path


# -- run enumeration ------------------------------------------------------------------
def _run_summary(manifest: dict, run_dir: Path) -> dict:
    """The one card in the run list — small, no per-query text."""
    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "created_at": manifest.get("created_at"),
        "label": manifest.get("label", ""),
        "kind": manifest.get("kind"),               # adhoc | eval | batch
        "status": manifest.get("status"),           # open | finalized
        "index_hash": manifest.get("index_hash"),
        "config_hash": manifest.get("config_hash"),
        "eval_hash": manifest.get("eval_hash"),
        "n_queries": manifest.get("n_queries", 0),
        "total_tokens": manifest.get("total_tokens", 0),
        "avg_latency_ms": manifest.get("avg_latency_ms"),
        "model": (manifest.get("config") or {}).get("model"),
        "eval_score": manifest.get("eval_score"),   # null until scored
        "has_eval": (run_dir / "eval" / "scores.json").exists(),
        "has_traces": (run_dir / "queries").is_dir()
        and any((run_dir / "queries").iterdir()),
        "note": manifest.get("note"),
    }


def list_runs(runs_dir: Path | str | None = None) -> list[dict]:
    """All runs, newest first. Robust to a half-written or malformed manifest (skipped)."""
    base = _runs_dir(runs_dir)
    if not base.exists():
        return []
    out = []
    for manifest_path in base.glob("*/manifest.json"):
        try:
            out.append(_run_summary(_read_json(manifest_path), manifest_path.parent))
        except (json.JSONDecodeError, OSError):
            continue                                # a run mid-write is not fatal
    out.sort(key=lambda r: r.get("created_at") or r["run_id"], reverse=True)
    return out


# -- one run --------------------------------------------------------------------------
def load_run(run_id: str, runs_dir: Path | str | None = None) -> dict:
    """A run's manifest + its compact per-query list (``queries.jsonl``)."""
    run_dir = _run_path(run_id, runs_dir)
    manifest = _read_json(run_dir / "manifest.json")
    return {
        "manifest": manifest,
        "queries": _read_jsonl(run_dir / "queries.jsonl"),
        "has_eval": (run_dir / "eval" / "scores.json").exists(),
    }


def load_trace(run_id: str, trace_id: str,
               runs_dir: Path | str | None = None) -> dict:
    """One full per-query trace: every stage, the exact prompt, the raw response."""
    run_dir = _run_path(run_id, runs_dir)
    if not _TRACE_ID_RE.match(trace_id):
        raise NotFound(f"bad trace id: {trace_id!r}")
    path = (run_dir / "queries" / f"{trace_id}.json").resolve()
    if path.parent != (run_dir / "queries").resolve() or not path.exists():
        raise NotFound(f"no such trace: {run_id}/{trace_id}")
    return _read_json(path)


# -- one run's eval -------------------------------------------------------------------
def load_eval(run_id: str, runs_dir: Path | str | None = None) -> dict | None:
    """A run's scoring output, or None if it was never scored.

    Returns the four eval artefacts as-is; the frontend joins them (per_query↔judge
    answers on ``query_id``; per_group↔judge agreement on ``group_id``). Keeping the
    join in the UI keeps this module a thin, obvious reader.
    """
    run_dir = _run_path(run_id, runs_dir)
    eval_dir = run_dir / "eval"
    scores_path = eval_dir / "scores.json"
    if not scores_path.exists():
        return None
    return {
        "scores": _read_json(scores_path),
        "per_group": _read_jsonl(eval_dir / "per_group.jsonl"),
        "per_query": _read_jsonl(eval_dir / "per_query.jsonl"),
        "judge_answers": _read_jsonl(eval_dir / "judge" / "answers.jsonl"),
        "judge_agreement": _read_jsonl(eval_dir / "judge" / "agreement.jsonl"),
    }


# -- budget indicator -----------------------------------------------------------------
def token_spend_today(runs_dir: Path | str | None = None) -> dict:
    """Pipeline tokens spent by runs created today — an *approximate* budget gauge.

    The Groq free-tier daily cap (100k/key/model, ~400k across the 4-key pool) is
    invisible in response headers, so this can only sum what our own runs recorded, not
    read a real remaining balance. Judge tokens live outside run manifests and are not
    counted here (they draw a separate per-model budget).
    """
    today = date.today().isoformat()
    spent = 0
    for run in list_runs(runs_dir):
        if (run.get("created_at") or "")[:10] == today:
            spent += run.get("total_tokens") or 0
    return {"date": today, "spent": spent, "day_budget_estimate": 400_000}


# -- gold sets (for the eval control) -------------------------------------------------
def list_gold_sets() -> list[dict]:
    """Available `eval/<name>.jsonl` gold sets with their group counts, name-sorted."""
    out = []
    for path in sorted(EVAL_DIR.glob("*.jsonl")):
        try:
            n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            continue
        out.append({"name": path.stem, "n_groups": n})
    return out


# -- the one writer: delete an ad-hoc run ---------------------------------------------
def delete_run(run_id: str, runs_dir: Path | str | None = None) -> dict:
    """Delete an ad-hoc run's folder and prune its ``index.jsonl`` line.

    The sole mutation in this module (see the module docstring). Guards, in order:
      * ``_run_path`` validates the id shape *and* that it resolves under ``runs/`` — a
        crafted ``../`` can never ``rmtree`` outside the runs dir (raises ``NotFound``).
      * only ``kind == "adhoc"`` runs may be removed; a non-adhoc or still-``open`` (being
        written) run raises ``NotDeletable``. Eval passes are protected: they anchor
        measured results (hashes, committed ``eval/``, the ``EXPERIMENTS.md`` ledger).

    Ad-hoc runs never touched the ledger (``trace.py`` only appends it for ``kind=="eval"``),
    so no ``EXPERIMENTS.md`` surgery is needed — just the folder and the ``index.jsonl`` row.
    """
    run_dir = _run_path(run_id, runs_dir)
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("kind") != "adhoc":
        raise NotDeletable(f"only ad-hoc runs may be deleted (run is {manifest.get('kind')!r})")
    if manifest.get("status") == "open":
        raise NotDeletable(f"run {run_id} is still being written")

    shutil.rmtree(run_dir)
    _prune_index(run_id, runs_dir)
    return {"run_id": run_id, "deleted": True}


def _prune_index(run_id: str, runs_dir: Path | str | None) -> None:
    """Drop ``run_id``'s line from ``runs/index.jsonl`` (a no-op if the file is absent).

    Tolerant of malformed lines (kept as-is, like ``_read_jsonl``) so one bad row can't
    lose the rest of the index. Rewritten atomically via a temp file + replace.
    """
    index_path = _runs_dir(runs_dir) / "index.jsonl"
    if not index_path.exists():
        return
    kept = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                if json.loads(stripped).get("run_id") == run_id:
                    continue                          # drop this run's row
            except json.JSONDecodeError:
                pass                                  # keep an unparseable line untouched
            kept.append(stripped)
    tmp = index_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
    tmp.replace(index_path)
