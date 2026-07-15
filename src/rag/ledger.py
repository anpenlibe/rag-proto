"""The EXPERIMENTS.md run-ledger: format owner + score write-back.

A `Run` closes before its scores exist — the harness computes metrics *after* the 240
queries are done. So a run appends its row with a `_pending_` score cell at close, and
this module patches that cell once the numbers land:

    from rag.ledger import update_eval_score
    update_eval_score(run_id, {"consistency": 0.62, "recall@k": 0.80, ...})

**This module owns the row format.** It used to be written in `trace.py` and parsed here
with no shared definition, so the two could drift and a column added on one side silently
clobbered a cell on the other. `trace.Run.close()` now calls `append_row()`; nothing else
constructs a row. (No import cycle: ledger -> config only.)

All reads/writes are scoped **between the RUN-LEDGER markers** — EXPERIMENTS.md holds
other tables whose rows are wide enough to match a naive parse.
"""
from __future__ import annotations

import json
import warnings

from .config import EXPERIMENTS_MD, RUNS_DIR

_LEDGER_START = "<!-- RUN-LEDGER:START -->"
_LEDGER_END = "<!-- RUN-LEDGER:END -->"

# The ledger's columns, in order. Changing this means editing the header in
# EXPERIMENTS.md to match; `_score_col()` warns if the two disagree.
_COLUMNS = (
    "run_id", "date", "label", "config_hash", "eval_hash",
    "n_queries", "total_tokens", "avg_latency_ms", "eval_score",
)
_SCORE_COL = "eval_score"
_PENDING = "_pending_"

_HEADLINE = "consistency"   # the metric shown in the ledger's eval_score column


# -- cell formatting ------------------------------------------------------------------
def _num(v):
    """Ledger cells are read by humans; raw floats render as 0.6183333333333333."""
    return round(v, 3) if isinstance(v, float) else v


def _fmt(scores: dict) -> str:
    """Ledger cell: headline metric first, others compact."""
    if _HEADLINE in scores:
        rest = " ".join(f"{k}={_num(v)}" for k, v in scores.items() if k != _HEADLINE)
        return f"**{_num(scores[_HEADLINE])}**" + (f" ({rest})" if rest else "")
    return " ".join(f"{k}={_num(v)}" for k, v in scores.items()) or "—"


# -- markdown table plumbing ----------------------------------------------------------
def _cells(line: str) -> list[str] | None:
    """Inner cells of a `| a | b |` row, or None if the line isn't such a row."""
    parts = line.split("|")
    if len(parts) < 3 or parts[0].strip() or parts[-1].strip():
        return None
    return [p.strip() for p in parts[1:-1]]


def _render(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |\n"


def _is_separator(cells: list[str]) -> bool:
    """`|---|---|` is table syntax, not data — never a candidate row."""
    return all(set(c) <= set("-: ") and c for c in cells)


def _bounds(lines: list[str]) -> tuple[int, int] | None:
    """Index range (start, end) of the lines strictly between the ledger markers."""
    start = end = None
    for i, line in enumerate(lines):
        if _LEDGER_START in line:
            start = i + 1
        elif _LEDGER_END in line:
            end = i
            break
    if start is None or end is None or end < start:
        return None
    return start, end


def _score_col(lines: list[str], lo: int, hi: int) -> int | None:
    """Locate the eval_score column from the header row rather than hardcoding it."""
    for i in range(lo, hi):
        cells = _cells(lines[i])
        if cells and cells[0] == "run_id":
            if tuple(cells) != _COLUMNS:
                warnings.warn(
                    f"ledger: EXPERIMENTS.md header {tuple(cells)} does not match "
                    f"rag.ledger._COLUMNS {_COLUMNS}", stacklevel=2)
            return cells.index(_SCORE_COL) if _SCORE_COL in cells else None
    return None


def _read_lines() -> list[str] | None:
    try:
        return EXPERIMENTS_MD.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as e:
        warnings.warn(f"ledger: cannot read {EXPERIMENTS_MD} ({e})", stacklevel=2)
        return None


def _write_lines(lines: list[str], run_id: str) -> bool:
    try:
        EXPERIMENTS_MD.write_text("".join(lines), encoding="utf-8")
        return True
    except OSError as e:
        warnings.warn(f"ledger: cannot write {EXPERIMENTS_MD} ({e}); "
                      f"row for run {run_id} NOT written", stacklevel=2)
        return False


# -- public API -----------------------------------------------------------------------
def append_row(row: dict) -> bool:
    """Insert a run's row (score cell `_pending_`) before the RUN-LEDGER:END marker.

    Best-effort — a docs problem must not crash a finished run — but never *silent*: a
    dropped row means an eval run that reports success while leaving no experiment record.
    """
    missing = [c for c in _COLUMNS if c != _SCORE_COL and c not in row]
    if missing:
        warnings.warn(f"ledger: row for run {row.get('run_id')} missing {missing}; "
                      f"NOT written", stacklevel=2)
        return False

    cells = [_PENDING if c == _SCORE_COL else str(row[c]) for c in _COLUMNS]
    lines = _read_lines()
    if lines is None:
        return False
    b = _bounds(lines)
    if b is None:
        warnings.warn(f"ledger: markers {_LEDGER_START!r}/{_LEDGER_END!r} missing in "
                      f"{EXPERIMENTS_MD}; row for run {row['run_id']} NOT written",
                      stacklevel=2)
        return False
    lines.insert(b[1], _render(cells))
    return _write_lines(lines, row["run_id"])


def update_manifest(run_id: str, scores: dict) -> bool:
    path = RUNS_DIR / run_id / "manifest.json"
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(f"ledger: cannot read manifest for run {run_id} ({e})", stacklevel=2)
        return False
    m["eval_score"] = scores
    # Serialise before writing: a non-JSON score value (numpy float, Decimal) must not
    # take down the harness *after* a full run's tokens have already been spent.
    try:
        blob = json.dumps(m, indent=2, ensure_ascii=False)
    except TypeError as e:
        warnings.warn(f"ledger: scores for run {run_id} are not JSON-serialisable ({e}); "
                      f"manifest NOT updated", stacklevel=2)
        return False
    try:
        path.write_text(blob, encoding="utf-8")
    except OSError as e:
        warnings.warn(f"ledger: cannot write manifest for run {run_id} ({e})", stacklevel=2)
        return False
    return True


def update_ledger_row(run_id: str, scores: dict) -> bool:
    """Replace the eval_score cell of the ledger row whose run_id cell matches."""
    lines = _read_lines()
    if lines is None:
        return False
    b = _bounds(lines)
    if b is None:
        warnings.warn(f"ledger: markers missing in {EXPERIMENTS_MD}; run {run_id} "
                      f"score NOT written", stacklevel=2)
        return False
    lo, hi = b

    col = _score_col(lines, lo, hi)
    if col is None:
        warnings.warn(f"ledger: no {_SCORE_COL!r} column found in the {EXPERIMENTS_MD} "
                      f"ledger header; run {run_id} score NOT written", stacklevel=2)
        return False

    for i in range(lo, hi):
        cells = _cells(lines[i])
        if cells and not _is_separator(cells) and len(cells) > col and cells[0] == run_id:
            cells[col] = _fmt(scores)
            lines[i] = _render(cells)
            return _write_lines(lines, run_id)

    warnings.warn(f"ledger: no row for run {run_id} in {EXPERIMENTS_MD} "
                  f"(was it an 'eval' kind run?)", stacklevel=2)
    return False


def update_eval_score(run_id: str, scores: dict) -> bool:
    """Write scores to both the run manifest and the EXPERIMENTS.md ledger row."""
    ok_m = update_manifest(run_id, scores)
    ok_l = update_ledger_row(run_id, scores)
    return ok_m and ok_l
