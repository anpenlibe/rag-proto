"""The EXPERIMENTS.md ledger: header-driven column lookup, marker scoping, write-back.

The old parser hardcoded `cells[8]` and scanned the whole document. Adding a column
clobbered avg_latency_ms and truncated eval_score away — silently. These tests pin the
contract that replaced it.

`ledger` binds EXPERIMENTS_MD/RUNS_DIR by value at import, so tests patch `rag.ledger`
(not `rag.config`).
"""
from __future__ import annotations

import json

import pytest

from rag import ledger

_HEADER = "| " + " | ".join(ledger._COLUMNS) + " |\n"
_SEP = "|" + "|".join(["---"] * len(ledger._COLUMNS)) + "|\n"


def _doc(rows: str = "") -> str:
    # A realistic document: another wide table BEFORE the ledger, which the old
    # table-agnostic parser matched by luck.
    return (
        "# Experiments\n\n"
        "## Results summary\n\n"
        "| Exp | Date | Lever | Model | gold_set | recall@k | mrr | consistency | "
        "faithful | cite_acc | config_hash |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| E0 | 2026-07-15 | baseline | llama | gold_v1 | – | – | – | – | – | `abc` |\n\n"
        "## Run ledger\n\n"
        f"{ledger._LEDGER_START}\n{_HEADER}{_SEP}{rows}{ledger._LEDGER_END}\n"
    )


@pytest.fixture
def exp(tmp_path, monkeypatch):
    p = tmp_path / "EXPERIMENTS.md"
    p.write_text(_doc(), encoding="utf-8")
    monkeypatch.setattr(ledger, "EXPERIMENTS_MD", p)
    monkeypatch.setattr(ledger, "RUNS_DIR", tmp_path / "runs")
    return p


def _row(run_id="r1"):
    return {"run_id": run_id, "date": "2026-07-15", "label": "E0", "config_hash": "`c`",
            "eval_hash": "`e`", "n_queries": 240, "total_tokens": 1000,
            "avg_latency_ms": 12.5}


def test_append_row_inserts_before_the_end_marker(exp):
    assert ledger.append_row(_row()) is True
    lines = exp.read_text().splitlines()
    i_row = next(i for i, l in enumerate(lines) if l.startswith("| r1 |"))
    i_end = next(i for i, l in enumerate(lines) if ledger._LEDGER_END in l)
    assert i_row < i_end


def test_appended_row_has_pending_score_cell(exp):
    ledger.append_row(_row())
    row = next(l for l in exp.read_text().splitlines() if l.startswith("| r1 |"))
    assert ledger._PENDING in row


def test_append_row_refuses_incomplete_rows(exp):
    bad = _row()
    del bad["eval_hash"]
    with pytest.warns(UserWarning, match="missing"):
        assert ledger.append_row(bad) is False


def test_update_ledger_row_patches_only_the_score_cell(exp):
    ledger.append_row(_row())
    assert ledger.update_ledger_row("r1", {"consistency": 0.5}) is True
    row = next(l for l in exp.read_text().splitlines() if l.startswith("| r1 |"))
    cells = ledger._cells(row)
    assert cells[ledger._COLUMNS.index("avg_latency_ms")] == "12.5", "clobbered latency"
    assert cells[ledger._COLUMNS.index("n_queries")] == "240"
    assert cells[ledger._COLUMNS.index("eval_hash")] == "`e`"
    assert "**0.5**" in cells[ledger._COLUMNS.index("eval_score")]
    assert len(cells) == len(ledger._COLUMNS), "row must not lose columns"


def test_score_column_is_found_by_header_not_by_position(exp, monkeypatch):
    """The point of the rewrite: reordering columns must not corrupt a cell."""
    cols = ("run_id", "date", "eval_score", "label", "config_hash", "eval_hash",
            "n_queries", "total_tokens", "avg_latency_ms")
    monkeypatch.setattr(ledger, "_COLUMNS", cols)
    header = "| " + " | ".join(cols) + " |\n"
    sep = "|" + "|".join(["---"] * len(cols)) + "|\n"
    exp.write_text(
        f"{ledger._LEDGER_START}\n{header}{sep}"
        f"| r1 | 2026-07-15 | _pending_ | E0 | `c` | `e` | 240 | 10 | 5 |\n"
        f"{ledger._LEDGER_END}\n", encoding="utf-8")
    assert ledger.update_ledger_row("r1", {"consistency": 0.9}) is True
    cells = ledger._cells(next(l for l in exp.read_text().splitlines()
                               if l.startswith("| r1 |")))
    assert "**0.9**" in cells[2]
    assert cells[3] == "E0", "label must survive"


def test_other_tables_are_never_touched(exp):
    ledger.append_row(_row("E0"))   # a run_id that collides with the summary table's cell
    before = next(l for l in _doc().splitlines() if l.startswith("| E0 | 2026-07-15 | baseline"))
    ledger.update_ledger_row("E0", {"consistency": 0.5})
    after = next(l for l in exp.read_text().splitlines()
                 if l.startswith("| E0 | 2026-07-15 | baseline"))
    assert before == after, "the Results-summary row must be out of scope"


def test_separator_row_is_never_mistaken_for_a_run(exp):
    with pytest.warns(UserWarning, match="no row for run"):
        assert ledger.update_ledger_row("---", {"consistency": 1.0}) is False


def test_missing_run_warns_and_returns_false(exp):
    with pytest.warns(UserWarning, match="no row for run"):
        assert ledger.update_ledger_row("nope", {"consistency": 1.0}) is False


def test_missing_markers_warn_and_do_not_write(exp):
    exp.write_text("# nothing here\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="markers"):
        assert ledger.append_row(_row()) is False


def test_header_mismatch_warns(exp):
    exp.write_text(
        f"{ledger._LEDGER_START}\n| run_id | date | eval_score |\n|---|---|---|\n"
        f"| r1 | 2026-07-15 | _pending_ |\n{ledger._LEDGER_END}\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="does not match"):
        ledger.update_ledger_row("r1", {"consistency": 0.1})


# -- formatting -----------------------------------------------------------------------
def test_fmt_bolds_the_headline_and_rounds_floats():
    out = ledger._fmt({"consistency": 0.6183333333333333, "recall@k": 0.8})
    assert out.startswith("**0.618**"), "raw float would render 16 decimals"
    assert "recall@k=0.8" in out


def test_fmt_without_headline_and_empty():
    assert ledger._fmt({"recall@k": 0.5}) == "recall@k=0.5"
    assert ledger._fmt({}) == "—"


# -- manifest -------------------------------------------------------------------------
def test_update_manifest_writes_scores(exp, tmp_path):
    d = tmp_path / "runs" / "r1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"run_id": "r1", "eval_score": None}))
    assert ledger.update_manifest("r1", {"consistency": 0.5}) is True
    assert json.loads((d / "manifest.json").read_text())["eval_score"] == {"consistency": 0.5}


def test_update_manifest_survives_unserialisable_scores(exp, tmp_path):
    """A bad score value must not crash AFTER a full run's tokens are already spent."""
    d = tmp_path / "runs" / "r1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"run_id": "r1", "eval_score": None}))
    with pytest.warns(UserWarning, match="not JSON-serialisable"):
        assert ledger.update_manifest("r1", {"consistency": {1, 2}}) is False
    assert json.loads((d / "manifest.json").read_text())["eval_score"] is None


def test_update_manifest_missing_run_warns(exp):
    with pytest.warns(UserWarning, match="cannot read manifest"):
        assert ledger.update_manifest("ghost", {"consistency": 0.5}) is False
