"""webui.store — the frontend's read-model over runs/.

Hermetic: every case builds a synthetic runs/ tree in tmp_path matching the real schema
(rag/trace.py + rag/eval/harness.py output) and reads it back. No network, no Qdrant, no
committed-run dependency — so it survives a fresh clone the same way the console does.
The path-safety cases are the ones that matter: run_id/trace_id arrive from URLs.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from webui import store


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def runs(tmp_path):
    """A scored eval run + an older unscored ad-hoc run. Returns (base, rid, tid, rid2)."""
    rid, tid = "20260101-120000-aaaa", "abc123abc123"
    d = tmp_path / rid
    _write(d / "manifest.json", {
        "run_id": rid, "created_at": "2026-01-01T12:00:00", "label": "fixture",
        "kind": "eval", "status": "finalized", "index_hash": "idx0", "config_hash": "cfg0",
        "eval_hash": "evl0", "config": {"model": "m", "top_k": 6, "candidate_k": 20},
        "n_queries": 1, "total_tokens": 100, "prompt_tokens": 90, "completion_tokens": 10,
        "avg_latency_ms": 5.0, "eval_score": {"consistency": 0.5, "recall@k": 0.9}})
    _write(d / "queries" / f"{tid}.json", {
        "trace_id": tid, "run_id": rid, "query": "q", "query_id": "g01:canonical:0",
        "status": "ok", "answer": "a [1]", "citations": [{"n": 1}], "invalid_citations": [],
        "stages": {"retrieve": {"retrieved": [{"id": "p-000", "page_id": "p"}]},
                   "select": {"selected": [{"id": "p-000"}]},
                   "assemble": {"sources": {"1": {"n": 1}}, "context_chars": 10}}})
    _write_jsonl(d / "queries.jsonl", [
        {"trace_id": tid, "run_id": rid, "query": "q", "answer": "a [1]", "total_tokens": 100}])
    _write(d / "eval" / "scores.json", {
        "run_id": rid, "gold_set": "gold_v1_small", "panel": {"consistency": 0.5, "recall@k": 0.9},
        "meta": {"n_groups_scored": 1}})
    _write_jsonl(d / "eval" / "per_group.jsonl", [{"group_id": "g01", "consistency": 0.5}])
    _write_jsonl(d / "eval" / "per_query.jsonl", [
        {"query_id": "g01:canonical:0", "group_id": "g01", "trace_id": tid,
         "expected_page_ids": ["p"], "selected_pages": ["p"], "recall@k": 1}])
    _write_jsonl(d / "eval" / "judge" / "answers.jsonl", [
        {"query_id": "g01:canonical:0", "faithfulness": 1, "citation_acc": 1}])
    _write_jsonl(d / "eval" / "judge" / "agreement.jsonl", [
        {"group_id": "g01", "consistent": [1], "score": 1.0}])

    rid2 = "20251231-120000-bbbb"          # older, unscored ad-hoc run
    d2 = tmp_path / rid2
    _write(d2 / "manifest.json", {
        "run_id": rid2, "created_at": "2025-12-31T12:00:00", "kind": "adhoc",
        "status": "finalized", "n_queries": 1, "total_tokens": 50, "eval_score": None})
    _write_jsonl(d2 / "queries.jsonl", [{"trace_id": "ffff", "query": "q2", "answer": ""}])
    return tmp_path, rid, tid, rid2


def test_list_runs_newest_first_and_eval_flag(runs):
    base, rid, _tid, rid2 = runs
    got = store.list_runs(base)
    assert [r["run_id"] for r in got] == [rid, rid2]           # sorted by created_at desc
    assert got[0]["has_eval"] is True and got[1]["has_eval"] is False
    assert got[0]["eval_score"]["consistency"] == 0.5
    assert got[0]["model"] == "m"                              # lifted from config


def test_list_runs_skips_malformed_manifest(runs):
    base, *_ = runs
    bad = base / "20260202-120000-cccc"
    bad.mkdir()
    (bad / "manifest.json").write_text("{ not json", encoding="utf-8")
    got = store.list_runs(base)                                # half-written run is not fatal
    assert len(got) == 2 and all(r["run_id"] != bad.name for r in got)


def test_list_runs_empty_dir(tmp_path):
    assert store.list_runs(tmp_path / "does-not-exist") == []


def test_load_run_and_trace(runs):
    base, rid, tid, _ = runs
    run = store.load_run(rid, base)
    assert run["manifest"]["run_id"] == rid and len(run["queries"]) == 1
    tr = store.load_trace(rid, tid, base)
    assert tr["status"] == "ok"
    assert tr["stages"]["retrieve"]["retrieved"][0]["page_id"] == "p"


def test_load_eval_joins_and_none_when_unscored(runs):
    base, rid, tid, rid2 = runs
    ev = store.load_eval(rid, base)
    assert ev["scores"]["panel"]["consistency"] == 0.5
    assert ev["per_query"][0]["trace_id"] == tid                # joins back to the trace
    assert ev["judge_answers"][0]["faithfulness"] == 1
    assert ev["judge_agreement"][0]["group_id"] == "g01"
    assert store.load_eval(rid2, base) is None                  # ad-hoc run was never scored


@pytest.mark.parametrize("bad", [
    "../etc", "not-a-run-id", "20260101-120000-aaaa/../..", "20260101-120000-zzzz",
])
def test_load_run_rejects_bad_ids(runs, bad):
    base, *_ = runs
    with pytest.raises(store.NotFound):
        store.load_run(bad, base)


def test_load_trace_rejects_traversal_and_missing(runs):
    base, rid, _tid, _ = runs
    with pytest.raises(store.NotFound):
        store.load_trace(rid, "../manifest", base)              # no escape from queries/
    with pytest.raises(store.NotFound):
        store.load_trace(rid, "deadbeef", base)                 # well-formed but absent


def test_list_gold_sets_reports_group_counts():
    # Reads the real eval/ dir (committed, deterministic, no network).
    sets = {s["name"]: s["n_groups"] for s in store.list_gold_sets()}
    assert sets.get("gold_v1_small") == 10
    assert "gold_v1" not in sets            # the original 40-group set was removed


def test_token_spend_today_sums_only_today(tmp_path):
    today = date.today().isoformat()
    _write(tmp_path / "20260101-120000-aaaa" / "manifest.json",
           {"run_id": "20260101-120000-aaaa", "created_at": f"{today}T10:00:00",
            "kind": "adhoc", "total_tokens": 1234})
    _write(tmp_path / "20251231-120000-bbbb" / "manifest.json",
           {"run_id": "20251231-120000-bbbb", "created_at": "2025-12-31T10:00:00",
            "kind": "adhoc", "total_tokens": 9999})
    out = store.token_spend_today(tmp_path)
    assert out["date"] == today and out["spent"] == 1234        # yesterday's 9999 excluded


# -- delete_run: the one guarded mutation ---------------------------------------------
def test_delete_run_removes_adhoc_and_prunes_index(runs):
    base, rid, _tid, rid2 = runs
    # seed an index.jsonl with both runs; only the adhoc line should be pruned.
    (base / "index.jsonl").write_text(
        json.dumps({"run_id": rid}) + "\n" + json.dumps({"run_id": rid2}) + "\n", encoding="utf-8")
    assert store.delete_run(rid2, base) == {"run_id": rid2, "deleted": True}
    assert not (base / rid2).exists()                          # folder gone
    assert all(r["run_id"] != rid2 for r in store.list_runs(base))
    kept = [json.loads(l) for l in (base / "index.jsonl").read_text().splitlines() if l.strip()]
    assert [r["run_id"] for r in kept] == [rid]                # eval line kept, adhoc pruned


def test_delete_run_refuses_eval(runs):
    base, rid, *_ = runs
    with pytest.raises(store.NotDeletable):
        store.delete_run(rid, base)                            # eval run: hashes/ledger anchor it
    assert (base / rid).exists()                               # left untouched


def test_delete_run_refuses_open_run(runs):
    base, *_ = runs
    rid3 = "20260303-120000-dddd"
    _write(base / rid3 / "manifest.json",
           {"run_id": rid3, "kind": "adhoc", "status": "open", "created_at": "2026-03-03T12:00:00"})
    with pytest.raises(store.NotDeletable):
        store.delete_run(rid3, base)                           # mid-write run is not deletable
    assert (base / rid3).exists()


def test_delete_run_no_index_file_is_fine(runs):
    base, _rid, _tid, rid2 = runs                              # fixture writes no index.jsonl
    assert store.delete_run(rid2, base)["deleted"] is True
    assert not (base / rid2).exists()


@pytest.mark.parametrize("bad", [
    "../etc", "not-a-run-id", "20260101-120000-aaaa/../..", "20260101-120000-zzzz",
])
def test_delete_run_rejects_bad_ids(runs, bad):
    base, *_ = runs
    with pytest.raises(store.NotFound):
        store.delete_run(bad, base)
