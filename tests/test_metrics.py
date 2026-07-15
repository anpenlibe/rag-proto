"""The metric panel. These numbers ARE the deliverable, so they get pinned by hand.

Each expected value below is computed manually in the docstring/comment — a test that
just re-runs the implementation would happily bless a wrong formula.
"""
from __future__ import annotations

import pytest

from rag.eval.gold import Group, Phrasing
from rag.eval.metrics import (
    jaccard,
    mean_pairwise_jaccard,
    offline_panel,
    recall_any,
    reciprocal_rank,
)
from rag.eval.traces import QueryTrace, dedup


# -- primitives -----------------------------------------------------------------------
def test_jaccard_identical_and_disjoint():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial():
    # |{a} n {a,b}| / |{a} u {a,b}| = 1/2
    assert jaccard({"a"}, {"a", "b"}) == 0.5


def test_mean_pairwise_jaccard_needs_two_sets():
    assert mean_pairwise_jaccard([]) is None
    assert mean_pairwise_jaccard([{"a"}]) is None


def test_mean_pairwise_jaccard_averages_all_pairs():
    # pairs: (ab,ab)=1, (ab,a)=1/2, (ab,a)=1/2 -> mean = 2/3
    got = mean_pairwise_jaccard([{"a", "b"}, {"a", "b"}, {"a"}])
    assert got == pytest.approx((1.0 + 0.5 + 0.5) / 3)


def test_mean_pairwise_jaccard_uses_all_15_pairs_for_six_sets():
    sets = [{"a"}] * 6
    assert mean_pairwise_jaccard(sets) == 1.0


def test_recall_any_hit_and_miss():
    assert recall_any(["p1", "p2"], ["p2"]) == 1
    assert recall_any(["p1"], ["p9"]) == 0


def test_recall_any_with_two_expected_pages_is_any_not_all():
    assert recall_any(["p1"], ["p1", "p2"]) == 1


def test_reciprocal_rank_positions():
    assert reciprocal_rank(["gold"], ["gold"]) == 1.0
    assert reciprocal_rank(["x", "gold"], ["gold"]) == 0.5
    assert reciprocal_rank(["x", "y", "gold"], ["gold"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_absent_is_zero():
    assert reciprocal_rank(["x", "y"], ["gold"]) == 0.0


def test_reciprocal_rank_takes_first_occurrence():
    assert reciprocal_rank(["gold", "x", "gold"], ["gold"]) == 1.0


def test_dedup_preserves_first_occurrence_order():
    """Rank must count distinct pages: [A, A, gold] puts gold at rank 2, not 3."""
    assert dedup(["a", "a", "b", None, "a"]) == ["a", "b"]
    assert reciprocal_rank(dedup(["a", "a", "gold"]), ["gold"]) == 0.5


# -- panel ----------------------------------------------------------------------------
def _group(gid="g01", expected=("gold",)):
    ph = [Phrasing(gid, "canonical", 0, "c")]
    ph += [Phrasing(gid, "paraphrase", i, f"p{i}") for i in range(4)]
    ph.append(Phrasing(gid, "weird", 0, "w"))
    return Group(gid, "Sec", "fact", ("u",), tuple(expected), tuple(ph))


def _trace(qid, pages, status="ok"):
    return QueryTrace(
        trace_id=qid, query_id=qid, query="q", status=status,
        selected_pages=list(pages), selected_chunks=[f"{p}-000" for p in pages],
        retrieved_pages=list(pages), reranked_pages=list(pages),
    )


def test_perfect_consistency_when_every_phrasing_selects_the_same_pages():
    g = _group()
    traces = [_trace(p.query_id, ["gold", "x"]) for p in g.phrasings]
    panel, _pq, _pg, meta = offline_panel([g], traces)
    assert panel["consistency"] == 1.0
    assert panel["consistency_weird"] == 1.0
    assert panel["recall@k"] == 1.0
    assert panel["mrr"] == 1.0
    assert meta["n_groups_scored"] == 1


def test_consistency_excludes_the_weird_framing_but_weird_variant_includes_it():
    """Headline is the 5 paraphrase-set phrasings; weird is the stress sub-score."""
    g = _group()
    traces = [_trace(p.query_id, ["gold"]) for p in g.paraphrase_set]
    traces.append(_trace("g01:weird:0", ["totally_other"]))
    panel, _pq, _pg, _m = offline_panel([g], traces)
    assert panel["consistency"] == 1.0, "the 5 paraphrases agree perfectly"
    assert panel["consistency_weird"] < 1.0, "weird disagrees, so the stress score drops"


def test_consistency_is_zero_when_every_phrasing_picks_a_different_page():
    g = _group()
    traces = [_trace(p.query_id, [f"page{i}"]) for i, p in enumerate(g.phrasings)]
    panel, _pq, _pg, _m = offline_panel([g], traces)
    assert panel["consistency"] == 0.0


def test_error_traces_still_count_for_retrieval_metrics():
    """The pipeline wraps only the generate span, so an error trace carries the full
    retrieval. Retrieval metrics must stay at full coverage even if generation died."""
    g = _group()
    traces = [_trace(p.query_id, ["gold"], status="error") for p in g.phrasings]
    panel, _pq, _pg, meta = offline_panel([g], traces)
    assert panel["recall@k"] == 1.0
    assert meta["n_retrieval_scored"] == 6
    assert meta["n_error"] == 6


def test_missing_traces_are_counted_not_silently_dropped():
    g = _group()
    traces = [_trace(p.query_id, ["gold"]) for p in g.phrasings[:3]]
    _panel, _pq, _pg, meta = offline_panel([g], traces)
    assert meta["n_missing"] == 3
    assert meta["n_retrieval_scored"] == 3


def test_recall_at_k_can_miss_while_recall_at_cand_hits():
    """The rerank headroom: gold is in the pool but didn't survive selection."""
    g = _group()
    traces = []
    for p in g.phrasings:
        t = _trace(p.query_id, ["other"])
        t.retrieved_pages = ["other", "gold"]
        t.reranked_pages = ["other", "gold"]
        traces.append(t)
    panel, _pq, _pg, _m = offline_panel([g], traces)
    assert panel["recall@k"] == 0.0
    assert panel["recall@cand"] == 1.0
    assert panel["mrr"] == 0.5


def test_crosscheck_catches_a_broken_recall_implementation():
    from rag.eval.metrics import crosscheck
    per_query = [{"query_id": "x", "recall@k": 1, "recall@cand": 0, "rr": 0.0}]
    problems = crosscheck({"recall@cand": 0.0, "recall@k": 1.0}, per_query)
    assert problems, "selecting a page that was never retrieved must be flagged"
