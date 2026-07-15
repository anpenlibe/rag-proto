"""The offline metric panel — pure functions, no LLM, no I/O.

Locked in ARCHITECTURE §9; don't redesign silently. Everything here is deterministic and
computable from the traces alone, so these numbers cost nothing and are reproducible
from a finished run forever.

Plain floats only — never numpy. `ledger.update_manifest` json-dumps the score dict, and
a numpy float would raise *after* a full run's tokens were already spent.
"""
from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence

from .gold import Group
from .traces import QueryTrace


# -- primitives -----------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    """|A n B| / |A u B|. Two empty sets are treated as identical (both retrieved
    nothing), but in practice selection always returns top_k."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(sets: list[set]) -> float | None:
    """Mean Jaccard over every unordered pair. None if fewer than 2 sets."""
    if len(sets) < 2:
        return None
    pairs = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    return statistics.fmean(pairs)


def recall_any(found: Sequence[str], expected: Sequence[str]) -> int:
    """1 if ANY expected page was found. gold_v1 has exactly 1 expected page per group,
    so any-hit and all-hit are provably identical on it; any-hit matches the locked
    panel's wording ("*an* expected page present")."""
    return int(bool(set(found) & set(expected)))


def reciprocal_rank(ranked: Sequence[str], expected: Sequence[str]) -> float:
    """1/rank of the first expected page in a page-deduped ranking; 0 if absent."""
    exp = set(expected)
    for i, page in enumerate(ranked, start=1):
        if page in exp:
            return 1.0 / i
    return 0.0


# -- panel ----------------------------------------------------------------------------
def _consistency(traces_by_qid, phrasings, level: str) -> float | None:
    attr = "selected_pages" if level == "page" else "selected_chunks"
    sets = []
    for p in phrasings:
        t = traces_by_qid.get(p.query_id)
        if t is not None and t.has_selection:
            sets.append(set(getattr(t, attr)))
    return mean_pairwise_jaccard(sets)


def group_consistency(traces_by_qid, group: Group, level: str = "page") -> dict:
    return {
        "consistency": _consistency(traces_by_qid, group.paraphrase_set, level),
        "consistency_weird": _consistency(traces_by_qid, group.all_set, level),
    }


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _pct(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile. Small n here (240), so no interpolation games."""
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[i]


def cost_panel(traces: list[QueryTrace]) -> dict:
    """Latency + token cost, reported beside the headline (ARCHITECTURE §9).

    The mean alone hides the tail, and the tail is the interesting part: a key rotation
    or a 429 backoff shows up as a slow query, not a failed one.
    """
    lat = sorted(t.latency_ms for t in traces if t.latency_ms)
    prompt = sum(t.usage.get("prompt_tokens", 0) for t in traces)
    completion = sum(t.usage.get("completion_tokens", 0) for t in traces)
    n = len(traces)

    # Per-stage means show WHERE the time goes (generation dominates; retrieval is ~ms).
    stages: dict[str, list[float]] = {}
    for t in traces:
        for k, v in t.stage_latency_ms.items():
            stages.setdefault(k, []).append(v)

    return {
        "avg_latency_ms": round(statistics.fmean(lat), 1) if lat else None,
        "p50_latency_ms": round(_pct(lat, 0.50), 1) if lat else None,
        "p95_latency_ms": round(_pct(lat, 0.95), 1) if lat else None,
        "max_latency_ms": round(lat[-1], 1) if lat else None,
        "total_latency_ms": round(sum(lat), 1) if lat else None,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "avg_tokens_per_query": round((prompt + completion) / n, 1) if n else None,
        "stage_avg_latency_ms": {
            k: round(statistics.fmean(v), 1) for k, v in sorted(stages.items())
        },
    }


def offline_panel(groups: list[Group], traces: list[QueryTrace]):
    """Return (panel, per_query_rows, per_group_rows). No LLM involved."""
    from .traces import by_query_id
    tq = by_query_id(traces)

    per_query, per_group = [], []
    for g in groups:
        for p in g.phrasings:
            t = tq.get(p.query_id)
            if t is None:
                per_query.append({"query_id": p.query_id, "group_id": g.group_id,
                                  "kind": p.kind, "status": "missing"})
                continue
            row = {
                "query_id": p.query_id, "group_id": g.group_id, "kind": p.kind,
                "trace_id": t.trace_id, "status": t.status,
                "expected_page_ids": list(g.expected_page_ids),
                "selected_pages": t.selected_pages,
                "n_selected": len(t.selected_chunks),
                "truncated": t.truncated,
                "latency_ms": t.latency_ms,
                "total_tokens": t.usage.get("total_tokens", 0),
            }
            if t.has_selection:
                row["recall@k"] = recall_any(t.selected_pages, g.expected_page_ids)
                row["recall@cand"] = recall_any(t.retrieved_pages, g.expected_page_ids)
                # MRR over the post-rerank ranking, untruncated: measuring *how far
                # down* the gold page sits is the whole point, and truncating at top_k
                # would just restate recall@k. In E0 rerank is passthrough.
                ranked = t.reranked_pages or t.retrieved_pages
                row["rr"] = reciprocal_rank(ranked, g.expected_page_ids)
            per_query.append(row)

        gp = group_consistency(tq, g, level="page")
        gc = group_consistency(tq, g, level="chunk")
        per_group.append({
            "group_id": g.group_id, "section": g.section,
            "consistency": gp["consistency"],
            "consistency_weird": gp["consistency_weird"],
            "consistency_chunks": gc["consistency"],
            "n_phrasings_scored": sum(
                1 for p in g.phrasings
                if (t := tq.get(p.query_id)) is not None and t.has_selection),
        })

    scored = [r for r in per_query if "recall@k" in r]
    panel = {
        "consistency": _mean(r["consistency"] for r in per_group),
        "consistency_weird": _mean(r["consistency_weird"] for r in per_group),
        "consistency_chunks": _mean(r["consistency_chunks"] for r in per_group),
        "recall@k": _mean(r["recall@k"] for r in scored),
        "recall@cand": _mean(r["recall@cand"] for r in scored),
        "mrr": _mean(r["rr"] for r in scored),
    }
    meta = {
        "n_groups": len(groups),
        "n_groups_scored": sum(1 for r in per_group if r["consistency"] is not None),
        "n_queries": len(per_query),
        "n_retrieval_scored": len(scored),
        "n_missing": sum(1 for r in per_query if r["status"] == "missing"),
        "n_error": sum(1 for r in per_query if r["status"] == "error"),
        "n_truncated": sum(1 for r in per_query if r.get("truncated")),
        **cost_panel(traces),
    }
    return panel, per_query, per_group, meta


def crosscheck(panel: dict, per_query: list[dict]) -> list[str]:
    """Cheap invariants that catch a broken metric implementation.

    recall@cand is by definition the fraction of queries whose gold page appears
    anywhere in the pool — which is exactly the fraction with rr > 0. If these two
    disagree, the metric code is wrong, not the pipeline.
    """
    problems = []
    scored = [r for r in per_query if "recall@k" in r]
    if scored:
        frac_rr = statistics.fmean(1.0 if r["rr"] > 0 else 0.0 for r in scored)
        if panel["recall@cand"] is not None and abs(frac_rr - panel["recall@cand"]) > 1e-9:
            problems.append(
                f"recall@cand ({panel['recall@cand']:.4f}) != fraction with rr>0 "
                f"({frac_rr:.4f}) — metric bug")
        bad = [r["query_id"] for r in scored if r["recall@k"] > r["recall@cand"]]
        if bad:
            problems.append(f"recall@k > recall@cand for {bad[:3]} — selected a page "
                            f"that was never retrieved")
    return problems
