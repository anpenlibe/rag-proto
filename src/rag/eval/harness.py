"""The scoring harness: run the gold set, score the run, write the score back.

    python -m rag.eval.harness --retrieve-only        # headline metric, 0 tokens
    python -m rag.eval.harness --limit 3 --label smoke # cheap end-to-end check
    python -m rag.eval.harness --label E0              # the real thing: generate + score
    python -m rag.eval.harness --score <run_id>        # re-score an existing run

Phase A (generate) runs the 240 gold phrasings through the pipeline as ONE
`Run(kind="eval")`, so every trace lands in one folder and the ledger gets one row.
Phase B (score) reads those traces back off disk, so it can be re-run without spending
the generation budget again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

from .. import ledger
from ..config import RUNS_DIR, Config
from ..llm import KeyPool
from ..pipeline import RagPipeline
from ..trace import Run
from . import gold as gold_mod
from . import metrics as metrics_mod
from .traces import load_run_traces, load_runs

# Keys that go in the ledger cell. The rest (per-query rows, judge output, counts) live
# in runs/<id>/eval/ — the cell must stay readable.
_LEDGER_KEYS = ("consistency", "consistency_weird", "recall@k", "recall@cand", "mrr",
                "answer_agreement", "faithfulness", "citation_acc")


def eval_dir(run_id: str):
    d = RUNS_DIR / run_id / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -- phase A: generate ----------------------------------------------------------------
def run_generation(cfg: Config, groups: list[gold_mod.Group], *, generate: bool = True,
                   label: str = "E0", pool: KeyPool | None = None) -> str:
    phrasings = gold_mod.phrasings_of(groups)
    # A complete pass over the gold set is a measured eval and earns a ledger row —
    # including a retrieval-only one, which produces the headline metric (`consistency`
    # is decided at selection, before generation). Partial/sampled passes are `batch`:
    # they must never look like a comparable result.
    full = len(groups) == len(gold_mod.load_gold(cfg.gold_set))
    kind = "eval" if full else "batch"
    run = Run(cfg, label=label, kind=kind)
    pipe = RagPipeline(cfg, run=run, pool=pool)

    print(f"run {run.run_id}  kind={kind}  {len(groups)} groups / {len(phrasings)} queries")
    print(f"config_hash={cfg.config_hash}  index_hash={cfg.index_hash}  "
          f"eval_hash={cfg.eval_hash}")
    if generate:
        print(f"model={cfg.model}  ~{len(phrasings) * 1700 // 1000}k tokens expected")

    t0 = time.perf_counter()
    errors = 0
    try:
        for i, p in enumerate(phrasings, 1):
            try:
                # One failure must not kill the run: answer() re-raises after emitting
                # an error trace, and that trace still carries the full retrieval — so
                # the query still counts toward the retrieval metrics.
                pipe.answer(p.text, generate=generate, query_id=p.query_id)
            except Exception as e:
                errors += 1
                print(f"  ! {p.query_id}: {type(e).__name__}: {str(e)[:100]}",
                      file=sys.stderr)
            if i % 20 == 0 or i == len(phrasings):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(phrasings)}  {el:5.1f}s  errors={errors}")
    finally:
        run.close()

    print(f"-> {run.dir}  ({errors} errors)")
    return run.run_id


# -- phase B: score -------------------------------------------------------------------
def _assert_same_pipeline(run_ids: list[str]) -> None:
    """Merging runs from different pipelines would average two different systems."""
    seen = {}
    for rid in run_ids:
        try:
            m = json.loads((RUNS_DIR / rid / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        seen.setdefault(m.get("config_hash"), []).append(rid)
    if len(seen) > 1:
        raise SystemExit(
            "refusing to merge runs from different pipelines:\n" +
            "\n".join(f"  config_hash={h}: {', '.join(r)}" for h, r in seen.items()))


def run_scoring(cfg: Config, run_id: str, *, judge: bool = True,
                pool: KeyPool | None = None, extra_runs: list[str] | None = None) -> dict:
    groups = gold_mod.load_gold(cfg.gold_set)
    run_ids = [run_id] + list(extra_runs or [])
    if len(run_ids) > 1:
        _assert_same_pipeline(run_ids)
        traces = load_runs(run_ids)
        print(f"merged {len(traces)} traces from {len(run_ids)} runs")
    else:
        traces = load_run_traces(run_id)

    # Score only the groups this run actually covered (--limit runs cover a subset).
    seen = {t.query_id for t in traces if t.query_id}
    groups = [g for g in groups if any(p.query_id in seen for p in g.phrasings)]
    if not groups:
        raise SystemExit(
            f"run {run_id} has no traces carrying a gold query_id — was it generated by "
            f"this harness? (ad-hoc runs can't be scored)")

    panel, per_query, per_group, meta = metrics_mod.offline_panel(groups, traces)

    for problem in metrics_mod.crosscheck(panel, per_query):
        print(f"  !! CROSSCHECK: {problem}", file=sys.stderr)

    if judge:
        from .judge import judge_run
        jpanel, jmeta = judge_run(cfg, groups, traces, eval_dir(run_id), pool=pool)
        panel.update(jpanel)
        meta.update(jmeta)

    scores = {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in panel.items() if v is not None}

    d = eval_dir(run_id)
    (d / "scores.json").write_text(json.dumps(
        {"run_id": run_id, "gold_set": cfg.gold_set, "eval_hash": cfg.eval_hash,
         "config_hash": cfg.config_hash, "top_k": cfg.top_k,
         "candidate_k": cfg.candidate_k, "judge_model": cfg.judge_model if judge else None,
         "panel": panel, "meta": meta},
        indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(d / "per_query.jsonl", per_query)
    _write_jsonl(d / "per_group.jsonl", per_group)

    _report(panel, meta)

    cell = {k: scores[k] for k in _LEDGER_KEYS if k in scores}
    cell["n"] = meta["n_queries"]
    # Speed rides along with quality; the row's own avg_latency_ms column comes from the
    # Run's aggregate, this is the scored-population figure.
    if meta.get("avg_latency_ms") is not None:
        cell["avg_ms"] = round(meta["avg_latency_ms"])
        cell["p95_ms"] = round(meta["p95_latency_ms"])
    # Only eval-kind runs own a ledger row; patching the manifest is always right, but
    # asking the ledger to patch a row a batch/smoke run never wrote would warn falsely.
    if _run_kind(run_id) == "eval":
        ledger.update_eval_score(run_id, cell)
        print(f"\n-> scores written to {d}/scores.json and the EXPERIMENTS.md ledger")
    else:
        ledger.update_manifest(run_id, cell)
        print(f"\n-> scores written to {d}/scores.json (no ledger row: not an eval run)")
    return panel


def _run_kind(run_id: str) -> str:
    try:
        m = json.loads((RUNS_DIR / run_id / "manifest.json").read_text(encoding="utf-8"))
        return m.get("kind", "")
    except (OSError, json.JSONDecodeError):
        return ""


def _report(panel: dict, meta: dict):
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"
    print("\n" + "=" * 58)
    print(f"  {'⭐ consistency (paraphrase)':<34} {fmt(panel.get('consistency'))}")
    print(f"  {'consistency (incl. weird)':<34} {fmt(panel.get('consistency_weird'))}")
    print(f"  {'consistency (chunk-level)':<34} {fmt(panel.get('consistency_chunks'))}")
    print("  " + "-" * 54)
    print(f"  {'recall@k (selected)':<34} {fmt(panel.get('recall@k'))}")
    print(f"  {'recall@cand (retrieved pool)':<34} {fmt(panel.get('recall@cand'))}")
    print(f"  {'mrr':<34} {fmt(panel.get('mrr'))}")
    if panel.get("recall@k") is not None and panel.get("recall@cand") is not None:
        head = panel["recall@cand"] - panel["recall@k"]
        print(f"  {'  -> rerank headroom':<34} {head:+.3f}")
    for k in ("answer_agreement", "faithfulness", "citation_acc"):
        if k in panel:
            print(f"  {k:<34} {fmt(panel.get(k))}")

    # Cost + speed sit beside quality: a lever that buys +0.05 consistency for 3x the
    # latency is a different decision than one that's free (ARCHITECTURE §9).
    def ms(v):
        return "—" if v is None else f"{v:,.0f} ms"
    print("  " + "-" * 54)
    print(f"  {'avg latency / query':<34} {ms(meta.get('avg_latency_ms'))}")
    print(f"  {'  p50 / p95 / max':<34} "
          f"{ms(meta.get('p50_latency_ms'))} / {ms(meta.get('p95_latency_ms'))} / "
          f"{ms(meta.get('max_latency_ms'))}")
    if meta.get("total_latency_ms"):
        print(f"  {'  wall time (sum)':<34} {meta['total_latency_ms'] / 1000:,.1f} s")
    stages = meta.get("stage_avg_latency_ms") or {}
    if stages:
        order = ["query_transform", "retrieve", "rerank", "select", "assemble", "generate"]
        parts = [f"{k.split('_')[0]} {stages[k]:.0f}" for k in order if k in stages]
        print(f"  {'  avg by stage (ms)':<34} {' · '.join(parts)}")
    print(f"  {'total tokens (cost)':<34} {meta.get('total_tokens', 0):,}"
          f"  (avg {meta.get('avg_tokens_per_query') or 0:.0f}/query)")

    print("=" * 58)
    print(f"  groups scored {meta['n_groups_scored']}/{meta['n_groups']} · "
          f"queries {meta['n_retrieval_scored']}/{meta['n_queries']} · "
          f"errors {meta['n_error']} · missing {meta['n_missing']} · "
          f"truncated {meta['n_truncated']}")
    if meta.get("n_judged") is not None:
        print(f"  judged {meta['n_judged']}/{meta.get('n_judgeable', '?')}"
              f"  (unusable verdicts: {meta.get('n_judge_failed', 0)})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag.eval.harness", description=__doc__.split("\n")[0])
    ap.add_argument("--retrieve-only", action="store_true",
                    help="skip generation: the headline metric for zero tokens")
    ap.add_argument("--score", metavar="RUN_ID[,RUN_ID...]",
                    help="score existing run(s); several are merged (a generated panel "
                         "spans days on the free tier, so it spans runs)")
    ap.add_argument("--no-judge", action="store_true", help="offline metrics only")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="N gold groups (all 6 phrasings each), seeded sample")
    ap.add_argument("--groups", metavar="g01,g02",
                    help="generate exactly these groups (resume a budget-capped panel)")
    ap.add_argument("--judge-sample", choices=sorted(gold_mod.JUDGE_SAMPLES),
                    help="which phrasings get per-answer judging (default: config)")
    ap.add_argument("--gold-set", default=None,
                    help="which gold set to score, e.g. gold_v1_small (default: config's gold_v1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="E0")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.judge_sample:
        cfg = replace(cfg, judge_sample=args.judge_sample)   # a config diff, not an edit
    if args.gold_set:
        cfg = replace(cfg, gold_set=args.gold_set)           # moves eval_hash, not config_hash

    if args.score:
        ids = [r.strip() for r in args.score.split(",") if r.strip()]
        run_scoring(cfg, ids[0], judge=not args.no_judge, extra_runs=ids[1:])
        return

    groups = gold_mod.load_gold(cfg.gold_set)
    if args.groups:
        want = {g.strip() for g in args.groups.split(",") if g.strip()}
        groups = [g for g in groups if g.group_id in want]
        missing = want - {g.group_id for g in groups}
        if missing:
            raise SystemExit(f"unknown group ids: {sorted(missing)}")
    groups = gold_mod.select_groups(groups, args.limit, args.seed)
    generate = not args.retrieve_only
    # One pool shared by the generator and the judge: separate pools would round-robin
    # independently and double the 429s on the same key budget.
    pool = KeyPool() if generate else None
    run_id = run_generation(cfg, groups, generate=generate, label=args.label, pool=pool)
    run_scoring(cfg, run_id, judge=generate and not args.no_judge, pool=pool)


if __name__ == "__main__":
    main()
