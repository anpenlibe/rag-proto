# rag-proto — Experiment Log

Every run is recorded here. An experiment = **one config diff** (± a new component
impl) against the architecture in `ARCHITECTURE.md`. Each entry must reference the
exact `config_hash`, the change made, and the resulting scores from the fixed metric
panel — so any number is reproducible and attributable.

**Ground rules**
- Change **one lever at a time** (chunker / retriever / reranker / model …) so a
  score delta has a single cause.
- Pin the `gold_set` version and keep `temperature=0`, fixed `k`.
- **Fixed metric panel** (headline = ⭐`consistency`):
  - `recall@k`, `mrr` — is an expected page in the selected/retrieved set.
  - ⭐`consistency` — per group, Jaccard overlap of the *selected chunk/page sets*
    across the 6 phrasings; reported paraphrase-only AND including the weird framing.
  - `answer_agreement`, `faithfulness`, `citation_acc` — via an **LLM-judge on a
    separate large-context model** (`openai/gpt-oss-120b`), run through the same
    key-cycling pool.
- Also tracked per run alongside the score: **total tokens (cost)** and **latency**.
- Log `model` + `key_id` per call; report the model(s) used in the run.

> ⚠️ **The free tier caps each key at 100k tokens/DAY/model** (invisible in the response
> headers — it only appears in the body of the 429). 4 keys ≈ 400k/day per model, while
> a generated 240-query pass costs ~508k and judging it ~800k. **Retrieval metrics are
> free and complete; judged metrics are budget-bound.** See "Token budget" below.

## Results summary

| Exp | Date | Lever changed | Model | gold_set | recall@k | mrr | ⭐consistency | faithful | cite_acc | config_hash |
|-----|------|---------------|-------|----------|----------|-----|--------------|----------|----------|-------------|
| E0 | 2026-07-15 | baseline | llama-3.3-70b-versatile | gold_v1 | **0.875** | **0.757** | **0.427** | 0.986 † | 0.819 † | `f22363afaf1d` |

> **E0 = the "before".** `recall@k`/`mrr`/⭐`consistency` are **complete (240/240)**,
> deterministic and free — selection happens before generation, so they need no LLM.
> † `faithfulness`/`citation_acc`/`answer_agreement` are from a **partial, section-biased
> fragment** (24/40 groups = 2 of 8 sections; `judge_sample=c+2p`, 72 verdicts) because
> the daily token quota ran out mid-generation. **Do not compare them to a future run's
> judged panel** until the full panel exists — on this same fragment `consistency` reads
> 0.465 vs the true 0.427, so the bias is real and material.

## Run ledger (auto)

Machine-written: each `Run` appends a row on close; **reset to header-only when
clearing logs**. One row per `runs/<run_id>/`. `eval_score` fills in once the harness
runs.

<!-- RUN-LEDGER:START -->
| run_id | date | label | config_hash | eval_hash | n_queries | total_tokens | avg_latency_ms | eval_score |
|--------|------|-------|-------------|-----------|-----------|--------------|----------------|------------|
| 20260715-195821-911f | 2026-07-15 | E0 retrieval panel (240/240, 0 tokens) | `f22363afaf1d` | `4a7e973b7ae9` | 240 | 0 | 7.7 | **0.427** (consistency_weird=0.399 recall@k=0.875 recall@cand=0.967 mrr=0.757 n=240 avg_ms=8 p95_ms=10) |
| 20260715-202245-2583 | 2026-07-15 | E0 retrieval panel (Docker Qdrant parity) | `f22363afaf1d` | `4a7e973b7ae9` | 240 | 0 | 11.0 | **0.427** (consistency_weird=0.399 recall@k=0.875 recall@cand=0.967 mrr=0.757 n=240 avg_ms=11 p95_ms=15) |
<!-- RUN-LEDGER:END -->

---

## E0 — baseline, measured

- **Date:** 2026-07-15
- **Hypothesis:** the parity baseline is inconsistent across paraphrasings. This is the
  "before"; it is *supposed* to score badly on the headline.
- **Lever changed:** none — first measured run. (Pre-run integrity fixes changed E0's
  identity: `config_hash 44f361c80b32 → f22363afaf1d`. See "Why the hash moved".)
- **Config:** `config_hash=f22363afaf1d` · `index_hash=2747344b6db6` ·
  `eval_hash=4a7e973b7ae9` (retrieval, `judge_sample=all`) / `cdeb9320556a` (`c+2p`).
  720 chunks · `candidate_k`=20 → passthrough rerank → `top_k`=6 ·
  `llama-3.3-70b-versatile` temp 0 · prompt `v1`.
- **Gold set:** `gold_v1` — 40 groups × 6 phrasings = 240.
- **Runs:** `20260715-195821-911f` (retrieval, 240/240, 0 tokens) ·
  `20260715-193039-cc9a` (generation fragment, 145 queries, 305k tokens, TPD-capped).

**Results**

| | recall@k | recall@cand | mrr | ⭐consistency | cons. weird | cons. chunk |
|---|---|---|---|---|---|---|
| **240/240, complete** | **0.875** | 0.967 | **0.757** | **0.427** | 0.399 | 0.310 |

| | answer_agreement | faithfulness | citation_acc |
|---|---|---|---|
| **fragment only †** | 0.868 | 0.986 | 0.819 |

† 24/40 groups (Study organisation + Admission only — 2 of 8 sections), `judge_sample=c+2p`,
72/72 usable verdicts. **Section-biased: not E0's judged panel.** Same fragment reads
`consistency`=0.465 vs the true 0.427.

**Cost / speed:** 2102 tokens/query · avg 3.9 s/query (p50 1.98 s, p95 8.8 s, max 85 s) ·
generation is >99% of latency (retrieval is 9 ms). Retrieval panel: 0 tokens, 8 ms/query.

**Observations**

1. **⭐ consistency 0.427 with recall@k 0.875.** The gold page reaches the prompt ~88% of
   the time, yet the *selected page-set* overlaps only ~43% across rephrasings — the
   instability is the other 5 of 6 slots churning, not the answer being unfindable.
2. **The paraphrase problem is mostly retrieval-side, and does not fully propagate.**
   `answer_agreement` 0.868 ≫ `consistency` 0.465 (same fragment): the context churns,
   but because gold usually survives, the answers still mostly agree. A retrieval lever
   should move `consistency` a lot and `answer_agreement` less.
3. **Rerank headroom = recall@cand − recall@k = +0.092.** A perfect reranker could buy
   at most ~9 pts of recall@k — real, bounded, and measurable *before* building it.
4. **`citation_acc` 0.819 ≪ `faithfulness` 0.986.** The model states supported facts but
   attaches the wrong `[n]` — a citation-mapping problem, not a grounding one.
5. **Judge noise is real.** Two judge passes over *identical* traces at temp 0 gave
   `citation_acc` 0.529 → 0.444 (one verdict flipped). Calibrate before trusting a small
   judged delta; `consistency` is immune (offline, deterministic — 3 runs gave 0.427).

**Decision:** E0 is the reference. Next lever = **cross-encoder rerank** (bounded upside
already measured at +0.092 recall) or **multi-query+RRF** (attacks the page-set churn
that `consistency` is actually measuring). One at a time.

### Why the hash moved (`44f361c80b32` → `f22363afaf1d`)

Not a pipeline change in spirit, but a real one in fact. Before any number was trusted:
- `config_hash` scoped (eval knobs no longer re-stamp pipeline identity) → new value.
- **Chunker v2** (`chunker_version` now in `index_hash`): common-prefix `heading_path`
  fixed 195 mis-anchored vectors (26.6% of the index); tail-window dedup removed 13
  duplicate chunks. **733 → 720 chunks.** The old E0 was measuring a partly broken index,
  so `44f361c80b32` has no comparable numbers — nothing is lost.

### Token budget (free tier) — plan around this

`100k tokens/day/key/model`, 4 keys ⇒ **~400k/day per model**. Invisible in headers; only
in the 429 body. Judge draws on a *separate* model's quota (a reason to keep
`judge_model != model`).

| Job | Cost | Fits in a day? |
|---|---|---|
| Retrieval panel, 240 | **0** | yes — seconds |
| Generate 240 answers | ~508k | **no** (~189/day max) |
| Judge 240 + 40 groups | ~800k | **no** |
| Judge `c+2p` (120) + 40 | ~490k | ~1 day |

**To finish E0's judged panel** (generation spans days by necessity; each day is its own
run, merged at scoring time — runs are never reopened):

```bash
# day 2: generate the groups the quota cut off
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --groups g25,g26,...,g40 --label "E0 gen fragment 2/2"
# then score both fragments as one population
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --score 20260715-193039-cc9a,<new_run_id> --judge-sample c+2p
```

---

## Template (copy per experiment)

### E<N> — <short title>
- **Date:** YYYY-MM-DD
- **Hypothesis:** what we expect this change to improve and why.
- **Lever changed (vs previous):** e.g. `retriever: dense → hybrid(dense+bm25)`.
- **Config:** `config_hash=…`; key non-default values (chunk size, top_k, query mode,
  reranker, model, prompt_version, temperature).
- **Models / keys:** model(s) trialled; key-pool size; any rate-limit events.
- **Gold set:** `gold_vX` (N questions / M paraphrase groups).
- **Results:**
  | recall@k | mrr | ⭐consistency | faithfulness | citation_acc |
  |----------|-----|--------------|--------------|--------------|
  |          |     |              |              |              |
- **Δ vs baseline:** which metrics moved, by how much.
- **Observations:** failures inspected via traces (trace_ids), surprises.
- **Decision:** keep / revert / needs follow-up. Next lever to try.

---

## Baseline definition (E0) — `config_hash f22363afaf1d`

The reference all deltas are measured against — see `ARCHITECTURE.md` for full spec:
single dense query (identity transform) · contextual heading-path chunks (~230 w) ·
bge-small · Qdrant local · retrieve `candidate_k`=20 → passthrough rerank → select
`top_k`=6 · Groq `llama-3.3-70b-versatile`, temp 0 · grounded cite-only prompt `v1` ·
per-run full-pipeline tracing.
