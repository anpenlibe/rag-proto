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
> headers — it only appears in the body of the 429). 4 keys ≈ 400k/day per model. The
> current gold set (`gold_v1_small`, 60 queries) is small enough that a **full generate +
> judge panel fits in a single day** (measured: ~142k generation + ~189k judge tokens).
> Retrieval metrics are free and instant regardless. See "Token budget" below.

## Results summary

| Exp | Date | Lever changed | Model | gold_set | recall@k | mrr | ⭐consistency | faithful | cite_acc | config_hash |
|-----|------|---------------|-------|----------|----------|-----|--------------|----------|----------|-------------|
| E0 | 2026-07-18 | baseline | llama-3.3-70b-versatile | gold_v1_small | **0.783** | **0.674** | **0.428** | 0.933 | 0.750 | `f22363afaf1d` |

> **E0 = the "before"** on `gold_v1_small` (10 groups × 6 = 60). Every panel metric is
> **complete (60/60) and fully judged** (0 judge failures, quota not exhausted). The
> `recall@k`/`mrr`/⭐`consistency` metrics are deterministic and free (selection precedes
> generation); the judged trio (`faithfulness`/`citation_acc`/`answer_agreement`) comes
> from the LLM-judge on `gpt-oss-120b`. This set is deliberately hard — half its groups are
> Study-organisation near-neighbour hard negatives — so `recall@k` runs lower than a
> section-balanced set would.

## Run ledger (auto)

Machine-written: each `Run` appends a row on close; **reset to header-only when
clearing logs**. One row per `runs/<run_id>/`. `eval_score` fills in once the harness
runs.

<!-- RUN-LEDGER:START -->
| run_id | date | label | config_hash | eval_hash | n_queries | total_tokens | avg_latency_ms | eval_score |
|--------|------|-------|-------------|-----------|-----------|--------------|----------------|------------|
| 20260718-151413-4593 | 2026-07-18 | baseline gold_v1_small (10x6, gen+judge) | `f22363afaf1d` | `6647ca36c217` | 60 | 142308 | 2664.5 | **0.428** (consistency_weird=0.404 recall@k=0.783 recall@cand=0.917 mrr=0.674 answer_agreement=0.7 faithfulness=0.933 citation_acc=0.75 n=60 avg_ms=2664 p95_ms=9674) |
<!-- RUN-LEDGER:END -->

---

## E0 — baseline, measured

- **Date:** 2026-07-18
- **Hypothesis:** the parity baseline is inconsistent across paraphrasings. This is the
  "before"; it is *supposed* to score badly on the headline.
- **Lever changed:** none — the reference run.
- **Config:** `config_hash=f22363afaf1d` · `index_hash=2747344b6db6` ·
  `eval_hash=6647ca36c217`. 720 chunks · `candidate_k`=20 → passthrough rerank →
  `top_k`=6 · `llama-3.3-70b-versatile` temp 0 · prompt `v1` · `judge_sample=all`.
- **Gold set:** `gold_v1_small` — 10 groups × 6 phrasings = 60.
- **Run:** `20260718-151413-4593` (generate + judge, 60/60, 142k pipeline + 189k judge
  tokens; 0 errors, 0 judge failures).

**Results (complete panel)**

| recall@k | recall@cand | mrr | ⭐consistency | cons. weird | answer_agree | faithful | cite_acc |
|---|---|---|---|---|---|---|---|
| 0.783 | 0.917 | 0.674 | **0.428** | 0.404 | 0.700 | 0.933 | 0.750 |

**Cost / speed:** 2372 tokens/query · avg 2.66 s/query (p50 0.60 s, p95 9.7 s) ·
generation is >99% of latency (retrieval 17 ms). Fully judged: 60/60, 0 failures.

**Observations**

1. **⭐ consistency 0.428 with recall@k 0.783.** The gold page reaches the prompt ~78% of
   the time, yet the *selected page-set* overlaps only ~43% across rephrasings — the
   instability is the other slots churning, not the answer being unfindable.
2. **The paraphrase problem is mostly retrieval-side.** `answer_agreement` 0.700 >
   `consistency` 0.428: the context churns, but because gold often survives, the answers
   agree more than their contexts do. A retrieval lever should move `consistency` most.
3. **Rerank headroom = recall@cand − recall@k = +0.134.** A perfect reranker could buy at
   most ~13 pts of recall@k — real, bounded, measurable *before* building it.
4. **`citation_acc` 0.750 ≪ `faithfulness` 0.933.** The model states supported facts but
   attaches the wrong `[n]` — a citation-mapping problem, not a grounding one.
5. **This set is deliberately hard.** 5 of 10 groups are Study-organisation near-neighbour
   hard negatives (minimum-credits vs prüfungsaktiv; the three registration pages), so
   `recall@k` here (0.783) runs below what a section-balanced set would show — by design.

**Decision:** E0 is the reference. Next lever = **multi-query + RRF** (attacks the page-set
churn `consistency` measures) or **cross-encoder rerank** (bounded upside +0.134 recall).
One at a time.

### Token budget (free tier) — plan around this

`100k tokens/day/key/model`, 4 keys ⇒ **~400k/day per model**. Invisible in headers; only
in the 429 body. Judge draws on a *separate* model's quota (a reason to keep
`judge_model != model`).

| Job (`gold_v1_small`, 60 queries) | Cost | Fits in a day? |
|---|---|---|
| Retrieval panel (`--retrieve-only`) | **0** | yes — seconds |
| Generate 60 answers | ~142k | yes |
| Judge 60 answers + 10 groups | ~189k | yes (separate model quota) |

A **full generate + judge panel on this set fits in one day** — measured above. (The
original 40-group set could not: ~508k generation + ~800k judge spanned days, which is why
it was distilled to these 10 groups.)

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
