# rag-proto — Architecture & Design Decisions

RAG prototype for a voice-agent company (onboarding assignment). Query/retrieval
over a knowledge base. This doc is the design record; build follows it.

## Problems the system must address

1. **Paraphrase robustness** — the same question, reframed, must give consistent &
   accurate answers. (Headline metric.)
2. **Traceability** — every hop of the flow must be inspectable/loggable.
3. **Citations** — the final answer must cite its sources.

Approach: ship a **strict-parity baseline** first, measure it, then turn on
**iteration levers** one at a time so each improvement is attributable.

## Architecture

```
                ┌──────────── TRACE LAYER (wraps every stage · one trace_id) ────────────┐
                │                                                                          │
 pages.jsonl ─►[1 CHUNK]─►[2 EMBED]─►[3 INDEX: Qdrant]                                     │
                                                                                          │
 user query ─►[4 QUERY]─►[5 RETRIEVE]─►[6 ASSEMBLE]─►[7 GENERATE: Groq]─► answer+citations │
                                                                                          │
                └──────────────── all stages scored by [8 EVAL] (separate run) ───────────┘
```

Requirement → where it lives:

| Goal | Addressed in |
|------|--------------|
| Paraphrase robustness | chunk context (1), query proc (4), retrieval hybrid/rerank (5), generation determinism (7) |
| Traceability | cross-cutting trace layer |
| Citations | chunk metadata (1/3) → prompt contract (7) → post-answer validation (7) |

## Planned layout

```
data/            corpus (pages.jsonl canonical) + pages/*.md mirror
scripts/         scrape.py
docs/            this doc
src/rag/         chunk.py embed.py index.py retrieve.py generate.py trace.py pipeline.py config.py
.venv/           Python 3.12
```

## Modularity & configuration

Every layer is a swappable component behind a small interface, wired by a single
`Config`. **An experiment = a config diff (± a new component impl), never a rewrite.**

| Component | Interface (informal) | Baseline impl | Swap-in variants |
|-----------|----------------------|---------------|------------------|
| Chunker | `chunk(pages) -> chunks` | heading-aware ~320 tok | fixed-size, semantic, no-prepend |
| Embedder | `embed(texts)`, `embed_query(text)` | bge-small (fastembed) | bge-base, e5, OpenAI 3-small |
| VectorStore | `upsert(points)`, `search(vec,k,filter)` | Qdrant local path | Qdrant Docker; +sparse |
| QueryTransform | `transform(q) -> [q...]` | identity | multi-query, HyDE |
| Retriever | `retrieve(q,k) -> candidates` | dense | hybrid dense+BM25 |
| Reranker | `rerank(q,cands) -> cands` | none (passthrough) | bge-reranker cross-encoder |
| ContextBuilder | `build(cands) -> prompt_ctx` | numbered top-k | dedupe, budget variants |
| **LLMProvider** | `chat(client,model,messages)`, `list_models()` | `GroqProvider` | OpenAI / local / any |
| **KeyPool** | `complete(model, messages) -> LLMResponse` | N keys, round-robin, rotate-on-429 | shared by generator + judge + expansion |
| Prompts | `system_prompt(version)` | `v1` cite-only contract | `v2`… (never edit a used version) |
| Generator | `generate(q,ctx,sources) -> Answer` | KeyPool + parse/validate `[n]` | any model (config) |
| Tracer | `Run` + `span(stage)` | per-run folders, full trace | Phoenix/Langfuse later |
| Ledger | `update_eval_score(run_id, scores)` | patches manifest + EXPERIMENTS row | — |

**Config is one flat dataclass; the *hash* is scoped into three** (2026-07-15). Every
field belongs to exactly one bucket, enforced by a module-level assert at import — with
explicit buckets, a new unbucketed field would silently affect nothing, which is worse
than hashing everything:

| Hash | Covers | Answers |
|------|--------|---------|
| `index_hash` | `chunker_version`, `target_words`, `overlap_words`, `min_chunk_words`, `embedder_id` | which on-disk index is this? |
| `config_hash` | **`index_hash`** + `query_instruction`, `candidate_k`, `top_k`, `query_mode`, `retriever_mode`, `rerank`, `model`, `prompt_version`, `temperature`, `max_tokens` | which pipeline produced this answer? |
| `eval_hash` | `gold_set`, `judge_model`, `judge_prompt_version`, `judge_max_tokens` | how was it scored? |

`collection` and `qdrant_url` are in none — they're physical *locations*, not identity
(the live name is `physical_collection` = `collection_<index_hash>`; the same vectors are
the same vectors in a local file or a Docker server — verified: both score `consistency`
0.426785). **`config_hash` nests `index_hash`**,
so two runs sharing a `config_hash` provably share an index — which is why the ledger
carries `config_hash` + `eval_hash` and needs no `index_hash` column. Two runs are
comparable iff `config_hash` matches; their *scores* are comparable iff `eval_hash`
matches too. `config_hash` + `trace_id` make any result reproducible.

## Pipeline — stages & decisions

Legend: **[B]** baseline (build now) · **[I]** iteration lever (later, measured).

### 1. Chunking  `pages.jsonl → chunks.jsonl`
- **[B]** `src/rag/chunk.py` is the **single chunker**, driven by `Config` (so chunk
  params are in `config_hash`). `scripts/scrape.py` produces **only** `pages.jsonl` +
  `pages/*.md` + `manifest.json` — it must never write `chunks.jsonl`. It used to carry
  a second, divergent chunker (350/60 vs Config's 230/40, different schema): re-scraping
  silently overwrote the chunks while traces still claimed the Config values, and
  indexing succeeded without error. Removed 2026-07-15 — don't reintroduce it.
- **[B]** Heading-aware: split on the existing `##`/`###` structure (corpus keeps it,
  and provides a pre-parsed `headings` outline).
- **[B]** Target **~320 tokens (~230 words)**. Merge tiny sections up, window-split
  large ones. Hard cap: stay under bge's **512-token** limit. Target is a **config
  knob** — re-chunking is cheap.
- **[B]** **`overlap_words` applies to the window-split path ONLY** (blocks bigger than
  `target_words`). The greedy packing path merges whole heading-blocks with no overlap —
  it respects semantic boundaries, and overlapping there would duplicate text. (The docs
  claimed "~50-word overlap" generally until 2026-07-15; that was never true of packing.)
  Consequence worth knowing: `overlap_words` is in `index_hash`, so changing it produces
  a new hash and a nearly identical index — you'd re-chunk, re-index, re-run and measure
  noise. Making packing overlap is a **lever**, not a fix.
- **[B]** Thin hub pages (~130 w) → single chunk.
- **[B]** **Heading-path prepend** (`title › H2 › H3: <text>`) into the embedded text.
  Mitigates small-chunk context loss; also serves as citation anchor + filter field.
- **[B]** A packed chunk's `heading_path` is the **common prefix** of its blocks' paths
  (`[A,B]+[A,C] → [A]`), not the first block's. The path is prepended into `embed_text`,
  so the old first-block rule mis-anchored the **vector** of every multi-block chunk
  (195/733 = 26.6%; ~84% of `top_k`=6 prompts carried at least one) and cited merged
  text under a sibling heading it never came from. `heading_paths` keeps the full list.
  Fixed 2026-07-15 (`chunker_version=v2`).
- **[B]** The window-split loop **stops once a window reaches the end** — it used to emit
  a final window wholly contained in its predecessor (13 duplicate chunks in the live
  index), burning `top_k` slots on identical text.
- Chunk payload: `text`, `url`, `title`, `section`, `breadcrumb`, `heading_path`,
  `heading_paths`, `page_id`, `chunk_index`. Stable ID = hash(`page_id`+`chunk_index`).
- **`data/chunks.meta.json`** (sidecar, written **after** `chunks.jsonl`) attests the
  `index_hash` + count that built the file; `rag.index` refuses to index if it disagrees.
  Together with the hash-named collection this closes both staleness edges — see §3.

### 2. Embedding
- **[B]** `BAAI/bge-small-en-v1.5` (384-d, cosine) via **fastembed** (ONNX/CPU, no torch).
  Local free stand-in for the company's OpenAI `text-embedding-3-small`.
- **[B]** **Asymmetric**: query gets bge instruction prefix, passages do not
  (`query_embed()` vs `embed()`). Getting this wrong silently degrades retrieval.

### 3. Indexing (Qdrant)
- **[B]** One collection, `size=384, distance=Cosine`. Full chunk payload stored
  (citation + trace substrate, no second lookup). Payload indexes on `section`,`language`.
- **[B]** **The collection name carries `index_hash`** (`univie_studying_<index_hash>`).
  Nothing used to bind the on-disk index to the config that built it: change
  `target_words`, forget to re-index, and the eval silently scored the **old** index
  while the ledger claimed the new hash — fabricated results, undetectable after the
  fact. The name makes that structurally impossible rather than a rule to remember.
  The two staleness edges need two guards, and neither is redundant:

  | Mistake | Caught by |
  |---|---|
  | re-chunk, forget to re-index | collection name (`..._<new>` missing → loud error) |
  | forget to re-chunk, **do** re-index | **sidecar only** (name right, contents stale) |
  | forget both | collection name |

- **[B]** **Run mode: Docker server** (`docker-compose.yml`; `QDRANT_URL=http://localhost:6333`)
  — done 2026-07-15. The local persistent path remains the zero-infra fallback (unset
  `QDRANT_URL`), but it takes an **exclusive file lock**: one process at a time, so a
  frontend holding it locks out every index/harness run. `get_client(cfg)` picks the
  backend; `qdrant_url` is `_UNHASHED` (a location, not identity) and both backends were
  verified bit-identical. Payload indexes only actually work on the server.

### 4. Query processing
- **[B]** Single query, embedded with bge query prefix. (Deliberately the flaky
  baseline — needed as the "before" measurement.)
- **[I]** Multi-query / HyDE expansion via Groq → retrieve per variant → **RRF fuse**.
  Primary lever for paraphrase robustness.

### 5. Retrieval
- **[B]** Dense vector search, **top-k 5–8**, no filter. Returns payload + score.
- **[I]** **Hybrid** dense+sparse/BM25 (Qdrant native) for jargon/exact tokens
  (STEOP, ECTS, €). **[I]** **Cross-encoder rerank** (`bge-reranker-base`) over top-20
  → top-5; usually the biggest single consistency win.

### 6. Context assembly
- **[B]** Number chunks `[1..k]`, each mapped to url/title/heading_path; include
  heading path in-context; optional per-page dedupe; keep to top 5–8.

### 7. Generation (Groq)
- **[B]** Model is a **config value, not hardcoded** — we A/B multiple Groq models.
  Default `llama-3.3-70b-versatile` (behavioral stand-in for the company's **GPT-4.1**),
  `llama-3.1-8b-instant` for voice-latency tests, plus whatever else we trial. Exact
  IDs verified vs Groq catalog at build time.
- **[B]** **Key pool — 4 API keys, round-robin, rotate-on-429** (`rag/llm.py`). On a
  429 it **tries every other key before sleeping** (a different key is usually
  healthy), and only backs off once a full cycle is exhausted — defaults ride out a
  per-minute window (~31 s across 6 cycles). Keys from `.env` (`GROQ_API_KEYS=…` or
  `k1..k4`); **never logged**. Each call logs `model` + `key_id` + rotation events.
- **[B]** **Provider seam** (`LLMProvider` / `GroqProvider` in `rag/llm.py`): every LLM
  caller — generator, LLM-judge, future query-expansion — shares one pool + provider,
  so key cycling is implemented once. Prompts live in `rag/prompts.py`, versioned.
- **[B]** **Prompt contract**: answer *only* from numbered sources; cite `[n]` for
  claims used; if the answer is absent, say "I don't know" and cite nothing →
  grounding kills hallucination & converges paraphrases.
- **[B]** **`temperature = 0`** → removes generation-side nondeterminism.
- **[B]** **Citation validation**: every `[n]` exists & was retrieved; resolve to
  source list (title+url).

### 8. Trace layer  (requirement #2) — per-run, full-pipeline
- **[B]** A **`Run`** owns `runs/<run_id>/` (`run_id = YYYYMMDD-HHMMSS-<4hex>`) and
  aggregates **latency + token cost** across its queries. A **`Tracer`** writes one
  full record per query into it. Files:
  `manifest.json` (run meta + aggregates + `eval_score` slot), `queries/<trace_id>.json`
  (full trace), `queries.jsonl` (compact list view); plus `runs/index.jsonl` enumerating
  all runs. Homegrown, zero external calls, replayable, OTel-ish field names.
- **[B]** Full per-query record captures the **entire pipeline** (the frontend replays
  it): raw query → transformed queries → **retrieved pool** (full chunk text) →
  **reranked** (scores) → **selected** chunks → **exact prompt** (system+user) → Groq
  `model`/`key_id`/`temperature`/`usage` → **rate-limit rotation events** → **raw
  response** → parsed answer + resolved citations → per-stage latency.
- **[B]** **Run `kind`** separates modes: `adhoc` (custom single query — folder + index,
  no ledger), `eval` (scored gold-set pass — folder + index + **EXPERIMENTS.md ledger
  row**), `batch`. Same mechanism, different downstream handling.
- **[B]** Every trace carries `status` — **`error` traces are emitted too** (LLM
  failure records `error`/`error_type` then re-raises). A failed query is the one you
  most want traced; losing it would defeat requirement #2.
- **[B]** `sources` is **str-keyed** (`"1"`, `"2"`…), not int: JSON object keys are
  always strings, so an int-keyed map silently changes shape on disk and a consumer
  doing `sources[citation["n"]]` would `KeyError`. In-memory == on-disk. Use
  `context.src_key(n)`.
- **[B]** **Score write-back** (`rag/ledger.py`): a run closes *before* its scores
  exist (the harness scores afterwards), so `update_eval_score(run_id, scores)` patches
  the manifest's `eval_score` and the ledger row's `_pending_` cell.
- **[I]** UI via Arize Phoenix / Langfuse — deferred; the frontend (next task) consumes
  `runs/` directly.

### 9. Evaluation  (gold set + harness built; E0 measured 2026-07-18)
- **Gold set `gold_v1_small`: 10 groups × 6 phrasings = 60 queries** — each group has a
  canonical question, **4 paraphrases**, and **1 weird framing**, tagged with expected
  source page(s). Keeps both near-neighbour hard-negative clusters; covers 6 of 8 sections.
  In `eval/gold_v1_small.jsonl`. (The original 40-group `gold_v1` was removed.)
- **Fixed scoring panel (full; identical every run for comparability):**
  | Metric | Definition |
  |--------|------------|
  | `recall@k` | an expected page present in the selected/retrieved set |
  | `mrr` | 1/rank of first expected page |
  | ⭐ `consistency` | per group, Jaccard overlap of the **selected chunk/page sets** across the 6 phrasings — deterministic; reported paraphrase-only AND incl. weird framing |
  | `answer_agreement` | do a group's phrasings yield equivalent answers (LLM-judge) |
  | `faithfulness` | answer supported by cited context (LLM-judge 0/1) |
  | `citation_acc` | every claim's `[n]` resolves & supports it (LLM-judge 0/1) |
- **Headline = `consistency`.** Components stay visible; each eval run also reports
  **total tokens (cost)** and **latency** (avg / p50 / p95 / max, plus per-stage) beside
  the score.
- **LLM-judge = a separate, larger-context model** (`judge_model`, default
  `openai/gpt-oss-120b`) so it doesn't grade itself, driven through the same
  **key-cycling** pool. Comparability guaranteed by pinned `gold_set`, `temperature=0`,
  fixed `k`, logged `config_hash` + `eval_hash`.
- **Built** (2026-07-15): `src/rag/eval/` — `gold.py` (groups/phrasings, `query_id` joins
  traces to gold), `traces.py` (read a finished run off disk), `metrics.py` (pure, plain
  floats), `judge.py`, `harness.py`.

**Exact definitions** (as implemented — `metrics.py`):

| Metric | Definition |
|---|---|
| ⭐`consistency` | mean over groups of **mean pairwise Jaccard** of selected **page**-sets over `canonical + 4 paraphrases` (m=5, 10 pairs). Canonical is included: the problem is the same question *reworded*, so paraphrase-vs-base is the key comparison |
| `consistency_weird` | same over all 6 phrasings (15 pairs) — the **stress** sub-score |
| `consistency_chunks` | same as `consistency` on chunk-id sets — diagnostic (high page + low chunk = chunker churn) |
| `recall@k` | fraction of queries with **any** expected page in the selected set (the gold set has exactly 1 expected page per group, so any-hit ≡ all-hit) |
| `recall@cand` | same over the retrieved pool. **`recall@cand − recall@k` is the rerank ceiling** — it says whether that lever can pay *before* building it |
| `mrr` | 1/rank of the first expected page in the **page-deduped post-rerank** list, untruncated (truncating at `top_k` would just restate `recall@k`) |
| `answer_agreement` | judge returns a per-answer consistency vector over a group's answers; score = mean ∈ [0,1] (40 binary verdicts would be too coarse) |

- **Retrieval metrics need no LLM** — selection precedes generation, so the headline is
  free, instant and deterministic. `harness --retrieve-only` measures it for 0 tokens,
  and **error traces still carry full retrieval**, so coverage survives generation
  failure. Cross-check: `recall@cand` must equal the fraction with `mrr > 0`.
- **Judged metrics are budget-bound.** 100k tokens/day/key/model ⇒ a generated 240-query
  panel spans days. `judge_sample` (`all` | `c+2p` | `c`) pins which phrasings are judged
  and lives in `eval_hash` so a cheaper panel is still comparable to itself.

## Decision log

| # | Decision | Choice | Why / alternatives | Revisit when |
|---|----------|--------|--------------------|--------------|
| 1 | Corpus | univie *studieren* EN, 121 pages | Clean, citation-ready metadata, natural topic overlap to stress paraphrases | — |
| 2 | Python | 3.12 venv (not system 3.14) | ML stack has no 3.14 wheels yet | — |
| 3 | Embeddings | bge-small-en-v1.5 / fastembed | Free/local stand-in for OpenAI 3-small; ONNX, no torch | want closer parity → OpenAI |
| 4 | Chunking | heading-aware, ~320 tok, path-prepend | Precise vectors + retained context; tunable | eval shows retrieval misses |
| 5 | Vector DB | Qdrant, local persistent path | Velocity; Docker parity is 1-line later | need server features/parity |
| 6 | Generation | Groq, default llama-3.3-70b, temp 0 | Behavioral stand-in for GPT-4.1; determinism | latency demo → 8b-instant |
| 7 | Citations | numbered payload + `[n]` + validation | Metadata already in schema | — |
| 8 | Tracing | per-run folders (`runs/<id>/`), full-pipeline record, run `kind` | Frontend replays entire pipeline; adhoc vs eval separated | want UI → Phoenix/Langfuse |
| 9 | Baseline scope | single dense query + contextual chunks | Clean before/after; path-prepend also aids index/citation | — |
| 10 | Eval | `gold_v1`: 40×6 (canonical+4 paraphrases+1 weird)=240; full panel. `gold_v1_small` = 20 groups verbatim (120) for cheap iteration | Measures the headline problem; balanced by section. A subset is a **new versioned set** (`--gold-set`, moves `eval_hash` only), never an in-place edit — keeps E0 comparable | — |
| 11 | Groq keys | pool of 4, round-robin + rotate-on-429; rotations logged | Survive free-tier limits; `key_id`/events logged, secret never | hitting limits on 4 |
| 12 | Modularity | component interfaces + Config; experiment = config diff | Each layer tampered independently, results attributable | — |
| 13 | Experiment tracking | `EXPERIMENTS.md` run-ledger (auto, eval runs only) + config_hash | Reproducible; adhoc queries don't spam the ledger | — |
| 14 | Retrieval pool | `candidate_k`(20) retrieved → rerank → `top_k`(6) selected | Real retrieve→rerank→select trace; rerank seam ready | — |
| 15 | LLM-judge | separate large-ctx model (`openai/gpt-oss-120b`), key-cycled | Doesn't grade itself; survives rate limits | — |
| 16 | Cost/latency | per-run token totals + latency beside `eval_score` | Cost/speed tracked with quality every run | — |
| 17 | LLM access | one `rag/llm.py`: `KeyPool` + `LLMProvider`/`GroqProvider` | Generator, judge & expansion share **one** pool (separate pools would double 429s on the same key budget); provider swappable | adding OpenAI/local |
| 18 | Prompts | `rag/prompts.py`, versioned; `prompt_version` in `config_hash` | Prompt is a lever; editing a used version invalidates results | — |
| 19 | Chunker ownership | `rag/chunk.py` only; `scrape.py` never chunks | A 2nd chunker silently clobbered `chunks.jsonl` with wrong-sized chunks; no error | — |
| 20 | Score write-back | `rag/ledger.py` `update_eval_score(run_id, …)` | Runs close before scores exist; harness needs a patch path | — |
| 21 | Trace `sources` | str-keyed + `status` on every trace (incl. errors) | JSON keys are strings — int keys break the frontend join; failed queries must still trace | — |
| 22 | Retry policy | rotate on 429 **and** transient 5xx/connection/timeout | One 503 mid-run would otherwise kill a 240-query eval | — |
| 23 | Hash scoping | three hashes (`index_hash` ⊂ `config_hash`; `eval_hash` apart), one flat Config, bucket-completeness assert at import | A single hash let `judge_model` re-stamp pipeline identity, invalidating results the change couldn't affect. Explicit buckets are *less* safe than hashing everything unless the assert forces every field into one | adding a Config field (the assert will tell you) |
| 24 | Index binding | collection name = `collection_<index_hash>` + `chunks.meta.json` sidecar asserted by `rag.index` | Nothing bound the index to the config that built it: change a chunk param, skip the re-index, and the eval scored the OLD index while the ledger claimed the new hash — fabricated, undetectable. Name catches a missing re-index; sidecar catches a missing re-chunk. Both needed | — |
| 25 | Chunker code version | `chunker_version` in `index_hash` | Chunk *params* don't describe the *algorithm*; fixing a chunker bug would otherwise rebuild the index under an unchanged hash — a stale collection with a valid name | any `chunk.py` behaviour change |
| 26 | Ledger format ownership | `rag/ledger.py` owns `_COLUMNS` + `append_row`; `trace.py` delegates; score column found by header, scoped between markers | Format was written in trace.py and parsed in ledger.py with no shared definition, matching rows by luck across 3 tables. Adding a column silently clobbered a cell | — |
| 27 | Eval phases | generate (costly, spans days) and score (reads traces off disk) are separate; `--score r1,r2` merges runs | The headline needs no LLM, so it is free and instant; judge prompts need iteration without re-generating; the daily quota forces a generated panel across runs. Runs are never reopened (close() would double the ledger row) | — |
| 28 | Judge failures | truncated/unparseable/refused ⇒ score `None`, **excluded**, never 0; `n_judged` reported | Scoring the pipeline 0 because our evaluator misbehaved manufactures a grounding failure it never committed | — |
| 29 | Judge budget | `judge_max_tokens` separate from `max_tokens`; `judge_sample` pins which phrasings get judged | gpt-oss is a *reasoning* model — at the generator's 1024 it spent the budget thinking and returned empty content. And 100k tok/day/key/model makes a full judged panel multi-day, so the sample must be pinned to stay comparable | paid tier |
| 30 | Vector store runtime | **Qdrant in Docker** (`docker-compose.yml`, `QDRANT_URL`); local path still the fallback. `qdrant_url` is `_UNHASHED` | The local path's exclusive file lock made the frontend and any harness/index run mutually exclusive (verified: 2nd process dies with `BlockingIOError`; against the server both succeed). Also closer to the company's stack, and payload indexes are a silent **no-op** in local mode. Not hashed because the backend holds the same vectors — verified bit-identical (`consistency` 0.426785 both ways), so E0 stays comparable across the switch | — |
| 31 | Frontend | **zero-dependency `src/webui/`**: stdlib `http.server` + one self-contained SPA (no framework, no build, no new deps) | requirements.txt stays at 4 packages, and it runs on a fresh clone. Split into a pure `store.py` read-model (testable, no network) and a thin `server.py`; the pipeline (Qdrant/fastembed/Groq) is imported **lazily**, only on a live query, so replaying traces needs nothing but the filesystem. Enumerates `runs/*/manifest.json` (not `index.jsonl`, which omits the richest demo run). Signature views: a **pipeline rail** (the 20→6→answer funnel, per-stage) and a **paraphrase-divergence strip** (which pages each phrasing selected — `consistency` drawn directly) | want hosted UI → Phoenix/Langfuse |

## Status
**E0 is measured** (2026-07-18) — `config_hash f22363afaf1d` · `index_hash 2747344b6db6`.
720 chunks → Qdrant (`univie_studying_2747344b6db6`) → retrieve 20 → passthrough
rerank → select 6 → Groq `llama-3.3-70b-versatile`. Per-run full-pipeline logging and the
`gold_v1_small` eval set (10×6=60) are in place. Code in `src/rag/` (14 modules + `eval/`);
a zero-dep console in `src/webui/` (decision 31); **124 tests** in `tests/`.

**⭐ consistency = 0.428** · recall@k 0.783 · recall@cand 0.917 · mrr 0.674 — complete
(60/60), deterministic, zero-token for the retrieval panel. **Complete judged panel**
(faithfulness 0.933, citation_acc 0.750, answer_agreement 0.700; 60/60, 0 failures) — the
60-query set fits a full generate + judge run in one day. See `EXPERIMENTS.md` → E0.

**The integrity gates closed before E0 was measured** (all were pre-conditions for
trusting any number): hash scoping (#8), index↔config binding (#2), chunker `heading_path`
mis-anchoring (#12 — 26.6% of vectors) and duplicate tail chunks (#11), monotonic latency
(#19), fail-loud `retriever_mode` (#4), `finish_reason` truncation reported (#16).
Remaining open findings → `HANDOFF.md`.

**The frontend is built** — `src/webui/` (decision 31): replay every hop of any logged
query, view a scored run's eval panel + divergence strip, ask a live question, or run the
retrieve-only eval panel. `PYTHONPATH=src .venv/bin/python -m webui.server`.

Next: **iteration levers** — multi-query+RRF (attacks the page-set churn `consistency`
measures) first, then cross-encoder rerank (bounded upside already measured: +0.134 recall
headroom). One lever at a time, each an `EXPERIMENTS.md` entry measured against E0.
