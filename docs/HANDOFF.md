# Handoff — rag-proto

Read this first, then `ARCHITECTURE.md` (design + decision log). This doc is the
"where things stand and what to do next" note; the architecture doc is the "why".

## The assignment in one paragraph

A RAG prototype for an onboarding assignment at a **voice-agent company**, doing
retrieval/QA over a knowledge base. The company's real pain points, in priority order:
**(1) the same question reworded gives inconsistent/inaccurate answers** (this is the
headline problem), **(2) no traceability** through the RAG flow, **(3) answers must
cite sources**. The approach: build a strict-parity **baseline (E0)** mirroring their
stack, measure it, then turn on improvements **one lever at a time** so each is
attributable. Everything must stay free/local (16 GB RAM, RTX 3060 6 GB).

## State: what's built and verified

| Piece | State |
|---|---|
| Corpus | ✅ 121 EN pages (`data/pages.jsonl`), 720 chunks (`data/chunks.jsonl`) |
| Index | ✅ Qdrant local (`univie_studying_2747344b6db6`), 720 points, 384-d cosine |
| Pipeline E0 | ✅ end-to-end, `config_hash f22363afaf1d` · `index_hash 2747344b6db6` |
| Logging | ✅ per-run folders, full-pipeline traces |
| Eval set | ✅ `eval/gold_v1.jsonl` — 40 groups × 6 phrasings = 240, validated |
| Scoring harness | ✅ `src/rag/eval/` — offline panel + LLM-judge, two-phase |
| **E0 measured** | ✅ **⭐ consistency 0.427** (240/240) · judged panel partial ⚠️ |
| Tests | ✅ 106 in `tests/` (`.venv/bin/python -m pytest`) |
| Frontend | ⬜ **next task** |
| Iteration levers | ⬜ measurement now exists — go |

**E0 flow:** `query → transform(identity) → retrieve 20 (dense, bge-small) → rerank
(passthrough) → select 6 → assemble numbered sources → Groq llama-3.3-70b-versatile
(temp 0, grounded cite-only prompt) → parse+validate [n] citations → trace`.

Run it: `PYTHONPATH=src .venv/bin/python -m rag.pipeline "how much is tuition?"`

## E0 — the "before" (2026-07-15)

| | recall@k | recall@cand | mrr | ⭐consistency | weird | chunk |
|---|---|---|---|---|---|---|
| **complete, 240/240, 0 tokens** | 0.875 | 0.967 | 0.757 | **0.427** | 0.399 | 0.310 |

Judged (⚠️ **section-biased fragment**, 24/40 groups, `judge_sample=c+2p`, 72 verdicts):
`answer_agreement` 0.868 · `faithfulness` 0.986 · `citation_acc` 0.819.
Cost/speed: 2102 tok/query · 3.9 s avg (p95 8.8 s); generation is >99% of latency.

**What E0 says — read this before picking a lever:**
1. **consistency 0.427 vs recall@k 0.875** — the gold page usually *does* reach the
   prompt; the churn is in the other 5 of 6 slots. The paraphrase problem is
   **retrieval-side page-set instability**, not "can't find the answer".
2. **answer_agreement 0.868 ≫ consistency 0.465** (same fragment) — the churn only
   partly propagates: different context, same answer, because gold usually survives.
   Expect a retrieval lever to move `consistency` far more than `answer_agreement`.
3. **Rerank headroom = recall@cand − recall@k = +0.092** — a *perfect* reranker buys at
   most ~9 pts of recall@k. Bounded, and known before building it.
4. **citation_acc 0.819 ≪ faithfulness 0.986** — claims are supported but the `[n]` is
   often attached to the wrong source. A citation-mapping problem, not a grounding one.

## ⚠️ Token budget — the constraint that shapes every eval

**100k tokens/day/key/model** on the free tier. 4 keys ⇒ **~400k/day per model**. It is
**invisible in the response headers** (those only expose the per-minute bucket) — it
appears *only* in the body of the 429. Don't re-derive this from headers and conclude
there's no daily cap; there is.

| Job | Cost | One day? |
|---|---|---|
| Retrieval panel (240) | **0** | yes, ~8 ms/query |
| Generate 240 answers | ~508k | **no** — ~189/day max |
| Judge 240 + 40 groups | ~800k | **no** |

⇒ A generated panel **necessarily spans runs**. Each day is its own `Run`; merge at
scoring time (`--score r1,r2`). **Never reopen a Run** — `close()` would append a second
ledger row (idempotency is in-memory, not on disk).

**To finish E0's judged panel** when the quota resets:
```bash
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --groups g25,g26,…,g40 --label "E0 gen fragment 2/2"
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --score 20260715-193039-cc9a,<new_id> --judge-sample c+2p
```

## The harness

```bash
python -m rag.eval.harness --retrieve-only     # headline panel, 0 tokens, ~30 s, no keys
python -m rag.eval.harness --limit 3           # smoke: 3 groups end-to-end (~50k tokens)
python -m rag.eval.harness --label E0          # full: generate + judge (quota-bound)
python -m rag.eval.harness --score <run_id>    # re-score off disk, no regeneration
```
Two phases on purpose: **generate** costs the day's budget; **score** reads traces off
disk. So judge prompts can be iterated, noise calibrated, and metrics recomputed without
regenerating — and scoring needs no Qdrant lock.

## ⚠️ Read these before you touch anything

1. **temp-0 generation is NOT deterministic on Groq.** The same query, same selected
   chunks, run 3× produced 3 different answers (two good, one "I don't know").
   **This is a real consistency problem — not a bug in the pipeline.**
   ⚠️ **Refined by E0 (2026-07-15) — the old note said "it's generation-side, not
   retrieval-side; don't chase a retrieval ghost". That is now half wrong.** Two
   *distinct* effects, don't conflate them:
   - *Same* query re-run → identical retrieval, different answers = **generation
     nondeterminism** (the original observation; still true).
   - *Reworded* query → **different retrieval**. This is the headline problem, and E0
     measures it at `consistency` **0.427**: retrieval itself is perfectly deterministic
     (3 retrieval-only runs gave 0.427 exactly), but paraphrases select ~57% different
     page-sets. **The retrieval lever is not a ghost — it's the main event.**
   - Mitigating evidence for the generator: `answer_agreement` 0.868 ≫ `consistency`
     0.465 on the same fragment, i.e. the generator is *more* paraphrase-robust than the
     retriever.
2. **Local Qdrant takes an exclusive file lock — one process at a time.** The frontend
   and a harness run cannot both hold `qdrant_storage/`. Options: run Qdrant in Docker
   (the documented 1-line parity upgrade), or serialise access behind one process.
   **Decide this before building the frontend.**
3. **`gold_v1` contains deliberate hard negatives** — near-neighbour pages that should
   punish sloppy retrieval (minimum-credits vs prüfungsaktiv, both "16 ECTS" but
   different windows/consequences; the three registration pages). Confusing them is a
   real failure, not a labelling error.
4. **`gold_v1` `key_fact`s quote time-sensitive fees/deadlines** (scraped 2026-07-14).
   Questions stay valid after a re-scrape; the facts may drift.
5. **Never edit a used prompt version in place.** `prompt_version` feeds `config_hash`;
   add `v2` to `rag/prompts.py` instead, or you silently invalidate logged results.

## Known issues — from the code review (2026-07-15)

A full review was run against E0. **Fixed already** (verified): duplicate chunker in
`scrape.py` silently clobbering `chunks.jsonl`; `sources` int keys not surviving the
trace JSON round-trip; `Run.close()` non-idempotent (duplicate ledger rows); failed
queries writing no trace; no write-back path for `eval_score`; key pool not retrying
5xx/connection blips, not deduping keys, not thread-safe, untestable (real sleeps);
prompts trapped in `generate.py`; silent ledger drops.

**Closed 2026-07-15, before E0 was measured** (each was a precondition for trusting any
number — see `ARCHITECTURE.md` decisions 23–29 and `EXPERIMENTS.md` → E0):

| # | Was | Now |
|---|---|---|
| **8** | `config_hash` conflated eval knobs with pipeline identity | **Three scoped hashes** + import-time bucket-completeness assert. Proven live: adding `judge_max_tokens` moved `eval_hash` only, leaving `config_hash` untouched — under the old scheme that one-line fix would have re-stamped E0. |
| **2** | Nothing bound the index to the config that built it | Collection = `univie_studying_<index_hash>` + `data/chunks.meta.json` sidecar asserted by `rag.index`. All three failure paths verified to fail loudly. |
| **12** | Packed chunks claimed their **first** block's `heading_path` | **Common prefix.** It was worse than filed: `heading_path` feeds `embed_text`, so **195/733 (26.6%) of vectors were mis-anchored**, not just mislabeled (~84% of top-6 prompts). `chunker_version=v2`. |
| **11** | 13 redundant tail chunks | Split loop breaks once a window reaches the end. **733 → 720 chunks**, 0 redundant (verified). |
| **4** | `retriever_mode` silently ignored | `Retriever` raises `NotImplementedError` for anything but `dense`. |
| **16** | `finish_reason=="length"` captured but unused | Truncated answers **excluded** from judged metrics; `n_truncated` reported. |
| **19** | Wall-clock vs monotonic mixed in tracing | `perf_counter` throughout. |
| **13** | `overlap_words` a no-op on the packing path | **Documented, not "fixed"** — it's a lever in disguise (see below). |
| — | `p.payload` could be `None` | Guarded; missing collection raises a directive error. |

**Still open:**

| # | Issue | Why it matters |
|---|---|---|
| **3** | **The multi-query/HyDE seam is fake.** `pipeline.py` does `retrieve(queries[0])` — variants are dropped, no RRF fuse step exists. | Now the **#1 lever**: E0 proves the paraphrase problem is retrieval-side page-set churn (`consistency` 0.427). Add `rag/fuse.py` (RRF) + loop over all variants — identity makes it a no-op today, so it can land without moving E0. Also: rerank scores against the original `query` while retrieval used `queries[0]` — decide explicitly. |
| **15** | **Local Qdrant takes an exclusive lock** — frontend + harness can't coexist. | Take the Docker parity upgrade **before the frontend**. Note scoring (`--score`) needs no lock — only generation does. |
| **13** | `overlap_words` applies to the window-split path only. | **Deliberate.** Packing merges whole heading-blocks; overlapping there duplicates text and re-creates #11. But it's in `index_hash`, so changing it yields a new hash + a near-identical index — you'd re-chunk, re-index, re-run and measure noise. |
| — | `index.py` deletes the collection before rebuilding (no rollback; moots the "idempotent" stable-UUID comment). Now scoped to one `index_hash`, so a failed rebuild only breaks that index. |
| — | **Judge noise is uncalibrated.** Two passes over *identical* traces gave `citation_acc` 0.529 → 0.444. Re-judge a fixed subset twice (~80 calls) to get a noise floor before trusting a small judged delta. `consistency` is immune (offline/deterministic). |
| — | **E0's judged panel is a section-biased fragment.** Finish it when the quota resets (commands above). |

**Still recommended:** `rag/fuse.py` (RRF — makes the multi-query seam real). `rag/eval/`
now exists. Keep `Config` **one flat dataclass** — splitting it just adds plumbing at
this size; the *hash* is scoped instead. Don't add `corpus/`/`retrieval/` subpackages at
~1k lines — gold-plating.

## Tests — `106 passing` (`.venv/bin/python -m pytest`)

`tests/{test_config_hash,test_chunk,test_metrics,test_ledger,test_judge,test_keypool}.py`.
No network, no Qdrant, no real sleeps. The two that matter most:
- **`test_config_hash.py`** — the bucket-completeness assert + the scoping contract
  (`judge_model` must NOT move `config_hash`; `target_words` MUST move both). This is the
  regression that would silently undo the whole refactor.
- **`test_chunk.py`** — packing invariants for #11/#12. Both defects were *silent*:
  plausible chunks that quietly cost recall. Nothing but a test notices their return.

Still untested: `parse_citations`, `sources` JSON round-trip, `RagPipeline` wiring
(now injectable via `pool=` — `Retriever` already took `client`/`embedder`).
Known test-env gotchas: `trace.py`/`ledger.py` bind `RUNS_DIR`/`EXPERIMENTS_MD` **by
value at import** — patch `rag.trace`/`rag.ledger`, not `rag.config`. `llm.py` calls
`load_dotenv()` at import.

## Layout

```
data/            pages.jsonl (canonical corpus) · pages/*.md (eyeball mirror)
                 · chunks.jsonl · chunks.meta.json (index_hash sidecar — see #2)
eval/            gold_v1.jsonl · validate_gold.py · README.md (schema + method)
scripts/         scrape.py (refresh corpus)
docs/            ARCHITECTURE.md · EXPERIMENTS.md · HANDOFF.md (this)
src/rag/         config · chunk · embed · index · query · retrieve · rerank · context
                 · prompts · llm · generate · trace · ledger · pipeline
src/rag/eval/    gold · traces · metrics · judge · harness   (the scoring harness)
tests/           106 tests — no network, no Qdrant
runs/            per-run traces (gitignored) · <id>/eval/ holds scores + judge output
qdrant_storage/  local Qdrant (gitignored)
```

**Key seams** (every layer is swappable via one `Config`; an experiment = a config diff):
- `rag/llm.py` — `KeyPool` (round-robin, rotate-on-429, backoff) + `LLMProvider`
  protocol + `GroqProvider`. **All LLM callers go through here** — generator, the
  upcoming judge, future query-expansion. Add a provider by implementing `chat` +
  `list_models`.
- `rag/prompts.py` — versioned templates (`SYSTEM_PROMPTS["v1"]`).
- `rag/query.py` `get_transform()` — identity today; multi-query/HyDE plugs in here.
- `rag/rerank.py` `get_reranker()` — passthrough today; cross-encoder plugs in here.
- `rag/trace.py` — `Run` (owns `runs/<run_id>/`, aggregates tokens+latency) +
  `Tracer` (per-query, `span()` per stage).
- `rag/ledger.py` — `update_eval_score(run_id, scores)` writes scores back onto a
  **finished** run (manifest + the ledger row's `_pending_` cell). The harness needs
  this: runs close before their scores exist.

## Logging model (the frontend's data source)

```
runs/<run_id>/manifest.json        config + config_hash + aggregates + eval_score slot
runs/<run_id>/queries/<tid>.json   FULL per-query trace (see below)
runs/<run_id>/queries.jsonl        one compact line per query (list view)
runs/index.jsonl                   one line per finalized run (enumerate runs)
```
`run_id = YYYYMMDD-HHMMSS-<4hex>`. Each query trace has stages
`query_transform → retrieve → rerank → select → assemble → generate`, carrying the
**full retrieved pool (20, with text)**, the **selected 6**, the **exact prompt
messages**, the **raw response**, `model`/`key_id`/`temperature`/`usage`,
**rate-limit rotation events**, per-stage `latency_ms`, and resolved `citations`.

**Run `kind` matters:**
- `adhoc` — a custom one-off query → run folder + `index.jsonl`, **no** ledger row.
- `eval` — a scored gold-set pass → folder + index + **appends a row to the
  EXPERIMENTS.md run ledger** (between the `RUN-LEDGER:START/END` markers).
- `batch` — anything else multi-query.

## ✅ Done: the scoring harness — how it's built

`src/rag/eval/`. Panel definitions are in **ARCHITECTURE §9** (exact formulas) — don't
redesign them silently. Design points worth not re-litigating:

- **Two phases.** `run_generation` (costly) → traces on disk → `run_scoring` (reads them
  back). So the judge prompt can be iterated, noise calibrated, and metrics recomputed
  **without regenerating**; and scoring needs no Qdrant lock. `--score r1,r2` merges runs
  because the daily quota forces generation across days.
- **The headline is free.** Selection precedes generation ⇒ `consistency`/`recall`/`mrr`
  need no LLM. `--retrieve-only` = 240 queries, ~30 s, 0 tokens, **no API key required**.
- **Error traces still score.** `pipeline.py` wraps *only* the generate span, so a failed
  generation still carries the full retrieved/selected sets. Retrieval coverage survives
  total generation failure — that's why E0's retrieval panel is 240/240.
- **`query_id`** (`g07:paraphrase:2`) is written into each trace and joins it back to
  gold. Without it, scoring could only match on exact query text — which breaks silently
  the moment a gold phrasing is edited.
- **One `KeyPool`** shared by generator + judge (`RagPipeline(pool=...)`). Separate pools
  would round-robin independently and double the 429s on the same budget.
- **A judge failure is never a 0.** Truncated/unparseable/refused ⇒ `None`, excluded from
  the mean, counted in `n_judged`. Judge output goes to `runs/<id>/eval/judge/` — **not**
  through `Run` (it's closed; `Run._record` would flip its manifest back to `open` and
  fold judge tokens into the pipeline's cost).
- **Judge tokens stay out of the run's `total_tokens`** — that column is the *pipeline's*
  cost, the thing E0 vs E1 compares.
- **`--limit` is group-atomic** (`consistency` needs a whole group) and seeded — gold is
  ordered by section, so a plain prefix is a section-biased, non-comparable sample.
  **This is exactly how E0's judged fragment got biased** — see the ⚠️ above.

## Then: the frontend

Reads `runs/` (see logging model above). Must support: run a **custom ad-hoc query**,
and **run the eval set**; and replay the full pipeline per query (retrieved → reranked
→ selected → prompt → answer → citations). Resolve the Qdrant single-process lock
first (see warning 2).

## Then: iteration levers (one at a time, each an EXPERIMENTS.md entry)

**Re-ordered by E0's evidence** (the old order was written before any measurement — it
led with prompt/self-consistency on the theory that the problem was generation-side. The
data says otherwise: `consistency` 0.427 is retrieval page-set churn, while
`answer_agreement` 0.868 says the generator is comparatively robust):

1. **Multi-query / HyDE + RRF fusion** — `get_transform()` seam (+ issue #3: the seam is
   currently fake). Directly attacks paraphrase divergence, which E0 pins as *the*
   problem. Retrieval-only ⇒ **measurable for 0 tokens.**
2. **Cross-encoder rerank** (`bge-reranker-base`) — `get_reranker()` seam. Upside is
   **already bounded at +0.092 recall@k** (= recall@cand − recall@k); it may still help
   `consistency` more than recall by stabilising *which* 6 survive. Local, 0 tokens.
3. **Hybrid dense+BM25** — Qdrant-native sparse; helps jargon (ECTS, STEOP, €). Note
   `recall@cand` is already 0.967, so the pool is rarely the problem — this is about
   ranking, not finding.
4. **Prompt / self-consistency** — majority-vote over n samples or a stricter contract.
   Demoted: `faithfulness` is already 0.986. But **`citation_acc` 0.819 is the real
   generation-side gap** — supported claims, wrong `[n]`. A cheap prompt `v2` targeting
   citation mapping is the highest-value generation lever. (Costs tokens; budget it.)

Levers 1–3 are **retrieval-side ⇒ scoreable with `--retrieve-only` for free**. Do the
headline comparison first, spend tokens only on the judged panel once a lever wins.

## Conventions

- Run everything `PYTHONPATH=src .venv/bin/python` (Python 3.12 venv; **not** system 3.14).
- `RAG_EMBED_THREADS` (default 6) caps onnxruntime; without it indexing pegs all cores.
- Rebuild corpus artifacts: `python -m rag.chunk` then `python -m rag.index`.
  **Both, in that order, after any `index_hash` change** — `rag.index` refuses stale chunks.
- Keys: `.env`, `GROQ_API_KEYS=a,b,c,d` **or** `k1..k4`. Secrets never logged — only `key_id`.
- **One lever per experiment**, logged in `EXPERIMENTS.md` with `config_hash` + `eval_hash`.
- **Adding a `Config` field?** The import-time assert will fail until you put it in a
  hash bucket (`_INDEX_FIELDS` / `_PIPELINE_FIELDS` / `_EVAL_FIELDS` / `_UNHASHED`).
  That's deliberate: choose consciously whether it changes the index, the pipeline, or
  only how a run was scored.
- Tests: `.venv/bin/python -m pytest` (106, fast, no network).
