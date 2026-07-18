# rag-proto

RAG prototype for an **onboarding assignment** at a company that builds **voice
agents**. The task is information retrieval / querying over a knowledge base.

## Goal

Stand up a **baseline RAG pipeline** (mirroring the company's stack) that the team
can then iterate on. Three problems the baseline must be built to expose and,
eventually, fix:

1. **Paraphrase robustness** — the same question asked different ways currently
   gives inconsistent / inaccurate answers. Consistency across rephrasings is the
   headline metric.
2. **Traceability** — every hop of the RAG flow (query → retrieval → prompt →
   answer → sources) must be inspectable/loggable.
3. **Citations** — the final answer must cite its sources.

Build the baseline first (get parity with the company's setup), *then* iterate.

## Stack (constraints: free/local · 16 GB RAM · RTX 3060 6 GB)

Matches the company's stack: **Qdrant + a small embedding model + Groq**. (Company
uses OpenAI `text-embedding-3-small`; local free substitute is bge-small below.)

- **Vector DB:** Qdrant. **Docker server** (`docker-compose.yml`, `QDRANT_URL=http://localhost:6333`)
  — the local persistent path is the zero-infra fallback but takes an **exclusive file
  lock** (one process at a time), and its payload indexes are a silent no-op. The backend
  is `_UNHASHED`: same vectors either way (verified bit-identical), so switching moves no
  hash.
- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-dim, cosine) via **fastembed**
  (ONNX/CPU, no torch). Instant for 121 pages. bge query-side instruction prefix
  applies for retrieval. Swappable — keep the embedder behind one interface.
- **LLM (generation):** Groq API. **Model is a config value** (we A/B several);
  default `llama-3.3-70b-versatile` (stand-in for company's GPT-4.1),
  `llama-3.1-8b-instant` for latency. **Key pool: 4 keys, round-robin + rotate-on-429**
  (keys in `.env` as `GROQ_API_KEYS=…` OR individually `k1..k4`; secrets never logged,
  `model`+`key_id` are).
- Everything must stay free and run on the local machine. No paid services.

## How we work

- **Docs map:** `docs/HANDOFF.md` (state, known issues + next steps — **start here**) →
  `README.md` (how to run) → `docs/ARCHITECTURE.md` (design + decision log, update it
  when a decision changes) → `docs/EXPERIMENTS.md` (run log). Code in `src/rag/`
  (14 modules) + `src/rag/eval/` (the scoring harness); tests in `tests/`.

**Invariants — don't break these:**
- **All LLM calls go through `rag/llm.py`** (`KeyPool` + `LLMProvider`/`GroqProvider`) —
  generator, LLM-judge, and future query-expansion share **one** pool instance
  (separate pools double the 429s on the same 4-key budget). Never add a second
  key-loading path.
- **`rag/chunk.py` is the ONLY chunker.** `scripts/scrape.py` must never write
  `chunks.jsonl` — a second chunker silently clobbered it with wrong-sized chunks while
  traces still claimed the `Config` values.
- **Prompts live in `rag/prompts.py`**, versioned — generator (`prompt_version` →
  `config_hash`) *and* judge (`judge_prompt_version` → `eval_hash`). Never edit a used
  version in place; add `v2`.
- **`sources` is str-keyed** (`context.src_key(n)`) — JSON object keys are strings, so
  int keys silently break the frontend's citation→source join.
- **Scores are written back via `rag/ledger.py`** (`update_eval_score`) — runs close
  before their scores exist. **`ledger.py` owns the ledger row format** (`_COLUMNS`,
  `append_row`); `trace.py` calls it. Never hand-build a row in two places again.
- **Three scoped hashes, one flat `Config`.** `index_hash` ⊂ `config_hash`; `eval_hash`
  separate. **Every field must be in exactly one bucket** — an import-time assert
  enforces it, and it is the only thing making explicit buckets safer than hashing
  everything. Adding a field? Choose its bucket consciously.
- **Never reopen a `Run`.** `close()` is idempotent only in memory — reopening would
  append a second ledger row. A panel spanning days = several runs merged at *scoring*
  time (`--score r1,r2`).
- **A judge failure is never a 0.** Truncated/unparseable/refused ⇒ excluded from the
  mean, counted in `n_judged`. Scoring the pipeline 0 for our evaluator's misbehaviour
  manufactures a failure it never committed.
- **Modular by contract:** every layer (chunk/embed/store/query/retrieve/rerank/
  context/generate/trace) is a swappable component behind a small interface, wired by
  one `Config`. **An experiment = a config diff, not a rewrite.**
- **Every run is logged in `docs/EXPERIMENTS.md`** — one lever changed at a time,
  citing `config_hash` + the change + fixed-panel scores. Headline metric =
  **`consistency`** (paraphrase robustness).
- Logging: each execution is a **`Run`** → `runs/<run_id>/` with `manifest.json`
  (config, token+latency aggregates, `eval_score` slot), `queries/<trace_id>.json`
  (full pipeline: retrieved→reranked→selected→exact prompt→raw response→citations,
  rate-limit rotations, latencies), `queries.jsonl`, + `runs/index.jsonl`. Run `kind`:
  `adhoc` (custom query) vs `eval` (scored pass → appends the `EXPERIMENTS.md` ledger).

## Environment / setup

- **Python 3.12.7** (via pyenv) in **`.venv/`**. NOT system Python 3.14 — the ML
  stack (onnxruntime/fastembed) has no 3.14 wheels yet.
- Run everything with `PYTHONPATH=src .venv/bin/python`. Deps in `requirements.txt`
  (`qdrant-client[fastembed]`, `groq`, `python-dotenv`, `tqdm`).
- bge-small (~130 MB) is cached locally; the built Qdrant index (720 chunks) lives in
  `qdrant_storage/`.

## Data — `data/`

Clean structured scrape of `studieren.univie.ac.at/en/` (University of Vienna
studies info: admission, tuition, exams, degree programmes, etc.). **English only,
121 pages, ~119k words.**

- **`data/pages.jsonl`** — **canonical corpus and chunking input**, one page/line.
  Fields: `id`, `url`, `title`, `section`, `language` (all `en`), `breadcrumb`,
  `headings` (pre-parsed level+text outline), `word_count`, `html_title`, `text`
  (clean markdown with `##`/`###` intact — split on headings directly).
- **`data/pages/*.md`** — human-readable mirror, 1:1 with `pages.jsonl`. For
  eyeballing only; **not** a pipeline input.
- **`data/manifest.json`** — counts, sections, skipped URLs.
- **`scripts/scrape.py`** — the scraper (re-run to refresh fees/deadlines). It
  **rewrites `data/` in place** (`pages.jsonl`, `pages/*.md`, `manifest.json`) and
  does **not** chunk — `src/rag/chunk.py` is the single chunker. After re-scraping:
  `python -m rag.chunk && python -m rag.index`.

**Citations** should be built from chunk metadata (`url` + `title` + `section` +
heading path) — provenance is already in the schema, don't bolt it on later.

## Known gaps / TODO

- **Gold eval set:** `gold_v1_small` — 10 paraphrase-grouped question groups × 6 phrasings
  = 60, each tagged with expected source page(s), corpus-grounded and validated
  (`eval/validate_gold.py`). It is the `Config` default `gold_set`. The original 40-group
  `gold_v1` was removed; a different `gold_set` moves `eval_hash` only, never `config_hash`.
- Corpus has natural topic overlap (`admission` vs `admission-procedure` vs
  `entrance-exam`; 5 tuition pages) — this is intentionally useful for stressing
  paraphrase retrieval, but means adjacent-wrong-page retrieval is the failure to
  watch. **`gold_v1_small` deliberately keeps near-neighbour hard negatives** (minimum
  credits vs prüfungsaktiv — both "16 ECTS", different windows/consequences; the three
  registration pages). Retrievers that confuse those should be penalised.
- **`gold_v1_small` `key_fact`s quote time-sensitive fees/deadlines** (corpus scraped
  2026-07-14) — questions stay valid on re-scrape, the facts may drift.
- `ABC of terminology` (20k words) was deliberately **excluded as a gold target** — it
  shallowly answers everything, making it an ambiguous retrieval target.
- A few thin hub pages (~130 words) produce weak chunks; some are link-farms with no
  standalone facts (see `eval/README.md` for which were swapped out).

## Status

- ✅ Data cleaned, English-only (121 pages), ready to chunk.
- ✅ Env set up (`.venv`, Python 3.12), deps installed & smoke-tested.
- ✅ Design locked — `docs/ARCHITECTURE.md` (+ `docs/EXPERIMENTS.md`).
- ✅ **Baseline pipeline built & verified end-to-end** (E0, config_hash `f22363afaf1d`):
  720 chunks → Qdrant (local) → retrieve 20 → passthrough rerank → select 6 → Groq
  `llama-3.3-70b-versatile`, grounded cited answers, 4-key round-robin pool.
- ✅ **Per-run full-pipeline logging** (`runs/`, adhoc/eval `kind`, token+latency cost).
- ✅ **`gold_v1_small` eval set** (10×6 = 60) in `eval/gold_v1_small.jsonl`
  (`eval/validate_gold.py`) — the `Config` default. Keeps both hard-negative clusters;
  covers 6 of 8 sections. The original 40-group `gold_v1` was removed.
- ✅ **Code-reviewed + refactored** (2026-07-15): `llm.py`, `prompts.py`, `ledger.py`.
- ✅ **Integrity gates closed before measuring** (2026-07-15): `config_hash` scoped into
  `index_hash`/`config_hash`/`eval_hash`; index bound to config (collection name carries
  `index_hash` + `chunks.meta.json` sidecar); chunker `heading_path` mis-anchoring fixed
  (26.6% of vectors) + duplicate tail chunks removed → **733 → 720 chunks**.
- ✅ **Scoring harness** — `src/rag/eval/` (gold · traces · metrics · judge · harness).
  Two-phase: generate (costly) → score (reads traces off disk, re-runnable).
- ✅ **E0 MEASURED (2026-07-18)** on `gold_v1_small` — **⭐ consistency 0.428** · recall@k
  0.783 · recall@cand 0.917 · mrr 0.674 · **complete judged panel** faithfulness 0.933,
  citation_acc 0.750, answer_agreement 0.700 (60/60, 0 judge failures). Run
  `20260718-151413-4593`, eval_hash `6647ca36c217`. See `docs/EXPERIMENTS.md` → E0.
- ✅ **124 tests** (`.venv/bin/python -m pytest`) — no network, no Qdrant.
- ✅ **Console (frontend)** — `src/webui/` (zero-dep stdlib `http.server` + SPA):
  replay every hop of any run, eval panel + paraphrase-divergence strip, live ad-hoc
  query, retrieve-only eval. `PYTHONPATH=src .venv/bin/python -m webui.server`. Replay
  needs only the filesystem; live query needs Qdrant + keys. Decision 31 in ARCHITECTURE.
- ⬜ Iteration levers — **next**. Measurement + console now exist. E0 says start **retrieval-side**.

⚠️ **Free-tier budget: 100k tokens/DAY/key/model** (~400k across 4 keys). It is invisible
in the response headers — it appears only in the body of the 429. On `gold_v1_small` (60
queries) a **full generate + judge panel fits in one day** (~142k gen + ~189k judge,
measured); retrieval metrics cost 0. (The old 40-group set couldn't — that's why it was
distilled.) Don't re-derive "there's no daily cap" from headers; there is.

Note: temp-0 generation is **not** deterministic on Groq — the same query can vary
(good answer ↔ "I don't know"). Real, but E0 refined the picture: the *headline* problem
is retrieval-side (paraphrases select ~57% different page-sets; retrieval itself is
perfectly deterministic), while the generator is comparatively robust
(`answer_agreement` 0.700 > `consistency` 0.428 on the same data).

## Running it
- Rebuild corpus artifacts: `PYTHONPATH=src .venv/bin/python -m rag.chunk` then `... -m rag.index`.
  Both, in that order, after any `index_hash` change — `rag.index` refuses stale chunks.
- Ask a question: `PYTHONPATH=src .venv/bin/python -m rag.pipeline "how much is tuition?"`.
- **Eval:** `... -m rag.eval.harness --retrieve-only` (headline panel, 0 tokens, ~30 s,
  no keys needed) · `--gold-set gold_v1_small` (20-group subset) · `--limit 3` (smoke) ·
  `--label E0` (full, quota-bound) · `--score <run_id>[,<run_id>]` (re-score off disk).
- Tests: `.venv/bin/python -m pytest` (106, fast, no network).
- Keys: `.env` holds 4 Groq keys (loader accepts `GROQ_API_KEYS=` OR `k1..k4`).
- `RAG_EMBED_THREADS` (default 6) caps onnxruntime so indexing doesn't peg all cores.
- **Qdrant:** `docker compose up -d` + `export QDRANT_URL=http://localhost:6333`. Without
  it you fall back to the local path, which allows only ONE process at a time (file lock)
  — that blocks running the frontend and a harness pass together.
