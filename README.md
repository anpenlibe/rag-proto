# rag-proto

A small, **modular, fully-traced RAG pipeline** with grounded citations, built over a
121-page University of Vienna "Studying" knowledge base. It's the baseline for
iterating on three problems that matter for a voice-agent KB:

1. **Paraphrase robustness** — the same question, reworded, should give consistent answers.
2. **Traceability** — every hop of the flow is logged and inspectable.
3. **Citations** — every answer cites its sources.

The design philosophy: ship a plain **baseline** that mirrors a typical
Qdrant + small-embedding + Groq stack, measure it, then flip on improvements one at a
time so each is attributable. Full design rationale in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); experiment log in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## How it works

```
pages.jsonl ─►[chunk]─►[embed]─►[index: Qdrant]           (offline, once)

query ─►[transform]─►[retrieve 20]─►[rerank]─►[select 6]─►[assemble]─►[generate: Groq]
                                                                    └─► grounded, cited answer
        └────── every hop written to runs/<run_id>/queries/<trace_id>.json ──────┘
```

- **Chunk** — heading-aware, ~230-word chunks with the heading path prepended (keeps
  small chunks meaningful and doubles as the citation anchor).
- **Embed** — `bge-small-en-v1.5` (384-d) via fastembed, local/free; asymmetric
  query vs passage encoding.
- **Index** — Qdrant, local persistent path, full chunk payload stored for citations.
- **Retrieve → rerank → select** — dense search returns a `candidate_k` (20) pool;
  reranking is a passthrough for now (the seam for a cross-encoder); the top `top_k`
  (6) are selected into the prompt. All three sets are traced separately.
- **Generate** — Groq (`llama-3.3-70b-versatile` by default), `temperature=0`, a
  grounded *cite-only* prompt (answers only from sources, else "I don't know") and
  citation validation. All LLM calls go through `rag/llm.py`: a **provider seam**
  (`GroqProvider`, swappable) plus a **4-key pool** that rotates on rate limits —
  trying every key before backing off — shared by the generator and the LLM-judge.
- **Trace** — each execution is a **`Run`** (`runs/<run_id>/`) that records the *full
  pipeline* per query (retrieved pool → reranked → selected → exact prompt → raw
  response → citations), plus per-stage latency, token cost, and rate-limit rotations.
  Runs are tagged `adhoc` (custom query) or `eval` (scored gold-set pass).

Every layer is swappable through one `Config`; **an experiment is a config diff**.

## Setup

Python 3.12 (the ML stack has no 3.14 wheels yet). Deps are already vendored into
`.venv`; to recreate:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Add your Groq keys to `.env` (gitignored). Either form works:

```bash
GROQ_API_KEYS=gsk_a,gsk_b,gsk_c,gsk_d      # comma-separated
# ...or individually named:
# k1=gsk_a
# k2=gsk_b
```

## Usage

```bash
# 0. start Qdrant (recommended — the local-file fallback allows only ONE process)
docker compose up -d
export QDRANT_URL=http://localhost:6333     # unset it to use qdrant_storage/ instead

# 1. build corpus artifacts
PYTHONPATH=src .venv/bin/python -m rag.chunk      # pages.jsonl -> data/chunks.jsonl
PYTHONPATH=src .venv/bin/python -m rag.index      # -> Qdrant (once per index_hash)

# 2. ask a question
PYTHONPATH=src .venv/bin/python -m rag.pipeline "How much is the tuition fee?"

# list live Groq models (confirm ids)
PYTHONPATH=src .venv/bin/python -m rag.generate --models

# validate the eval set against the corpus
.venv/bin/python eval/validate_gold.py

# refresh the corpus (fees/deadlines change), then ALWAYS re-chunk + re-index:
.venv/bin/python scripts/scrape.py            # rewrites data/ in place; does NOT chunk
PYTHONPATH=src .venv/bin/python -m rag.chunk && PYTHONPATH=src .venv/bin/python -m rag.index
```

Chunk + index need no keys; only generation does. Retrieval is ~6 ms; a full answer is
~0.6–0.9 s. Both `chunk` and `index` are idempotent (stable ids), so re-running is safe.

## Layout

```
data/            corpus: pages.jsonl (canonical) + pages/*.md mirror + manifest
eval/            gold_v1.jsonl — 40 groups × 6 phrasings = 240 eval queries
scripts/         scrape.py (refresh the corpus from studieren.univie.ac.at)
src/rag/         config, chunk, embed, index, query, retrieve, rerank, context,
                 prompts, llm (provider + key pool), generate, trace, ledger, pipeline
docs/            HANDOFF.md (start here: state, known issues, next steps),
                 ARCHITECTURE.md (design + decisions), EXPERIMENTS.md (run log + ledger)
runs/            per-run trace folders runs/<run_id>/ (gitignored)
qdrant_storage/  local Qdrant data (gitignored)
```

## Notes / gotchas

- **Local Qdrant allows one process at a time** (exclusive file lock) — you can't index
  and query concurrently. **Fix: run Qdrant in Docker** (below); parity is exact (both
  backends score `consistency` 0.426785) and no hash moves, since the backend is a
  location, not part of the index's identity. Payload indexes are also a silent no-op in
  local mode and only work on the server.
- **`temperature=0` is not deterministic on Groq** — the same question can return a
  good answer or "I don't know" across runs. That's a real consistency problem, not a
  bug. (But E0 shows the *headline* problem is retrieval-side, not generation-side.)
- **Groq free tier: 100k tokens/DAY/key/model** (~400k across the 4 keys). It does **not**
  appear in the response headers — only in the body of the 429. Generating all 240 gold
  queries costs ~508k, so it cannot finish in one day; the retrieval metrics cost 0.
- **`src/rag/chunk.py` is the only chunker** — `scrape.py` intentionally doesn't chunk.
  Always re-run `rag.chunk` + `rag.index` after a scrape (`rag.index` refuses stale chunks).
- `RAG_EMBED_THREADS` (default 6) caps onnxruntime so indexing doesn't peg all cores.
- The corpus is English-only and time-sensitive (fees/deadlines, scraped 2026-07-14).

## Evaluate it

```bash
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --retrieve-only  # headline, 0 tokens, no keys
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --limit 3        # smoke, end-to-end
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --score <run_id> # re-score off disk
.venv/bin/python -m pytest                                           # 106 tests
```

## Status

**E0 — the baseline — is measured** (`config_hash f22363afaf1d`, `gold_v1`, 40×6 = 240):

| ⭐ consistency | recall@k | recall@cand | mrr | | faithfulness † | citation_acc † |
|---|---|---|---|---|---|---|
| **0.427** | 0.875 | 0.967 | 0.757 | | 0.986 | 0.819 |

Retrieval metrics are complete (240/240), deterministic and free — selection happens
before generation, so the headline needs no LLM. † judged metrics come from a
section-biased fragment (daily token quota); see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

**What it says:** the gold page reaches the prompt ~88% of the time, yet paraphrases of
the same question select ~57% *different* page-sets. The paraphrase problem is
retrieval-side page-set churn — and it only partly reaches the answers
(`answer_agreement` 0.868). Rerank headroom is bounded at **+0.092**.

**Next steps and open findings live in [`docs/HANDOFF.md`](docs/HANDOFF.md)** — read it
first. Then: a **frontend** over `runs/`, and iteration levers (multi-query/HyDE + RRF
first, on E0's evidence) — each measured against E0 in `docs/EXPERIMENTS.md`.
