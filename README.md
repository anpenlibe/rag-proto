# rag-proto

A small, **modular, fully-traced RAG pipeline** with grounded citations, evaluated
against a measured baseline. It answers natural-language questions over a 121-page
University of Vienna "Studying" knowledge base — admission, tuition, exams, degree
programmes — and logs every hop of how it got there.

Built as an onboarding assignment for a **voice-agent company** whose retrieval stack
is Qdrant + a small embedding model + an LLM API. Everything here runs **free and
local** (16 GB RAM, RTX 3060 6 GB) — no paid services.

```
$ python -m rag.pipeline "How much is the tuition fee for non-EU students?"

Non-EU/EEA degree students pay €726.72 in tuition plus the €26.20 Students' Union
(ÖH) fee — €752.92 per semester in total [1].

  [1] Tuition fees → Amount non-EU/EEA   studieren.univie.ac.at/en/…/tuition-fees
```

---

## The problem it's built to expose

A voice agent that answers from a knowledge base has three failure modes this project
targets, in priority order:

1. **Paraphrase robustness** — the same question, reworded, should give the *same*
   answer. Today it doesn't. **This is the headline metric.**
2. **Traceability** — every hop (query → retrieval → prompt → answer → sources) must be
   inspectable and loggable, not a black box. *([See it in the console below.](#the-console--track-every-hop))*
3. **Citations** — every answer must cite the sources it drew from.

The strategy is deliberate: ship a **plain baseline** that mirrors the company's stack,
**measure it honestly**, then flip on improvements **one at a time** so each is
attributable. No improvement lands without a number next to it.

## The console — track every hop

The traceability requirement isn't just log files: there's a **zero-dependency web console**
over every run where you can actually watch what the pipeline did. Replay any logged query
end to end — pipeline rail → the 20 retrieved candidates → the 6 selected → the *exact*
prompt sent to the model → the raw answer → resolved citations — plus a scored run's eval
panel and paraphrase-divergence strip. You can also ask a live question or run the eval set
from the UI.

![The trace console replaying one query: the pipeline rail (transform → retrieve → rerank → select → assemble → generate) above the grounded answer, whose inline [n] markers resolve to the numbered sources with their heading path and URL.](docs/img/console-trace.png)

```bash
PYTHONPATH=src .venv/bin/python -m webui.server   # http://127.0.0.1:8000
```

**It works on a fresh clone with nothing else running** — replay needs only the filesystem,
no Qdrant and no keys, and the committed E0 baseline ships 60 real answers to browse. A live
*Ask* or *Run eval* additionally needs Qdrant + keys; the console shows both statuses and a
running token tally. Built on stdlib `http.server`, localhost only. *(Setup for live queries
is in [Quickstart](#quickstart) below.)*

## What the baseline scores

**E0** is the measured "before" — a strict-parity baseline, scored on a purpose-built
gold set of 10 question groups × 6 paraphrases = 60 queries (`config_hash f22363afaf1d`,
2026-07-18):

| ⭐ consistency | recall@k | recall@cand | mrr | | faithfulness | citation_acc | answer_agree |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.428** | 0.783 | 0.917 | 0.674 | | 0.933 | 0.750 | 0.700 |

A **complete, fully-judged panel** — 60/60 queries, 0 judge failures. Retrieval metrics
are deterministic and free; the judged trio (faithfulness / citation accuracy / answer
agreement) comes from an LLM-judge.

**What the numbers say:** the correct page reaches the prompt ~78% of the time, yet
paraphrases of the same question pull ~57% *different* page-sets. So the paraphrase
problem is **retrieval-side page-set churn**, not "can't find the answer" — and it only
partly reaches the answers (`answer_agreement` 0.700 > `consistency` 0.428, i.e. the
generator is more robust than the retriever). The gold set is deliberately hard: half
its groups are near-neighbour hard negatives designed to punish sloppy retrieval.

Full run log and interpretation in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## How it works

```
pages.jsonl ─►[chunk]─►[embed]─►[index: Qdrant]                        (offline, once)

query ─►[transform]─►[retrieve 20]─►[rerank]─►[select 6]─►[assemble]─►[generate: Groq]
                                                                    └─► grounded, cited answer
        └────── every hop written to runs/<run_id>/queries/<trace_id>.json ──────┘
```

- **Chunk** — heading-aware, ~230-word chunks with the heading path prepended, so a small
  chunk stays meaningful *and* carries its own citation anchor.
- **Embed** — `bge-small-en-v1.5` (384-d) via fastembed — ONNX on CPU, no torch, no GPU
  needed. Asymmetric query-vs-passage encoding.
- **Index** — Qdrant (Docker server, or a local-file fallback), full chunk payload stored
  for citations.
- **Retrieve → rerank → select** — dense search returns a 20-candidate pool; rerank is a
  passthrough today (the seam where a cross-encoder plugs in); the top 6 go to the prompt.
  All three sets are traced separately.
- **Generate** — Groq (`llama-3.3-70b-versatile`, `temperature=0`), a grounded *cite-only*
  prompt (answer only from the sources, else "I don't know") with citation validation.
  Every LLM call flows through one provider seam and a **4-key pool** that round-robins and
  rotates on rate limits — shared by the generator and the judge.
- **Trace** — each run records the *full* pipeline per query (retrieved pool → selected →
  exact prompt → raw response → citations), plus per-stage latency, token cost, and any
  rate-limit rotations.

**The design principle that ties it together:** every layer is a swappable component
behind a small interface, wired by one `Config`. **An experiment is a config diff, not a
rewrite** — that's what makes each improvement cleanly attributable. Design rationale and
the full decision log are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quickstart

Requires Python 3.12 (the ML stack has no 3.14 wheels yet) and Docker for Qdrant.

```bash
# 1. environment
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo 'GROQ_API_KEYS=gsk_a,gsk_b,gsk_c,gsk_d' > .env   # your Groq keys (gitignored)

# 2. start Qdrant
docker compose up -d
export QDRANT_URL=http://localhost:6333

# 3. build the index (once)
PYTHONPATH=src .venv/bin/python -m rag.chunk      # pages.jsonl -> data/chunks.jsonl
PYTHONPATH=src .venv/bin/python -m rag.index      # -> Qdrant

# 4. ask a question
PYTHONPATH=src .venv/bin/python -m rag.pipeline "How much is the tuition fee?"
```

Chunk and index need no API keys — only generation does. Retrieval is ~6 ms; a full
answer is ~0.6–0.9 s. Both `chunk` and `index` are idempotent, so re-running is safe.

## Evaluate it

```bash
# headline retrieval panel (60 queries, 0 tokens, no keys, ~30 s)
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --retrieve-only

# full generate + judge panel (60 queries, fits in one day's free quota)
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --label E0

# re-score an existing run off disk, no regeneration
PYTHONPATH=src .venv/bin/python -m rag.eval.harness --score <run_id>

.venv/bin/python -m pytest        # 132 tests — no network, no Qdrant
```

Evaluation is two-phase on purpose: **generate** spends the token budget and writes traces
to disk; **score** reads them back, so metrics and judge prompts can be iterated without
regenerating.

## Repo layout

```
data/            corpus: pages.jsonl (canonical) + pages/*.md mirror + manifest
eval/            gold_v1_small.jsonl — 10 groups × 6 phrasings = 60 eval queries
scripts/         scrape.py (refresh the corpus from studieren.univie.ac.at)
src/rag/         config, chunk, embed, index, query, retrieve, rerank, context,
                 prompts, llm (provider + key pool), generate, trace, ledger, pipeline
src/rag/eval/    gold · traces · metrics · judge · harness — the scoring harness
src/webui/       zero-dep console over runs/
docs/            HANDOFF.md · ARCHITECTURE.md · EXPERIMENTS.md
runs/            per-run trace folders (gitignored)
```

## Where to read next

- **[`docs/HANDOFF.md`](docs/HANDOFF.md)** — start here: current state, open findings, and
  what to build next (the data says: retrieval-side levers first).
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — the design and the decision log (the
  *why* behind every seam).
- **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)** — the run ledger; one lever per entry,
  each measured against E0.

## Good to know

- **Groq free tier: 100k tokens/day/key/model** (~400k across 4 keys). This cap is
  invisible in the response headers — it only shows in the body of a 429. A full
  generate + judge panel over the 60-query gold set fits in one day.
- **`temperature=0` isn't deterministic on Groq** — the same question can return a good
  answer or "I don't know" across runs. That's a real consistency problem, not a bug (and
  E0 shows it's secondary to the retrieval-side one).
- **Local Qdrant allows one process at a time** (an exclusive file lock), which blocks
  indexing and querying concurrently — hence the Docker server above. The two backends are
  bit-identical and score the same, so switching moves no hash.
- The corpus is **English-only and time-sensitive** (fees/deadlines, scraped 2026-07-14).
  Re-run `scripts/scrape.py`, then always `rag.chunk` + `rag.index` — `scrape.py` never
  chunks, and `rag.index` refuses stale chunks.
</content>
</invoke>
