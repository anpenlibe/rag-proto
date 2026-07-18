# `gold_v1_small` — grounded retrieval evaluation set

A hand-built, corpus-grounded evaluation set for the RAG system over the University of
Vienna **Studying** knowledge base (`../data/pages.jsonl`, 121 English pages).

**10 question groups × 6 phrasings = 60 queries.** Every group is anchored to a real page
whose text actually answers the question; nothing is invented. The 10 groups were distilled
from an original 40-group draft (`gold_v1`, since removed) to keep evaluation cheap while
staying a hard test — see *Selection* below.

## Files

| File | What it is |
| --- | --- |
| `gold_v1_small.jsonl` | The evaluation set. One JSON object per line, 10 lines, UTF-8 (non-ASCII kept as-is). |
| `validate_gold.py` | Structural + grounding validator. Run it after any edit. |

```bash
.venv/bin/python eval/validate_gold.py     # default gold_v1_small; exits non-zero on failure
```

## Schema

One JSON object per line:

```jsonc
{
  "group_id": "g37",                    // from the original g01..g40 numbering (non-contiguous)
  "section": "Tuition fee",             // corpus section the group comes from
  "canonical": "How much is the tuition fee at the University of Vienna?",
  "paraphrases": ["...", "...", "...", "..."],   // EXACTLY 4, same intent, varied surface form
  "weird_framing": "asking for a friend — that uni vienna money thing per semester, ballpark? ...",
  "expected_urls": ["https://studieren.univie.ac.at/en/tuition-fee/amount-tuition-fee/"],
  "expected_page_ids": ["0f5bfccf9812"],
  "key_fact": "EU/EEA/Switzerland citizens pay only the ~26.20 EUR Students' Union fee ..."
}
```

| Field | Meaning / contract |
| --- | --- |
| `group_id` | Stable id from the original `g01`–`g40` numbering; the subset is non-contiguous. |
| `section` | A corpus landing section. Always equals the `section` of every gold page in the group (enforced by the validator). |
| `canonical` | The base question as a real student would ask it. |
| `paraphrases` | Exactly 4 genuine rephrasings — same information need, different wording, length, register and question structure. Not synonym swaps. |
| `weird_framing` | One oblique phrasing of the *same* need: colloquial, scenario-embedded, or voice/ASR-style (lowercase, filler, mild disfluency). Still answerable from the same page. |
| `expected_urls` | 1–2 gold pages that actually answer the question. Present **verbatim** in the corpus. |
| `expected_page_ids` | The matching `id`(s) from `pages.jsonl`, positionally aligned with `expected_urls`. |
| `key_fact` | A short factual answer, supported by the gold page's text. A grading reference for answer quality, not a string to match exactly. |

The 6 phrasings per group are `canonical` + 4 `paraphrases` + `weird_framing`. Together they
form one retrieval-robustness cluster: **all six should retrieve the same gold page(s).**

## Composition (10 groups)

| Section | Groups | Group ids |
| --- | ---: | --- |
| Study organisation | 5 | g01, g03, g04, g06, g07 |
| Admission | 1 | g17 |
| Degree programmes | 1 | g25 |
| Accessible studies | 1 | g32 |
| Tuition fee | 1 | g37 |
| Web services | 1 | g40 |
| **Total** | **10** | |

Study organisation is over-weighted on purpose: it holds **both deliberate hard-negative
clusters** (below), which are the point of the set. That leaves 5 slots for one group from
each of five other sections; **Entrance exam** and **Resumption of studies** drop out at this
size. 10 distinct gold pages, **zero `expected_url` reuse** across groups.

### Near-neighbour hard negatives (intentional)

Some gold pages are topically adjacent and make good hard negatives for each other — a
retriever that confuses them should be penalised, which is the point:

- **`g03` minimum credits** (16 ECTS / 4 semesters, binding) ↔ **`g04` prüfungsaktiv**
  (16 ECTS / academic year, no personal consequence) — same "16 ECTS", different window and
  consequence.
- **The three registration pages** — `g01` (register for courses/exams), `g06` (registration
  open dates), `g07` (missing a continuous-assessment first session).

## How it was built

1. **Corpus survey** — enumerated all 121 pages by section/title/URL/word-count to map topics.
2. **Page selection** — one distinct primary page per group, favouring pages with concrete,
   checkable facts; thin hub/link-farm pages were deliberately avoided.
3. **Grounding** — read the full `text` of each selected page before writing; each `canonical`
   targets a fact actually on the page, and each `key_fact` is lifted from that page's wording.
   Numbers (fees, ECTS, scores, dates) were transcribed, not recalled.
4. **Phrasing** — 4 paraphrases per group varying length/formality/vocabulary/structure, plus
   1 weird framing (colloquial / scenario / ASR-style).
5. **Generation** — a script emitted the JSONL, looking each `expected_url` up from the corpus
   by page id, so URL/id agreement is correct by construction.
6. **Validation** — `validate_gold.py` re-checks everything against the corpus.

### Caveats

- **Time-sensitive content.** Many `key_fact`s quote fees, deadlines and semester dates
  (scraped 2026-07-14). If the corpus is re-scraped, re-verify; the *questions* stay valid,
  but the `key_fact`s may drift.
- **Editing in place moves no hash.** `eval_hash` keys on the gold-set *name*, not its
  contents, so changing `gold_v1_small.jsonl` yields runs that look comparable but aren't. It's
  fine here (older runs on other contents were wiped); for a lasting change, make a new
  versioned set (`gold_v2`, …) rather than editing this file.

The measured baseline on this set lives in [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

## What the validator checks

`python eval/validate_gold.py [gold_set]` (default `gold_v1_small`). All checks run for any set:

- each line valid JSON, all 8 fields present, unique `group_id`s
- exactly 4 paraphrases per group; non-empty `canonical`, `weird_framing`, `key_fact`
- all 6 phrasings within a group are distinct → **6 × groups** total phrasings
- 1–2 `expected_urls` per group, length-matched to `expected_page_ids`
- every `expected_url` exists **verbatim** among corpus urls
- every `expected_page_id` exists in the corpus **and** maps to its paired url
- each gold page's corpus `section` equals the group's declared `section`
- reports distinct gold pages used and any `expected_url` reused across groups
