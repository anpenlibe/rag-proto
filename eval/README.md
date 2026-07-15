# `gold_v1` — grounded retrieval evaluation set

A hand-built, corpus-grounded evaluation set for the RAG system over the University of
Vienna **Studying** knowledge base (`../data/pages.jsonl`, 121 English pages).

**40 question groups × 6 phrasings = 240 queries.** Every group is anchored to a real page
whose text actually answers the question; nothing is invented.

## Files

| File | What it is |
| --- | --- |
| `gold_v1.jsonl` | The evaluation set. One JSON object per line, 40 lines, UTF-8 (non-ASCII kept as-is). |
| `validate_gold.py` | Structural + grounding validator. Run it after any edit. |

```bash
.venv/bin/python eval/validate_gold.py     # exits non-zero on any failure
```

## Schema

One JSON object per line:

```jsonc
{
  "group_id": "g37",                    // g01..g40, unique, zero-padded
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
| `group_id` | Stable id `g01`–`g40`. |
| `section` | One of the corpus's 9 landing sections. Always equals the `section` of every gold page in the group (enforced by the validator). |
| `canonical` | The base question as a real student would ask it. |
| `paraphrases` | Exactly 4 genuine rephrasings — same information need, different wording, length, register and question structure. Not synonym swaps. |
| `weird_framing` | One oblique phrasing of the *same* need: colloquial, scenario-embedded, or voice/ASR-style (lowercase, filler, mild disfluency). Still answerable from the same page. |
| `expected_urls` | 1–2 gold pages that actually answer the question. Present **verbatim** in the corpus. |
| `expected_page_ids` | The matching `id`(s) from `pages.jsonl`, positionally aligned with `expected_urls`. |
| `key_fact` | A short factual answer, supported by the gold page's text. Intended as a grading reference for answer quality, not a string to match exactly. |

The 6 phrasings per group are `canonical` + 4 `paraphrases` + `weird_framing`. Together they
form one retrieval-robustness cluster: **all six should retrieve the same gold page(s).**

## Section balance

Allocated proportionally to corpus size, so retrieval scores aren't dominated by one area.

| Section | Corpus pages | Groups | Share of set |
| --- | ---: | ---: | ---: |
| Study organisation | 45 | 15 | 37.5% |
| Admission | 28 | 9 | 22.5% |
| Degree programmes | 21 | 7 | 17.5% |
| Accessible studies | 9 | 3 | 7.5% |
| Entrance exam | 7 | 2 | 5.0% |
| Tuition fee | 5 | 2 | 5.0% |
| Resumption of studies | 4 | 1 | 2.5% |
| Web services | 1 | 1 | 2.5% |
| **Total** | **121** | **40** | **100%** |

`Web services` was chosen over `Graduates` for the single slot in that either/or bucket:
u:space is the tool students actually ask about, and the page carries concrete, checkable facts.

## Coverage

- **40 distinct gold pages** across 40 groups — a **distinct primary page per group**.
- **Zero `expected_url` reuse** across groups: no two groups share a gold page, so a retriever
  cannot score well by over-fitting to a handful of popular pages.
- Gold pages span **33% of the 121-page corpus**.

Within sections, groups spread deliberately across subtopics rather than clustering. Study
organisation, for example, covers registration (general / continuous-assessment rules /
concrete 2026S phase dates), STEOP, minimum credits, examination activity, semester dates,
four distinct AI topics (permitted use, citing, exams, the u:ai tool), confirmations,
personal data, campus safety and health services.

## How it was built

1. **Corpus survey.** Enumerated all 121 pages by section, title, URL and word count to map
   the topic space and pick the per-section allocation.
2. **Page selection.** Chose one distinct primary page per group, favouring pages with
   concrete, checkable facts and spreading across subtopics. Thin hub/link-farm pages were
   deliberately swapped out during drafting (e.g. *Studying remotely* and the *admission
   procedure* landing page were replaced by *Registration for courses with continuous
   assessment* and *Closing a degree programme*, which carry real answers).
3. **Grounding.** Read the **full `text`** of all 40 selected pages before writing anything.
   Each `canonical` was written against a fact actually present on its page, and each
   `key_fact` was lifted from that page's wording. Numbers (fees, ECTS, scores, dates,
   deadlines) were transcribed from the page, not recalled.
4. **Phrasing.** Wrote 4 paraphrases per group varying length, formality, vocabulary and
   question structure, plus 1 weird framing (colloquial / scenario / ASR-style).
5. **Generation.** A script emitted the JSONL, looking each `expected_url` up **from the
   corpus by page id** rather than transcribing URLs by hand — so URL/id agreement is correct
   by construction.
6. **Validation.** `validate_gold.py` re-checks everything from scratch against the corpus.

### Grounding notes / caveats

- **Time-sensitive content.** Many `key_fact`s quote fees, deadlines and semester dates
  (scraped 2026-07-14) — e.g. the 2026/27 application periods and the 2026S registration
  phases. If the corpus is re-scraped, re-verify these groups; the *questions* stay valid,
  but the `key_fact`s may drift.
- **Pages that resisted good questions.** Several study-organisation pages are hubs of
  outbound links with little standalone fact content (*Studying remotely*, *Studying and
  exams*, *Health during your studies*). *Health* was kept, anchored to its one concrete
  fact (the two vegetarian/vegan cafeteria locations); the other hubs were dropped in favour
  of richer pages. The huge *ABC of terminology* page (20k words) was skipped as a gold
  target: it answers a great many questions shallowly, so it makes an ambiguous retrieval
  target rather than a clean one.
- **List-heavy pages.** Some pages (*Admission to master programmes*, *Knowledge of Foreign
  Languages*) are mostly programme tables. Questions for these target the prose rules around
  the tables (eligibility, Visiting Master, whether a language level is binding or advisory)
  rather than asking the model to read a row out of a long list.
- **Near-neighbour pairs are intentional.** A few gold pages are topically adjacent and make
  good hard negatives for each other — e.g. *minimum number of credits* (16 ECTS / 4
  semesters, binding) vs *prüfungsaktiv* (16 ECTS / academic year, no personal consequence),
  and the three registration pages. Retrievers that confuse them should be penalised, which
  is the point.

## What the validator checks

- exactly 40 lines, each valid JSON, all 8 fields present, unique `group_id`s
- exactly 4 paraphrases per group; non-empty `canonical`, `weird_framing`, `key_fact`
- all 6 phrasings within a group are distinct → **240 total phrasings**
- 1–2 `expected_urls` per group, length-matched to `expected_page_ids`
- every `expected_url` exists **verbatim** among corpus urls
- every `expected_page_id` exists in the corpus **and** maps to its paired url
- each gold page's corpus `section` equals the group's declared `section`
- section counts match the allocation above
- reports distinct gold pages used and any `expected_url` reused across groups
