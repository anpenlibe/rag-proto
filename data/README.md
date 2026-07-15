# University of Vienna – Studying (structured scrape)

Clean structured scrape of **https://studieren.univie.ac.at/en/** — all 9 landing
sections and their sub-pages.

Scraped 2026-07-14 · 121 pages · ~119k words · English only.

## Files
- **`pages.jsonl`** — one page per line. **The canonical corpus** and the chunker's input.
- **`pages/*.md`** — same content as readable markdown, for eyeballing (1:1 with `pages.jsonl`).
  Not a pipeline input.
- **`chunks.jsonl`** — **generated**, don't hand-edit. Produced by `rag.chunk` from
  `pages.jsonl` (720 chunks); regenerate with `PYTHONPATH=src python -m rag.chunk`.
- **`manifest.json`** — counts, sections, skipped URLs.
- **`../scripts/scrape.py`** — the scraper. Re-run to refresh fees/deadlines; it
  **rewrites this directory in place** and deliberately does **not** chunk. Always
  follow it with `python -m rag.chunk && python -m rag.index`.

## `pages.jsonl` fields
```json
{
  "id":        "0f5bfccf9812",           // stable md5-based page id
  "url":       "https://studieren.univie.ac.at/en/tuition-fee/amount-tuition-fee/",
  "title":     "Amount of the tuition fee/Students' Union fee",   // page H1
  "section":   "Tuition fee",            // one of the 9 landing sections
  "language":  "en",                      // always "en" (German pages removed)
  "breadcrumb":["University of Vienna","Studying at the University of Vienna","..."],
  "headings":  [{"level":2,"text":"Citizenship EU/EEA/CH"}],       // structural outline
  "word_count":2014,
  "html_title":"Amount ... | Studying",
  "text":      "…full clean page as markdown…"                     // your chunking input
}
```

## Chunking
`src/rag/chunk.py` is the **single chunker** — driven by `Config` (`target_words`,
`overlap_words`), so chunk params are part of `config_hash`. Don't add a second one
(see `docs/ARCHITECTURE.md` decision #19).

It works because `text` is markdown with `##`/`###` intact, so it splits on headings
directly (`headings` is the same outline pre-parsed, if you'd rather walk structure
without re-parsing). Every other field becomes chunk metadata — and `url` + `title` +
`section` + heading path are what the answer's citations are built from.

## Notes
- 2 URLs skipped (see `manifest.json`): one external redirect, one empty stub.
- German-only pages have been removed — the corpus is now English only (was 124 pages / 3 de).
- Scope: kept to the `studieren.univie.ac.at/en/` tree. Some sections link out to
  separate subdomains (`aufnahmeverfahren`, `doktorat`, `lehramt`, `postgraduatecenter`)
  that weren't crawled.
