#!/usr/bin/env python3
"""Validate eval/gold_v1.jsonl against the corpus in data/pages.jsonl.

Usage:  .venv/bin/python eval/validate_gold.py
Exits non-zero if any check fails.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "pages.jsonl"
GOLD = ROOT / "eval" / "gold_v1.jsonl"

REQUIRED_FIELDS = [
    "group_id", "section", "canonical", "paraphrases",
    "weird_framing", "expected_urls", "expected_page_ids", "key_fact",
]

TARGET_SECTIONS = {
    "Study organisation": 15,
    "Admission": 9,
    "Degree programmes": 7,
    "Accessible studies": 3,
    "Entrance exam": 2,
    "Tuition fee": 2,
    "Resumption of studies": 1,
}
# One group must come from Web services OR Graduates.
EITHER_OR = ({"Web services", "Graduates"}, 1)

errors = []
def check(cond, msg):
    if not cond:
        errors.append(msg)

# ---- load corpus ----
pages = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
url_by_id = {p["id"]: p["url"] for p in pages}
all_urls = {p["url"] for p in pages}
section_by_id = {p["id"]: p["section"] for p in pages}

# ---- load gold ----
raw_lines = [l for l in GOLD.open(encoding="utf-8").read().splitlines() if l.strip()]
print(f"lines in gold_v1.jsonl: {len(raw_lines)}")
check(len(raw_lines) == 40, f"FAIL: expected exactly 40 lines, got {len(raw_lines)}")

groups = []
for n, line in enumerate(raw_lines, start=1):
    try:
        groups.append(json.loads(line))
    except json.JSONDecodeError as e:
        errors.append(f"FAIL: line {n} is not valid JSON: {e}")

# ---- per-group checks ----
phrasing_count = 0
seen_ids = set()
for g in groups:
    gid = g.get("group_id", "<missing>")
    for f in REQUIRED_FIELDS:
        check(f in g, f"FAIL: {gid}: missing field '{f}'")
    check(gid not in seen_ids, f"FAIL: duplicate group_id {gid}")
    seen_ids.add(gid)

    # exactly 4 paraphrases, non-empty canonical + weird_framing
    ps = g.get("paraphrases", [])
    check(isinstance(ps, list) and len(ps) == 4,
          f"FAIL: {gid}: expected 4 paraphrases, got {len(ps)}")
    check(all(isinstance(p, str) and p.strip() for p in ps),
          f"FAIL: {gid}: empty paraphrase")
    check(bool(str(g.get("canonical", "")).strip()), f"FAIL: {gid}: empty canonical")
    check(bool(str(g.get("weird_framing", "")).strip()), f"FAIL: {gid}: empty weird_framing")
    check(bool(str(g.get("key_fact", "")).strip()), f"FAIL: {gid}: empty key_fact")
    phrasing_count += 1 + len(ps) + 1  # canonical + paraphrases + weird_framing

    # all 6 phrasings distinct within the group
    all_p = [g.get("canonical", "")] + ps + [g.get("weird_framing", "")]
    check(len(set(all_p)) == len(all_p), f"FAIL: {gid}: duplicate phrasings within group")

    urls = g.get("expected_urls", [])
    pids = g.get("expected_page_ids", [])
    check(1 <= len(urls) <= 2, f"FAIL: {gid}: expected 1-2 expected_urls, got {len(urls)}")
    check(len(urls) == len(pids), f"FAIL: {gid}: expected_urls/expected_page_ids length mismatch")

    for u in urls:
        check(u in all_urls, f"FAIL: {gid}: url not in corpus verbatim: {u}")
    for pid in pids:
        check(pid in url_by_id, f"FAIL: {gid}: page id not in corpus: {pid}")
    # id <-> url must match, and page section must equal the group's section
    for u, pid in zip(urls, pids):
        check(url_by_id.get(pid) == u,
              f"FAIL: {gid}: id {pid} maps to {url_by_id.get(pid)}, not {u}")
        check(section_by_id.get(pid) == g.get("section"),
              f"FAIL: {gid}: page {pid} is section '{section_by_id.get(pid)}' "
              f"but group says '{g.get('section')}'")

# ---- section distribution ----
sec_counts = Counter(g.get("section") for g in groups)
print("\nsection distribution:")
for s, c in sec_counts.most_common():
    print(f"  {c:2d}  {s}")

for s, want in TARGET_SECTIONS.items():
    got = sec_counts.get(s, 0)
    check(got == want, f"FAIL: section '{s}': expected {want}, got {got}")
either_got = sum(sec_counts.get(s, 0) for s in EITHER_OR[0])
check(either_got == EITHER_OR[1],
      f"FAIL: sections {EITHER_OR[0]}: expected {EITHER_OR[1]} total, got {either_got}")

# ---- cross-group duplicate phrasings ----
# The same phrasing in two groups would corrupt per-group consistency scoring.
all_phrasings = [p.strip().lower()
                 for g in groups
                 for p in [g.get("canonical", "")] + g.get("paraphrases", [])
                          + [g.get("weird_framing", "")]
                 if p and p.strip()]
cross_dupes = [q for q, c in Counter(all_phrasings).items() if c > 1]
check(not cross_dupes,
      f"FAIL: {len(cross_dupes)} phrasing(s) reused across groups, e.g. {cross_dupes[:3]}")

# ---- coverage ----
distinct_pages = {pid for g in groups for pid in g.get("expected_page_ids", [])}
all_gold_urls = [u for g in groups for u in g.get("expected_urls", [])]
dupe_urls = [u for u, c in Counter(all_gold_urls).items() if c > 1]

print(f"\ntotal groups:            {len(groups)}")
print(f"total phrasings:         {phrasing_count}  (expect 240)")
print(f"distinct expected pages: {len(distinct_pages)} of {len(pages)} corpus pages")
print(f"expected_url reuse across groups: {len(dupe_urls)} url(s) used by >1 group")
check(phrasing_count == 240, f"FAIL: expected 240 phrasings, got {phrasing_count}")

# ---- report ----
print()
if errors:
    for e in errors:
        print(e)
    print(f"\n{len(errors)} CHECK(S) FAILED")
    sys.exit(1)
print("ALL CHECKS PASSED")
