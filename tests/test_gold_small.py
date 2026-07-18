"""`gold_v1_small` — the project's gold eval set (10 groups × 6 = 60 phrasings).

It was carved from the original 40-group set (since removed) to keep evaluation cheap
while still a hard test: the deliberate near-neighbour hard-negative clusters must survive,
a spread of sections stays represented, and each group keeps 6 distinct phrasings + a
grounded expected page. Pure file checks — no corpus, no network. (Corpus-grounding lives
in `eval/validate_gold.py`.)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# With only 10 groups and both hard-negative clusters (5 Study-org groups) kept, coverage
# is 6 of the corpus's 8 gold sections — Entrance exam and Resumption drop out.
EXPECTED_SECTIONS = {
    "Study organisation", "Admission", "Degree programmes",
    "Accessible studies", "Tuition fee", "Web services",
}


def _load(name="gold_v1_small"):
    text = (ROOT / "eval" / f"{name}.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_ten_groups_with_unique_ids():
    groups = _load()
    ids = [g["group_id"] for g in groups]
    assert len(groups) == 10
    assert len(set(ids)) == 10


def test_preserves_the_hard_negative_clusters():
    ids = {g["group_id"] for g in _load()}
    assert {"g03", "g04"} <= ids          # minimum-credits vs prüfungsaktiv (both "16 ECTS")
    assert {"g01", "g06", "g07"} <= ids   # the three near-neighbour registration pages


def test_expected_sections_represented():
    assert {g["section"] for g in _load()} == EXPECTED_SECTIONS


def test_six_distinct_phrasings_and_grounded_pages():
    for g in _load():
        phrasings = [g["canonical"], *g["paraphrases"], g["weird_framing"]]
        assert len(phrasings) == 6 and len(set(phrasings)) == 6, g["group_id"]
        assert 1 <= len(g["expected_page_ids"]) <= 2, g["group_id"]
        assert len(g["expected_page_ids"]) == len(g["expected_urls"]), g["group_id"]
