"""The gold set as objects: 40 groups x 6 phrasings = 240 queries.

Each group is one information need asked six ways (canonical + 4 paraphrases + 1 weird
framing) against a known gold page. All six *should* retrieve the same page — the spread
across them is the headline metric.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from ..config import EVAL_DIR

CANONICAL = "canonical"
PARAPHRASE = "paraphrase"
WEIRD = "weird"


@dataclass(frozen=True)
class Phrasing:
    group_id: str
    kind: str          # canonical | paraphrase | weird
    idx: int
    text: str

    @property
    def query_id(self) -> str:
        """Stable join key, written into the trace at generation time.

        Without it, scoring could only match traces to gold by exact query text — which
        breaks silently the moment a gold phrasing is edited.
        """
        return f"{self.group_id}:{self.kind}:{self.idx}"


@dataclass(frozen=True)
class Group:
    group_id: str
    section: str
    key_fact: str
    expected_urls: tuple[str, ...]
    expected_page_ids: tuple[str, ...]
    phrasings: tuple[Phrasing, ...]      # exactly 6, deterministic order

    @property
    def paraphrase_set(self) -> tuple[Phrasing, ...]:
        """Canonical + 4 paraphrases (m=5) — the headline `consistency` population.

        Canonical belongs here: the problem is "the same question *reworded* gives
        different answers", so paraphrase-vs-canonical is the comparison that matters
        most. The weird framing is the stress sub-score, not part of this.
        """
        return tuple(p for p in self.phrasings if p.kind in (CANONICAL, PARAPHRASE))

    @property
    def all_set(self) -> tuple[Phrasing, ...]:
        return self.phrasings


def gold_path(gold_set: str = "gold_v1_small"):
    return EVAL_DIR / f"{gold_set}.jsonl"


def _phrasings(rec: dict) -> tuple[Phrasing, ...]:
    gid = rec["group_id"]
    out = [Phrasing(gid, CANONICAL, 0, rec["canonical"])]
    out += [Phrasing(gid, PARAPHRASE, i, t) for i, t in enumerate(rec["paraphrases"])]
    out.append(Phrasing(gid, WEIRD, 0, rec["weird_framing"]))
    return tuple(out)


def load_gold(gold_set: str = "gold_v1_small") -> list[Group]:
    path = gold_path(gold_set)
    try:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError as e:
        raise SystemExit(f"cannot read gold set {path}: {e}") from None

    groups = []
    for line in lines:
        r = json.loads(line)
        groups.append(Group(
            group_id=r["group_id"],
            section=r["section"],
            key_fact=r["key_fact"],
            expected_urls=tuple(r["expected_urls"]),
            expected_page_ids=tuple(r["expected_page_ids"]),
            phrasings=_phrasings(r),
        ))
    groups.sort(key=lambda g: g.group_id)
    return groups


def select_groups(groups: list[Group], limit: int | None = None,
                  seed: int = 0) -> list[Group]:
    """Pick `limit` groups — whole groups only, via a seeded shuffle.

    Group-atomic because `consistency` is defined across a group's phrasings: half a
    group scores nothing. Shuffled because the gold set is ordered by section, so a
    plain prefix would be a section-biased sample that isn't comparable to a full run.
    """
    if limit is None or limit >= len(groups):
        return groups
    picked = random.Random(seed).sample(groups, limit)
    picked.sort(key=lambda g: g.group_id)
    return picked


def phrasings_of(groups: list[Group]) -> list[Phrasing]:
    return [p for g in groups for p in g.phrasings]


# Pinned judge subsamples. Deterministic by construction (a fixed rule, not a sample) so
# two runs of the same `judge_sample` judge exactly the same phrasings and stay
# comparable. Keyed into `eval_hash` via Config.judge_sample.
JUDGE_SAMPLES = {
    "all": lambda g: g.phrasings,
    "c+2p": lambda g: tuple(p for p in g.phrasings
                            if p.kind == CANONICAL
                            or (p.kind == PARAPHRASE and p.idx in (0, 1))),
    "c": lambda g: tuple(p for p in g.phrasings if p.kind == CANONICAL),
}


def judge_phrasings(group: Group, sample: str = "all") -> tuple[Phrasing, ...]:
    """Which of a group's phrasings get per-answer judging."""
    try:
        return tuple(JUDGE_SAMPLES[sample](group))
    except KeyError:
        raise KeyError(
            f"unknown judge_sample {sample!r}; known: {sorted(JUDGE_SAMPLES)}"
        ) from None
