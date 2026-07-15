"""The LLM-judge: faithfulness, citation_acc (per query) and answer_agreement (per group).

Runs on `cfg.judge_model` — deliberately a *different* model from the generator so it
isn't grading itself — through the same `rag.llm.KeyPool`. Groq rate-limits per model,
so a distinct judge also draws on its own token budget rather than competing with
generation for one.

**A judge failure is not a pipeline failure.** If the judge truncates, refuses, or emits
unparseable output, that verdict scores `None` and is excluded from the mean — never 0.
Scoring the pipeline 0 because our evaluator misbehaved would manufacture exactly the
kind of fake result this codebase spends so much effort preventing. `n_judged` is
reported so a thin denominator is visible.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

from ..config import Config
from ..llm import KeyPool
from ..prompts import (
    judge_agreement_prompt,
    judge_agreement_user_prompt,
    judge_prompt,
    judge_user_prompt,
)
from .gold import Group, judge_phrasings
from .traces import QueryTrace

OK = "ok"
UNPARSEABLE = "unparseable"
TRUNCATED = "truncated"
ERROR = "error"

# Groq exposes the per-minute bucket in response headers but the per-DAY quota ONLY in
# the body of the 429 it refuses with. A TPM limit is worth waiting out (the pool does);
# a TPD limit is not — it resets tomorrow.
_QUOTA_MARKERS = ("tokens per day", "TPD")


def is_quota_exhausted(detail: str) -> bool:
    return bool(detail) and any(m in detail for m in _QUOTA_MARKERS)


@dataclass
class JudgeVerdict:
    query_id: str = ""
    faithfulness: int | None = None
    citation_acc: int | None = None
    reason: str = ""
    status: str = OK
    raw: str = ""
    usage: dict = field(default_factory=dict)
    key_id: int | None = None

    @property
    def usable(self) -> bool:
        return self.status == OK and self.faithfulness is not None


@dataclass
class AgreementVerdict:
    group_id: str = ""
    consistent: list[int] = field(default_factory=list)
    score: float | None = None
    reason: str = ""
    status: str = OK
    raw: str = ""
    usage: dict = field(default_factory=dict)


def extract_json(text: str) -> dict | None:
    """Pull the last top-level {...} out of a response and parse it.

    `gpt-oss-120b` is a reasoning model: it may emit analysis, a markdown fence, or both
    around the object. Scanning for balanced braces and preferring the LAST candidate
    survives all three without depending on `response_format` support.
    """
    if not text:
        return None
    candidates, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
    for blob in reversed(candidates):
        try:
            out = json.loads(blob)
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            continue
    return None


def _bit(v) -> int | None:
    """Accept 1/0, true/false, "1"/"yes" — models are inconsistent about JSON types."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return int(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes"):
            return 1
        if s in ("0", "false", "no"):
            return 0
    return None


class Judge:
    def __init__(self, cfg: Config | None = None, pool: KeyPool | None = None):
        self.cfg = cfg or Config()
        self._pool = pool

    @property
    def pool(self) -> KeyPool:
        if self._pool is None:
            self._pool = KeyPool()
        return self._pool

    def _call(self, system: str, user: str):
        return self.pool.complete(
            model=self.cfg.judge_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=self.cfg.judge_max_tokens,
        )

    def grade_answer(self, question: str, answer: str, context: str,
                     query_id: str = "") -> JudgeVerdict:
        try:
            r = self._call(judge_prompt(self.cfg.judge_prompt_version),
                           judge_user_prompt(question, answer, context))
        except Exception as e:
            return JudgeVerdict(query_id=query_id, status=ERROR,
                                reason=f"{type(e).__name__}: {str(e)[:120]}")
        v = JudgeVerdict(query_id=query_id, raw=r.text, usage=r.usage, key_id=r.key_id)
        if r.finish_reason == "length":
            v.status = TRUNCATED
            return v
        obj = extract_json(r.text)
        if obj is None:
            v.status = UNPARSEABLE
            return v
        v.faithfulness = _bit(obj.get("faithfulness"))
        v.citation_acc = _bit(obj.get("citation_acc"))
        v.reason = str(obj.get("reason", ""))[:300]
        if v.faithfulness is None and v.citation_acc is None:
            v.status = UNPARSEABLE
        return v

    def grade_agreement(self, group: Group, answers: list[str]) -> AgreementVerdict:
        try:
            r = self._call(judge_agreement_prompt(self.cfg.judge_prompt_version),
                           judge_agreement_user_prompt(group.key_fact, answers))
        except Exception as e:
            return AgreementVerdict(group_id=group.group_id, status=ERROR,
                                    reason=f"{type(e).__name__}: {str(e)[:120]}")
        v = AgreementVerdict(group_id=group.group_id, raw=r.text, usage=r.usage)
        if r.finish_reason == "length":
            v.status = TRUNCATED
            return v
        obj = extract_json(r.text)
        if obj is None:
            v.status = UNPARSEABLE
            return v
        bits = [_bit(x) for x in (obj.get("consistent") or [])]
        bits = [b for b in bits if b is not None]
        v.reason = str(obj.get("reason", ""))[:300]
        # A vector of the wrong length means the judge lost track of which answer is
        # which — the per-answer scores can't be trusted, so throw the verdict away.
        if len(bits) != len(answers):
            v.status = UNPARSEABLE
            return v
        v.consistent = bits
        v.score = sum(bits) / len(bits)
        return v


def judge_run(cfg: Config, groups: list[Group], traces: list[QueryTrace], out_dir,
              pool: KeyPool | None = None):
    """Judge a finished run's traces. Returns (panel_fragment, meta_fragment)."""
    from .traces import by_query_id
    tq = by_query_id(traces)
    judge = Judge(cfg, pool=pool)

    # Only answers that actually exist and are complete can be judged. A truncated
    # answer is cut mid-sentence by max_tokens — scoring it unfaithful would report a
    # budget artifact as a grounding failure (known issue #16).
    judgeable = [
        (g, p, t) for g in groups for p in judge_phrasings(g, cfg.judge_sample)
        if (t := tq.get(p.query_id)) is not None
        and t.status == "ok" and t.answer and not t.truncated
    ]

    n_calls = len(judgeable) + len(groups)
    print(f"\njudge: {cfg.judge_model} · sample={cfg.judge_sample!r} · "
          f"{len(judgeable)} answers + {len(groups)} groups = {n_calls} calls "
          f"(~{n_calls * 3000 // 1000}k tokens)")

    verdicts: list[JudgeVerdict] = []
    t0 = time.perf_counter()
    exhausted = False
    for i, (g, p, t) in enumerate(judgeable, 1):
        v = judge.grade_answer(p.text, t.answer, t.context, p.query_id)
        verdicts.append(v)
        # A per-DAY quota can't be waited out inside a run. Once every key is out, each
        # further call would burn a full rotation (n_keys x max_cycles requests plus
        # ~61s of backoff) to fail anyway — stop, and report the honest denominator.
        if is_quota_exhausted(v.reason):
            exhausted = True
            print(f"  !! daily token quota exhausted after {i}/{len(judgeable)} answers "
                  f"— stopping early rather than grinding through guaranteed failures",
                  file=sys.stderr)
            break
        if i % 40 == 0 or i == len(judgeable):
            bad = sum(1 for v in verdicts if not v.usable)
            print(f"  answers {i}/{len(judgeable)}  {time.perf_counter() - t0:5.1f}s  "
                  f"unusable={bad}")

    agreements: list[AgreementVerdict] = []
    for i, g in enumerate(groups, 1):
        if exhausted:
            break
        # agreement always sees every answer the group has — it's one call per group
        # either way, so sampling it would save nothing and only weaken the metric.
        answers = [t.answer for p in g.phrasings
                   if (t := tq.get(p.query_id)) is not None and t.status == "ok"
                   and t.answer]
        if len(answers) < 2:
            agreements.append(AgreementVerdict(group_id=g.group_id, status=ERROR,
                                               reason="fewer than 2 answers to compare"))
            continue
        a = judge.grade_agreement(g, answers)
        agreements.append(a)
        if is_quota_exhausted(a.reason):
            exhausted = True
            print(f"  !! daily token quota exhausted after {i}/{len(groups)} groups",
                  file=sys.stderr)
            break
        if i % 20 == 0 or i == len(groups):
            print(f"  groups  {i}/{len(groups)}  {time.perf_counter() - t0:5.1f}s")

    _dump(out_dir, verdicts, agreements)

    faith = [v.faithfulness for v in verdicts if v.usable and v.faithfulness is not None]
    cite = [v.citation_acc for v in verdicts if v.usable and v.citation_acc is not None]
    agree = [a.score for a in agreements if a.score is not None]

    panel = {
        "faithfulness": sum(faith) / len(faith) if faith else None,
        "citation_acc": sum(cite) / len(cite) if cite else None,
        "answer_agreement": sum(agree) / len(agree) if agree else None,
    }
    n_failed = sum(1 for v in verdicts if not v.usable)
    meta = {
        "judge_sample": cfg.judge_sample,
        "n_judgeable": len(judgeable),
        "n_judged": len(faith),
        "n_judge_failed": n_failed,
        "judge_quota_exhausted": exhausted,
        "n_groups_agreement_scored": len(agree),
        "judge_tokens": (sum(v.usage.get("total_tokens", 0) for v in verdicts)
                         + sum(a.usage.get("total_tokens", 0) for a in agreements)),
        "judge_status_counts": _counts(v.status for v in verdicts),
    }
    # The judged metrics are only as good as their denominator; say so loudly rather
    # than quietly reporting a mean over a handful of verdicts.
    if judgeable and len(faith) / len(judgeable) < 0.95:
        print(f"  !! judge produced usable verdicts for only {len(faith)}/{len(judgeable)}"
              f" answers — faithfulness/citation_acc are SUSPECT", file=sys.stderr)
    return panel, meta


def _counts(statuses) -> dict:
    out: dict[str, int] = {}
    for s in statuses:
        out[s] = out.get(s, 0) + 1
    return dict(sorted(out.items()))


def _dump(out_dir, verdicts, agreements):
    """Judge calls are LLM calls — traceability (requirement #2) applies to them too.

    These go to plain files under runs/<id>/eval/, NOT through the run's Tracer: the run
    is already closed, and Run._record would rewrite its manifest back to status "open"
    and fold judge tokens into the pipeline's cost.
    """
    d = out_dir / "judge"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "answers.jsonl").open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v.__dict__, ensure_ascii=False) + "\n")
    with (d / "agreement.jsonl").open("w", encoding="utf-8") as f:
        for a in agreements:
            f.write(json.dumps(a.__dict__, ensure_ascii=False) + "\n")
