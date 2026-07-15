"""Judge output parsing + the "a judge failure is not a pipeline failure" rule.

Both failure modes here were observed live against `openai/gpt-oss-120b`: it is a
reasoning model, so it wraps its JSON in prose/fences, and at the generator's
max_tokens it burned the whole budget thinking and returned EMPTY content with
finish_reason="length".
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from rag.config import Config
from rag.eval.gold import Group, Phrasing
from rag.eval.judge import TRUNCATED, UNPARSEABLE, Judge, _bit, extract_json
from rag.llm import LLMResponse


# -- tolerant JSON extraction ---------------------------------------------------------
def test_extract_plain_object():
    assert extract_json('{"faithfulness": 1}') == {"faithfulness": 1}


def test_extract_from_markdown_fence():
    assert extract_json('```json\n{"faithfulness": 0}\n```') == {"faithfulness": 0}


def test_extract_ignores_leading_reasoning_prose():
    raw = 'We must check each claim. The answer cites [1].\n{"faithfulness": 1}'
    assert extract_json(raw) == {"faithfulness": 1}


def test_extract_prefers_the_last_object():
    """Reasoning models often restate a draft object before the final one."""
    raw = 'Draft: {"faithfulness": 0}\nFinal answer:\n{"faithfulness": 1}'
    assert extract_json(raw) == {"faithfulness": 1}


def test_extract_handles_nested_objects():
    assert extract_json('{"a": {"b": 1}, "faithfulness": 1}')["faithfulness"] == 1


def test_extract_returns_none_on_no_json():
    assert extract_json("I cannot judge this.") is None
    assert extract_json("") is None


def test_extract_returns_none_on_broken_json():
    assert extract_json("{not valid json at all") is None


def test_bit_accepts_the_shapes_models_actually_emit():
    assert _bit(1) == 1 and _bit(0) == 0
    assert _bit(True) == 1 and _bit(False) == 0
    assert _bit("1") == 1 and _bit("yes") == 1 and _bit("true") == 1
    assert _bit("0") == 0 and _bit("no") == 0
    assert _bit(None) is None and _bit(2) is None and _bit("maybe") is None


# -- verdicts -------------------------------------------------------------------------
class FakePool:
    """Stands in for KeyPool — no keys, no network."""
    def __init__(self, text="", finish_reason="stop"):
        self.text, self.finish_reason = text, finish_reason
        self.calls = []

    def complete(self, **kw):
        self.calls.append(kw)
        return LLMResponse(text=self.text, model="fake", key_id=0,
                           finish_reason=self.finish_reason,
                           usage={"total_tokens": 10})


def _judge(text, finish_reason="stop"):
    pool = FakePool(text, finish_reason)
    return Judge(Config(), pool=pool), pool


def test_grade_answer_parses_both_metrics():
    j, _ = _judge('{"faithfulness": 1, "citation_acc": 0, "reason": "cite [2] is wrong"}')
    v = j.grade_answer("q", "a", "ctx", "g01:canonical:0")
    assert (v.faithfulness, v.citation_acc, v.status) == (1, 0, "ok")
    assert v.usable and "wrong" in v.reason


def test_truncated_judge_scores_none_not_zero():
    """The rule. A truncated verdict means OUR evaluator ran out of room — scoring the
    pipeline 0 for that would manufacture a grounding failure it never committed."""
    j, _ = _judge("", finish_reason="length")
    v = j.grade_answer("q", "a", "ctx")
    assert v.status == TRUNCATED
    assert v.faithfulness is None and not v.usable


def test_unparseable_judge_scores_none_not_zero():
    j, _ = _judge("I refuse to evaluate this content.")
    v = j.grade_answer("q", "a", "ctx")
    assert v.status == UNPARSEABLE
    assert v.faithfulness is None and not v.usable


def test_judge_exception_is_captured_not_raised():
    class Boom:
        def complete(self, **kw):
            raise RuntimeError("all keys failed")
    v = Judge(Config(), pool=Boom()).grade_answer("q", "a", "ctx")
    assert v.status == "error" and not v.usable


def test_judge_uses_judge_model_and_judge_max_tokens_not_the_generator_budget():
    cfg = replace(Config(), model="gen-model", judge_model="judge-model",
                  max_tokens=1024, judge_max_tokens=2048)
    pool = FakePool('{"faithfulness": 1, "citation_acc": 1}')
    Judge(cfg, pool=pool).grade_answer("q", "a", "ctx")
    assert pool.calls[0]["model"] == "judge-model", "must not grade itself"
    assert pool.calls[0]["max_tokens"] == 2048, "reasoning needs its own headroom"
    assert pool.calls[0]["temperature"] == 0.0


# -- agreement ------------------------------------------------------------------------
def _group(n=6):
    ph = [Phrasing("g01", "canonical", 0, "c")]
    ph += [Phrasing("g01", "paraphrase", i, f"p{i}") for i in range(4)]
    ph.append(Phrasing("g01", "weird", 0, "w"))
    return Group("g01", "Sec", "the fee is 20 EUR", ("u",), ("pg",), tuple(ph[:n]))


def test_agreement_scores_the_mean_of_the_vector():
    j, _ = _judge('{"consistent": [1,1,1,0,0,1], "reason": "two say I do not know"}')
    v = j.grade_agreement(_group(), ["a"] * 6)
    assert v.score == pytest.approx(4 / 6)
    assert v.consistent == [1, 1, 1, 0, 0, 1]


def test_agreement_all_consistent_is_one():
    j, _ = _judge('{"consistent": [1,1,1,1,1,1]}')
    assert j.grade_agreement(_group(), ["a"] * 6).score == 1.0


def test_agreement_rejects_a_wrong_length_vector():
    """A short vector means the judge lost track of which answer is which — the
    per-answer scores can't be trusted, so the verdict is thrown away rather than
    silently averaged over the wrong denominator."""
    j, _ = _judge('{"consistent": [1,1]}')
    v = j.grade_agreement(_group(), ["a"] * 6)
    assert v.status == UNPARSEABLE and v.score is None
