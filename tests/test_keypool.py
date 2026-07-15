"""KeyPool rotation + backoff. No network, no real sleeps.

The pool is what lets a 240-query eval survive a free tier. Its failure mode is nasty:
give up 29 seconds early and a run dies at query ~170 with the day's tokens spent.
"""
from __future__ import annotations

import pytest

from rag.llm import KeyPool, RateLimited, TransientError


class FakeProvider:
    """Replays a scripted outcome per call. `None` = success."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def make_client(self, api_key):
        return f"client::{api_key}"

    def chat(self, client, *, model, messages, **kw):
        self.calls.append(client)
        exc = self.script.pop(0) if self.script else None
        if exc:
            raise exc
        return {"text": "ok", "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def list_models(self, client):
        return ["m"]


def _pool(script, keys=("k1", "k2", "k3", "k4"), **kw):
    slept = []
    p = KeyPool(keys=list(keys), provider=FakeProvider(script),
                sleep_fn=slept.append, **kw)
    return p, slept


def _call(pool):
    return pool.complete(model="m", messages=[{"role": "user", "content": "hi"}])


# -- construction ---------------------------------------------------------------------
def test_duplicate_keys_are_deduped():
    """Duplicates make rotation a no-op — you'd think you had 4 keys' quota while
    hammering one."""
    pool, _ = _pool([], keys=["a", "a", "b", "b", "a"])
    assert pool.keys == ["a", "b"]


def test_empty_keys_raises():
    with pytest.raises(RuntimeError, match="No API keys"):
        KeyPool(keys=[], provider=FakeProvider([]))


def test_blank_keys_are_dropped():
    pool, _ = _pool([], keys=["a", "", None, "b"])
    assert pool.keys == ["a", "b"]


# -- rotation -----------------------------------------------------------------------
def test_round_robin_advances_across_calls():
    pool, _ = _pool([None, None, None])
    assert _call(pool).key_id == 0
    assert _call(pool).key_id == 1
    assert _call(pool).key_id == 2


def test_round_robin_wraps():
    pool, _ = _pool([None] * 5, keys=("k1", "k2"))
    assert [_call(pool).key_id for _ in range(4)] == [0, 1, 0, 1]


def test_rate_limit_rotates_to_the_next_key():
    pool, slept = _pool([RateLimited("429"), None])
    r = _call(pool)
    assert r.key_id == 1, "must move to the next key, not retry the limited one"
    assert not slept, "no backoff until a whole cycle is exhausted"
    assert r.rate_limit_events[0]["error"] == "rate_limit"


def test_transient_error_also_rotates():
    """One 503 mid-run must not kill a 240-query eval (decision #22)."""
    pool, _ = _pool([TransientError("503"), None])
    r = _call(pool)
    assert r.key_id == 1
    assert r.rate_limit_events[0]["error"] == "transient"


def test_non_retryable_errors_propagate_immediately():
    """A bad model id is a bug, not weather — failing fast beats 28 pointless retries."""
    pool, _ = _pool([ValueError("bad model id")])
    with pytest.raises(ValueError, match="bad model id"):
        _call(pool)


def test_events_record_key_and_cycle_but_never_the_secret():
    pool, _ = _pool([RateLimited("429 on key"), None], keys=("SECRET_A", "SECRET_B"))
    r = _call(pool)
    blob = str(r.rate_limit_events)
    assert "SECRET_A" not in blob and "SECRET_B" not in blob
    assert r.rate_limit_events[0]["key_id"] == 0


# -- backoff ------------------------------------------------------------------------
def test_backoff_only_after_a_full_cycle():
    pool, slept = _pool([RateLimited("x")] * 4 + [None])
    r = _call(pool)
    assert slept == [1], "one sleep after all 4 keys failed once"
    assert r.key_id == 0, "cycle 2 starts back at the first key"


def test_backoff_is_exponential_and_capped():
    pool, slept = _pool([RateLimited("x")] * 100, max_backoff_s=30)
    with pytest.raises(RuntimeError):
        _call(pool)
    assert slept == [1, 2, 4, 8, 16, 30], "1,2,4,8,16 then capped at 30"


def test_default_backoff_outlasts_a_60s_rate_limit_window():
    """The reason max_cycles is 7, not 6. TPM windows reset after 60s; 6 cycles wait
    1+2+4+8+16 = 31s and would give up ~29s early, killing a long eval run."""
    pool, slept = _pool([RateLimited("x")] * 200)
    with pytest.raises(RuntimeError):
        _call(pool)
    assert sum(slept) > 60, f"only waited {sum(slept)}s — a 60s window would outlast us"


def test_exhaustion_raises_with_context():
    pool, _ = _pool([RateLimited("x")] * 100, keys=("a", "b"))
    with pytest.raises(RuntimeError, match="All 2 keys failed after 7 cycles"):
        _call(pool)


def test_recovery_on_a_later_cycle_returns_normally():
    pool, slept = _pool([RateLimited("x")] * 8 + [None])
    r = _call(pool)
    assert r.text == "ok"
    assert slept == [1, 2], "backed off twice, then the third cycle succeeded"


def test_sleep_fn_is_injectable_so_tests_never_really_sleep():
    pool, slept = _pool([RateLimited("x")] * 4 + [None])
    _call(pool)
    assert slept, "if this is empty the test suite is sleeping for real"
