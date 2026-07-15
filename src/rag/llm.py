"""LLM access — provider seam + API-key cycling.

Everything that talks to an LLM goes through here: the answer `Generator`, the
(upcoming) LLM-judge on `judge_model`, and future query-expansion. Keeping it in one
place means key cycling, rate-limit handling, and rotation logging are implemented
once and shared.

- `KeyPool` — provider-agnostic: round-robin across N keys per call, and on a
  rate-limit **tries every other key before sleeping**, then backs off exponentially.
  Secrets never leave this module; only `key_id` (the index) is ever logged/returned.
- `LLMProvider` — the seam. `GroqProvider` is the only impl today; an OpenAI/other
  provider implements the same two methods.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from dotenv import load_dotenv

from .config import ROOT

load_dotenv(ROOT / ".env")

# Env var names tried, in order, to find the key pool.
_KEY_LIST_VARS = ("GROQ_API_KEYS", "GROQ_API_KEY")
_KEY_INDEXED_PREFIXES = ("k", "GROQ_API_KEY_", "GROQ_KEY_")


def load_keys() -> list[str]:
    """Collect API keys from the environment.

    Accepts a comma-separated list (`GROQ_API_KEYS=a,b,c,d`) or individually named
    keys (`k1..kN`, `GROQ_API_KEY_1..N`). Returns [] if none found.
    """
    for var in _KEY_LIST_VARS:
        raw = os.environ.get(var) or ""
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            return keys
    for prefix in _KEY_INDEXED_PREFIXES:
        collected, i = [], 1
        while (v := os.environ.get(f"{prefix}{i}")):
            collected.append(v.strip())
            i += 1
        if collected:
            return collected
    return []


@dataclass
class LLMResponse:
    """Provider-agnostic completion result."""
    text: str
    model: str
    key_id: int
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)          # prompt/completion/total tokens
    rate_limit_events: list[dict] = field(default_factory=list)


class RateLimited(Exception):
    """Raised by a provider when a key is rate-limited (429) — rotate to another key."""


class TransientError(Exception):
    """Raised by a provider for a retryable blip (connection/timeout/5xx).

    Not the key's fault, but retrying on another key is the cheapest recovery. Groq's
    free tier serves these routinely; without retrying, one 503 at query 173 of a
    240-query eval run would kill the whole run.
    """


class LLMProvider(Protocol):
    """The seam. Implement these two to add a provider (OpenAI, local, …)."""

    def chat(self, client, *, model: str, messages: list[dict], **kw): ...
    def list_models(self, client) -> list[str]: ...


class GroqProvider:
    name = "groq"

    def make_client(self, api_key: str):
        from groq import Groq
        return Groq(api_key=api_key)

    def chat(self, client, *, model: str, messages: list[dict], **kw):
        """Call the provider, mapping SDK errors onto the pool's retry vocabulary.

        Note RateLimitError ⊂ APIStatusError in the Groq SDK, so it must be caught
        first; APIConnectionError/APITimeoutError are NOT APIStatusError.
        """
        from groq import (APIConnectionError, APIStatusError, APITimeoutError,
                          RateLimitError)
        try:
            resp = client.chat.completions.create(model=model, messages=messages, **kw)
        except RateLimitError as e:
            raise RateLimited(str(e)) from e
        except (APIConnectionError, APITimeoutError) as e:
            raise TransientError(f"connection/timeout: {e}") from e
        except APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code == 429:
                raise RateLimited(str(e)) from e
            if code is not None and code >= 500:
                raise TransientError(f"server {code}: {e}") from e
            raise                              # 4xx (bad model/auth) — a real bug, fail loudly
        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            pt, ct = resp.usage.prompt_tokens, resp.usage.completion_tokens
            usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        return {
            "text": choice.message.content or "",
            "finish_reason": getattr(choice, "finish_reason", "") or "",
            "usage": usage,
        }

    def list_models(self, client) -> list[str]:
        return sorted(m.id for m in client.models.list().data)


class KeyPool:
    """Round-robin over API keys, rotating on rate limits.

    On a 429 we move to the next key **immediately** — a different key is usually
    healthy, so sleeping first just wastes time. Only once every key has been tried in
    the current cycle do we back off (exponentially, capped), then cycle again.
    """

    # Defaults sized to ride out a per-MINUTE rate-limit window. The backoff is only
    # slept between cycles, so N cycles wait sum(1,2,4,8,16,...) capped at
    # max_backoff_s. 6 cycles = 1+2+4+8+16 = 31s — which is NOT enough: if every key is
    # rate-limited at t=0 the pool would give up ~29s before the window resets, killing
    # a 240-query run partway through. 7 cycles = 1+2+4+8+16+30 = 61s > 60s window.
    def __init__(self, keys: list[str] | None = None,
                 provider: LLMProvider | None = None, max_cycles: int = 7,
                 max_backoff_s: int = 30, sleep_fn=time.sleep):
        self.provider = provider or GroqProvider()
        keys = keys if keys is not None else load_keys()
        # Dedupe (order-preserving): duplicate keys make rotation a no-op — you'd think
        # you had 4 keys' worth of quota while hammering one.
        self.keys = list(dict.fromkeys(k for k in keys if k))
        if not self.keys:
            raise RuntimeError(
                "No API keys found. Set GROQ_API_KEYS=a,b,c,d (or k1..k4) in .env"
            )
        self._clients = [self.provider.make_client(k) for k in self.keys]
        self._idx = 0
        self._lock = threading.Lock()      # a harness may parallelise queries
        self.max_cycles = max_cycles
        self.max_backoff_s = max_backoff_s
        self._sleep = sleep_fn             # injectable: tests must not really sleep

    def _next_idx(self) -> int:
        """Atomically take the current index and advance the cursor."""
        with self._lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self._clients)
            return idx

    def complete(self, *, model: str, messages: list[dict], **kw) -> LLMResponse:
        """Try each key in turn; back off only after a full cycle is exhausted.

        Retries rate limits AND transient blips (connection/timeout/5xx). Other errors
        (bad model id, auth) propagate immediately — those are bugs, not weather.
        Returns an LLMResponse carrying the key_id used and every rotation event.
        """
        n = len(self._clients)
        events: list[dict] = []
        last_err: Exception | None = None

        for cycle in range(self.max_cycles):
            for _ in range(n):                     # try every key before sleeping
                idx = self._next_idx()
                try:
                    out = self.provider.chat(self._clients[idx], model=model,
                                             messages=messages, **kw)
                    return LLMResponse(
                        text=out["text"], model=model, key_id=idx,
                        finish_reason=out["finish_reason"], usage=out["usage"],
                        rate_limit_events=events,
                    )
                except (RateLimited, TransientError) as e:
                    last_err = e
                    events.append({
                        "cycle": cycle, "key_id": idx,
                        "error": "rate_limit" if isinstance(e, RateLimited) else "transient",
                        "detail": str(e)[:120], "backoff_s": 0,
                    })
            # every key failed this cycle → back off before trying again
            if cycle < self.max_cycles - 1:
                backoff = min(2 ** cycle, self.max_backoff_s)
                events.append({"cycle": cycle, "key_id": None,
                               "error": "all_keys_failed", "backoff_s": backoff})
                self._sleep(backoff)

        raise RuntimeError(
            f"All {n} keys failed after {self.max_cycles} cycles "
            f"({len(events)} events; last: {last_err})"
        ) from last_err

    def list_models(self) -> list[str]:
        return self.provider.list_models(self._clients[0])
