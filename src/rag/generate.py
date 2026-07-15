"""Answer generation — grounded, cited, from the selected sources.

Only concern here is the answer contract: build the prompt, call the LLM (via the
shared `KeyPool` in `rag.llm`), then parse and **validate** the `[n]` citations. Key
cycling and provider specifics live in `rag.llm`; prompt text lives in `rag.prompts`.

CLI: `python -m rag.generate --models`  (list available model ids — needs keys)
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

from .config import Config
from .context import src_key
from .llm import KeyPool
from .prompts import system_prompt, user_prompt

_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    citations: list[dict]
    model: str
    key_id: int
    usage: dict = field(default_factory=dict)          # prompt/completion/total tokens
    invalid_citations: list[int] = field(default_factory=list)
    prompt_messages: list[dict] = field(default_factory=list)  # exact system+user sent
    finish_reason: str = ""
    rate_limit_events: list[dict] = field(default_factory=list)


def parse_citations(text: str, sources: dict[str, dict]) -> tuple[list[dict], list[int]]:
    """Resolve `[n]` markers to sources. Returns (resolved, invalid_ns).

    `sources` is str-keyed (see rag.context) so it survives the trace JSON round-trip.
    `invalid_ns` = numbers the model cited that were never shown to it — a
    hallucinated citation, worth surfacing rather than silently dropping.
    """
    cited = sorted({int(m) for m in _CITE_RE.findall(text)})
    resolved = [{**sources[src_key(n)], "n": n} for n in cited if src_key(n) in sources]
    invalid = [n for n in cited if src_key(n) not in sources]
    return resolved, invalid


class Generator:
    def __init__(self, cfg: Config | None = None, pool: KeyPool | None = None):
        self.cfg = cfg or Config()
        self._pool = pool  # lazy: don't require keys until we actually generate

    @property
    def pool(self) -> KeyPool:
        if self._pool is None:
            self._pool = KeyPool()
        return self._pool

    def generate(self, query: str, context: str, sources: dict[str, dict]) -> Answer:
        messages = [
            {"role": "system", "content": system_prompt(self.cfg.prompt_version)},
            {"role": "user", "content": user_prompt(query, context)},
        ]
        resp = self.pool.complete(
            model=self.cfg.model, messages=messages,
            temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
        )
        citations, invalid = parse_citations(resp.text, sources)
        return Answer(
            text=resp.text, citations=citations, model=resp.model, key_id=resp.key_id,
            usage=resp.usage, invalid_citations=invalid, prompt_messages=messages,
            finish_reason=resp.finish_reason, rate_limit_events=resp.rate_limit_events,
        )


def _cli():
    if "--models" in sys.argv:
        pool = KeyPool()
        print(f"{len(pool.keys)} key(s) loaded. Available models:")
        for m in pool.list_models():
            print("  ", m)
    else:
        print("usage: python -m rag.generate --models")


if __name__ == "__main__":
    _cli()
