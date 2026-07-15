"""Wire the components into the baseline RAG flow and trace every hop.

  transform -> retrieve(pool) -> rerank -> select -> assemble -> generate

One CLI invocation = one `Run` of a single query; pass a shared `Run` to log a whole
batch (e.g. the eval harness) into one folder. Every stage — including the full
retrieved pool, the selected chunks, the exact prompt, the raw response, and any
rate-limit rotations — lands in `runs/<run_id>/queries/<trace_id>.json`.

CLI: `python -m rag.pipeline "how much is the tuition fee?"`
"""
from __future__ import annotations

import sys

from .config import Config
from .context import build_context
from .generate import Generator
from .llm import KeyPool
from .query import get_transform
from .rerank import get_reranker
from .retrieve import Candidate, Retriever
from .trace import Run


def _chunk(c: Candidate) -> dict:
    return {
        "id": c.id, "page_id": c.payload.get("page_id"),   # eval joins gold on page
        "score": round(c.score, 4), "url": c.url, "title": c.title,
        "section": c.section, "heading_path": c.heading_path,
        "word_count": c.payload.get("word_count"), "text": c.text,
    }


class RagPipeline:
    def __init__(self, cfg: Config | None = None, run: Run | None = None,
                 pool: KeyPool | None = None):
        self.cfg = cfg or Config()
        self.transform = get_transform(self.cfg)
        self.retriever = Retriever(self.cfg)
        self.reranker = get_reranker(self.cfg)
        # An injected pool keeps every LLM caller on one key budget once a lever (e.g.
        # query expansion) starts calling the LLM *inside* the pipeline, and lets tests
        # drive the wiring with a fake instead of real keys.
        self.generator = Generator(self.cfg, pool=pool)   # key pool is lazy
        self.run = run                          # shared run (batch) or None (ad-hoc)

    def answer(self, query: str, generate: bool = True, query_id: str | None = None):
        run = self.run or Run(self.cfg, label="ad-hoc")
        owns_run = self.run is None
        try:
            return self._answer_in_run(query, run, generate=generate, query_id=query_id)
        finally:
            if owns_run:
                run.close()

    def _answer_in_run(self, query: str, run: Run, generate: bool = True,
                       query_id: str | None = None):
        tr = run.new_trace(query)
        if query_id:
            tr.set(query_id=query_id)   # joins the trace back to its gold phrasing

        with tr.span("query_transform") as s:
            queries = self.transform.transform(query)
            s["queries"] = queries

        with tr.span("retrieve") as s:
            retrieved = self.retriever.retrieve(queries[0], k=self.cfg.candidate_k)
            s["retrieved"] = [_chunk(c) for c in retrieved]

        with tr.span("rerank") as s:
            reranked = self.reranker.rerank(query, retrieved)
            s["enabled"] = self.reranker.enabled
            s["method"] = self.reranker.method
            s["reranked"] = [_chunk(c) for c in reranked]

        with tr.span("select") as s:
            selected = reranked[: self.cfg.top_k]
            s["selected"] = [_chunk(c) for c in selected]

        with tr.span("assemble") as s:
            context, sources = build_context(selected)
            s["context"] = context
            s["sources"] = sources
            s["context_chars"] = len(context)

        # Retrieval-only: selection is complete, so consistency/recall/mrr are already
        # determined. Stopping here yields the headline metric for zero tokens and
        # without any API key — see rag/eval/harness.py --retrieve-only.
        if not generate:
            tr.set(status="retrieval_only", answer="", citations=[],
                   invalid_citations=[], model=None, usage={})
            return None, tr.emit()

        # A failed generation is the single most important thing to trace — if the LLM
        # errors (rate-limit exhaustion, 5xx), emit an error trace rather than losing
        # the record, then re-raise so the caller decides whether to continue.
        try:
            with tr.span("generate") as s:
                s["model"] = self.cfg.model
                s["temperature"] = self.cfg.temperature
                ans = self.generator.generate(query, context, sources)
                s["prompt_messages"] = ans.prompt_messages
                s["key_id"] = ans.key_id
                s["usage"] = ans.usage
                s["rate_limit_events"] = ans.rate_limit_events
                s["raw_response"] = ans.text      # verbatim model output, pre-parsing
                s["finish_reason"] = ans.finish_reason
        except Exception as e:
            tr.set(status="error", error=str(e), error_type=type(e).__name__,
                   answer="", citations=[], model=self.cfg.model)
            tr.emit()
            raise

        tr.set(status="ok", answer=ans.text, citations=ans.citations,
               invalid_citations=ans.invalid_citations,
               model=ans.model, key_id=ans.key_id, usage=ans.usage)
        trace_path = tr.emit()
        return ans, trace_path


def _cli():
    if len(sys.argv) < 2:
        print('usage: python -m rag.pipeline "your question"')
        raise SystemExit(1)
    query = " ".join(sys.argv[1:])
    ans, trace_path = RagPipeline().answer(query)
    run_id = trace_path.parents[1].name

    print(f"\nQ: {query}\n")
    print(ans.text)
    print("\nSources:")
    for c in ans.citations:
        hp = " > ".join(c["heading_path"]) if c["heading_path"] else ""
        print(f"  [{c['n']}] {c['title']}" + (f" — {hp}" if hp else ""))
        print(f"       {c['url']}")
    if ans.invalid_citations:
        print(f"\n  ! model cited non-existent sources: {ans.invalid_citations}")
    tok = ans.usage.get("total_tokens", "?")
    print(f"\n(model={ans.model} key_id={ans.key_id} tokens={tok} · run={run_id}"
          f" · trace: {trace_path.name})")


if __name__ == "__main__":
    _cli()
