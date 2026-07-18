"""Central configuration + paths.

Every knob the pipeline exposes lives here. An experiment is a `Config` diff, never a
rewrite. Hashes are stamped into every trace and experiment log entry so any result is
reproducible and attributable to an exact setup (see docs/ARCHITECTURE.md).

Three scoped hashes, not one (a single hash over every field meant changing `judge_model`
re-stamped the pipeline's identity, invalidating results the change could not affect):

  index_hash   what determines the on-disk index (chunking + embedding)
  config_hash  pipeline identity: index_hash + everything that shapes an answer
  eval_hash    how a run was *scored* (gold set + judge) — never touches config_hash

`config_hash` nests `index_hash`, so two runs sharing a `config_hash` provably share an
index — that is what lets the ledger carry `config_hash` alone.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from dataclasses import asdict, dataclass, field, fields

# --- paths (all relative to the repo root, resolved from this file) ---------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PAGES_JSONL = DATA / "pages.jsonl"
CHUNKS_JSONL = DATA / "chunks.jsonl"
CHUNKS_META = DATA / "chunks.meta.json"  # attests which index_hash built chunks.jsonl
QDRANT_PATH = ROOT / "qdrant_storage"
RUNS_DIR = ROOT / "runs"                 # per-run trace folders (runs/<run_id>/)
DOCS = ROOT / "docs"
EVAL_DIR = ROOT / "eval"                 # gold sets live outside the package
EXPERIMENTS_MD = DOCS / "EXPERIMENTS.md"  # run-ledger auto-append target

# --- hash buckets ------------------------------------------------------------------
# Every Config field belongs to exactly one bucket; `_assert_buckets_complete()` below
# enforces it at import. Without that assert this scoping is *less* safe than hashing
# everything: a new field would silently count for nothing, which is the exact failure
# this split exists to fix.
_INDEX_FIELDS = ("chunker_version", "target_words", "overlap_words", "min_chunk_words",
                 "embedder_id")
_PIPELINE_FIELDS = (
    "query_instruction", "candidate_k", "top_k", "query_mode", "retriever_mode",
    "rerank", "model", "prompt_version", "temperature", "max_tokens",
)
_EVAL_FIELDS = ("gold_set", "judge_model", "judge_prompt_version", "judge_max_tokens",
                "judge_sample")
# Physical location, not identity: the same index_hash means the same vectors whether
# they live in a local file or a Docker server, so neither may move a hash.
_UNHASHED = ("collection", "qdrant_url")


def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Config:
    # -- chunking --
    # Chunker *code* version, same contract as `prompt_version`: the params below don't
    # capture the algorithm, so a chunker change would otherwise rebuild the index under
    # an unchanged index_hash — a silently stale collection with a "valid" name.
    # v1 = pre-2026-07-15 (first-block heading_path; redundant tail chunks)
    # v2 = common-prefix heading_path; tail-window dedup
    chunker_version: str = "v2"
    target_words: int = 230       # ~300 tokens for bge; safely under the 512 cap
    # Applies to the window-split path ONLY (blocks larger than target_words); the
    # greedy packing path merges whole heading-blocks with no overlap. Changing this
    # therefore moves index_hash while barely moving the index — you would re-chunk,
    # re-index, re-run and measure noise. Making packing overlap is a lever, not a fix.
    overlap_words: int = 40
    min_chunk_words: int = 8      # drop fragments smaller than this

    # -- embedding --
    embedder_id: str = "BAAI/bge-small-en-v1.5"
    # bge-v1.5 retrieval: instruction on the QUERY side only, not passages.
    query_instruction: str = "Represent this sentence for searching relevant passages:"

    # -- vector store --
    collection: str = "univie_studying"   # base name; the live one is physical_collection
    # Empty => local persistent path (zero infra, but an EXCLUSIVE file lock: one
    # process at a time, so a long-lived frontend locks out every harness/index run).
    # Set QDRANT_URL (e.g. http://localhost:6333) to use a server instead — no lock,
    # concurrent readers, and closer parity with the company's stack. Not hashed: the
    # vectors are identical either way. Recorded in the manifest so a trace still says
    # which backend served it.
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", ""))

    # -- retrieval --
    candidate_k: int = 20             # vector-search pool size (what we retrieve)
    top_k: int = 6                    # how many of the pool reach the prompt (selected)
    query_mode: str = "identity"      # identity | multiquery | hyde  (levers, later)
    retriever_mode: str = "dense"     # dense | hybrid                (lever, later)
    rerank: bool = False              # cross-encoder rerank          (lever, later)

    # -- generation (Groq) --
    model: str = "llama-3.3-70b-versatile"   # config value; we A/B several
    prompt_version: str = "v1"
    temperature: float = 0.0                 # determinism
    max_tokens: int = 1024

    # -- evaluation (eval_hash only — must never move config_hash) --
    # gold_v1_small (20 groups × 6 = 120): the project's gold set. The original 40-group
    # gold_v1 was removed; a different gold_set moves eval_hash only, never config_hash.
    gold_set: str = "gold_v1_small"
    # LLM-judge for the scoring harness: a separate, larger-context model so it doesn't
    # grade itself; driven through the same `rag.llm.KeyPool` cycling. Groq rate-limits
    # per model, so a distinct judge also draws from its own token budget.
    judge_model: str = "openai/gpt-oss-120b"
    judge_prompt_version: str = "v1"
    # Separate from `max_tokens`: the judge emits a tiny JSON object but gpt-oss is a
    # *reasoning* model, and its reasoning counts against completion tokens. At the
    # generator's 1024 it spent the whole budget thinking and returned empty content
    # (finish_reason="length"), which scores as an unusable verdict, not a bad answer.
    judge_max_tokens: int = 2048
    # Which phrasings get per-answer judging (faithfulness/citation_acc). The free tier
    # caps each key at 100k tokens/DAY/model, so judging all 6 phrasings of all 40
    # groups (~800k) spans days. A PINNED subsample keeps a judged panel affordable and
    # still comparable across experiments — as long as it never changes silently, which
    # is why it lives in eval_hash.
    #   all   — all 6 phrasings (the full ARCHITECTURE §9 panel; multi-day)
    #   c+2p  — canonical + paraphrases 0,1 (3/group = 120 queries)
    #   c     — canonical only (1/group = 40 queries; cheapest signal)
    # `answer_agreement` always uses every answer a group has: it is one call per group
    # regardless, so sampling it would save nothing and only weaken the metric.
    judge_sample: str = "all"

    # NOTE: all three hashes derive from explicit field tuples over `asdict(self)` —
    # never from `to_dict()`, which injects the hashes themselves.
    @property
    def index_hash(self) -> str:
        """Identity of the on-disk index: chunking + embedding only."""
        d = asdict(self)
        return _hash({k: d[k] for k in _INDEX_FIELDS})

    @property
    def config_hash(self) -> str:
        """Identity of the pipeline that produces an answer. Nests `index_hash`."""
        d = asdict(self)
        payload = {k: d[k] for k in _PIPELINE_FIELDS}
        payload["index_hash"] = self.index_hash
        return _hash(payload)

    @property
    def eval_hash(self) -> str:
        """Identity of the *scoring* setup — gold set + judge. Not pipeline identity."""
        d = asdict(self)
        return _hash({k: d[k] for k in _EVAL_FIELDS})

    @property
    def physical_collection(self) -> str:
        """Live Qdrant collection name — index identity baked in.

        Makes scoring a stale index structurally impossible: change a chunk param and
        the old collection is simply not found, instead of being silently queried while
        the ledger claims the new hash.
        """
        return f"{self.collection}_{self.index_hash}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["index_hash"] = self.index_hash
        d["config_hash"] = self.config_hash
        d["eval_hash"] = self.eval_hash
        return d


def _assert_buckets_complete() -> None:
    declared = {f.name for f in fields(Config)}
    bucketed = set(_INDEX_FIELDS) | set(_PIPELINE_FIELDS) | set(_EVAL_FIELDS) | set(_UNHASHED)
    if declared != bucketed:
        raise AssertionError(
            "Config fields are not all assigned to exactly one hash bucket "
            f"(symmetric difference: {sorted(declared ^ bucketed)}). Add the field to "
            "_INDEX_FIELDS, _PIPELINE_FIELDS, _EVAL_FIELDS or _UNHASHED in rag/config.py."
        )
    counts = len(_INDEX_FIELDS) + len(_PIPELINE_FIELDS) + len(_EVAL_FIELDS) + len(_UNHASHED)
    if counts != len(bucketed):
        raise AssertionError("a Config field appears in more than one hash bucket")


_assert_buckets_complete()
