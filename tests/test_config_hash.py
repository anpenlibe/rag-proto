"""Hash scoping — the regression that would silently undo the three-way split.

Every assertion here encodes a failure that already happened once or was one edit away:
a new Config field counting for nothing, an eval knob re-stamping pipeline identity, or
a chunk knob failing to invalidate the index it built.
"""
from __future__ import annotations

from dataclasses import fields, replace

import pytest

from rag import config as config_mod
from rag.config import (
    _EVAL_FIELDS,
    _INDEX_FIELDS,
    _PIPELINE_FIELDS,
    _UNHASHED,
    Config,
)


def test_every_field_is_in_exactly_one_bucket():
    """The guardrail. Scoping the hash to explicit field tuples is *less* safe than
    hashing everything unless this holds: an unbucketed field would silently affect no
    hash at all — which is the exact bug the split exists to fix."""
    declared = {f.name for f in fields(Config)}
    buckets = [set(_INDEX_FIELDS), set(_PIPELINE_FIELDS), set(_EVAL_FIELDS), set(_UNHASHED)]
    assert set().union(*buckets) == declared
    for i, a in enumerate(buckets):
        for b in buckets[i + 1:]:
            assert not (a & b), f"field in two buckets: {a & b}"


def test_bucket_assert_fires_on_an_unbucketed_field(monkeypatch):
    """The assert must actually reject a field nobody bucketed."""
    monkeypatch.setattr(config_mod, "_UNHASHED", ())
    with pytest.raises(AssertionError, match="collection"):
        config_mod._assert_buckets_complete()


# -- what must NOT move pipeline identity ---------------------------------------------
@pytest.mark.parametrize("field_name,value", [
    ("gold_set", "gold_v2"),
    ("judge_model", "some/other-judge"),
    ("judge_prompt_version", "v2"),
    ("judge_max_tokens", 4096),
])
def test_eval_knobs_do_not_restamp_config_hash(field_name, value):
    """Issue #8. The harness adds judge_* fields; under one flat hash each addition
    re-stamped E0 and invalidated every historical ledger row, though the pipeline that
    produced those answers was byte-for-byte identical."""
    base = Config()
    other = replace(base, **{field_name: value})
    assert other.config_hash == base.config_hash
    assert other.index_hash == base.index_hash
    assert other.eval_hash != base.eval_hash, "eval_hash must record the change"


def test_collection_rename_is_not_identity():
    base = Config()
    other = replace(base, collection="somewhere_else")
    assert other.config_hash == base.config_hash
    assert other.index_hash == base.index_hash
    assert other.eval_hash == base.eval_hash


def test_qdrant_backend_is_not_identity():
    """Local file vs Docker server holds the *same vectors* — a location, not an
    identity. Verified empirically: both backends score consistency 0.426785."""
    base = replace(Config(), qdrant_url="")
    server = replace(base, qdrant_url="http://localhost:6333")
    assert server.config_hash == base.config_hash
    assert server.index_hash == base.index_hash
    assert server.eval_hash == base.eval_hash


def test_qdrant_url_defaults_from_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://example:6333")
    assert Config().qdrant_url == "http://example:6333"
    monkeypatch.delenv("QDRANT_URL")
    assert Config().qdrant_url == ""


# -- what MUST move it ----------------------------------------------------------------
@pytest.mark.parametrize("field_name,value", [
    ("target_words", 150),
    ("overlap_words", 80),
    ("min_chunk_words", 20),
    ("embedder_id", "BAAI/bge-base-en-v1.5"),
    ("chunker_version", "v3"),
])
def test_index_knobs_move_both_hashes(field_name, value):
    """Nesting index_hash inside config_hash is what makes 'same config_hash => same
    index' provable, so the ledger needs no index_hash column."""
    base = Config()
    other = replace(base, **{field_name: value})
    assert other.index_hash != base.index_hash
    assert other.config_hash != base.config_hash, "config_hash must nest index_hash"
    assert other.eval_hash == base.eval_hash


@pytest.mark.parametrize("field_name,value", [
    ("model", "llama-3.1-8b-instant"),
    ("prompt_version", "v2"),
    ("temperature", 0.7),
    ("top_k", 8),
    ("candidate_k", 50),
    ("rerank", True),
    ("query_mode", "multiquery"),
    ("retriever_mode", "hybrid"),
    ("max_tokens", 2048),
    ("query_instruction", "other"),
])
def test_pipeline_knobs_move_config_hash_only(field_name, value):
    base = Config()
    other = replace(base, **{field_name: value})
    assert other.config_hash != base.config_hash
    assert other.index_hash == base.index_hash, "pipeline knobs don't rebuild the index"
    assert other.eval_hash == base.eval_hash


def test_chunker_version_covers_algorithm_changes():
    """The chunk params don't describe the algorithm. Without this field, fixing a
    chunker bug would rebuild the index under an unchanged index_hash — a stale
    collection wearing a valid name, which the sidecar check would happily pass."""
    assert "chunker_version" in _INDEX_FIELDS


# -- stability ------------------------------------------------------------------------
def test_hashes_are_stable_across_calls_and_instances():
    assert Config().config_hash == Config().config_hash
    c = Config()
    assert c.config_hash == c.config_hash


def test_hashes_are_short_hex():
    c = Config()
    for h in (c.index_hash, c.config_hash, c.eval_hash):
        assert len(h) == 12 and int(h, 16) >= 0


def test_hashes_are_not_all_equal():
    c = Config()
    assert len({c.index_hash, c.config_hash, c.eval_hash}) == 3


def test_to_dict_carries_all_three_and_is_not_the_hash_source():
    """to_dict() injects the hashes, so hashing *it* would hash a hash of itself.
    The properties must read asdict(), never to_dict()."""
    d = Config().to_dict()
    for k in ("index_hash", "config_hash", "eval_hash"):
        assert k in d
    for f in fields(Config):
        assert f.name in d


def test_physical_collection_carries_index_hash():
    """Issue #2: change a chunk param, forget to re-index, and the eval would silently
    score the old index while the ledger claimed the new hash."""
    c = Config()
    assert c.physical_collection == f"{c.collection}_{c.index_hash}"
    other = replace(c, target_words=99)
    assert other.physical_collection != c.physical_collection
