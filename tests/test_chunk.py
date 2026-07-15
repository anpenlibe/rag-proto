"""Chunker packing invariants — issues #11 and #12, encoded so they can't come back.

Both defects were silent: they produced plausible chunks that quietly cost recall and
manufactured citation failures. Nothing but a test notices their return.
"""
from __future__ import annotations

from rag.chunk import common_prefix, pack_blocks, split_by_headings


def _texts(n, prefix="w"):
    """n DISTINCT words. Identical words would make substring/containment checks
    vacuously true — a 210-word run of "w" is a substring of a 230-word run of "w"."""
    return " ".join(f"{prefix}{i}" for i in range(n))


# -- #12: heading_path must describe ALL the text in the chunk ------------------------
def test_common_prefix_of_sibling_paths():
    assert common_prefix([["A", "B"], ["A", "C"]]) == ["A"]


def test_common_prefix_of_unrelated_paths_is_empty():
    assert common_prefix([["A"], ["B"]]) == []


def test_common_prefix_of_single_path_is_itself():
    assert common_prefix([["A", "B"]]) == ["A", "B"]


def test_common_prefix_with_a_pathless_block_is_empty():
    assert common_prefix([[], ["A"]]) == []


def test_packed_chunk_heading_path_is_a_prefix_of_every_merged_block():
    """The invariant. heading_path is prepended into embed_text, so claiming the first
    block's heading mis-anchors the VECTOR of every multi-block chunk, not just its
    citation."""
    blocks = [
        (["Fees", "Exemption"], _texts(20)),
        (["Fees", "Refunds"], _texts(20)),
        (["Fees", "Deadlines"], _texts(20)),
    ]
    out = pack_blocks(blocks, target=230, overlap=40, min_words=5)
    assert len(out) == 1, "small sibling blocks should pack into one chunk"
    path, _text, paths = out[0]
    assert path == ["Fees"], "must demote to the common ancestor, not claim 'Exemption'"
    for p in paths:
        assert p[:len(path)] == path


def test_packing_unrelated_headings_yields_no_false_anchor():
    blocks = [(["Alpha"], _texts(20)), (["Beta"], _texts(20))]
    (path, _t, paths), = pack_blocks(blocks, target=230, overlap=40, min_words=5)
    assert path == [], "no shared ancestor => no heading claim at all"
    assert paths == [["Alpha"], ["Beta"]]


def test_single_block_keeps_its_full_path():
    blocks = [(["A", "B"], _texts(20))]
    (path, _t, _p), = pack_blocks(blocks, target=230, overlap=40, min_words=5)
    assert path == ["A", "B"], "nothing was merged, so nothing is over-claimed"


# -- #11: no chunk wholly contained in its neighbour ----------------------------------
def test_window_split_emits_no_wholly_contained_tail():
    """The split loop used to emit a final window that started inside the previous one
    and reached the same end — a duplicate burning a top_k slot on identical text."""
    blocks = [(["H"], _texts(400))]
    out = pack_blocks(blocks, target=230, overlap=40, min_words=5)
    texts = [t for _p, t, _ps in out]
    for a, b in zip(texts, texts[1:]):
        assert b not in a, "tail window is wholly contained in its predecessor"
    assert len(texts) == 2


def test_window_split_still_covers_all_the_text():
    """Dedup must not silently drop content off the end of a long block."""
    words = [f"w{i}" for i in range(400)]
    out = pack_blocks([(["H"], " ".join(words))], target=230, overlap=40, min_words=5)
    covered = set()
    for _p, t, _ps in out:
        covered.update(t.split())
    assert covered == set(words), "every word must survive somewhere"


def test_window_split_exact_multiple_has_no_empty_tail():
    out = pack_blocks([(["H"], _texts(230))], target=230, overlap=40, min_words=5)
    assert len(out) == 1


def test_long_block_windows_respect_target():
    out = pack_blocks([(["H"], _texts(1000))], target=230, overlap=40, min_words=5)
    for _p, t, _ps in out:
        assert len(t.split()) <= 230


# -- general packing ------------------------------------------------------------------
def test_fragments_below_min_words_are_dropped():
    out = pack_blocks([(["H"], "one two")], target=230, overlap=40, min_words=8)
    assert out == []


def test_packing_respects_target_budget():
    blocks = [(["H"], _texts(100)) for _ in range(5)]
    out = pack_blocks(blocks, target=230, overlap=40, min_words=5)
    for _p, t, _ps in out:
        assert len(t.split()) <= 230


def test_split_by_headings_tracks_h2_h3():
    md = "intro\n## A\natext\n### B\nbtext\n## C\nctext"
    blocks = split_by_headings(md)
    paths = [p for p, _ in blocks]
    assert paths == [[], ["A"], ["A", "B"], ["C"]]
