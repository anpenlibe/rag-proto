"""Versioned prompt templates.

The prompt is a **lever** — changing it changes results, so each version is named and
`prompt_version` is part of `Config` (and therefore of `config_hash`). Never edit a
version in place once it has been used for a logged eval run; add `v2` instead.
"""
from __future__ import annotations

SYSTEM_PROMPTS: dict[str, str] = {
    "v1": (
        "You are a concise assistant for prospective and current University of Vienna "
        "students. Answer the question using ONLY the numbered sources provided.\n"
        "Rules:\n"
        "- Base every statement on the sources; never add outside knowledge or guess.\n"
        "- Cite each claim inline with the bracket number(s) of the source(s) you used, "
        "e.g. [1] or [2][3]. Cite only sources you actually relied on.\n"
        "- If the sources do not contain the answer, reply that you don't know, and do "
        "NOT cite any sources.\n"
        "- Keep the answer short and direct (it may be read aloud by a voice assistant)."
    ),
}


def system_prompt(version: str) -> str:
    try:
        return SYSTEM_PROMPTS[version]
    except KeyError:
        raise KeyError(
            f"unknown prompt_version {version!r}; known: {sorted(SYSTEM_PROMPTS)}"
        ) from None


def user_prompt(query: str, context: str) -> str:
    return f"Sources:\n{context}\n\nQuestion: {query}"


# --- LLM-judge (rag/eval/judge.py) ---------------------------------------------------
# Same contract as above: `judge_prompt_version` is in `Config` and feeds `eval_hash`
# (NOT `config_hash` — re-judging doesn't change the pipeline). Never edit a used
# version in place; add "v2", or every score logged under v1 becomes unreproducible.

JUDGE_PROMPTS: dict[str, str] = {
    "v1": (
        "You are a strict evaluator of a retrieval-augmented answer. You are given a "
        "question, the numbered sources that were shown to the model, and the model's "
        "answer.\n"
        "Judge exactly two things:\n"
        "1. faithfulness: 1 if EVERY factual claim in the answer is supported by the "
        "sources shown; 0 if any claim is unsupported, contradicted, or invented. An "
        "honest \"I don't know\" with no claims is faithful (1).\n"
        "2. citation_acc: 1 if every inline [n] marker points to a source that actually "
        "supports the claim it is attached to; 0 if any citation is irrelevant, wrong, "
        "or points to a source that doesn't support the claim. An answer with no claims "
        "and no citations is 1. An answer that makes claims but cites nothing is 0.\n"
        "Judge ONLY against the sources shown — not your own knowledge of the topic. "
        "The answer being incomplete is not itself unfaithful.\n"
        "Respond with ONLY a json object, no prose and no markdown fence:\n"
        '{"faithfulness": 0 or 1, "citation_acc": 0 or 1, "reason": "<one short sentence>"}'
    ),
}

JUDGE_AGREEMENT_PROMPTS: dict[str, str] = {
    "v1": (
        "You are evaluating whether a QA system answers the SAME question consistently "
        "when it is phrased differently.\n"
        "You are given a reference fact and N answers, each produced from a different "
        "phrasing of one question. Decide which answers are mutually consistent — i.e. "
        "they convey the same substantive information and would not mislead a reader "
        "who got one instead of another.\n"
        "Rules:\n"
        "- Wording, length, formality and detail may differ freely; that is NOT "
        "inconsistency.\n"
        "- Contradicting facts, different numbers/amounts/deadlines, or answering a "
        "different question ARE inconsistency.\n"
        "- An \"I don't know\" / refusal is INCONSISTENT with any answer that does "
        "answer the question.\n"
        "- Judge the answers against each other and the reference fact, majority "
        "meaning wins.\n"
        "Respond with ONLY a json object, no prose and no markdown fence. `consistent` "
        "must have exactly one 0/1 per answer, in the order given (1 = agrees with the "
        "majority meaning):\n"
        '{"consistent": [1, 0, ...], "reason": "<one short sentence>"}'
    ),
}


def judge_prompt(version: str) -> str:
    try:
        return JUDGE_PROMPTS[version]
    except KeyError:
        raise KeyError(
            f"unknown judge_prompt_version {version!r}; known: {sorted(JUDGE_PROMPTS)}"
        ) from None


def judge_agreement_prompt(version: str) -> str:
    try:
        return JUDGE_AGREEMENT_PROMPTS[version]
    except KeyError:
        raise KeyError(
            f"unknown judge_prompt_version {version!r}; "
            f"known: {sorted(JUDGE_AGREEMENT_PROMPTS)}"
        ) from None


def judge_user_prompt(question: str, answer: str, sources_block: str) -> str:
    return (f"Question: {question}\n\nSources shown to the model:\n{sources_block}\n\n"
            f"Model's answer:\n{answer}")


def judge_agreement_user_prompt(key_fact: str, answers: list[str]) -> str:
    block = "\n\n".join(f"Answer {i + 1}:\n{a}" for i, a in enumerate(answers))
    return (f"Reference fact: {key_fact}\n\n{block}\n\n"
            f"Return exactly {len(answers)} values in `consistent`.")
