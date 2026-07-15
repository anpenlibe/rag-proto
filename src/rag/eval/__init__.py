"""Scoring harness over a paraphrase-grouped gold set.

    gold.py     the gold set as objects (groups of 6 phrasings)
    traces.py   read a finished run's traces back off disk
    metrics.py  the offline panel — pure functions, no LLM, no I/O
    judge.py    the LLM-judge (faithfulness / citation_acc / answer_agreement)
    harness.py  the CLI: generate a run, score a run

Two phases, deliberately separable (`harness.py --score <run_id>` re-scores an existing
run without regenerating it):

  A. generate — 240 queries through the pipeline into one Run. Costs the day's tokens.
  B. score    — reads the traces off disk. Retrieval metrics need no LLM at all.

The headline metric (`consistency`) falls out of phase A alone, because selection
happens before generation: `harness.py --retrieve-only` measures it for zero tokens.
"""
