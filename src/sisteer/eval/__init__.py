"""eval — the original, in-process evaluation harness. The name is a historical misnomer.

``eval`` and its sibling :mod:`sisteer.capeval` do not split along "safety"
versus "capability". This package holds the in-process jailbreak and
StrongREJECT battery and its attacks (``jailbreak``, ``attacks/``), the
DefenseCell identity and its controls (``defense_cell``, ``cell_controls``,
``sampled_controls``, the over-refusal controls), attacker serving
(``attacks/attacker_serving``), the held-out agentic-misalignment suite
(``agentic_misalignment/``), and also capability, training-budget and shared
serving / judging code (``capability*``, ``train_budget``, ``runner``,
``suites``, ``static_grading``).

The served-model harness lives in :mod:`sisteer.capeval`: vLLM / proxy / lens
target serving, the served capability suite (GSM8K, MATH-500, MMLU, IFEval,
BigCodeBench, BFCL, tinycap), Petri, the misalignment evals and the
registry-facing contracts. The two packages import each other.

The paths are kept as they are because recorded run identities (harness
versions, pins and registry records) reference these module paths; renaming
either package would break the join between results and code.
"""
