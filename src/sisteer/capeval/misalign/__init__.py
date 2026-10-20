"""Misalignment evals as a Petri complement (em-misalign, 2026-09-10).

Two eval families run over the campaign's SERVED cells (the routing proxy's
virtual models, one request per sample), judged offline, filed into the
capeval store under the display domain ``misalign`` and paired per item
against the same target's ``none`` cell:

- ``agentic`` — Anthropic's agentic-misalignment scenarios (behaviors
  leaking + murder; blackmail omitted, no base headroom), sampled at T=1.
- ``em8q`` — the eight Betley et al. free-form questions
  (``sisteer.data.safety_texts.BETLEY_8Q``), 600 new tokens, T=1.

Modules: ``runner`` (served-path generation, resumable per-cell
``responses.jsonl``), ``rows`` (judged JSONL -> ``ResultRow`` cells ->
``rows.parquet``), ``analyze`` (rates, paired contrasts, figures),
``classify`` (the structural leak check, WP-A).

Every module here imports without torch: the runner runs on a serving node
against loopback, the rows/analysis steps on the grading host.
"""
