"""Held-out OOD suite: agentic-misalignment blackmail / leaking / murder scenarios.

The scenario templates and the minimal generation code under ``vendored/``
are vendored from ``anthropic-experimental/agentic-misalignment`` @ commit
``ea0630e1a3eaae7f9f9740fd2703229d3854ccda`` (MIT, (c) 2025 Aengus Lynch;
full license in ``vendored/LICENSE``). Only the prompt-generation path is
vendored — none of their classifier/API code — because this campaign judges
free-form responses with its own Sonnet rubric, never their classifier.

Two frozen artifacts share one deterministic generator
(``loader.generate_scenarios``):

* v1 ``scenarios_v1.jsonl`` — blackmail + leaking (the realign suite; the
  murder / lethal-inaction family was excluded by its spec)
* v2 ``scenarios_v2.jsonl`` — leaking + murder (the misalign campaign:
  blackmail omitted for lack of headroom, murder added), with the sidecar
  ``scenarios_v2_pruned.json`` naming the unrenderable conditions that were
  pruned.

``loader.load_scenarios`` / ``loader.load_scenarios_v2`` read the committed
artifacts. A test regenerates each and asserts byte-equality, so the
generator and the artifacts can never drift.
"""
