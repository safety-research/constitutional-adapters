"""TEMPLATE — copy this to ``src/sisteer/eval/attacks/<name>.py`` to add a new
HELD-OUT family. This file is a fixture, not a test (no ``test_`` prefix), so
pytest never collects it; it is here as a copy-paste starting point.

Adding a held-out family is THREE steps:

  1. Write the module below (rename ``example`` -> your family name).
  2. Register it in ``src/sisteer/eval/attacks/__init__.py``:
       - add ``"<name>": ("...attacks.<name>", "run_<name>")`` to ``JB_ATTACKS``
       - add ``"<name>": ("...attacks.<name>", "SPEC")`` to ``HELD_OUT_ATTACKS``
       (the disjointness assertion there refuses a name that is also a training
        family — held-out families must NEVER enter a training pool.)
     If it should be a default sweep leg member, also add it to
     ``sweep.HELD_OUT_FAMILIES``.
  3. Add a fixture/goal set if it is the rewriter/ingested kinds (a stub-built corpus, or a pinned
     artifact). ``tests/test_heldout_conformance.py`` then covers it AUTOMATICALLY
     — it is data-driven over ``HELD_OUT_FAMILIES`` and will assert determinism,
     variant count, no leakage, manifest completeness, and replay-resume
     idempotence for your family with no new test code.

Pick the class:
  A  pure deterministic transform of the goal (no model)  -> fill in ``_variants``
                                                             below; ``build_fn=None``.
  B  needs an LLM at build time                            -> add a ``build_<name>``
                                                             that calls
                                                             ``llm_rewriter.build_llm_corpus``
                                                             and freezes a corpus;
                                                             ``_variants`` reads the
                                                             frozen corpus; set
                                                             ``build_fn=build_<name>``.
  C  an external pinned artifact                           -> you almost certainly
                                                             do NOT need a new
                                                             module: register it
                                                             with
                                                             ``frozen_corpus.make_frozen_corpus_family``
                                                             (see ``frozen_demo.py``).

The example below is the DETERMINISTIC build kind.
"""

from __future__ import annotations

from sisteer.eval.attacks.heldout_common import HeldOutFamily
from sisteer.eval.attacks.static_common import run_static


def example_variants(goal, seed: int = 0):
    """Deterministic, DEFENSE-INDEPENDENT ``[(variant_name, prompt), ...]`` —
    same seed must give byte-identical prompts (the matched-pool property). The
    template text carries NO harmful content; the harmful ``goal.text`` is
    spliced at run time and kept only in the 0700 private dir (via run_static).

    ``seed`` is accepted for signature parity even if the transform has no
    randomness. Return >= 1 variant per goal."""
    return [
        ("variant_one", f"[example transform A] {goal.text}"),
        ("variant_two", f"[example transform B of] {goal.text}"),
    ]


def run_example(defense, goals, *, out_dir, trial: int = 0, seed: int = 0,
                max_new_tokens: int = 512, resume: bool = True) -> dict:
    """Replay the transform over ``goals`` against ``defense``, persisting to the
    private ``out_dir`` and resuming per goal. Grading is the separate grading host
    StrongREJECT post-step (the ONE shared grader — do NOT add your own success
    criterion)."""
    return run_static(defense, goals, out_dir=out_dir, family="example",
                      variants_fn=example_variants, trial=trial, seed=seed,
                      max_new_tokens=max_new_tokens, resume=resume)


SPEC = HeldOutFamily(
    name="example",
    build_kind="deterministic",
    variants_fn=example_variants,
    run_fn=run_example,
    transfer=False,  # the deterministic/rewriter kinds: a transform of held-out goals. ingested build kind ingested
                     # artifacts set transfer=True (they measure TRANSFER).
    provenance={"method": "deterministic_example", "paper": "0000.00000"},
    # build_fn=build_example,   # rewriter build kind only
)
