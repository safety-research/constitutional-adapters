"""Stage-3 pool SAMPLING lane — materialise the prompt inventory on CPU, then
generate every (prompt, SI, candidate) cell on a vLLM engine.

Why this exists as its own lane rather than reusing ``run_static``: that loop
calls ``defense.generate([prompt])`` ONE prompt at a time under HF
transformers, which is fine for a few hundred BoN variants and hopeless for
the stage-3 scale-up (the raised ``fit_floor`` wants ~45k matched pairs per
SI, and each pair costs a ``none`` response plus K candidates under each of
three SIs). Two properties of the existing code make the decoupling safe:

* every static family's ``variants_fn(goal, seed)`` is DETERMINISTIC and
  DEFENSE-INDEPENDENT by contract (``static_common.run_static``'s docstring),
  so the attack prompts can be materialised with no model loaded at all; and
* the two long SIs are a ~6.5k-token system prefix that is IDENTICAL across
  every prompt sampled under that SI, so vLLM's automatic prefix caching
  prefills it once per engine instead of once per request.

``sources`` does the CPU half, ``engine`` the GPU half. Row schemas are the
ones ``scripts/build_two_si_pool.py`` already reads (``_row_to_graded`` /
``_variant_key``), so grading and manifest freezing are unchanged.
"""

from .sources import (                                          # noqa: F401
    SamplePrompt, build_inventory, inventory_counts,
    HARMFUL_ATTACK_FAMILIES, BENIGN_FAMILIES,
)

__all__ = ["SamplePrompt", "build_inventory", "inventory_counts",
           "HARMFUL_ATTACK_FAMILIES", "BENIGN_FAMILIES"]
