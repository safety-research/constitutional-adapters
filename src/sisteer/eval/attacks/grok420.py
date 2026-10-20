"""G0DM0D3 semantic-inversion jailbreaks as a static training family.

Two templates read by :mod:`sisteer.eval.attacks_godmode`, which fetches them
at run time from the pinned upstream commit (they are not distributed with
this repository; see that module for provenance, licence and the single-turn
fold). Both are fixed
scaffolds with a goal slot, so they replay deterministically like the other
static transforms -- no attacker model, no sampling.

Registered as ONE family with two variants rather than two families:
``grok420`` and ``hermes_fast`` share the invert-the-refusal mechanism, and
the pool's per-family accounting is the unit a defense is judged on. Two
mechanisms would warrant two families; two spellings of one do not.
"""
from __future__ import annotations

from ..attacks_godmode import GODMODE_TEMPLATES, render
from .static_common import run_static


def grok420_variants(goal, seed: int = 0) -> list[tuple[str, str]]:
    """``[(variant_key, prompt), ...]`` -- deterministic, seed-independent."""
    text = (goal if isinstance(goal, str)
            else getattr(goal, "text", None) or getattr(goal, "goal", str(goal)))
    return [(name, render(name, text)) for name in sorted(GODMODE_TEMPLATES)]


def run_grok420(defense, goals, *, out_dir, trial: int = 0, seed: int = 0,
                max_new_tokens: int = 512, resume: bool = True,
                decode=None, seed_base=None, padding=None) -> dict:
    """Replay both G0DM0D3 framings over ``goals`` against ``defense``.
    Grading is the separate grading host StrongREJECT post-step. ``decode`` /
    ``seed_base`` pass through to ``run_static`` (the full22 SAMPLED re-run's
    vendor preset; absent, the call is byte-identical to before), as does
    ``padding`` (a ``static_padding.PadPlan``; absent, nothing is padded)."""
    return run_static(defense, goals, out_dir=out_dir, family="grok420",
                      variants_fn=grok420_variants, trial=trial, seed=seed,
                      max_new_tokens=max_new_tokens, resume=resume,
                      decode=decode, seed_base=seed_base, padding=padding)
