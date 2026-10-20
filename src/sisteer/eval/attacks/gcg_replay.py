"""GCG-by-replay — replay previously-found GCG suffixes (Zou et al. 2307.15043)
WITHOUT re-running the optimiser.

This is the GCG attack with the expensive discrete-optimisation step REMOVED: it
takes a BANK of already-found adversarial suffixes and, for each goal, appends
each banked suffix to the request and runs one real generation through the SAME
``defense.generate([f"{goal} {suffix}"])`` boundary ``gcg.run_gcg`` uses for its
periodic/final generations. Persistence + resume mirror ``gcg.py`` exactly (the
mirrored helpers live in ``static_common``, following gcg.py's own
"mirror, don't import" note that keeps this path torch-free) — every real
generation is appended in order to the private ``responses.jsonl`` and a
terminal per-goal record lands in ``progress.jsonl``.

Because it is a fixed, replayable transform, GCG-by-replay is STATIC/replayable
and qualifies for the two-SI pool. Each banked suffix is a distinct variant
(sample_idx 0/1/2… -> variant_key "0:0"/"1:0"/…; ``variant_name`` = the bank
label), and the same suffix produces the same prompt under every defense
condition.

COST ACCOUNTING (JB7). Replay runs NO optimiser, so its MARGINAL optimiser cost
is zero. It reuses the GCG cost-class field names (``n_optimizer_steps`` /
``n_optimizer_forwards``, surfaced by the existing ``gcg_optimizer_cost``
aggregate) reporting 0 for both — the optimiser forwards were already spent when
the banked suffixes were found. Like GCG it is EXEMPT from ``forward_cap``
(``capped: False``); ``n_target_forwards`` counts only the real replays.

The DEFAULT bank is deliberately BENIGN (the paper's "! …" init plus a neutral
filler) so this module ships no working exploit; real banked suffixes (found by
a prior ``run_gcg`` sweep, or a published transferable suffix) are passed in via
``suffixes=``. Non-gated: no attacker LLM, no PyRIT, no sub-env.
"""

from __future__ import annotations

from .static_common import run_static

# Benign default bank — placeholder suffixes only (NO working exploit in-repo).
# Production passes real found suffixes via ``suffixes=``.
DEFAULT_SUFFIX_BANK = (
    " ".join(["!"] * 20),
    " ".join(["..."] * 10),
)


def gcg_replay_variants(goal, seed: int = 0, *, suffixes=DEFAULT_SUFFIX_BANK):
    """Deterministic ordered ``[(bank_label, prompt), ...]`` — one per banked
    suffix, prompt = ``f"{goal} {suffix}"`` (the GCG splice). ``bank_label`` is
    an opaque positional id (``bank0``, ``bank1``, …); ``seed`` unused (replay
    has no randomness)."""
    out = []
    for i, suffix in enumerate(suffixes):
        out.append((f"bank{i}", f"{goal.text} {suffix}"))
    return out


def run_gcg_replay(defense, goals, *, out_dir, suffixes=DEFAULT_SUFFIX_BANK,
                   trial: int = 0, seed: int = 0, max_new_tokens: int = 512,
                   resume: bool = True) -> dict:
    """Replay ``suffixes`` (a bank of previously-found GCG suffixes) over
    ``goals`` against ``defense`` — no optimiser is run. Persists to the private
    ``out_dir`` and resumes per goal; grading is the separate grading host
    StrongREJECT post-step. Adds the GCG cost-class fields (both 0) and a
    ``replayed`` marker to each per-goal record."""
    def variants_fn(goal, s):
        return gcg_replay_variants(goal, s, suffixes=suffixes)

    def progress_extra(_goal):
        # Reuse the GCG cost-class shape; replay's MARGINAL optimiser cost is 0.
        return {"n_optimizer_steps": 0, "n_optimizer_forwards": 0,
                "replayed": True, "n_suffixes": len(suffixes)}

    result = run_static(defense, goals, out_dir=out_dir, family="gcg_replay",
                        variants_fn=variants_fn, trial=trial, seed=seed,
                        max_new_tokens=max_new_tokens, resume=resume,
                        progress_extra_fn=progress_extra)
    result["n_optimizer_forwards_total"] = 0  # no optimiser run (replay)
    return result
