"""Shared base-scale calibration for the closed-form steering arms.

DiD / AdaSteer / AlphaSteer produce an intervention whose magnitude has no
absolute rule from the source paper (DiD is this repository's own
construction; AlphaSteer/AdaSteer sweep a model-specific strength grid).
The campaign needs ONE stored base scale per arm that the eval factor grid
then multiplies — picked, exactly as the antidote arm does it, by walking
a geometric ladder under the harness degeneration guard.

The guard (mirrors extraction.antidote.calibrate_base_alpha):
  * distinct-3 of the steered generations >= rep_threshold (default 0.35 —
    the frontier's distinct-3 budget), AND
  * perplexity of the steered GENERATIONS under the UNSTEERED model <=
    ppl_guard_factor x the unsteered generations' perplexity (default 3x).

Scored on GENERATIONS (not a teacher-forced NLL) so it is valid for every
application mode — a last-token / prefill-only edit is invisible to a
full-forward NLL whose steered positions drop out, but always shows up in
what the model actually generates.

base_scale = midpoint of the coherent range [0, hi], where hi is the last
ladder rung before the first degenerate one. LOUD failure when even the
smallest rung is degenerate: there is no coherent range and a silent tiny
scale would fake a null result.
"""

from __future__ import annotations

from typing import Callable

DEFAULT_SCALE_LADDER = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def calibrate_base_scale(
    model,
    tok,
    rendered_prompts: list[str],
    steered_generate: Callable[[float], list[str]],
    *,
    scale_ladder=DEFAULT_SCALE_LADDER,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    max_new_tokens: int = 128,
    batch_size: int = 8,
    log_fn=None,
    arm: str = "arm",
) -> dict:
    """Walk ``scale_ladder`` upward; the first rung that trips the
    degeneration guard ends the coherent range. Returns a dict with
    ``base_scale`` (= hi/2) plus the full sweep for the audit trail.

    ``steered_generate(scale) -> list[str]`` produces the steered
    generations at raw magnitude ``scale`` (the arm supplies the closure so
    this stays intervention-agnostic — a residual vector, an input-dependent
    matrix, or a two-direction adaptive edit all fit).
    """
    from ..eval.capability import distinct_ngram_ratio, perplexity
    from ..generate import generate_batch

    ladder = sorted(float(s) for s in scale_ladder)
    if not ladder or ladder[0] <= 0:
        raise ValueError(f"scale_ladder must be positive, got {scale_ladder}")

    base_texts = generate_batch(
        model, tok, rendered_prompts, batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    base_gen_ppl = perplexity(model, tok, texts=[t for t in base_texts if t])

    sweep = []
    scale_hi = None
    bracketed = False
    for i, scale in enumerate(ladder):
        steered_texts = steered_generate(scale)
        rep = distinct_ngram_ratio(steered_texts)
        gen_ppl = perplexity(model, tok, texts=[t for t in steered_texts if t])
        degenerate = bool(
            rep < rep_threshold or gen_ppl > ppl_guard_factor * base_gen_ppl
        )
        rung = {
            "scale": scale,
            "distinct3": rep,
            "gen_ppl": gen_ppl,
            "degenerate": degenerate,
        }
        sweep.append(rung)
        if log_fn is not None:
            log_fn({f"{arm}_calib/{k}": v for k, v in rung.items()}, step=i)
        print(f"[{arm}] scale={scale:g}: distinct3={rep:.3f} "
              f"gen_ppl={gen_ppl:.2f} (base {base_gen_ppl:.2f}) "
              f"{'DEGENERATE' if degenerate else 'coherent'}")
        if degenerate:
            bracketed = True
            break
        scale_hi = scale

    if scale_hi is None:
        raise RuntimeError(
            f"{arm} calibration found NO coherent scale: the smallest ladder "
            f"rung ({ladder[0]:g}) already fails the degeneration guard "
            f"(distinct3 < {rep_threshold} or gen_ppl > {ppl_guard_factor}x "
            f"base). Sweep: {sweep}"
        )
    warnings = []
    if not bracketed:
        warnings.append(
            f"incoherence bound not bracketed: every ladder rung up to "
            f"{ladder[-1]:g} stayed coherent — base_scale rests on a ladder "
            "ceiling, not a measured transition"
        )
        print(f"[{arm}] WARNING: {warnings[-1]}")
    return {
        "base_scale": scale_hi / 2.0,
        "scale_bounds": [0.0, scale_hi],
        "scale_ladder": ladder,
        "rep_threshold": rep_threshold,
        "ppl_guard_factor": ppl_guard_factor,
        "base_gen_ppl": base_gen_ppl,
        "calib_max_new_tokens": max_new_tokens,
        "n_calib_prompts": len(rendered_prompts),
        "sweep": sweep,
        "warnings": warnings,
    }
