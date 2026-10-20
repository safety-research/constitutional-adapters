"""Sample-count-vs-fitted-dimension floor for closed-form fits.

The failure this exists to kill: the Class B
content fit fed 41 harmful prompts (HarmBench-val's 80 rows silently
reduced to the standard-only functional subset by ``load_harmbench``'s
default) into AlphaSteer's d x d ridge at d = 4096 — catastrophically
underdetermined — and NOTHING said so: the solve is regularized, so it
returns a matrix either way and the failure only surfaces at eval.

Two layers of defense, deliberately split:

- the HARD floor lives in the data layer
  (``data.content_sets.build_content_contrast``'s ``min_prompts``, lifted
  from an earlier development branch's seam) — it stops a 41-sample set
  reaching a fit at all;
- THIS module is the fit-side n-vs-d check: every closed-form trainer that
  solves for d-dimensional (or d x d) parameters calls
  :func:`check_sample_floor` with the realised sample count and the fitted
  dimension, and an n < d fit WARNS PROMINENTLY (stderr + ``log_fn``) with
  the realised n and d in the message, and records the regime in the
  object's metadata so the cell is legible after the fact.

Why warn rather than raise: the AlphaSteer authors' OWN composition is
n_harmful = 2,000 against d = 4096 — n < d, made workable by the ridge
regularizer — so a hard raise at n < d would refuse the paper-faithful
dataset itself. The hard raise belongs to the ``min_prompts`` floor
upstream; this check is the honest label on everything above it.
"""

from __future__ import annotations

import sys


def check_sample_floor(n: int, d: int, *, what: str, tag: str,
                       log_fn=None) -> dict:
    """Compare the realised sample count ``n`` against the fitted dimension
    ``d`` for one closed-form fit (``what`` names it, e.g. "ridge rows
    (harmful)"; ``tag`` names the arm/cell). Returns a metadata record
    ``{"what", "n", "d", "underdetermined"}`` and, when ``n < d``, prints a
    PROMINENT warning naming both numbers and logs
    ``{tag}/underdetermined_{what-slug}: 1`` through ``log_fn``.
    """
    under = n < d
    record = {"what": what, "n": int(n), "d": int(d),
              "underdetermined": bool(under)}
    if under:
        msg = (f"[{tag}] UNDERDETERMINED FIT: {what} has n = {n} samples "
               f"for a {d}-dimensional fit (n < d). The solve is "
               "regularized, so it will return parameters either way — "
               "interpret this cell with that label, and prefer a fit set "
               "with n >= d (ideally n >> d).")
        print(msg, file=sys.stderr)
        print(msg)
        if log_fn is not None:
            slug = "".join(c if c.isalnum() else "_" for c in what).strip("_")
            try:
                log_fn({f"{tag}/underdetermined_{slug}": 1,
                        f"{tag}/n_{slug}": int(n),
                        f"{tag}/d_{slug}": int(d)})
            except Exception:  # noqa: BLE001 — a logger must not fail a fit
                pass
    return record
