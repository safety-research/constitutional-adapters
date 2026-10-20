#!/usr/bin/env python
"""positive_control_verdict.py — the steer-is-live positive control's VERDICT from its result JSON, one line on stdout:

    PASS                                          every check PASS
    PASS_WITH_ANCHOR_NONDETERMINISM ...           only `determinism` failed, and only on the UNHOOKED anchor: numeric noise
                                                  under batching on the bare route; steered + twin legs byte-stable, every
                                                  steer-is-live check PASS -> traffic proceeds, recorded
    PASS_WITH_ANCHOR_NONDETERMINISM ...           the same anchor noise reaching `noise_floor` too: only `determinism`
                                                  (anchor-only) + `noise_floor` failed, the twin is byte-stable and the differing
                                                  probes are a MINORITY (<= n/2) -> the same recorded warning; a majority, a
                                                  non-stable twin, or any other failing check still refuses
    PASS_WITH_LENS_NONDETERMINISM ...             OPT-IN (F22_PC_TOLERATE_LENS_NONDETERMINISM=1): only `determinism` failed,
                                                  on lens legs too, but EVERY steer-is-live check passed (noise_floor,
                                                  steering_bites, intervention_hash_distinct, greedy_divergence) and NOT EVERY
                                                  probe flipped on any leg -> traffic proceeds, recorded. (Greedy repeats
                                                  differ on 16-85% of prompts run-to-run under vLLM batching — a text diff is
                                                  a hypersensitive readout — so a count threshold below "all" would encode a
                                                  number the noise does not respect; "all probes flipped" is the shape
                                                  of unstable served arithmetic, and the steer-live checks carry the burden.)
                                                  Under the merged serving tree every route (anchor, twin, steered) repeats
                                                  non-identically on a few probes — the same near-tie flips under batch /
                                                  resumed-prefill composition, on all legs alike — while steering bites on 5/5
                                                  probes with divergences at positions 0..71 and the noise floor holds 8/8. That
                                                  is instrument noise, not a served object silently no-op'd (what this control is
                                                  for). Default OFF: without the opt-in this case REFUSES exactly as before.
    FAIL <check,...>                              anything else

Exit 0 on a PASS* verdict, 1 on FAIL. `f22_positive_control` (common.sh) calls this; tests drive it on recorded results."""
from __future__ import annotations

import json
import os
import sys

STEER_LIVE_CHECKS = ("noise_floor", "steering_bites", "intervention_hash_distinct", "greedy_divergence")


def verdict(record: dict, tolerate_lens: bool) -> str:
    checks = record.get("checks", [])
    failing = [c for c in checks if c.get("status") != "PASS"]
    if not failing:
        return "PASS"
    if len(failing) == 1 and failing[0].get("check") == "determinism":
        f = failing[0]
        nd = f.get("nondeterministic_prompts") or {}
        n = int(f.get("n_prompts") or 0)
        counts = {leg: len(nd.get(leg) or []) for leg in ("steered", "anchor", "twin")}
        if not counts["steered"] and not counts["twin"] and counts["anchor"]:
            return f"PASS_WITH_ANCHOR_NONDETERMINISM anchor_nondeterministic={counts['anchor']}/{n}"
        by_name = {c.get("check"): c.get("status") for c in checks}
        live_ok = all(by_name.get(k) == "PASS" for k in STEER_LIVE_CHECKS)
        not_all = n > 0 and all(v < n for v in counts.values())
        if tolerate_lens and live_ok and not_all:
            return ("PASS_WITH_LENS_NONDETERMINISM " + " ".join(f"{leg}_nondeterministic={v}/{n}" for leg, v in counts.items())
                    + " steer_live_checks=PASS")
    # the SAME anchor noise reaches the noise_floor check, which compares the unhooked anchor with the scale-0 twin
    # byte-for-byte (2026-09-14, llama it_d0p1 at 9ffc03f0, cluster B jobs 100540/100541/100542). On a non-batch-invariant
    # anchor the two draws differ on the anchor's near-tie prompts (haiku / paperclip on llama) while the twin repeats
    # identically and every steer-is-live check passes — anchor noise, not a scale-0 leak. Admitted as the same recorded
    # warning when: the only failing checks are determinism (anchor-only) + noise_floor, the twin is byte-stable, and the
    # differing prompts are a minority of the probes (a majority would say the routes differ, not that the anchor is noisy).
    # This is the rule common.sh's inline fallback has always carried; the helper, which takes precedence whenever it
    # exists, did not — so this case refused on every tree that ships this file (2026-09-15: three llama8b lens legs,
    # 518075 / 518076 / 518077, refused on exactly haiku + paperclip). The two readers now agree.
    by = {c.get("check"): c for c in failing}
    det = by.get("determinism")
    nd = (det or {}).get("nondeterministic_prompts") or {}
    anchor_only = bool(det) and not nd.get("steered") and not nd.get("twin") and bool(nd.get("anchor"))
    nf = by.get("noise_floor")
    if anchor_only and nf and set(by) == {"determinism", "noise_floor"}:
        n = int(nf.get("n_prompts") or det.get("n_prompts") or 0)
        diff = nf.get("differing_prompts") or []
        if n and 0 < len(diff) <= n // 2:
            return (f"PASS_WITH_ANCHOR_NONDETERMINISM anchor_nondeterministic={len(nd['anchor'])}/{det.get('n_prompts')} "
                    f"noise_floor_anchor_vs_twin_differing={len(diff)}/{n} (twin byte-stable; D179)")
    return "FAIL " + ",".join(c.get("check", "?") for c in failing)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: positive_control_verdict.py <positive_control_*.json>", file=sys.stderr)
        return 2
    tolerate = os.environ.get("F22_PC_TOLERATE_LENS_NONDETERMINISM", "") == "1"
    try:
        v = verdict(json.load(open(argv[0], encoding="utf-8")), tolerate)
    except Exception as e:  # noqa: BLE001 — a verdict the caller can read, never a traceback swallowed as PASS
        v = f"FAIL verdict_error:{type(e).__name__}"
    print(v)
    return 0 if v.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
