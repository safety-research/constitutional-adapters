#!/usr/bin/env python3
"""Post-merge gate: assert the stage3 correctness fixes survived the merge.

Run this on the MERGED tree before pushing. Every check below corresponds to a
bug that shipped in a real dataset, and every one lives in a file the
twosweep<-stage3 merge conflicts on — so a plausible-looking resolution can
silently drop it and nothing else will notice until an artifact is wrong.

Exit 0 = all invariants intact. Exit 1 = a fix was lost in the merge.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
fails: list[str] = []


def need(path: str, must: str, why: str) -> None:
    p = REPO / path
    if not p.exists():
        fails.append(f"{path}: MISSING — {why}")
        return
    if must not in p.read_text():
        fails.append(f"{path}: lost `{must}` — {why}")


def need_absent(path: str, bad: str, why: str) -> None:
    p = REPO / path
    if p.exists() and bad in p.read_text():
        fails.append(f"{path}: reintroduced `{bad}` — {why}")


# 1. template-opened <think>: the trace is in the PROMPT, so the response has
#    only the closing tag. Without this, strip_think is a no-op and the judge
#    scores the reasoning. Cost: the whole qwen38_think8k pool's grades.
need("src/sisteer/sampling/engine.py", '"<think>" not in out and "</think>" in out',
     "strip_think must treat a bare </think> as a trace boundary")

# 2. ...and the two gates that decide whether strip_think is even CALLED.
#    Fixing the function alone is useless if the caller is gated on the
#    absent opening tag.
src = (REPO / "scripts/build_two_si_pool.py")
if src.exists():
    n = src.read_text().count('"<think>" in response or "</think>" in response')
    if n < 2:
        fails.append(f"scripts/build_two_si_pool.py: {n}/2 grade gates accept a "
                     "bare </think> — a gate keyed only on the opening tag makes "
                     "the strip unreachable")

# 3. 429 policy: throttle down on the SAME org, never hop to High-Prio.
need("src/sisteer/throttle.py", "class AdaptiveGate",
     "the shared throttle gate (user rule 2026-08-21, floor 50)")
need("src/sisteer/capeval/judging.py", "def _select_org",
     "the org-selection seam that refuses a rate-limit-caused High-Prio hop")
jud = (REPO / "src/sisteer/capeval/judging.py")
if jud.exists():
    t = jud.read_text()
    body = t[t.find("def _create("):t.find("def _note_rate_limit(")] if "def _create(" in t else t
    if "highprio" in body:
        fails.append("src/sisteer/capeval/judging.py: _create names highprio — "
                     "the D17 org-hop is back; 429s must throttle, not relocate")

# 4. request timeout: without it a 500-wide grade wedges silently for 10 min
#    per hung socket, with no 429 and no exception to diagnose from.
need("scripts/grade_bon_live.py", "timeout=timeout",
     "make_client must pass an explicit timeout; the SDK default is 600s")

# 5. degeneracy flags ride on every pair (flag, never filter).
need("src/sisteer/eval/degeneracy.py", "def classify",
     "the shared incoherent/repetitious detector")
# NB the field names are BUILT (f"pos_{k}"), so grep the constructor, not the
# literal — checking for "pos_degenerate" is a false negative on correct code.
need("scripts/build_two_si_pool.py", 'f"pos_{k}"',
     "emit-pairs must stamp the degeneracy flag on the pos side")
need("scripts/build_two_si_pool.py", 'f"neg_{k}"',
     "emit-pairs must stamp the degeneracy flag on the neg side")

# 6. sharding is round-robin, not contiguous — contiguous left GPUs idle.
need("scripts/pool_sample.py", "[idx::n]",
     "round-robin sharding; contiguous slices starve later shards")

if fails:
    print("MERGE INVARIANTS VIOLATED — do not push this merge:\n")
    for f in fails:
        print("  ✗ " + f)
    print(f"\n{len(fails)} invariant(s) lost. Each maps to a bug that already "
          "shipped in a dataset once.")
    sys.exit(1)
print("all merge invariants intact")
