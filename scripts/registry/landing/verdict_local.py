#!/usr/bin/env python3
"""Gate verdict from two LOCAL summary.json files — the failure-set verdict.

usage: verdict_local.py <baseline-summary.json> <merge-summary.json>

Refuses a non-terminal summary (status not PASS/FAIL, or no counts: an empty failure list there is NOT a pass). INTRODUCED =
M - B by node id is what protects the landing: an identical failure set LANDs; a VANISHED-only set LANDs under ANY layout with the
vanished nodes named (a module that stops collecting is its own `unknown::` INTRODUCED node, so vanished nodes cannot hide a
regression — they are a baseline flake or a fix); a mixed shard layout WITH an introduced node is a HOLD (order-dependent tests
can flip; re-run before blaming); otherwise an introduced node HOLDs unless it is the one sanctioned flaky node.
Prints the lines land.sh reuses a recorded verdict by (`^MERGE sha=<8 hex>`, `^VERDICT: LAND`); gate_slurm.sh writes them to
$STATE/verdicts/gate-land-<branch>.out (STATE = ~/.local/state/registry-landing by default)."""
import json
import sys

FLAKY_ALLOWED = {"wide_bf16"}   # the one sanctioned flaky node (substring match on the node id), allowed ALONE
#: COLLECTION-FLAKE PAIRS (2026-09-28): each frozenset is a set of collection-error nodes of which exactly ONE fails per run, and
#: WHICH one depends on what shares the shard. When the only INTRODUCED node and the only VANISHED node are two members of one
#: set, the failure set did not move: LAND with a caveat naming the flip.
#: RETIRED 2026-09-29: the one pair this held -- the ifeval r2/r3 collection flip -- was shard
#: co-tenancy (tests/ifeval_synthetic_store.py imported a scripts/ module without putting scripts/ on sys.path, so r2/r3 collected
#: only beside a shard-mate that did); the fix landed on main in 3c06e215f. Kept as an EMPTY tuple so collection_flip()
#: stays a no-op: a collection error on either module is a real INTRODUCED node from here on.
ALLOWED_COLLECTION_FLIPS = ()


def collection_flip(intro, vanished):
    """The ALLOWED_COLLECTION_FLIPS set that explains a one-in / one-out swap, else None."""
    if len(intro) != 1 or len(vanished) != 1 or intro[0] == vanished[0]:
        return None
    pair = {intro[0], vanished[0]}
    return next((s for s in ALLOWED_COLLECTION_FLIPS if pair <= s), None)


def nodes(p):
    d = json.load(open(p)); v = d.get("failures") or []
    return d, {f["node"] if isinstance(f, dict) else f for f in v}


if len(sys.argv) != 3:
    sys.exit(__doc__)
pb, pm = sys.argv[1], sys.argv[2]
db, B = nodes(pb); dm, M = nodes(pm)
bad = False
for tag, d, s, p in (("BASELINE", db, B, pb), ("MERGE", dm, M, pm)):
    print(f"{tag} sha={d.get('sha','')[:8]} status={d.get('status')} pytest_rc={d.get('pytest_rc')} counts={d.get('counts')} "
          f"elapsed={d.get('elapsed_s')} nodes={len(s)} shards={d.get('shards')} layout={ (d.get('shard_layout') or {}).get('map_sha256','')[:12] } file={p}")
    if d.get("status") not in ("PASS", "FAIL") or not d.get("counts"):
        print(f"  REFUSE: {tag} summary is not a terminal gate result (status={d.get('status')}); an empty failure list here is NOT a pass.")
        bad = True
    if d.get("substrate") and d.get("substrate") != "slurm":
        print(f"  NOTE: {tag} summary's substrate is {d.get('substrate')!r}, not slurm")
if (db.get("shards"), (db.get("shard_layout") or {}).get("map_sha256")) != (dm.get("shards"), (dm.get("shard_layout") or {}).get("map_sha256")):
    print("\nWARNING: MIXED SHARD LAYOUTS (shards/map_sha256 differ) — order-dependent tests can flip; an INTRODUCED node needs a re-run before it is blamed.")
    layout_mixed = True
else:
    layout_mixed = False
intro, vanished = sorted(M - B), sorted(B - M)
print(f"\nINTRODUCED ({len(intro)}):"); print("\n".join("  " + n for n in intro) or "  (none)")
print(f"VANISHED ({len(vanished)}):"); print("\n".join("  " + n for n in vanished) or "  (none)")
print("\nBASELINE nodes:"); print("\n".join("  " + n for n in sorted(B)))
print("\nMERGE nodes:"); print("\n".join("  " + n for n in sorted(M)))
unexplained = [n for n in intro if not any(k in n for k in FLAKY_ALLOWED)]
flip = collection_flip(intro, vanished)
if bad: verdict = "HOLD (non-terminal summary; re-run before reading)"
elif not intro and not vanished: verdict = "LAND" + (" (failure set identical to baseline; layout differs because the branch changed the test count — D64)" if layout_mixed else "")
elif flip is not None:
    print(f"\nCAVEAT: COLLECTION FLAKE PAIR — the one INTRODUCED node {intro[0]} and the one VANISHED node {vanished[0]} are the two faces "
          f"of one collection error (ALLOWED_COLLECTION_FLIPS); which module raises depends on shard co-tenancy, so the failure set did not move.")
    verdict = f"LAND (collection-flake pair: {vanished[0]} -> {intro[0]}; failure set otherwise identical to baseline" + ("; layout differs — D64)" if layout_mixed else ")")
# 2026-09-29: a VANISHED-only set is LAND under any layout — INTRODUCED = M - B by node id is what protects the
# landing, and a module that stops collecting shows up as its own `unknown::` INTRODUCED node, so vanished nodes cannot hide a
# regression; they are a baseline flake or a fix. Under a mixed layout the caveat names them (the 3c06e215f baseline carried a
# timing-flaky precheck node that vanished on a clean tip and read HOLD under the old order, which put the layout HOLD first).
elif not intro: verdict = "LAND (baseline nodes vanished: %s%s)" % (vanished, "; layout differs — D64" if layout_mixed else "")
elif layout_mixed: verdict = "HOLD (mixed shard layout AND the failure set moved; re-run before reading)"
elif not unexplained: verdict = f"LAND (only the sanctioned flaky node introduced: {intro})"
else: verdict = "HOLD"
print("\nVERDICT:", verdict)
