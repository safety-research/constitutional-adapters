#!/usr/bin/env python
"""alphasteer-full22 Phase C.0 — seed a PRIVATE judge cache and scrub it GLOBALLY by PRODUCER
STAMP. A precondition of the first ladder standup.

Why: the shared cache (~/.local/state/capeval/judge_cache.sqlite) holds 450 verdicts PRODUCED by
grok-4.6 through an OpenRouter fallback but filed under sonnet-5 protocol keys; they label exactly
the items sonnet-5 abstains on, so consuming them biases a benign REFUSAL rate downward and lets an
over-refusing lambda pass condition (a). The key cannot tell them apart — only the producer stamp
can (verdict_json.fallback == true, or org_used not an Anthropic pool).

What this does (thin wrapper — the rule is `seed_private_cache.global_scrub` @ c88e3daa, imported,
never re-implemented):
  1. VACUUM INTO a private copy from a READ-ONLY connection (the shared cache is never written);
  2. run `global_scrub` (producer stamp, NEVER shape — a shape rule across all protocols once
     deleted 66,772 legitimate pairwise verdicts);
  3. ASSERT the removed count equals the catalogue's n_foreign_rows AND the removed key set equals
     the catalogue's `rows` key set — a difference either way is a finding, never accepted;
  4. write judge_cache_scrub.json (private path + sha256, catalogue sha256, counts by reason /
     org_used / judge_model, module shas) tmp+replace, and stage it where `ladder.sbatch` refuses
     without it.

usage: python scripts/alphasteer_f22/scrub_judge_cache.py --out <private.sqlite> \
          --judge-tree <the repo checkout holding the judge> [--catalogue ...] [--attestation ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SHARED = Path("~/.local/state/capeval/judge_cache.sqlite").expanduser()
#: under the site's state root ($STATE); expanded after parsing through the site config (scripts/site_paths.py loads the site
#: module by file path: this script imports the judge tree's sisteer, never this checkout's)
DEFAULT_CATALOGUE = Path("$STATE/romp/candidate-si-refit/run-state/full22/judge_cache_foreign_rows.json")
JUDGE_TREE_SHA = "c88e3daa97dde153283acafd3bdde30b6a232f18"   # halfdose-controls: the gating grader + global_scrub
_ANTHROPIC_ORGS = {"lowprio", "highprio", "batch"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict) -> None:
    body = json.dumps(payload, indent=1, sort_keys=True).encode()
    payload = dict(payload, sha256=hashlib.sha256(body).hexdigest())
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def import_global_scrub(judge_tree: Path):
    """Import `global_scrub` from the judge tree's seed_private_cache.py (c88e3daa's file — the
    caller verifies the tree's copy is byte-identical to that commit's)."""
    mod_path = judge_tree / "scripts/sdfbase/full22/judge/seed_private_cache.py"
    committed = subprocess.run(["git", "-C", str(judge_tree), "show",
                                f"{JUDGE_TREE_SHA}:scripts/sdfbase/full22/judge/seed_private_cache.py"],
                               capture_output=True, check=True).stdout
    if hashlib.sha256(committed).hexdigest() != sha256_file(mod_path):
        raise SystemExit(f"{mod_path} differs from {JUDGE_TREE_SHA[:8]}'s copy; refusing to import a drifted scrub")
    for p in (judge_tree / "src", judge_tree / "scripts", mod_path.parent):
        sys.path.insert(0, str(p))
    import seed_private_cache as spc  # noqa: E402
    return spc, sha256_file(mod_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    ap.add_argument("--out", type=Path, required=True, help="the PRIVATE copy to create (refuses to overwrite)")
    ap.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    ap.add_argument("--judge-tree", type=Path, required=True,
                    help="checkout holding scripts/sdfbase/full22/judge/seed_private_cache.py @ c88e3daa")
    ap.add_argument("--attestation", type=Path, default=None,
                    help="where to write judge_cache_scrub.json (default: beside --out)")
    args = ap.parse_args(argv)
    scripts = str(Path(__file__).resolve().parents[1])
    if scripts not in sys.path:
        sys.path.append(scripts)
    from site_paths import site_path
    args.catalogue = site_path(args.catalogue, "--catalogue")   # an unset $STATE refuses by name

    shared, out = args.shared.expanduser(), args.out.expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite a private cache (delete it deliberately first)")
    if not shared.is_file():
        raise SystemExit(f"no shared cache at {shared}")
    cat = json.loads(args.catalogue.expanduser().read_text())
    cat_sha = sha256_file(args.catalogue.expanduser())
    want_n = int(cat["n_foreign_rows"])
    want_keys = set(cat["rows"])
    if len(want_keys) != want_n:
        raise SystemExit(f"catalogue inconsistent: n_foreign_rows={want_n} but {len(want_keys)} distinct row keys")

    spc, spc_sha = import_global_scrub(args.judge_tree.expanduser())
    if spc._ANTHROPIC_ORGS != _ANTHROPIC_ORGS:
        raise SystemExit(f"seed_private_cache._ANTHROPIC_ORGS={spc._ANTHROPIC_ORGS} != {_ANTHROPIC_ORGS}")

    out.parent.mkdir(parents=True, exist_ok=True)
    src_sha_before = sha256_file(shared)
    sc = sqlite3.connect(f"file:{shared}?mode=ro", uri=True)
    n_shared = sc.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    sc.execute("VACUUM INTO ?", (str(out),))
    sc.close()
    os.chmod(out, 0o600)

    dc = sqlite3.connect(out, timeout=60)
    n_before = dc.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    rec = spc.global_scrub(dc)
    n_after = dc.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    left = dc.execute("SELECT COUNT(*) FROM verdicts WHERE org_used NOT IN ('lowprio','highprio','batch') "
                      "OR verdict_json LIKE '%\"fallback\": true%' OR verdict_json LIKE '%\"fallback\":true%'").fetchone()[0]
    dc.execute("VACUUM"); dc.commit(); dc.close()

    got_keys = set(rec["keys"])
    problems = []
    if rec["n_removed"] != want_n:
        problems.append(f"removed {rec['n_removed']} rows but the catalogue names {want_n}")
    if got_keys != want_keys:
        problems.append(f"removed key set != catalogue: {len(got_keys - want_keys)} removed-but-uncatalogued, "
                        f"{len(want_keys - got_keys)} catalogued-but-not-removed")
    if n_before - n_after != rec["n_removed"]:
        problems.append(f"row delta {n_before - n_after} != n_removed {rec['n_removed']}")
    if left:
        problems.append(f"{left} producer-stamped foreign rows survive the scrub")
    if n_before != n_shared:
        problems.append(f"VACUUM INTO copied {n_before} rows but the shared cache holds {n_shared}")
    if sha256_file(shared) != src_sha_before:
        problems.append("the SHARED cache changed during the scrub (it must never be written by this step)")

    att = {
        "schema": "alphasteer_f22_judge_cache_scrub/1",
        "private_cache": str(out), "private_cache_sha256": sha256_file(out),
        "seeded_from": str(shared), "shared_cache_sha256_at_seed": src_sha_before,
        "n_entries_shared": n_shared, "n_entries_private_before_scrub": n_before,
        "n_entries_private": n_after, "n_removed": rec["n_removed"],
        "by_reason": rec["by_reason"], "by_org_used": rec["by_org_used"], "by_judge_model": rec["by_judge_model"],
        "removed_keys_sha256": hashlib.sha256("\n".join(sorted(got_keys)).encode()).hexdigest(),
        "catalogue": str(args.catalogue.expanduser()), "catalogue_sha256": cat_sha,
        "catalogue_n_foreign_rows": want_n, "catalogue_written_at": cat.get("written_at"),
        "scrub_rule": "PRODUCER STAMP — verdict_json.fallback true OR org_used not in {lowprio,highprio,batch}; never shape (D23)",
        "scrub_module": "scripts/sdfbase/full22/judge/seed_private_cache.py:global_scrub",
        "scrub_module_sha256": spc_sha, "scrub_module_commit": JUDGE_TREE_SHA,
        "this_script_sha256": sha256_file(Path(__file__).resolve()),
        "matches_catalogue": not problems, "problems": problems, "written_at": _utc(),
    }
    att_path = (args.attestation or out.with_name("judge_cache_scrub.json")).expanduser()
    _write_json(att_path, att)
    print(json.dumps({k: v for k, v in att.items() if k not in ("by_reason",)}, indent=1))
    if problems:
        print("SCRUB REFUSED:", *problems, sep="\n  ", file=sys.stderr)
        return 3
    print(f"OK: private cache {out} ({n_after} rows; removed {rec['n_removed']} == catalogue); attestation {att_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
