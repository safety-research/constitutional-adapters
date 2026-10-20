#!/usr/bin/env python3
"""Re-key capability cell stores whose probe WIDTH predates the --parity-n seam.

Before the seam, the driver computed the store key (harness_config_hash)
before reading the probe width, so a widened run (--n-gsm8k, or an edited
PARITY_N) wrote into the SAME store key as the 200-row probe — silent
width-mixing. The seam records a non-default width in the resolved config,
so widened runs key their own stores. This script brings EXISTING stores in
line, deriving everything from what the store itself recorded:

* the widths actually run live in rows.jsonl (distinct s2parity item_ids
  per probe; under --limit, the item INDEX bound tells default from wide);
* the store key is the trailing __<chash8> of the cell dir name;
* the correct new key is harness_config_hash(resolved_config + parity_n),
  with parity_n derived by inverting realized widths (each probe capped at
  its own pool): 0 canonical for the full pools, a scalar for all probes at
  one width, and the driver's per-task dict form ({"gsm8k": 0}) when the
  widths differ across probes — the old --n-gsm8k's product.

DRY-RUN by default: prints one line per store (OK / MOVE / MIXED / ...)
and moves nothing. --apply performs each MOVE copy-then-verify-then-remove
— never a rewrite-in-place: the store is COPIED to its new key, the copy's
manifest.json is rewritten so the store validates and resumes there
(validate_resume compares the manifest's harness_config_hash) with the
original manifest bytes preserved as manifest.json.premigration, every
other file is verified byte-identical to the source, the inverse mapping
is appended to width_migration_inverse.json next to the cells dir, and
only then is the source directory removed. A failure at any step leaves
the source untouched (at worst alongside a partial copy to delete).
Nothing is ever deleted. A mixed instrument (gsm8k widened, mmlu not —
the old --n-gsm8k's product) is representable since the per-task seam, and
re-keys onto the same dict form the driver records, so it can never again
conflate with its default-width twin. Only a store whose widths are
UNINVERTIBLE — a probe with zero rows (it never ran, so its requested
width is unknowable and recording a guess would bind future resumes to a
width nobody asked for), or more rows than its pinned pool holds — is
REPORTED and left alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from sisteer.capeval.store import harness_config_hash
from sisteer.capeval.tiers.parity import (PARITY_N, PARITY_TASK_GSM8K,
                                          PARITY_TASK_MMLU)

# The pinned pools' sizes (pins.yaml revisions): gsm8k-platinum test and
# cais/mmlu 'all' validation. These are the widths --parity-n 0 realizes.
FULL_POOLS = {PARITY_TASK_GSM8K: 1209, PARITY_TASK_MMLU: 1531}
PROBES = (PARITY_TASK_GSM8K, PARITY_TASK_MMLU)


def _parity_ids(store_dir: Path) -> dict[str, set[str]]:
    """Distinct per-probe item_ids from rows.jsonl (aggregate rows excluded)."""
    ids: dict[str, set[str]] = {t: set() for t in PROBES}
    rows = store_dir / "rows.jsonl"
    if not rows.exists():
        return ids
    with rows.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            task, iid = r.get("task", ""), r.get("item_id", "")
            if task in ids and iid and not iid.startswith("__"):
                ids[task].add(iid)
    return ids


def _max_index(ids: set[str]) -> int:
    """item_ids are f'{prefix}{i:04d}-{hash8}' with i the selection index —
    the index BOUND survives a seeded --limit subset, so it distinguishes a
    subset of the 200-row instrument from a subset of a wider pool."""
    out = -1
    for iid in ids:
        digits = iid[1:5]
        if digits.isdigit():
            out = max(out, int(digits))
    return out


def _requested_width(g: int, m: int):
    """Invert realized (gsm8k, mmlu) widths to the CANONICAL --parity-n
    request that produces them (each probe caps at its own pool) — the
    same canonical forms the driver's _parity_width records, so a
    migrated key and a freshly-generated key of the same instrument can
    never disagree:

    * None — the default instrument (no key entry);
    * a scalar — every probe at one request: 0 for the full pools
      (canonical — any request >= the larger pool realizes the same
      widths and hashes differently, so the migration must pick one
      form), else the request itself. The scalar is PREFERRED whenever
      one realizes the pair (e.g. 1300 -> gsm8k capped at 1209): it is
      the form a pre-seam scalar knob could actually have requested;
    * a per-task dict holding only the non-default probes (0 = that
      probe's own full pool) — the previously-unrepresentable mixed
      instrument, e.g. old --n-gsm8k output: gsm8k=1209 mmlu=200 ->
      {"gsm8k": 0}.

    Raises ValueError when a probe's width is UNINVERTIBLE: zero rows
    (the probe never ran, so its requested width is unknowable —
    recording a guess would bind future resumes to a width nobody asked
    for) or more rows than its pinned pool holds."""
    fg, fm = FULL_POOLS[PARITY_TASK_GSM8K], FULL_POOLS[PARITY_TASK_MMLU]
    if (g, m) == (PARITY_N, PARITY_N):
        return None
    if (g, m) == (fg, fm):
        return 0
    for n in sorted({g, m}):
        if n > 0 and (min(n, fg), min(n, fm)) == (g, m):
            return n
    per_task = {}
    for probe, realized, pool in (("gsm8k", g, fg), ("mmlu", m, fm)):
        if realized == 0 or realized > pool:
            raise ValueError(
                f"no --parity-n request realizes gsm8k={g} mmlu={m} "
                f"({probe} ran {realized} rows against a pool of {pool})")
        if realized != PARITY_N:
            per_task[probe] = 0 if realized == pool else realized
    return per_task


def _spec(requested) -> str:
    """The --parity-n spelling of a recorded width (for the report)."""
    if isinstance(requested, dict):
        return ",".join(f"{k}={v}" for k, v in sorted(requested.items()))
    return str(requested)


def classify(store_dir: Path) -> dict:
    """One store's verdict: status + (for MOVE) the re-key plan."""
    out: dict = {"dir": store_dir}
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        out["status"] = "NOMANIFEST"
        return out
    manifest = json.loads(manifest_path.read_text())
    rc = dict(manifest.get("resolved_config") or {})
    if not rc:
        out["status"] = "NORESOLVED"
        return out
    ids = _parity_ids(store_dir)
    counts = {t: len(ids[t]) for t in PROBES}
    out["counts"] = counts
    if not any(counts.values()):
        out["status"] = "NOPARITY"  # judged-only store; width never ran
        return out

    limit = rc.get("limit")
    if limit:
        # a --limit subset: counts are min(limit, width), so the width is
        # recovered from the index bound instead. Every index < PARITY_N
        # == a subset of the default instrument; anything beyond means a
        # subset of a WIDER pool, whose exact width the subset does not
        # determine — report, don't guess.
        if all(_max_index(ids[t]) < PARITY_N for t in PROBES if ids[t]):
            widths = (PARITY_N, PARITY_N)
        else:
            out["status"] = "WIDE-SUBSET"
            return out
    else:
        widths = (counts[PARITY_TASK_GSM8K], counts[PARITY_TASK_MMLU])
    out["widths"] = widths

    try:
        requested = _requested_width(*widths)
    except ValueError as e:
        out["status"], out["why"] = "MIXED", str(e)
        return out
    out["requested"] = requested

    prefix, _, cur_chash8 = store_dir.name.rpartition("__")
    expected_rc = {k: v for k, v in rc.items() if k != "parity_n"}
    if requested is not None:
        expected_rc["parity_n"] = requested
    expected_chash = harness_config_hash(expected_rc)
    expected_name = f"{prefix}__{expected_chash[:8]}"
    out["expected_rc"], out["expected_chash"] = expected_rc, expected_chash
    out["expected_name"] = expected_name

    # Only a WIDTH change is this script's to make. A store whose recorded
    # key does not even match a recomputation of its own stored resolved
    # config has drifted for some OTHER reason (e.g. a deliberate post-hoc
    # manifest edit, like the judge-switch annotations in the grok-judged
    # lanes) — report it, never "repair" it here.
    self_consistent = (
        harness_config_hash(rc) == manifest.get("harness_config_hash")
        and manifest.get("harness_config_hash", "").startswith(cur_chash8))
    if rc == expected_rc:
        out["status"] = "OK" if self_consistent else "DRIFT"
        return out
    if not self_consistent:
        out["status"] = "DRIFT"  # width also wrong, but not safely movable
        return out
    out["status"] = "MOVE"
    return out


def _verify_copy(src: Path, dst: Path) -> None:
    """Every file travelled byte-identically — manifest.json excepted (it
    is deliberately rewritten; its ORIGINAL bytes must sit at the copy's
    manifest.json.premigration instead). Refuses on any missing, extra,
    or differing file, BEFORE the source is removed."""
    rel = lambda root: {q.relative_to(root)  # noqa: E731
                        for q in root.rglob("*") if q.is_file()}
    src_files, dst_files = rel(src), rel(dst)
    expected = src_files | {Path("manifest.json.premigration")}
    if dst_files != expected:
        raise SystemExit(
            f"copy verify FAILED at {dst}: missing "
            f"{sorted(map(str, expected - dst_files))}, extra "
            f"{sorted(map(str, dst_files - expected))}")
    for f in src_files:
        against = (Path("manifest.json.premigration")
                   if f == Path("manifest.json") else f)
        if (src / f).read_bytes() != (dst / against).read_bytes():
            raise SystemExit(f"copy verify FAILED at {dst}: {f} differs")


def apply_move(verdict: dict) -> Path:
    """Copy-then-verify-then-remove, never a rewrite-in-place: the store
    is copied to its new key, the COPY's manifest is rewritten (original
    bytes preserved at manifest.json.premigration), every other file is
    verified byte-identical, the inverse mapping is appended to
    width_migration_inverse.json beside the cells dir, and only then is
    the source removed — so a failure at any step leaves the source
    untouched."""
    src: Path = verdict["dir"]
    dst = src.with_name(verdict["expected_name"])
    if dst.exists():
        raise SystemExit(f"refusing: {dst} already exists")
    if (src / "manifest.json.premigration").exists():
        raise SystemExit(
            f"refusing: {src} already carries manifest.json.premigration "
            "(a prior migration); re-migrating would bury its provenance "
            "chain — resolve by hand")
    shutil.copytree(src, dst)
    manifest_path = dst / "manifest.json"
    original = manifest_path.read_text()
    (dst / "manifest.json.premigration").write_text(original)
    manifest = json.loads(original)
    manifest["resolved_config"] = verdict["expected_rc"]
    manifest["harness_config_hash"] = verdict["expected_chash"]
    g, m = verdict["widths"]
    manifest["parity_n"] = {
        "requested": verdict["requested"],
        "realized": {"gsm8k": g, "mmlu": m},
        "migrated_from": src.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    _verify_copy(src, dst)
    inverse = src.parent.parent / "width_migration_inverse.json"
    log = json.loads(inverse.read_text()) if inverse.exists() else {}
    log[str(dst)] = str(src)
    inverse.write_text(json.dumps(log, indent=1, sort_keys=True))
    shutil.rmtree(src)
    return dst


def verify(store_dir: Path) -> None:
    """The migrated store must be discoverable at its new key: dir suffix,
    manifest hash, and a fresh hash of the manifest's resolved config all
    agree (what validate_resume and _find_cell_store read)."""
    manifest = json.loads((store_dir / "manifest.json").read_text())
    chash = manifest["harness_config_hash"]
    recomputed = harness_config_hash(manifest["resolved_config"])
    name8 = store_dir.name.rpartition("__")[2]
    if not (chash == recomputed and chash.startswith(name8)):
        raise SystemExit(f"verify FAILED at {store_dir}: dir {name8}, "
                         f"manifest {chash}, recomputed {recomputed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path,
                    help="sweep roots; every */cells/*__controls__* under "
                         "each is classified")
    ap.add_argument("--apply", action="store_true",
                    help="perform the MOVEs (default: dry-run, print only)")
    args = ap.parse_args()

    stores = sorted({p for root in args.roots
                     for p in root.glob("**/cells/*__controls__*")
                     if p.is_dir()})
    if not stores:
        raise SystemExit(f"no cell stores under {args.roots}")
    moves, n_ok = [], 0
    for store in stores:
        v = classify(store)
        n_ok += v["status"] == "OK"
        line = f"{v['status']:12s} {store}"
        if v["status"] == "MOVE":
            line += (f"\n{'':12s} -> {v['expected_name']}"
                     f"  (widths gsm8k={v['widths'][0]} mmlu={v['widths'][1]}"
                     f" => --parity-n {_spec(v['requested'])})")
            moves.append(v)
        elif v["status"] in ("MIXED", "WIDE-SUBSET"):
            line += f"  counts={v.get('counts')} {v.get('why', '')}"
        print(line)
    print(f"\n{len(stores)} stores: {len(moves)} to move, "
          f"{n_ok} already keyed correctly")
    if not args.apply:
        if moves:
            print("dry-run: nothing moved (pass --apply)")
        return
    for v in moves:
        dst = apply_move(v)
        verify(dst)
        print(f"moved    {v['dir'].name} -> {dst.name}  (verified)")


if __name__ == "__main__":
    main()
