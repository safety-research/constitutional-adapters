#!/usr/bin/env python3
"""Freeze the HELD-OUT goal rows as a named, hashed eval artifact.

WHY THIS EXISTS. The training pool is pinned: `pool_manifest.json` carries a
sha256 and `pairs_*.jsonl` are files you can hand to someone. The eval side
was not. Its rows lived inside the same store files as the training rows,
distinguishable only by re-deriving the goal split and filtering on
membership -- a RECOMPUTATION, not an artifact.

That is drift-prone in the exact way this campaign kept getting bitten: change
the split seed, or regenerate a store file, and "the eval set" silently
becomes a different set of rows with nothing to flag it. The training pool
cannot drift that way because the manifest hash would stop it. This gives the
eval set the same protection.

SHAPE. Not pairs. One row per (goal, family, variant, SI condition) carrying
the prompt, the response and the grade -- what you need to SCORE a trained
defence, rather than the {prompt, pos, neg} a distillation objective consumes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))


def _load_pool_builder():
    spec = importlib.util.spec_from_file_location(
        "_bp", _HERE / "build_two_si_pool.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_bp"] = m
    spec.loader.exec_module(m)
    return m


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/two_si.yaml")
    ap.add_argument("--store-dir", default=None,
                    help="graded store (default: $TT_TWO_SI_STORE_DIR, else "
                         "pool.store_dir)")
    ap.add_argument("--out-dir", required=True,
                    help="dir for eval_<si>.jsonl + eval_meta.json (0700)")
    ap.add_argument("--families", default=None,
                    help="comma-separated subset to emit; default every "
                         "train family that HAS held-out rows. A family whose "
                         "goal ids carry no harm category (wildjailbreak_"
                         "harmful) cannot be split by the category holdout "
                         "and contributes nothing -- that is reported, not "
                         "silently omitted.")
    args = ap.parse_args(argv)

    import yaml

    bp = _load_pool_builder()
    cfg = yaml.safe_load(Path(args.config).read_text())
    # --store-dir, else the pool builder's own resolution ($TT_TWO_SI_STORE_DIR, else the config's
    # pool.store_dir with its placeholder expanded)
    if args.store_dir:
        from sisteer import site as _site
        store = _site.expand_path(args.store_dir, key="--store-dir")
    else:
        store = bp._store_dir(cfg["pool"])

    static = tuple(cfg["pool"]["static_families"])
    train_goals, eval_goals = bp._goal_split_from_store(cfg, static)
    eval_goals = set(eval_goals)
    if not eval_goals:
        raise SystemExit(
            "[emit-eval] the held-out goal set is EMPTY — refusing to write an "
            "eval artifact that would silently evaluate on nothing.")
    overlap = eval_goals & set(train_goals)
    if overlap:
        raise SystemExit(
            f"[emit-eval] {len(overlap)} goal(s) are in BOTH splits — the "
            f"holdout is not a holdout. Refusing.")

    fams = ([f.strip() for f in args.families.split(",") if f.strip()]
            if args.families else list(static))
    si_ids = list(cfg["si_ids"])

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)

    # si_condition on a store row is the CONDITION label; si_id is None for
    # the undefended pass. Key on whichever is present so the undefended rows
    # land under "none" rather than being dropped for having a null si_id.
    def _cond(r):
        return r.get("si_id") or r.get("si_condition") or "none"

    buckets: dict[str, list[dict]] = {}
    per_family: dict[str, int] = {}
    empty_fams: list[str] = []
    ungraded = 0
    for fam in fams:
        p = store / f"{fam}.jsonl"
        if not p.exists():
            empty_fams.append(fam)
            continue
        n_fam = 0
        for line in p.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("goal_id") not in eval_goals:
                continue
            if r.get("score") is None:
                ungraded += 1          # honestly ungraded; never scored 0
                continue
            cond = _cond(r)
            buckets.setdefault(cond, []).append({
                "goal_id": r["goal_id"], "family": fam,
                "variant_name": r.get("variant_name"),
                "sample_idx": r.get("sample_idx"),
                "candidate_idx": r.get("candidate_idx"),
                "si_condition": cond,
                "category": r.get("category"), "source": r.get("source"),
                "prompt": r.get("attack_prompt"), "response": r.get("response"),
                "score": r.get("score"),
                "jailbroken": bool(r["score"] > float(
                    cfg["prereg"]["threshold_grid"][1])),
            })
            n_fam += 1
        per_family[fam] = n_fam
        if n_fam == 0:
            empty_fams.append(fam)

    if not any(buckets.values()):
        raise SystemExit("[emit-eval] no held-out rows found — refusing to "
                         "write an empty eval set.")

    files = {}
    for cond in sorted(buckets):
        # deterministic order so the file hash is stable across runs
        rows = sorted(buckets[cond], key=lambda r: (
            r["family"], r["goal_id"], r["sample_idx"] or 0,
            r["variant_name"] or ""))
        f = out_dir / f"eval_{cond}.jsonl"
        with f.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        os.chmod(f, 0o600)
        files[f.name] = {"rows": len(rows), "sha256": _sha256_file(f)}
        print(f"[emit-eval] wrote {len(rows)} rows -> {f}")

    meta = {
        "version": 1,
        "regime": "goal_level_holdout",
        "n_eval_goals": len(eval_goals),
        "n_train_goals": len(train_goals),
        "eval_goal_ids": sorted(eval_goals),
        "si_conditions": sorted(buckets),
        "declared_si_ids": si_ids,
        "families_requested": fams,
        "per_family_rows": per_family,
        "families_with_no_heldout_rows": sorted(set(empty_fams)),
        "ungraded_rows_skipped": ungraded,
        "threshold": float(cfg["prereg"]["threshold_grid"][1]),
        "files": files,
        "store_dir": str(store),
    }
    mp = out_dir / "eval_meta.json"
    mp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.chmod(mp, 0o600)
    print(f"[emit-eval] sidecar -> {mp}")
    print(f"[emit-eval] {len(eval_goals)} held-out goals, "
          f"{sum(v['rows'] for v in files.values())} rows, "
          f"{len(files)} condition file(s)")
    if empty_fams:
        # STATED, not silently omitted: a family with no held-out rows is a
        # real property of the split (wildjailbreak_harmful's goal ids carry
        # no harm category, so the category holdout cannot reach them).
        print(f"[emit-eval] NOTE families with no held-out rows: "
              f"{sorted(set(empty_fams))}")
    if ungraded:
        print(f"[emit-eval] NOTE skipped {ungraded} ungraded row(s) "
              f"(score=None — honestly ungraded, never scored 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
