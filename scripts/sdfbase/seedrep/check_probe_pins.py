#!/usr/bin/env python
"""Probe-pin leak check for a seed replicate's held-out split.

The contrastive probes (``probe_v2``, ``probe_v3_genre``) are continuations
of HELD-OUT documents; a seed's re-carved split is only a valid probe
substrate if every probe source document sits on its held-out side AND
none of them is trained on. This check REFUSES (exit 2) when

- any probe ``source == heldout_doc`` ``source_id`` is a ``doc_id`` of a
  training pool named by ``--train-pools`` (a probe leak);
- any held-out doc of the split at all is a ``doc_id`` of a training pool
  (the broader split leak — the probe leak is a special case of it);
- any probe source id is OUTSIDE the split's held-out set;
- either family's realized held-out fraction is outside
  ``[--frac, --frac + --window)`` (the lane's 0.085 with a 0.02 whole-group
  overshoot allowance, decisions O2);
- the split carries no ``pins`` block (an unpinned carve is not a seed
  split) — ``--allow-unpinned`` relaxes only this, for reading the
  campaign's seed-0 split.

It prints the pinned / seed-drawn group counts per family and the
seed-drawn share of the held-out mass from the split's ``pins`` block.
Read-only; no corpus text is loaded (pools are streamed for ``doc_id``).

  python scripts/sdfbase/seedrep/check_probe_pins.py \\
      --split data/sdfbase/heldout_split.s1.json \\
      --probes data/sdfbase/probe_v2.jsonl data/sdfbase/probe_v3_genre.jsonl \\
      --train-pools $STAGE/sdfbase/seedrep/pools_s1/pairs_constv2-mix-*.jsonl \\
      --frac 0.085
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SDFBASE = _HERE.parent
if str(_SDFBASE) not in sys.path:
    sys.path.insert(0, str(_SDFBASE))

import build_heldout_split as hsplit  # noqa: E402

#: Whole-group overshoot allowance above the requested fraction.
WINDOW = 0.02
FAMILIES = ("advice", "stories")


class ProbePinError(RuntimeError):
    """A refusal — the split is not a valid probe substrate."""


def _tilde(p) -> str:
    s = str(p)
    home = str(Path.home())
    return "~" + s[len(home):] if s.startswith(home + "/") else s


def pool_doc_ids(pool_path) -> list[str]:
    """The ``doc_id`` of every row of a pool jsonl, streamed."""
    ids: list[str] = []
    with open(pool_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            did = row.get("doc_id")
            if not did:
                raise ProbePinError(f"{pool_path}: row {lineno} has no doc_id")
            ids.append(str(did))
    return ids


def check(split_path, probe_paths, train_pools=(), *, frac: float,
          window: float = WINDOW, allow_unpinned: bool = False) -> dict:
    """Run every check; returns a report dict with ``ok`` and the named
    findings (never raises on a finding — the CLI turns ``ok: False`` into
    the exit-2 refusal; a malformed input raises)."""
    split_path = Path(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    held = {i for fam in FAMILIES for i in split[fam]["ids"]}
    source_ids = hsplit.load_probe_source_ids(probe_paths)
    src = set(source_ids)

    outside = sorted(src - held)
    probe_leaks: dict[str, list[str]] = {}
    heldout_leaks: dict[str, int] = {}
    pool_counts: dict[str, int] = {}
    for p in train_pools:
        p = Path(p)
        ids = set(pool_doc_ids(p))
        pool_counts[_tilde(p)] = len(ids)
        hit = sorted(src & ids)
        if hit:
            probe_leaks[_tilde(p)] = hit
        n_held = len(held & ids)
        if n_held:
            heldout_leaks[_tilde(p)] = n_held

    frac_realized = {fam: split["token_totals"][fam]["heldout_frac_realized"]
                     for fam in FAMILIES}
    frac_ok = {fam: (frac <= v < frac + window)
               for fam, v in frac_realized.items()}

    pins = split.get("pins")
    pin_stats = None
    if pins:
        pin_stats = {
            "n_pinned_source_ids": pins.get("n_pinned_source_ids"),
            "probe_files": [p.get("name") for p in pins.get("probe_files", [])],
            **{fam: {
                "pinned_groups": len(pins[fam]["pinned_groups"]),
                "seed_drawn_groups": len(pins[fam]["seed_drawn_groups"]),
                "pinned_records": pins[fam].get("pinned_records"),
                "seed_drawn_records": pins[fam].get("seed_drawn_records"),
                "pinned_frac_of_family": pins[fam].get("pinned_frac_of_family"),
                "seed_drawn_share_of_heldout":
                    pins[fam].get("seed_drawn_share_of_heldout"),
            } for fam in FAMILIES},
        }
        # The pins block's own list must agree with the probe files handed
        # in — a split pinned from OTHER probes is not this lane's split.
        pinned_in_split = set(pins.get("pinned_source_ids") or [])
        pins_agree = pinned_in_split == src
    else:
        pins_agree = None

    findings = []
    if outside:
        findings.append(f"{len(outside)} probe source id(s) outside the "
                        f"held-out set (first: {outside[:5]})")
    for pool, hit in probe_leaks.items():
        findings.append(f"{len(hit)} probe source id(s) inside training pool "
                        f"{pool} (first: {hit[:5]})")
    for pool, n in heldout_leaks.items():
        findings.append(f"{n} held-out doc(s) inside training pool {pool}")
    for fam, ok in frac_ok.items():
        if not ok:
            findings.append(
                f"{fam}: frac_realized {frac_realized[fam]:.4f} outside "
                f"[{frac}, {frac + window})")
    if pins is None and not allow_unpinned:
        findings.append("split carries no 'pins' block — not a pinned seed "
                        "split (pass --allow-unpinned to read an unpinned "
                        "carve)")
    if pins_agree is False:
        findings.append("the split's pinned_source_ids differ from the probe "
                        "files given — pinned from other probes?")

    return {
        "split": _tilde(split_path),
        "split_sha256": hsplit.sha256_file(split_path),
        "salt": split.get("salt"),
        "eval_salt": (split.get("eval_subset") or {}).get("salt"),
        "frac_requested": split.get("frac"),
        "frac_window": [frac, frac + window],
        "frac_realized": frac_realized,
        "frac_ok": frac_ok,
        "n_heldout_docs": len(held),
        "n_probe_source_ids": len(source_ids),
        "n_probe_sources_in_heldout": len(src & held),
        "probe_sources_outside_heldout": outside,
        "train_pools": pool_counts,
        "probe_leaks": probe_leaks,
        "heldout_docs_in_training_pools": heldout_leaks,
        "pins": pin_stats,
        "pins_agree_with_probes": pins_agree,
        "findings": findings,
        "ok": not findings,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--probes", type=Path, nargs="+", required=True)
    ap.add_argument("--train-pools", type=Path, nargs="*", default=[],
                    help="training pool jsonl files (shell-expanded glob)")
    ap.add_argument("--frac", type=float, required=True,
                    help="requested constitution fraction (lane value 0.085)")
    ap.add_argument("--window", type=float, default=WINDOW)
    ap.add_argument("--allow-unpinned", action="store_true")
    ap.add_argument("--report", type=Path, default=None,
                    help="write the JSON report here")
    args = ap.parse_args(argv)
    try:
        rep = check(args.split, args.probes, args.train_pools, frac=args.frac,
                    window=args.window, allow_unpinned=args.allow_unpinned)
    except (ProbePinError, hsplit.SplitError, OSError, KeyError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=1) + "\n",
                               encoding="utf-8")
    print(f"split {rep['split']} salt={rep['salt']} eval_salt={rep['eval_salt']} "
          f"held-out docs {rep['n_heldout_docs']}; probe sources "
          f"{rep['n_probe_source_ids']} ({rep['n_probe_sources_in_heldout']} in "
          f"held-out); training pools checked: {len(rep['train_pools'])}")
    for fam in FAMILIES:
        line = (f"  {fam}: frac_realized {rep['frac_realized'][fam]:.4f} "
                f"in [{args.frac}, {args.frac + args.window}) -> "
                f"{'ok' if rep['frac_ok'][fam] else 'MISS'}")
        if rep["pins"]:
            ps = rep["pins"][fam]
            line += (f"; pinned groups {ps['pinned_groups']} "
                     f"({ps['pinned_records']} docs, "
                     f"{ps['pinned_frac_of_family']:.4f} of family), "
                     f"seed-drawn groups {ps['seed_drawn_groups']} "
                     f"({ps['seed_drawn_records']} docs, "
                     f"{ps['seed_drawn_share_of_heldout']:.3f} of held-out "
                     "tokens)")
        print(line)
    if not rep["ok"]:
        for f in rep["findings"]:
            print(f"REFUSED: {f}", file=sys.stderr)
        return 2
    print("probe pins OK: every probe source is held out, none is trained "
          "on, fractions inside the window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
