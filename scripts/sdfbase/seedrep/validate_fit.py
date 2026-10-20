#!/usr/bin/env python3
"""validate_fit — the post-fit check of ONE seed-replicate cell,
run on every DONE wake before the cell is recorded in the fits ledger.

    validate_fit.py <cell_dir> --seed <s> --expect-key <key> \\
        [--config configs/sdfbase_<key>_fits.yaml --pairs <pool>/pairs_<si>.jsonl]
        [--expected-fingerprint <64hex>] [--n-docs N] [--effective-batch 64]
        [--ce-ratio-max 1.1] [--cluster cluster_a] [--bank-dir <remote dir>]
        [--out <verdict.json>] [--ledger RS/reseed/fits_ledger.jsonl | --no-ledger]

Checks, each filed by name in the verdict (``checks.<name>.ok``), never
short-circuited so one run prints every problem:

* ``sidecars``: ``train_budget_meta.json`` and ``COMPLETE`` parse; the
  marker's ``object_sha256`` equals the sidecar's, and ``object.pt`` (when
  present) hashes to it.
* ``key``: the cell dir sits under ``<bank>/<key>/cells_all/<cell>`` for the
  expected key (and ``--config`` is that key's ``sdfbase_<key>_fits.yaml``).
* ``fingerprint``: ``COMPLETE.fit_fingerprint`` equals the EXPECTED fit
  fingerprint — derived through the trainer's own functions
  (``fit_cell_identity.expected_fit_fingerprint(config, pairs, si, arm)``,
  which needs the config and the pool bytes) or supplied as
  ``--expected-fingerprint`` by a caller that already derived it.
* ``effective_batch``: the sidecar's ``effective_batch`` is 64 and
  ``resolved_batch_size × accum_steps`` equals it (bs 1 / accum 64 is
  admitted — the memprobe fallback).
* ``seed``: the sidecar records ``seed == s``, written top-level.
* ``scratch``: no ``continued_from`` block anywhere in the sidecar or the
  marker (a scratch fit, never a continuation lineage).
* ``realized_batch``: ``sisteer.extraction.realized_batch.validate_fit_record``
  on the sidecar's ``fit_record`` with the ACTUAL ``n_docs`` (the pool
  manifest's ``n_docs`` beside ``--pairs``, or ``--n-docs``), ``bs``,
  ``accum`` and ``seed`` — exact first-epoch reproduction for an unresumed
  run, the semantic floor for a resumed one, REFUSAL when the record carries
  no ``resumed_from_step`` marker (a batch-64 vector reads ``{50, 63, 64}``
  on the control corpus and ``{8, 64}`` on the constitution corpus; "all 64"
  was never possible and this predicate is the criterion).
* ``ce_ratio``: ``fit_record.ce_ratio <= 1.1``.
* ``stopped_early`` is INFORMATION, not a check: ``true`` → fine; ``false``
  → ``cap_hit: true`` with ``heldout_delta_last4 = curve[-1] - curve[-5]``
  printed (validate_fit flags a cap hit, never silently publishes it and
  never re-fits — expected on the constitution cells).

Exit 0 when every check passes, 4 when one fails (the verdict is still
written and, unless ``--no-ledger``, appended to the ledger with
``ok: false`` so a failed cell is on record too), 3 when the inputs cannot
be read at all. Needs the repo venv (``realized_batch`` imports the trainer).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO / "src", REPO / "scripts" / "sdfbase"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RS_DEFAULT = "$STATE/romp/candidate-si-refit/run-state/full22"   # under the site's state root ($STATE); expanded through sisteer.site where the ledger is written
DEFAULT_EFFECTIVE_BATCH = 64
DEFAULT_CE_RATIO_MAX = 1.1

__all__ = ["validate_cell", "find_continued_from", "manifest_n_docs", "main"]


class InputError(RuntimeError):
    """The cell cannot be read at all (exit 3)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path, what: str) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise InputError(f"{what} missing at {path}")
    except (ValueError, OSError) as e:
        raise InputError(f"{what} at {path} unreadable: {e}")
    if not isinstance(doc, dict):
        raise InputError(f"{what} at {path} is not a JSON object")
    return doc


def find_continued_from(doc, path: str = "") -> list[str]:
    """Every JSON path in ``doc`` whose key is ``continued_from`` with a
    truthy value (the block sits under the optimizer record; the
    marker may carry its own copy — search everywhere)."""
    hits: list[str] = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            p = f"{path}.{k}" if path else str(k)
            if k == "continued_from" and v:
                hits.append(p)
            hits.extend(find_continued_from(v, p))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            hits.extend(find_continued_from(v, f"{path}[{i}]"))
    return hits


def manifest_n_docs(pairs: Path) -> tuple[int | None, str]:
    """``n_docs`` from the pool manifest beside ``pairs`` (either spelling:
    ``<pool>.jsonl.manifest.json`` or ``pairs_<cell>.manifest.json``)."""
    for cand in (Path(str(pairs) + ".manifest.json"),
                 pairs.with_suffix(".manifest.json")):
        if cand.is_file():
            try:
                m = json.loads(cand.read_text(encoding="utf-8"))
            except (ValueError, OSError) as e:
                return None, f"{cand} unreadable: {e}"
            n = m.get("n_docs") if isinstance(m, dict) else None
            if isinstance(n, int) and n > 0:
                return n, str(cand)
            return None, f"{cand} records no n_docs"
    return None, f"no manifest beside {pairs} (either spelling)"


def validate_cell(cell_dir, *, seed: int, expect_key: str, config=None,
                  pairs=None, expected_fingerprint: str | None = None,
                  n_docs: int | None = None,
                  effective_batch: int = DEFAULT_EFFECTIVE_BATCH,
                  ce_ratio_max: float = DEFAULT_CE_RATIO_MAX,
                  cluster: str | None = None, bank_dir: str | None = None,
                  hash_object: bool = True) -> dict:
    cell_dir = Path(cell_dir)
    meta = _read_json(cell_dir / "train_budget_meta.json", "train_budget_meta.json")
    marker = _read_json(cell_dir / "COMPLETE", "COMPLETE")
    checks: dict[str, dict] = {}

    def file(name: str, ok: bool, **detail):
        checks[name] = {"ok": bool(ok), **detail}

    cell_id = meta.get("cell_id") or cell_dir.name
    si, arm = meta.get("si_id"), meta.get("arm")
    fr = meta.get("fit_record") if isinstance(meta.get("fit_record"), dict) else {}

    # --- sidecars: the marker and the sidecar name ONE object ------------------
    problems = []
    if marker.get("cell_id") not in (None, cell_id):
        problems.append(f"COMPLETE names cell {marker.get('cell_id')!r}, the "
                        f"sidecar {cell_id!r}")
    m_sha, s_sha = marker.get("object_sha256"), meta.get("object_sha256")
    if not m_sha or not s_sha:
        problems.append("object_sha256 absent from the marker or the sidecar")
    elif m_sha != s_sha:
        problems.append(f"COMPLETE object_sha256 {m_sha[:12]}… != sidecar "
                        f"{s_sha[:12]}…")
    obj = cell_dir / "object.pt"
    obj_sha = None
    if obj.is_file() and hash_object:
        obj_sha = _sha256(obj)
        if m_sha and obj_sha != m_sha:
            problems.append(f"object.pt hashes to {obj_sha[:12]}…, the marker "
                            f"records {m_sha[:12]}…")
    elif not obj.is_file():
        problems.append("object.pt absent (hash not verified)")
    if not marker.get("fit_fingerprint"):
        problems.append("COMPLETE carries no fit_fingerprint")
    file("sidecars", not problems, problems=problems, object_sha256=obj_sha or m_sha)

    # --- key: the dir sits under <bank>/<key>/cells_all/<cell> -----------------
    problems = []
    try:
        key_dir = cell_dir.resolve().parents[1]
        parent_name = cell_dir.resolve().parent.name
    except IndexError:
        key_dir, parent_name = None, None
    if parent_name != "cells_all" or key_dir is None or key_dir.name != expect_key:
        problems.append(f"cell dir is {cell_dir}, expected …/{expect_key}/"
                        f"cells_all/{cell_id}")
    if config is not None:
        cfg_name = Path(config).name
        if cfg_name != f"sdfbase_{expect_key}_fits.yaml":
            problems.append(f"--config {cfg_name} is not sdfbase_{expect_key}_fits.yaml")
    file("key", not problems, problems=problems, expect_key=expect_key)

    # --- fingerprint -------------------------------------------------------------
    exp_fp, fp_source, problems = expected_fingerprint, "argument", []
    if exp_fp is None:
        if config is None or pairs is None:
            problems.append("no --expected-fingerprint and no --config/--pairs "
                            "to derive it from")
        elif not si or not arm:
            problems.append("sidecar records no si_id/arm to derive the "
                            "fingerprint for")
        else:
            try:
                from fit_cell_identity import expected_fit_fingerprint
                exp_fp = expected_fit_fingerprint(config, pairs, si, arm)
                fp_source = "fit_cell_identity.expected_fit_fingerprint"
            except Exception as e:  # noqa: BLE001 — filed, never hidden
                problems.append(f"fingerprint derivation failed: "
                                f"{type(e).__name__}: {e}")
    got_fp = marker.get("fit_fingerprint")
    if exp_fp is not None and got_fp != exp_fp:
        problems.append(f"COMPLETE.fit_fingerprint {str(got_fp)[:12]}… != expected "
                        f"{exp_fp[:12]}… ({fp_source})")
    file("fingerprint", not problems and exp_fp is not None, problems=problems,
         expected=exp_fp, got=got_fp, source=fp_source)

    # --- effective batch ----------------------------------------------------------
    problems = []
    eb, bs, accum = meta.get("effective_batch"), meta.get("resolved_batch_size"), \
        meta.get("accum_steps")
    if eb != effective_batch:
        problems.append(f"effective_batch {eb} != {effective_batch}")
    if not isinstance(bs, int) or not isinstance(accum, int):
        problems.append(f"resolved_batch_size {bs!r} / accum_steps {accum!r} "
                        "not both recorded")
    elif bs * accum != effective_batch:
        problems.append(f"resolved_batch_size {bs} × accum_steps {accum} = "
                        f"{bs * accum} != {effective_batch}")
    file("effective_batch", not problems, problems=problems, effective_batch=eb,
         resolved_batch_size=bs, accum_steps=accum)

    # --- seed ------------------------------------------------------------------------
    got_seed = meta.get("seed", fr.get("seed"))
    file("seed", got_seed is not None and int(got_seed) == int(seed),
         expected=int(seed), got=got_seed,
         problems=([] if got_seed is not None and int(got_seed) == int(seed) else
                   [f"sidecar seed {got_seed!r} != expected {seed}"
                    if got_seed is not None else
                    "sidecar records no seed (a pre-W1 sidecar, or the "
                    "seed never reached the trainer)"]))

    # --- scratch (no continuation lineage) -----------------------------------------
    hits = find_continued_from(meta) + [f"COMPLETE.{h}" for h in
                                        find_continued_from(marker)]
    file("scratch", not hits, problems=([f"continued_from at {h}" for h in hits]))

    # --- realized batch -------------------------------------------------------------
    problems, rb_verdict, n_docs_source = [], None, "argument"
    if n_docs is None and pairs is not None:
        n_docs, n_docs_source = manifest_n_docs(Path(pairs))
    if n_docs is None:
        problems.append(f"n_docs unknown ({n_docs_source}); pass --n-docs or "
                        "--pairs with its manifest")
    elif not isinstance(bs, int) or not isinstance(accum, int):
        problems.append("bs/accum not recorded — realized-batch predicate not applied")
    else:
        try:
            from sisteer.extraction.realized_batch import validate_fit_record
            v = validate_fit_record(fr, n_docs=int(n_docs), bs=bs, accum=accum,
                                    seed=int(seed))
            rb_verdict = v.as_dict()
            if not v.ok:
                problems.extend(v.problems)
        except Exception as e:  # noqa: BLE001
            problems.append(f"realized_batch.validate_fit_record failed: "
                            f"{type(e).__name__}: {e}")
    vec = fr.get("realized_eff_batch") if isinstance(fr.get("realized_eff_batch"), list) else []
    file("realized_batch", not problems, problems=problems, n_docs=n_docs,
         n_docs_source=n_docs_source,
         distinct_sizes=sorted({int(x) for x in vec}) if vec else None,
         resume_provenance=(rb_verdict or {}).get("resume_provenance"),
         exact_ok=(rb_verdict or {}).get("exact_ok"),
         G=(rb_verdict or {}).get("G"))

    # --- ce ratio -------------------------------------------------------------------
    ce = fr.get("ce_ratio")
    ok = isinstance(ce, (int, float)) and float(ce) <= ce_ratio_max
    file("ce_ratio", ok, ce_ratio=ce, max=ce_ratio_max,
         problems=([] if ok else [f"ce_ratio {ce!r} > {ce_ratio_max}"
                                  if isinstance(ce, (int, float)) else
                                  "fit_record.ce_ratio absent"]))

    # --- cap hit: information, flagged, never a failure ----------------------------
    stopped_early = fr.get("stopped_early")
    curve = fr.get("heldout_curve") if isinstance(fr.get("heldout_curve"), list) else []
    cap_hit = stopped_early is False
    delta = None
    if cap_hit and len(curve) >= 5:
        delta = float(curve[-1]) - float(curve[-5])
    cap = {"cap_hit": cap_hit, "stopped_early": stopped_early,
           "epochs_run": fr.get("epochs_run"), "best_epoch": fr.get("best_epoch"),
           "heldout_delta_last4": delta,
           "note": (None if not cap_hit else
                    ("CAP HIT — flagged, never re-fit (criterion 1); "
                     f"held-out CE moved {delta:+.6f} over the last four epochs"
                     if delta is not None else
                     "CAP HIT — flagged; the curve has fewer than five points, "
                     "no last-four delta"))}
    if stopped_early is None:
        cap["note"] = "fit_record.stopped_early absent"

    ok_all = all(c["ok"] for c in checks.values())
    return {
        "kind": "seedrep_fit_verdict",
        "cell": cell_id, "si": si, "arm": arm, "key": expect_key,
        "seed": int(seed), "cluster": cluster,
        "bank_path": bank_dir or str(cell_dir),
        "object_sha256": obj_sha or m_sha,
        "fit_fingerprint": got_fp, "expected_fit_fingerprint": exp_fp,
        "effective_batch": eb, "resolved_batch_size": bs, "accum_steps": accum,
        "ok": ok_all,
        "failed_checks": sorted(k for k, c in checks.items() if not c["ok"]),
        "checks": checks,
        **cap,
        "realized_batch": rb_verdict,
        "checked_at": _now(),
        "checker": "scripts/sdfbase/seedrep/validate_fit.py",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cell_dir")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--expect-key", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pairs", default=None, help="the training pool the cell "
                    "was fit on (its manifest supplies n_docs; its bytes the "
                    "fingerprint)")
    ap.add_argument("--expected-fingerprint", default=None)
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--effective-batch", type=int, default=DEFAULT_EFFECTIVE_BATCH)
    ap.add_argument("--ce-ratio-max", type=float, default=DEFAULT_CE_RATIO_MAX)
    ap.add_argument("--cluster", default=None, choices=("cluster_a", "cluster_b"))
    ap.add_argument("--bank-dir", default=None, help="the cell dir ON THE "
                    "CLUSTER when validating a synced copy")
    ap.add_argument("--no-hash-object", action="store_true")
    ap.add_argument("--out", default=None, help="verdict JSON (default "
                    "<cell_dir>/VALIDATE.json)")
    ap.add_argument("--ledger", default=None,
                    help="fits ledger JSONL (default $RS/reseed/fits_ledger.jsonl)")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args(argv)
    try:
        v = validate_cell(a.cell_dir, seed=a.seed, expect_key=a.expect_key,
                          config=a.config, pairs=a.pairs,
                          expected_fingerprint=a.expected_fingerprint,
                          n_docs=a.n_docs, effective_batch=a.effective_batch,
                          ce_ratio_max=a.ce_ratio_max, cluster=a.cluster,
                          bank_dir=a.bank_dir, hash_object=not a.no_hash_object)
    except InputError as e:
        print(f"[validate-fit] REFUSED: {e}", file=sys.stderr)
        return 3
    out = Path(a.out) if a.out else Path(a.cell_dir) / "VALIDATE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=1, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    if not a.no_ledger:
        from sisteer import site as _site
        rs = _site.expand_path(os.environ.get("RS", RS_DEFAULT), key="RS")
        ledger = Path(a.ledger) if a.ledger else rs / "reseed" / "fits_ledger.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(v, sort_keys=True, default=str) + "\n")
    state = "OK" if v["ok"] else f"FAILED {v['failed_checks']}"
    print(f"[validate-fit] {v['key']}/{v['cell']} seed={v['seed']}: {state}"
          + (f"; {v['note']}" if v.get("note") else "") + f" -> {out}")
    for name, c in v["checks"].items():
        if not c["ok"]:
            for pr in c.get("problems") or ["(no detail)"]:
                print(f"  - {name}: {pr}")
    return 0 if v["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
