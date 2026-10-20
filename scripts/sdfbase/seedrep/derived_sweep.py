#!/usr/bin/env python3
"""derived_sweep — the PERSISTED lr-multiplier ladder of one residual
distillation tree (``sweep.json``) and its DURABLE pre-distillation
assessment (``assess.json``); stdlib only, so the batch shell runs it under
the node's ``python3`` before the venv is touched.

Why this exists: ``sdfbase_derived_build.sbatch`` used to build the
``finish`` stage's parts and ``--lr-mult`` from a fixed in-script
``SWEEP=(0.3 1 3)``, so an extension rung calibrated under
``calib_parts/lrmult0.1/`` was never merged, and ``merge-calibration``
(which writes ``selected_lr_mult``) ran back to back with ``distill`` in ONE
stage — the selection was decided and consumed in the same job, with nothing
durable in between. Now:

* ``prep`` DECLARES the ladder: ``sweep.json = {"lr_mults": [0.3, 1, 3],
  "declared_by": "prep", "extensions": []}`` (``declare``; identical
  re-declaration is a no-op, a different one refuses).
* ``calibrate <m>`` with an explicit multiplier performs an idempotent
  "ensure declared" (``ensure-extension``): ``m`` already in the ladder →
  no-op, so an interrupted extension job's ``escalate()`` self-resubmit
  passes the declaration step and resumes its part; a DIFFERENT extension
  already declared → REFUSE (one extension rung only); absent →
  append by atomic rewrite.
* ``assess`` (a NEW stage) merges EXACTLY ``lr_mults ∪ extensions`` (the
  sbatch calls ``merge-calibration`` with the ``ladder`` this tool prints,
  which refuses any extra or missing part) and then writes ``assess.json``
  — the selected multiplier, the ladder, the edge decision and the
  extension recommendation (``0.1`` below, ``10`` above, ``null`` when
  interior or when the one extension has already been run), plus the sha256
  of every part and of the merged file — WITHOUT distilling.
* ``finish`` runs ``check-finish`` first: ``assess.json`` must exist, every
  part and the merged file must hash to what ``assess.json`` recorded (an
  exact "nothing moved since the assessment", never an mtime comparison —
  MooseFS mtimes are not a clock to build on), and the merged selection
  must equal the assessed one; then it distills, and ``check-distilled``
  confirms the published ``derived_meta.json`` trainer multiplier IS the
  assessed one.

The multiplier spelling is ``psr_residual_distill.format_mult``'s (``f"{m:g}"``:
``1.0 -> "1"``, ``0.3 -> "0.3"``), re-implemented here because this file
must import nothing beyond the stdlib; ``tests/test_sdfbase_derived_full.py``
pins the two spellings equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SWEEP_NAME = "sweep.json"
ASSESS_NAME = "assess.json"
ASSESS_HISTORY_NAME = "assess_history.jsonl"
MERGED_NAME = "calibration_controls.json"
PARTS_DIRNAME = "calib_parts"
DEFAULT_LR_MULTS = (0.3, 1.0, 3.0)
#: The ONE extension rung per edge: extend one rung on an edge selection —
#: the Llama precedent's 0.1 below the ladder, 10 above it — never a second
#: one; a persisting edge is published with the alarm stated.
EXTENSION_BELOW = 0.1
EXTENSION_ABOVE = 10.0

__all__ = ["SweepError", "format_mult", "parse_mults", "ladder_of",
           "load_sweep", "declare", "ensure_extension", "assess",
           "check_finish", "check_distilled", "part_path", "main"]


class SweepError(RuntimeError):
    """A refusal — the reason is the message; the CLI exits 3 on it."""


def format_mult(m) -> str:
    """The ONE spelling of an lr multiplier in file keys (psr_residual_distill
    .format_mult: ``f"{float(m):g}"``)."""
    return f"{float(m):g}"


def parse_mults(spec) -> list[float]:
    """``0.3,1,3`` → ``[0.3, 1.0, 3.0]``; positive, de-duplicated, order kept."""
    if spec is None:
        return list(DEFAULT_LR_MULTS)
    if isinstance(spec, (list, tuple)):
        raw = [float(x) for x in spec]
    else:
        raw = [float(x) for x in str(spec).replace(";", ",").split(",")
               if x.strip()]
    out: list[float] = []
    for v in raw:
        if v <= 0:
            raise SweepError(f"lr multiplier must be positive, got {v}")
        if v not in out:
            out.append(v)
    if not out:
        raise SweepError(f"no lr multipliers in {spec!r}")
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    """tempfile in the target dir + os.replace: a reader never sees a torn
    file, and two writers of IDENTICAL content cannot corrupt each other."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path, what: str) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SweepError(f"{what} missing at {path}")
    except (ValueError, OSError) as e:
        raise SweepError(f"{what} at {path} unreadable: {e}")
    if not isinstance(doc, dict):
        raise SweepError(f"{what} at {path} is a JSON {type(doc).__name__}, "
                         "not an object")
    return doc


# --------------------------------------------------------------------------- #
# sweep.json                                                                   #
# --------------------------------------------------------------------------- #

def sweep_path(resid) -> Path:
    return Path(resid) / SWEEP_NAME


def load_sweep(resid) -> dict | None:
    """The declaration, validated; None when the tree carries none (a tree
    prepped before this tool existed)."""
    p = sweep_path(resid)
    if not p.exists():
        return None
    doc = _read_json(p, "sweep.json")
    try:
        mults = parse_mults(doc["lr_mults"])
        exts = parse_mults(doc["extensions"]) if doc.get("extensions") else []
    except (KeyError, TypeError, ValueError) as e:
        raise SweepError(f"{p} malformed ({type(e).__name__}: {e}); expected "
                         "{lr_mults: [...], extensions: [...], declared_by: ...}")
    if set(mults) & set(exts):
        raise SweepError(f"{p}: extensions {exts} overlap the base ladder "
                         f"{mults}")
    return doc


def ladder_of(sweep: dict) -> list[float]:
    """``lr_mults ∪ extensions``, sorted ascending — the EXACT set the merge
    must cover."""
    return sorted(set(parse_mults(sweep["lr_mults"]))
                  | set(parse_mults(sweep["extensions"])
                        if sweep.get("extensions") else []))


def ladder_spec(ladder: list[float]) -> str:
    return ",".join(format_mult(m) for m in ladder)


def part_path(resid, m) -> Path:
    return (Path(resid) / PARTS_DIRNAME / f"lrmult{format_mult(m)}"
            / MERGED_NAME)


def declare(resid, lr_mults=None, *, declared_by: str = "prep") -> dict:
    """Write ``sweep.json`` (extensions empty). An existing declaration with
    the SAME base ladder is a no-op (its extensions are kept — a re-run prep
    after an extension must not forget it); a different ladder refuses."""
    mults = parse_mults(lr_mults)
    existing = load_sweep(resid)
    if existing is not None:
        if parse_mults(existing["lr_mults"]) != mults:
            raise SweepError(
                f"{sweep_path(resid)} already declares lr_mults "
                f"{existing['lr_mults']}, not {mults} — a ladder is declared "
                "once; refusing to rewrite it")
        print(f"[derived-sweep] declaration unchanged at {sweep_path(resid)}: "
              f"lr_mults {ladder_spec(mults)}, extensions "
              f"{ladder_spec(parse_mults(existing['extensions'])) if existing.get('extensions') else '-'}")
        return existing
    doc = {"lr_mults": mults, "extensions": [], "declared_by": declared_by,
           "declared_at": _now(), "extension_events": []}
    _atomic_write(sweep_path(resid), _dump(doc))
    print(f"[derived-sweep] declared lr_mults {ladder_spec(mults)} -> "
          f"{sweep_path(resid)}")
    return doc


def ensure_extension(resid, m, *, default_ladder=None,
                     declared_by: str = "calibrate") -> str:
    """The idempotent "ensure declared" a ``calibrate <m>`` job runs before
    its part. Returns one of ``"base"`` (m is a base rung), ``"declared"``
    (already the declared extension), ``"appended"`` (declared now), or
    ``"undeclared-base"`` (no sweep.json — a pre-declaration tree — and m is
    a default-ladder rung, so today's behaviour continues). Refuses an
    extension on a tree with no declaration, and a SECOND extension."""
    m = float(m)
    sweep = load_sweep(resid)
    if sweep is None:
        base = parse_mults(default_ladder)
        if m in base:
            print(f"[derived-sweep] no {SWEEP_NAME} under {resid}: lr_mult "
                  f"{format_mult(m)} is a base rung of {ladder_spec(base)} — "
                  "proceeding without a declaration (tree prepped before the "
                  "declaration existed)")
            return "undeclared-base"
        raise SweepError(
            f"no {SWEEP_NAME} under {resid}: an extension rung "
            f"{format_mult(m)} has nowhere to be declared — run prep under "
            "this sbatch first (it writes the declaration)")
    base = parse_mults(sweep["lr_mults"])
    exts = parse_mults(sweep["extensions"]) if sweep.get("extensions") else []
    if m in base:
        print(f"[derived-sweep] lr_mult {format_mult(m)} is a base rung")
        return "base"
    if m in exts:
        print(f"[derived-sweep] extension {format_mult(m)} already declared "
              "(no-op)")
        return "declared"
    if exts:
        raise SweepError(
            f"{sweep_path(resid)} already declares the extension "
            f"{ladder_spec(exts)}; a second extension ({format_mult(m)}) is "
            "not allowed (decisions D7: one rung, then publish with the "
            "alarm stated)")
    doc = dict(sweep)
    doc["extensions"] = [m]
    doc.setdefault("extension_events", []).append(
        {"lr_mult": m, "declared_by": declared_by, "declared_at": _now()})
    _atomic_write(sweep_path(resid), _dump(doc))
    print(f"[derived-sweep] declared extension {format_mult(m)} -> "
          f"{sweep_path(resid)} (ladder now {ladder_spec(ladder_of(doc))})")
    return "appended"


def _require_sweep(resid) -> dict:
    sweep = load_sweep(resid)
    if sweep is None:
        raise SweepError(
            f"no {SWEEP_NAME} under {resid} — the ladder was never declared; "
            "run prep under this sbatch (it declares the ladder) before "
            "assess/finish")
    return sweep


# --------------------------------------------------------------------------- #
# parts                                                                        #
# --------------------------------------------------------------------------- #

def part_status(path: Path, m: float) -> tuple[bool, str]:
    """(complete?, why) for one ``calib_parts/lrmult<m>/calibration_controls
    .json`` — the ``sweep.complete`` completeness the merge requires, plus
    the payload for THIS multiplier present."""
    if not path.is_file():
        return False, f"missing ({path})"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return False, f"unreadable ({e})"
    sweep = doc.get("sweep") if isinstance(doc, dict) else None
    if not isinstance(sweep, dict):
        return False, "not a sweep file (no `sweep`)"
    if not sweep.get("complete", False):
        return False, (f"INCOMPLETE sweep (finished "
                       f"{list(sweep.get('per_mult') or {})} of "
                       f"{sweep.get('lr_mults')})")
    per = sweep.get("per_mult") or {}
    if format_mult(m) not in per:
        return False, (f"carries no payload for lr_mult {format_mult(m)} "
                       f"(has {list(per)})")
    return True, "complete"


def parts_report(resid, ladder: list[float]) -> dict:
    """Per multiplier: path, complete?, why, sha256 (None when incomplete)."""
    rep = {}
    for m in ladder:
        p = part_path(resid, m)
        ok, why = part_status(p, m)
        rep[format_mult(m)] = {"path": str(p), "complete": ok, "why": why,
                               "sha256": _sha256(p) if ok else None}
    return rep


def _require_parts(resid, ladder: list[float]) -> dict:
    rep = parts_report(resid, ladder)
    bad = {k: v["why"] for k, v in rep.items() if not v["complete"]}
    if bad:
        raise SweepError(f"calibration parts not complete for lr_mult "
                         f"{sorted(bad)}: {bad}")
    stray = sorted(p.name for p in (Path(resid) / PARTS_DIRNAME).glob("lrmult*")
                   if p.is_dir()
                   and p.name[len("lrmult"):] not in rep)
    if stray:
        raise SweepError(
            f"stray calibration part dir(s) {stray} under "
            f"{Path(resid) / PARTS_DIRNAME} are not in the declared ladder "
            f"{ladder_spec(ladder)} — declare them (calibrate <m>) or remove "
            "them; the merge covers EXACTLY the declaration")
    return rep


# --------------------------------------------------------------------------- #
# assess                                                                       #
# --------------------------------------------------------------------------- #

def _merged_selection(resid, ladder: list[float]) -> tuple[dict, str]:
    merged_path = Path(resid) / MERGED_NAME
    merged = _read_json(merged_path, "merged calibration_controls.json")
    sweep = merged.get("sweep")
    if not isinstance(sweep, dict):
        raise SweepError(f"{merged_path} carries no `sweep` block — not a "
                         "merged sweep file")
    if not sweep.get("complete", False):
        raise SweepError(f"{merged_path} holds an INCOMPLETE sweep")
    got = sorted(parse_mults(sweep.get("lr_mults") or []))
    if got != ladder:
        raise SweepError(
            f"{merged_path} covers lr_mults {ladder_spec(got)}, the "
            f"declaration is {ladder_spec(ladder)} — the merged file is "
            "stale; re-run assess (which re-merges) before reading it")
    return merged, _sha256(merged_path)


def assess(resid, *, out=None) -> dict:
    """Read the declaration, the complete parts and the MERGED file the
    sbatch just wrote, and record the selection + edge decision durably in
    ``assess.json`` (also appended to ``assess_history.jsonl``). Never
    distills; never writes the merged file."""
    resid = Path(resid)
    sweep = _require_sweep(resid)
    ladder = ladder_of(sweep)
    parts = _require_parts(resid, ladder)
    merged, merged_sha = _merged_selection(resid, ladder)
    msweep = merged["sweep"]
    sel = msweep.get("selected_lr_mult")
    selection = dict(msweep.get("selection") or {})
    exts = parse_mults(sweep["extensions"]) if sweep.get("extensions") else []
    doc = {
        "kind": "psr_residual_assess",
        "ladder": ladder,
        "base_lr_mults": parse_mults(sweep["lr_mults"]),
        "extensions": exts,
        "selected_lr_mult": (float(sel) if sel is not None else None),
        "selection_refused": sel is None,
        "selection_verdict": selection.get("verdict"),
        "selection_reason": selection.get("reason"),
        "on_edge": None, "edge_side": None,
        "recommended_extension": None,
        "alarm": None,
        "parts_sha256": {k: v["sha256"] for k, v in parts.items()},
        "calibration_sha256": merged_sha,
        "assessed_at": _now(),
        "assessor": "scripts/sdfbase/seedrep/derived_sweep.py assess",
    }
    if sel is None:
        doc["alarm"] = ("the sweep REFUSED to select a multiplier "
                        f"({selection.get('verdict')}: {selection.get('reason')}); "
                        "finish will refuse — a decision for the user")
    else:
        s = float(sel)
        if s not in ladder:
            raise SweepError(f"merged selection {format_mult(s)} is not in the "
                             f"ladder {ladder_spec(ladder)}")
        lo, hi = ladder[0], ladder[-1]
        if s == lo:
            doc.update(on_edge=True, edge_side="below")
        elif s == hi:
            doc.update(on_edge=True, edge_side="above")
        else:
            doc.update(on_edge=False, edge_side=None)
        if doc["on_edge"]:
            if exts:
                doc["alarm"] = (
                    f"edge selection persists after the one D7 extension "
                    f"({ladder_spec(exts)}): argmin {format_mult(s)} is the "
                    f"{doc['edge_side']} end of {ladder_spec(ladder)} — publish "
                    "with this alarm stated; no second extension")
            else:
                doc["recommended_extension"] = (
                    EXTENSION_BELOW if doc["edge_side"] == "below"
                    else EXTENSION_ABOVE)
    out_path = Path(out) if out else resid / ASSESS_NAME
    _atomic_write(out_path, _dump(doc))
    with open(resid / ASSESS_HISTORY_NAME, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, sort_keys=True) + "\n")
    print(f"[derived-sweep] assess: ladder {ladder_spec(ladder)} selected "
          f"{format_mult(doc['selected_lr_mult']) if doc['selected_lr_mult'] is not None else 'NOTHING'} "
          f"on_edge={doc['on_edge']} edge_side={doc['edge_side']} "
          f"recommended_extension={doc['recommended_extension']} -> {out_path}"
          + (f"\n[derived-sweep] ALARM: {doc['alarm']}" if doc["alarm"] else ""))
    return doc


def check_finish(resid) -> float:
    """Everything ``finish`` needs to hold BEFORE it distills; returns the
    assessed multiplier (printed alone on stdout by the CLI)."""
    resid = Path(resid)
    sweep = _require_sweep(resid)
    ladder = ladder_of(sweep)
    a_path = resid / ASSESS_NAME
    a = _read_json(a_path, "assess.json")
    problems: list[str] = []
    if sorted(parse_mults(a.get("ladder") or [])) != ladder:
        problems.append(f"assess.json ladder {a.get('ladder')} != declared "
                        f"{ladder_spec(ladder)} (the declaration changed after "
                        "the assessment)")
    parts = parts_report(resid, ladder)
    for k, v in parts.items():
        if not v["complete"]:
            problems.append(f"part lr_mult {k}: {v['why']}")
        elif (a.get("parts_sha256") or {}).get(k) != v["sha256"]:
            problems.append(f"part lr_mult {k} hashes to "
                            f"{str(v['sha256'])[:12]}…, assess.json recorded "
                            f"{str((a.get('parts_sha256') or {}).get(k))[:12]}… — "
                            "a part changed after the assessment")
    merged_path = resid / MERGED_NAME
    if not merged_path.is_file():
        problems.append(f"{merged_path} missing")
    else:
        msha = _sha256(merged_path)
        if a.get("calibration_sha256") != msha:
            problems.append(f"{MERGED_NAME} hashes to {msha[:12]}…, assess.json "
                            f"recorded {str(a.get('calibration_sha256'))[:12]}… "
                            "— the merged file changed after the assessment")
    sel = a.get("selected_lr_mult")
    if sel is None:
        problems.append("assess.json records NO selection (the sweep refused); "
                        "finish does not build on a refusal — a user decision")
    elif merged_path.is_file():
        try:
            merged = _read_json(merged_path, MERGED_NAME)
            msel = (merged.get("sweep") or {}).get("selected_lr_mult")
            if msel is None or format_mult(msel) != format_mult(sel):
                problems.append(f"merged selection {msel} != assessed "
                                f"{format_mult(sel)}")
        except SweepError as e:
            problems.append(str(e))
    if problems:
        raise SweepError("finish refused — re-run assess if the parts or the "
                         "declaration legitimately changed:\n  - "
                         + "\n  - ".join(problems))
    return float(sel)


def check_distilled(resid) -> float:
    """After ``distill``: the published ``derived_meta.json`` trainer
    multiplier must be the assessed one."""
    resid = Path(resid)
    a = _read_json(resid / ASSESS_NAME, "assess.json")
    meta = _read_json(resid / "derived_meta.json", "derived_meta.json")
    got = (meta.get("trainer") or {}).get("lr_mult")
    sel = a.get("selected_lr_mult")
    if got is None or sel is None or format_mult(got) != format_mult(sel):
        raise SweepError(f"distilled object records trainer.lr_mult {got}, "
                         f"assess.json selected {sel} — the publication is "
                         "not at the assessed multiplier")
    return float(got)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _resid(p):
        p.add_argument("--resid", required=True,
                       help="the distillation tree (…/psr_residual/<key>/"
                            "<mix>__psr_ce_all_minus_ctl_distilled)")
    p = sub.add_parser("declare", help="prep: write sweep.json (idempotent)")
    _resid(p)
    p.add_argument("--lr-mults", default=None,
                   help=f"comma list; default {ladder_spec(DEFAULT_LR_MULTS)}")
    p.add_argument("--declared-by", default="prep")
    p = sub.add_parser("ensure-extension",
                       help="calibrate <m>: idempotent ensure-declared")
    _resid(p)
    p.add_argument("--lr-mult", required=True)
    p.add_argument("--default-ladder", default=None,
                   help="the base ladder a pre-declaration tree is assumed "
                        "to run (the sbatch's SWEEP); default "
                        f"{ladder_spec(DEFAULT_LR_MULTS)}")
    p = sub.add_parser("ladder", help="print lr_mults ∪ extensions, sorted, "
                                      "comma-joined (the merge's --lr-mult)")
    _resid(p)
    p = sub.add_parser("parts", help="print the part files of the ladder, "
                                     "comma-joined (the merge's --parts)")
    _resid(p)
    p = sub.add_parser("assess", help="record the selection + edge decision "
                                      "in assess.json (never distills)")
    _resid(p)
    p.add_argument("--out", default=None)
    p = sub.add_parser("check-finish", help="finish's preconditions; prints "
                                            "the assessed multiplier")
    _resid(p)
    p = sub.add_parser("check-distilled", help="derived_meta.json is at the "
                                               "assessed multiplier")
    _resid(p)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "declare":
            declare(a.resid, a.lr_mults, declared_by=a.declared_by)
        elif a.cmd == "ensure-extension":
            ensure_extension(a.resid, a.lr_mult, default_ladder=a.default_ladder)
        elif a.cmd == "ladder":
            print(ladder_spec(ladder_of(_require_sweep(a.resid))))
        elif a.cmd == "parts":
            ladder = ladder_of(_require_sweep(a.resid))
            print(",".join(str(part_path(a.resid, m)) for m in ladder))
        elif a.cmd == "assess":
            assess(a.resid, out=a.out)
        elif a.cmd == "check-finish":
            print(format_mult(check_finish(a.resid)))
        elif a.cmd == "check-distilled":
            print(format_mult(check_distilled(a.resid)))
    except SweepError as e:
        print(f"[derived-sweep] REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
