#!/usr/bin/env python
"""STRENGTH FRONTIER: build the dose-scaled copies of the two LoRA objects at every rung of the
dose ladder, in every paper lane where the bank holds a parent, from the grading host's bank — then
stage_strength.py pushes them.

Objects (config cell id -> producer):
  constv2-mix-full__lora_r4_ce             constitution LoRA r4       scale_lora_dose.py (B x scale)
  constv2-mix-full__lora_r4_ce_minus_ctl   DERIVED LoRA subtraction   scale_lora_dose.py (B_pos, B_neg x scale)

Ladder: -1, 0.1, 0.25, 0.5, 1, 1.5. On the four 2026-09-10 targets 1.0 IS the parent (served by the
main / it view) and 0.5 already exists as the halfdose__ / halfdoseit__ copies, so this driver built the
remaining four rungs there. On a target whose parents are NOT staged under the campaign's own ids in this bank
(llama8b) every rung is built here: 0.5 as an ordinary scaled copy and 1.0 as the BYTE-IDENTICAL re-serve
ANCHOR (scale_lora_dose.py --anchor; same sha as the parent, its own staged id) — `--anchor` admits dose 1.0.

FAMILIES (FAMILIES below; the roster's `dose_families`): a target's ladder is one or two families, each a
(view prefix, copy-lane suffix, {cell id: parent lane}) triple:
  instruct-direct  itdoc__ parents               -> d<dose>it__<cell_id>  served by view it_d<dose>
  base-fit         base__ / derived__ parents    -> d<dose>__<cell_id>    served by view d<dose>
gemma4e4b / qwen35_4b carry the instruct family off the EFFECTIVE-BATCH-64 instruct-direct refits (`64`
names the effective batch those fits standardised on): parents `itdoc64__` / `derived_itdoc64__`
(`derived_<parent prefix>__` is the Llama precedent), copies `d<dose>it64__<cell_id>` (COPY_SUFFIX), still served by
view it_d<dose>. The bank's OLD `itdoc__` parent there (the batch-8 fit) is superseded, NOT overwritten — historical
stores carry its sha and its `d<dose>it__` copies' shas, so the refit's copies take NEW ids; 0.5 is baked as its own
`d0p5it64__` copy (`halfdoseit__` descends from the batch-8 parent) and 1.0 stays the parent's own cell (no anchor).
nemotron3nano4b / glm46vflash carry the base family; llama8b BOTH — one served instruct model, the base family
transferred onto it.

PARENTS OUTSIDE THE BANK (`--parent-root`, `--parent-origins`, `--parent-state`): the Llama parents were consumed
from their owning lane's bank before that lane staged them here, so they are read from a pulled read-only dir,
the authoritative path each was pulled from is recorded per parent (`--parent-origins <json>`: parent staged id ->
origin), and `--parent-state` says what state they were in — all of it lands in every copy's provenance.

Dose spelling in a lane / view name: 0.1 -> 0p1, 0.25 -> 0p25, 1 -> 1, 1.5 -> 1p5, -1 -> m1, -0.5 -> m0p5 (no '.'
or '-' in a staged id; `dose_of_view()` inverts it and the paper reader uses the same rule).

ENUMERATED FROM THE BANK, never from a list. Every copy is verified by its producer (tensor / served-op /
served-activation levels) and the ledger records parent sha, copy sha, object type, scale, scaling method
and the verification summary, ONE ROW PER (target, staged_id) with its own `scale`.

  python scripts/sdfbase/full22/strength/build_strength.py [--bank $STAGE/sdfbase/full22_objects]
        [--targets gemma4e4b,qwen35_4b,nemotron3nano4b,glm46vflash] [--doses -1,0.1,0.25,1.5] [--anchor]
        [--parent-root <dir of <parent staged id>/>] [--parent-origins <json>] [--parent-state <text>]
        [--only <staged_id>,...] [--ledger <path>] [--python <interpreter with torch>] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SDFBASE = HERE.parents[1]
#: the bank mirror under the staging directory ($STAGE) and the build ledger under the state root ($STATE), expanded through
#: sisteer.site after parsing
DEFAULT_BANK = Path("$STAGE/sdfbase/full22_objects")
DEFAULT_LEDGER = Path("$STATE/romp/strength/objects/build_ledger.json")
TARGETS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash")
DEFAULT_DOSES = (-1.0, 0.1, 0.25, 1.5)
LORA = "constv2-mix-full__lora_r4_ce"
SUB = "constv2-mix-full__lora_r4_ce_minus_ctl"
CTL = "constv2ctl2-full__lora_r4_ce"      # the CONTROL LoRA (the constv2ctl2 corpus): the third object of the base->instruct sweep
#: cell id -> (producer script, object type label)
OBJECT_TYPES = {
    LORA: ("scale_lora_dose.py", "constitution LoRA r4"),
    SUB: ("scale_lora_dose.py", "derived LoRA subtraction"),
    CTL: ("scale_lora_dose.py", "control LoRA r4"),
}
#: family name -> (view prefix, copy-lane suffix)
FAMILY_SPELLING = {"it": ("it_", "it"), "main": ("", "")}
#: (target, family) -> copy-lane suffix overriding FAMILY_SPELLING's: the eff-batch-64 instruct-direct refits' copies
#: are `d<dose>it64__`, never the `d<dose>it__` ids whose batch-8 shas historical stores carry. The VIEW spelling is
#: not overridden: it_d<dose> serves whichever copy the roster maps to it.
COPY_SUFFIX = {("gemma4e4b", "it"): "it64", ("qwen35_4b", "it"): "it64"}
#: per target, its ladder families: family -> {cell id: parent lane}. A cell id absent from a family has no parent
#: of that family.
FAMILIES = {
    # the BASE->INSTRUCT family: the paper's base-fit objects (`main` view: base__ fits, derived__
    # subtraction) TRANSFERRED onto the instruct target, THREE objects — constitution, control, subtraction — on gemma / qwen
    # beside their instruct-direct family, and the control LoRA added to llama's existing base family
    "gemma4e4b": {"it": {LORA: "itdoc64", SUB: "derived_itdoc64"}, "main": {LORA: "base", SUB: "derived", CTL: "base"}},
    "qwen35_4b": {"it": {LORA: "itdoc64", SUB: "derived_itdoc64"}, "main": {LORA: "base", SUB: "derived", CTL: "base"}},
    "nemotron3nano4b": {"main": {LORA: "base", SUB: "derived"}},
    "glm46vflash": {"main": {LORA: "base", SUB: "derived"}},
    # both families, one served instruct model: the it-lane subtraction is the Llama lane's `derived_itdoc__`
    "llama8b": {"it": {LORA: "itdoc", SUB: "derived_itdoc"}, "main": {LORA: "base", SUB: "derived", CTL: "base"}},
}
#: (target, cell id) -> reason, for lanes whose parents are ABSENT by construction (printed instead of the generic
#: "no parent lane"). Empty since the eff-batch-64 instruct-direct control fits landed (2026-09-13): gemma / qwen now
#: declare the `derived_itdoc64__` subtraction, and a parent not yet in the bank reports as "parent dir absent".
ABSENT_REASON: dict[tuple[str, str], str] = {}


def dose_token(d: float) -> str:
    """0.25 -> '0p25', -1 -> 'm1', 1 -> '1', 1.5 -> '1p5' (the staged-id / view grammar admits [A-Za-z0-9_] only)."""
    s = f"{abs(d):g}".replace(".", "p")
    return ("m" if d < 0 else "") + s


def dose_of_view(view: str) -> float:
    """Inverse of dose_token over a view name: 'd0p25' -> 0.25, 'it_dm1' -> -1.0, 'it_d1p5' -> 1.5, 'd1' -> 1.0,
    'd0p25_relu' -> 0.25. The optional trailing `_<object family>` segment (make_full22_configs' dose-view grammar,
    2026-09-16) says which family's ladder the view is on; it never changes the dose the name spells."""
    m = re.fullmatch(r"(?:it_)?d(m?\d+(?:p\d+)?)(?:_[a-z][a-z0-9]*)?", view)
    if not m:
        raise ValueError(f"not a dose view: {view!r}")
    v = m.group(1)
    sign = -1.0 if v.startswith("m") else 1.0
    return sign * float(v.lstrip("m").replace("p", "."))


def lane_and_view(family: str, dose: float, target: str | None = None) -> tuple[str, str]:
    """(copy lane, view) for one rung: `d<tok><suffix>`, `<prefix>d<tok>`. The suffix is COPY_SUFFIX's for
    (target, family) when one is declared, else FAMILY_SPELLING's; the view never depends on the target."""
    prefix, suffix = FAMILY_SPELLING[family]
    suffix = COPY_SUFFIX.get((target, family), suffix)
    tok = dose_token(dose)
    return f"d{tok}{suffix}", f"{prefix}d{tok}"


def enumerate_bank(bank: Path, targets: list[str], doses: list[float], *, parent_root: Path | None = None,
                   origins: dict | None = None) -> tuple[list[dict], list[dict]]:
    found, missing = [], []
    for target in targets:
        if target not in FAMILIES:
            raise SystemExit(f"{target}: no ladder families declared (FAMILIES)")
        cells = bank / target / "cells"
        if not cells.is_dir() and parent_root is None:
            raise SystemExit(f"no bank cells dir at {cells}")
        proot = parent_root or cells
        for family, parents in FAMILIES[target].items():
            for cid, (producer, label) in OBJECT_TYPES.items():
                plane = parents.get(cid)
                for dose in doses:
                    copy_lane, view = lane_and_view(family, dose, target)
                    row = {"target": target, "family": family, "cell_id": cid, "parent_lane": plane,
                           "parent_staged_id": f"{plane}__{cid}" if plane else None,
                           "staged_id": f"{copy_lane}__{cid}", "view": view, "scale": float(dose),
                           "anchor": dose == 1.0, "producer": producer, "object_type": label}
                    if plane is None:
                        row["why"] = ABSENT_REASON.get((target, cid), f"no {family}-family parent lane for {cid}")
                        missing.append(row); continue
                    d = proot / row["parent_staged_id"]
                    row["parent_dir"] = str(d)
                    row["parent_origin"] = (origins or {}).get(row["parent_staged_id"])
                    if (d / "object.pt").is_file():
                        found.append(row)
                    else:
                        row["why"] = "parent dir absent" if not d.is_dir() else "parent dir has no object.pt"
                        missing.append(row)
    return found, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--doses", default=",".join(f"{d:g}" for d in DEFAULT_DOSES))
    ap.add_argument("--anchor", action="store_true",
                    help="admit dose 1.0 as the BYTE-IDENTICAL re-serve anchor copy (D64; a target whose parents are not "
                         "staged under the campaign's own ids in this bank)")
    ap.add_argument("--parent-root", type=Path, default=None,
                    help="read parents from <dir>/<parent staged id>/ instead of <bank>/<target>/cells/ (a pulled read-only copy)")
    ap.add_argument("--parent-origins", type=Path, default=None,
                    help="JSON {parent staged id: authoritative origin path} recorded in every copy's provenance")
    ap.add_argument("--parent-state", default=None,
                    help="the owning lane's parent state at consumption, recorded verbatim (e.g. final bytes, not yet staged)")
    ap.add_argument("--only", default="", help="comma list of copy staged ids to (re)build; default all found")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if str(HERE.parents[2]) not in sys.path:
        sys.path.append(str(HERE.parents[2]))
    from site_paths import site_path   # the site config (scripts/site_paths.py); an unset placeholder refuses by name
    args.bank, args.ledger = site_path(args.bank, "--bank"), site_path(args.ledger, "--ledger")
    targets = [t for t in args.targets.split(",") if t]
    doses = [float(d) for d in args.doses.split(",") if d]
    for d in doses:
        if d == 0.0:
            raise SystemExit("dose 0 is not a copy (0 = the none cell)")
        if d == 1.0 and not args.anchor:
            raise SystemExit("dose 1.0 is the parent — pass --anchor to build it as the byte-identical re-serve anchor (D64)")
    if args.parent_root is not None and not args.parent_root.is_dir():
        raise SystemExit(f"--parent-root {args.parent_root} is not a directory")
    origins = json.loads(args.parent_origins.read_text(encoding="utf-8")) if args.parent_origins else {}
    found, missing = enumerate_bank(args.bank, targets, doses, parent_root=args.parent_root, origins=origins)
    if origins:
        for r in found:
            if not r["parent_origin"]:
                raise SystemExit(f"--parent-origins names no origin for parent {r['parent_staged_id']}")
    only = {s for s in args.only.split(",") if s}
    print(f"bank {args.bank}: {len(found)} copies to build, {len(missing)} absent"
          + (f"; parents from {args.parent_root}" if args.parent_root else ""))
    seen = set()
    for r in missing:
        key = (r["target"], r["family"], r["cell_id"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  ABSENT  {r['target']:<16} {r['family']:<5} {str(r['parent_staged_id']):<60} ({r['why']})")
    ledger = {"bank": str(args.bank), "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "doses": doses, "parent_root": str(args.parent_root) if args.parent_root else None,
              "parent_state": args.parent_state, "cells": [], "absent": missing}
    if args.ledger.is_file():
        try:
            prev = json.loads(args.ledger.read_text(encoding="utf-8"))
            ledger["cells"] = [c for c in prev.get("cells", []) if not (only and c["staged_id"] in only and c["target"] in targets)]
        except ValueError:
            pass
    built_keys = {(c["target"], c["staged_id"]) for c in ledger["cells"]}
    rc = 0
    for r in found:
        if only and r["staged_id"] not in only:
            continue
        out = args.bank / r["target"] / "cells" / r["staged_id"]
        if not only and (r["target"], r["staged_id"]) in built_keys and (out / "object.pt").is_file():
            print(f"  KEEP    {r['target']:<16} {r['staged_id']:<60} (already built; --only to rebuild)")
            continue
        cmd = [args.python, str(SDFBASE / r["producer"]), "--parent", r["parent_dir"],
               "--scale", f"{r['scale']:g}", "--out", str(out), "--target", r["target"], "--staged-id", r["staged_id"],
               "--parent-staged-id", r["parent_staged_id"]]
        if r["anchor"]:
            cmd.append("--anchor")
        if r["parent_origin"]:
            cmd += ["--parent-origin", r["parent_origin"]]
        if args.parent_state:
            cmd += ["--parent-state", args.parent_state]
        what = "ANCHOR " if r["anchor"] else "BUILD  "
        print(f"  {what} {r['target']:<16} {r['staged_id']:<60} <- {r['parent_staged_id']} x{r['scale']:g} [{r['producer']}]")
        if args.dry_run:
            continue
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"  FAILED  {r['target']} {r['staged_id']}: rc={p.returncode}\n{p.stdout[-1500:]}\n{p.stderr[-2500:]}")
            rc = 2
            continue
        summary = json.loads(p.stdout.strip().rsplit("\n{", 1)[-1] if not p.stdout.strip().startswith("{") else p.stdout)
        sidecar = "derived_meta.json" if (out / "derived_meta.json").is_file() else "train_budget_meta.json"
        meta = json.loads((out / sidecar).read_text(encoding="utf-8"))
        block = meta["dose_scaled_from"]
        assert float(block["scale"]) == r["scale"], (block["scale"], r["scale"])
        assert bool(block.get("anchor")) == r["anchor"], (block.get("anchor"), r["anchor"])
        if r["anchor"]:
            assert summary["object_sha256"] == summary["parent_object_sha256"], "an anchor is the parent's bytes"
        row = {**r, "parent_object_sha256": summary["parent_object_sha256"], "object_sha256": summary["object_sha256"],
               "scaling_method": block["scaling_method"], "tensors_scaled": block["tensors_scaled"],
               "identity_sidecar": sidecar, "verification": block["verification"], "out": str(out),
               "parent_consumed_state": block.get("parent_consumed_state"),
               "producer_commit": block.get("producer_commit"), "built_at": block["written_at"]}
        ledger["cells"] = [c for c in ledger["cells"] if (c["target"], c["staged_id"]) != (r["target"], r["staged_id"])] + [row]
        v = {c["check"]: c["status"] for c in block["verification"]["checks"]}
        print(f"          sha {summary['object_sha256'][:16]} <- parent {summary['parent_object_sha256'][:16]}  verification {v}  "
              f"max|diff|={summary['max_abs_diff_fp32']}")
    if not args.dry_run:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger["cells"].sort(key=lambda c: (c["target"], c["view"], c["cell_id"]))
        args.ledger.write_text(json.dumps(ledger, indent=1) + "\n", encoding="utf-8")
        print(f"ledger: {args.ledger} ({len(ledger['cells'])} cells built, {len(missing)} absent)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
