#!/usr/bin/env python
"""Build (or verify) ONE (target, view) bank VIEW of the full22 wave: the object directory the
served standup and every driver consume, in the code's cell-id grammar, over the staging agent's
lane-prefixed layout.

  <bank>/cells/<staged_id>/...                      the staged copy (lane-prefixed; read-only here)
  <bank>/cells_view/<view>/<cell_id>  ->  ../../cells/<staged_id>     one entry per roster cell

Every entry is admitted only when: the roster maps the cell; the staged object.pt hashes to the
pin in configs/sdfbase_full22_pins.json (the staging manifest's sha, verified on both clusters);
staged_from.json is present; and the dir's DECLARED IDENTITY validates.

TWO ROSTER FORMS PER CELL. A cell maps either to a STRING — a staged id of THIS wave's bank,
everything above, byte for byte as before — or to a MAPPING that names a PIN in another
authority:

    arditi: {pins: configs/misalign_baseline_pins.json, pin: gemma4e4b.arditi}

and nothing else: the mapping form has exactly those two keys, so a roster can point at a pin but
can never name a rung (there is no staged_id / bank / sha field to override the pin with: naming
`ar_r0p1__arditi` beside the production pin's `ar_r0p05__arditi` is the substitution this form
exists to make impossible). The builder reads `bank`, `source`, `staged_id`,
`object_sha256`, `kind`, `served_layer` and `identity_sidecar` FROM the pin, resolves the SOURCE
bank's staged copy under the same object root this bank lives in, and admits the entry only when
the on-disk object.pt re-hashes to the pin, the pin's own `source` view path resolves to that
staged copy, the declared identity sidecar is the pin's and validates, the blob's stored kind is
`ARM_OBJECT_KINDS[<arm>]` == the pin's kind, and the blob's served block layer is the pin's. The
source bank and the pin path are recorded in VIEW_MANIFEST.json (`form: "pin"`), so a view entry
never hides which authority chose the object. Certification-only by design: no production roster
uses the form, and every production view's manifest is byte-identical to before.

THREE IDENTITY SIDECARS, one dispatch (the same one train_budget.object_identity_kind makes):

* train_budget_meta.json — the TRAINING LANE's fit (the base / itdoc cells; gemma's scale-12 si12
  copies). Must name this cell (or its declared legacy spelling, methods.LEGACY_ARM_ALIASES); a
  pre-stamp copy (no object_sha256) is stamped on the record, below.
* fit_provenance.json — the HPO-PROMOTION lineage's fit (qwen's scale-12 si12 copies: tune_arm.py
  trials promoted into the bank). The SAME fit-time facts as the budget sidecar under the stage-3
  mirror gate's native spelling (resolved model id + revision + measured chat-template hash +
  writer), read through the ONE normaliser sweep.read_stage3_fit_provenance and NEVER translated
  into a second file on disk. It records no cell id and no sha — the roster's dir mapping names
  the cell and the PIN binds the bytes; its companion provenance.json (si + arm) is checked when
  present. Neither lineage carries the other's file anywhere upstream, so nothing exists to
  re-stage from: two spellings of one record, both admitted. A plain symlink.
* derived_meta.json — a DERIVED object (sdf-base-steering round 2: the corpus fit minus its control
  fit), never trained, so no budget record by construction; admitted through the shared derived
  verdict (train_budget.derived_object_problems — bytes <-> sidecar sha, sidecar <-> the provenance
  embedded in the bytes, two complete and mutually consistent parent identities). A plain symlink.

A dir carrying more than one of these is ambiguous and refuses rather than being read as any.

THE ONE RECORDED MIGRATION. A sidecar that records NO object_sha256 (the scale-12 safety-SI bank's
pre-stamp legacy sidecars) is refused by train_budget.staged_cell_problems with "migrate it
deliberately". This builder does exactly that, deliberately and on the record: the view entry
becomes a real directory whose files are symlinks to the staged copy EXCEPT train_budget_meta.json,
which is a copy stamped with the VERIFIED on-disk sha (plus a `full22_view_stamp` block naming the
original sidecar's sha and the reason), and EXCEPT COMPLETE (its publication manifest names the
original sidecar bytes, so it would read as torn next to the stamped copy; the staged copy keeps
it). full22_view.json in the entry records what was done. A sidecar whose recorded sha DIFFERS from
the artifact is never "fixed": that is a stale pair and the build refuses.

Idempotent: an existing entry is re-verified and replaced only if it is a symlink or a directory
carrying full22_view.json (never anything else). --check verifies without writing (exit 2 on any
problem). VIEW_MANIFEST.json summarises the view for the standup log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f22_roster import PIN_FORM_KEYS, served_cells, staged_cells  # noqa: E402,F401  (the ONE reader of a view's served cells; the view holds the STAGED subset — a prompted__<si> baseline has no object)

REPO = Path(__file__).resolve().parents[3]
ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"
PINS = REPO / "configs" / "sdfbase_full22_pins.json"
#: canonical arm -> legacy spelling accepted in a sidecar (methods.LEGACY_ARM_ALIASES inverted;
#: restated so this builder needs no torch import — the wave test asserts equality)
CANONICAL_TO_LEGACY_ARM = {"lora_r4": "lora_r1_all"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def accepted_sidecar_ids(cell_id: str) -> set[str]:
    si, arm = cell_id.split("__", 1)
    out = {cell_id}
    if arm in CANONICAL_TO_LEGACY_ARM:
        out.add(f"{si}__{CANONICAL_TO_LEGACY_ARM[arm]}")
    return out


def default_shared() -> Path:
    """The cluster's shared root, ``$SHARED`` through sisteer.site: the job's SISTEER_SHARED_ROOT, else CLUSTER_SHARED_ROOT,
    else the site config's cluster block; refused by name when none is set."""
    from sisteer import site as _site
    try:
        return _site.expand_path("$SHARED", key="--bank")
    except (_site.PlaceholderError, _site.SiteConfigError) as exc:
        raise SystemExit(f"cannot resolve the shared root: {exc}; or pass --bank")


def derived_entry_problems(staged_id: str, cell_id: str, obj: Path, meta_p: Path,
                           sha: str, rec: dict, *, derived_id: str | None = None) -> list[str]:
    """Every problem with a DERIVED staged copy, through the ONE shared
    verdict (``train_budget.derived_object_problems``) plus this builder's
    own roster-mapping check.

    The verdict is imported rather than restated — unlike the legacy-arm
    alias table above, whose three entries are cheap to mirror and pinned
    equal by the wave test. A derived object's identity is a dozen
    cross-checks between the bytes, the sidecar and two parent records; a
    second copy of those here is exactly the gate that can pass while
    quietly wrong. It costs a torch import, which the jobs that run this
    (common.sh f22_bank_view, under the driver venv) already pay."""
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from sisteer.eval.train_budget import derived_object_problems

    # THE EXPECTED derived_id (2026-09-17, the paired write): the string form holds the sidecar to the
    # CELL id, as ever — a wave cell IS the derived object it is named after. The cross-bank pin form
    # may serve a derived object under a cell id that is a ROUTE name (`<si>__psr_ce_all_minus_ctl`
    # serves `constv2-mix-full__psr_ce_all_minus_ctl`), so there the PIN declares the object's
    # own derived_id and the sidecar is held to THAT — a pin declaring the wrong one still refuses, and
    # a pin declaring none falls back to the cell id (the old rule, unchanged for every existing entry).
    expected = derived_id or cell_id
    rec["derived_id_expected"] = expected
    problems = [f"{staged_id}: {p}" for p in
                derived_object_problems(obj, derived_id=expected)]
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except ValueError as e:
        return problems + [f"{staged_id}: unreadable derived sidecar: {e}"]
    rec["derived_id"] = meta.get("derived_id")
    rec["derived_kind"] = meta.get("kind")
    # the verdict checks derived_id against what we pass; this records the
    # roster mapping in the view manifest so a mis-mapped entry is legible
    # from the artifact rather than only from an exit code
    if meta.get("object_sha256") != sha:
        problems.append(f"{staged_id}: derived sidecar object_sha256 "
                        f"{str(meta.get('object_sha256'))[:12]} != on-disk {sha[:12]}")
    return problems


#: every identity sidecar a staged dir may declare — restated from train_budget.IDENTITY_SIDECARS
#: so this builder needs no torch import on its light path; the wave test pins the two equal
IDENTITY_SIDECARS = ("train_budget_meta.json", "derived_meta.json", "fit_provenance.json")


def fit_provenance_entry_problems(staged_id: str, cell_id: str, obj: Path, rec_p: Path,
                                  rec: dict) -> list[str]:
    """Every problem with an HPO-promotion-lineage staged copy (``fit_provenance.json``), through
    the ONE shared verdict (``train_budget.fit_provenance_problems`` — the record normalises via
    ``sweep.read_stage3_fit_provenance``, ``provenance.json`` when present names this cell, the
    blob's fit leg agrees with the record's, a JSON COMPLETE manifest is verified while the
    lineage's plain-text promotion note is accepted as a note). Imported rather than restated,
    for the same reason as the derived verdict. The view manifest records the normalised
    fit-time facts so the two si12 lineages (gemma's budget sidecars, qwen's fit records) are
    legible side by side."""
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from sisteer.eval.sweep import read_stage3_fit_provenance
    from sisteer.eval.train_budget import fit_provenance_problems

    problems = [f"{staged_id}: {p}" for p in fit_provenance_problems(obj, cell_id)]
    norm = read_stage3_fit_provenance(rec_p.parent, object_path=obj)
    if norm is not None:
        rec["fit_record"] = {k: norm.get(k) for k in ("resolved_model_id", "resolved_revision",
                                                       "chat_template_hash", "positions_at_fit",
                                                       "source", "status")}
    return problems


def plan_entry(cell_id: str, staged_id: str, bank: Path, pin: dict | None) -> tuple[dict, list[str]]:
    """What the view entry for ``cell_id`` must be, and every problem with the staged copy."""
    src = bank / "cells" / staged_id
    problems: list[str] = []
    rec = {"cell_id": cell_id, "staged_id": staged_id, "source": str(src)}
    obj = src / "object.pt"
    if not obj.is_file():
        return rec, [f"{staged_id}: no object.pt at {obj}"]
    sha = sha256_file(obj)
    rec["object_sha256"] = sha
    if pin is None or not pin.get("object_sha256"):
        problems.append(f"{staged_id}: not pinned in {PINS.name} — refusing an unpinned object")
    elif pin["object_sha256"] != sha:
        problems.append(f"{staged_id}: object.pt sha {sha[:12]} != pinned {pin['object_sha256'][:12]}")
    if not (src / "staged_from.json").is_file():
        problems.append(f"{staged_id}: no staged_from.json (not a staged copy)")
    problems.extend(identity_entry_problems(cell_id, staged_id, src, obj, sha, rec))
    return rec, problems


def identity_entry_problems(cell_id: str, staged_id: str, src: Path, obj: Path,
                            sha: str, rec: dict, *, derived_id: str | None = None) -> list[str]:
    """The IDENTITY half of an entry's plan, shared by the string form
    (:func:`plan_entry`) and the cross-bank pin form (:func:`plan_pin_entry`):
    which identity contract the staged dir declares, its verdict, and the
    entry ``mode`` that follows (``symlink``, or ``stamped`` for the ONE
    recorded migration). Fills ``rec``; returns the problems."""
    problems: list[str] = []
    # WHICH IDENTITY the staged dir declares picks the verdict — the same
    # dispatch train_budget.object_identity_kind makes. Three sidecars,
    # exactly one per dir:
    #   train_budget_meta.json  the training lane's fit (gemma's si12 copies) — stamped below
    #                           when it records no object_sha256;
    #   fit_provenance.json     the HPO-promotion lineage's fit (qwen's si12 copies) — the SAME
    #                           fit-time facts under the stage-3 gate's native spelling, read
    #                           through sweep.read_stage3_fit_provenance; it records no sha, so
    #                           the PIN binds the bytes; a plain symlink, never a translated copy;
    #   derived_meta.json       a derived object (never trained; no budget record by construction),
    #                           held to the derived verdict; a plain symlink.
    # A dir declaring more than one is ambiguous and refuses.
    present = [s for s in IDENTITY_SIDECARS if (src / s).is_file()]
    if len(present) > 1:
        problems.append(f"{staged_id}: carries more than one identity sidecar {present} — a dir "
                        "declares exactly one; refusing to guess")
        return problems
    derived_p = src / "derived_meta.json"
    budget_p = src / "train_budget_meta.json"
    fitprov_p = src / "fit_provenance.json"
    if derived_p.is_file():
        rec["identity"] = "derived"
        rec["mode"] = "symlink"     # nothing to stamp: the sidecar records the sha itself
        problems.extend(derived_entry_problems(staged_id, cell_id, obj, derived_p, sha, rec, derived_id=derived_id))
        return problems
    if fitprov_p.is_file():
        rec["identity"] = "fit_provenance"
        rec["mode"] = "symlink"     # never a translated second file (the mirror gate's rule)
        problems.extend(fit_provenance_entry_problems(staged_id, cell_id, obj, fitprov_p, rec))
        return problems
    rec["identity"] = "staged"
    meta_p = budget_p
    if not meta_p.is_file():
        problems.append(f"{staged_id}: no train_budget_meta.json sidecar")
        return problems
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except ValueError as e:
        return problems + [f"{staged_id}: unreadable sidecar: {e}"]
    recorded = meta.get("cell_id")
    if recorded not in accepted_sidecar_ids(cell_id):
        problems.append(f"{staged_id}: sidecar cell_id={recorded!r} is not {cell_id!r} (or its "
                        "legacy spelling) — a mis-mapped roster entry or a mis-staged copy")
    rec["sidecar_cell_id"] = recorded
    rec_sha = meta.get("object_sha256")
    if rec_sha is None:
        rec["mode"] = "stamped"
        rec["stamp_reason"] = ("pre-stamp legacy sidecar (records no object_sha256); stamped with the "
                               "verified on-disk artifact sha; COMPLETE omitted from the view (its "
                               "file manifest names the original sidecar bytes)")
    elif rec_sha != sha:
        problems.append(f"{staged_id}: sidecar object_sha256 {str(rec_sha)[:12]} != on-disk {sha[:12]} "
                        "— a stale object/sidecar pair; never stamped over")
    else:
        rec["mode"] = "symlink"
    return problems


# ---------------------------------------------------------------------------------------
# the CROSS-BANK MAPPING FORM: a roster cell that names a PIN
# ---------------------------------------------------------------------------------------

# The EXACT key set of the roster's mapping form is ``f22_roster.PIN_FORM_KEYS`` (imported above):
# a cell may name only the pins FILE and the entry in it. There is deliberately NO `staged_id` /
# `bank` / `sha` field: every identity fact comes FROM the pin, so a roster cannot select a
# different rung of a dose ladder (naming `ar_r0p1__arditi` beside the production pin's
# `ar_r0p05__arditi` is exactly the substitution this form exists to make impossible).
#: what the builder READS from the pin entry it is pointed at (misalign_baseline_pins.json:16-38's
#: shape): the bank the object lives in, the VIEW path that publishes it there, the staged id, the
#: bytes' digest, the object kind, the served block layer, and which identity sidecar the staged
#: dir declares. Every one is checked against the disk below; none is taken on trust.
PIN_ENTRY_KEYS = ("bank", "source", "staged_id", "object_sha256", "kind", "served_layer",
                  "identity_sidecar")
#: every object bank's shared parent — `<SHARED>/exp/objects/<bank>/...`. The wave's own
#: `bank_root` starts with it (`exp/objects/sdfbase/full22`), and a pin's `source` is spelled
#: relative to it (`arditi_strength/<target>/strength/cells_view/<rung>/<cell>`).
OBJECTS_SUBDIR = "exp/objects"


def objects_root_of(bank: Path, roster: dict, target: str) -> Path:
    """The shared OBJECT-BANK root (:data:`OBJECTS_SUBDIR`) derived from the wave bank the builder
    was given — ONE derivation, so the cross-bank resolution follows `--bank` (and therefore the
    cluster's `$SHARED`, or a test's temporary root) instead of re-deriving a shared root of its
    own. `<bank>` is `<root>/<roster.bank_root>/<target>` by construction; a `--bank` that does not
    end that way, or a `bank_root` outside `exp/objects`, refuses by name rather than resolving a
    cross-bank path from a guess."""
    bank_root = str(roster["bank_root"]).strip("/")
    if not (bank_root + "/").startswith(OBJECTS_SUBDIR + "/"):
        raise SystemExit(f"roster bank_root {bank_root!r} is not under {OBJECTS_SUBDIR!r} — the "
                         "cross-bank pin form resolves every bank under that one root")
    rel = Path(bank_root) / target
    root = Path(bank)
    for _ in rel.parts:
        root = root.parent
    if (root / rel) != Path(bank):
        raise SystemExit(f"--bank {bank} does not end with the roster's {rel} — the cross-bank pin "
                         f"form derives the object root from it (got {root / rel})")
    return root / OBJECTS_SUBDIR


def plan_pin_entry(cell_id: str, mapping: dict, bank: Path, roster: dict, target: str,
                   repo: Path = REPO) -> tuple[dict, list[str]]:
    """What the view entry for a cell named through a PIN must be, and every problem with it.

    WHICH PATH IS RESOLVED, and why. The entry points at the SOURCE BANK'S STAGED COPY —
    `<objects root>/<the pin's bank root>/cells/<the pin's staged_id>` — not at the pin's
    `source` view entry, because the staged copy is where that lane's identity lives
    (`staged_from.json` + the identity sidecar) and it is the only path whose NAME carries the
    rung, so a substituted rung is caught by its sha under the name the pin gave. The pin's
    `source` view path is then required to EXIST and to resolve to that same directory: the view
    is how the owning lane publishes the rung, and a view symlink re-aimed at another rung must
    not pass unnoticed either. Both paths, one entry.

    Gates, all named: the mapping form's exact keys (no rung may be named); the pin file and
    entry; the pin's own required fields; `ARM_OBJECT_KINDS[<arm>] == pin.kind`; object.pt
    re-hashed EQUAL to the pin; the view path resolving to the staged dir; `staged_from.json`;
    exactly one identity sidecar, equal to the pin's `identity_sidecar`, and its lineage verdict;
    the BLOB's own stored kind equal to the pin's; and the blob's served block layer (its stored
    layer, shifted one earlier at `site='pre'` — `lens_adapters._block_output_layer`) equal to
    the pin's `served_layer`."""
    problems: list[str] = []
    rec: dict = {"cell_id": cell_id, "form": "pin"}
    unknown = sorted(set(mapping) - set(PIN_FORM_KEYS))
    missing = sorted(set(PIN_FORM_KEYS) - set(mapping))
    if unknown or missing:
        return rec, [f"{cell_id}: the roster's mapping form takes exactly {list(PIN_FORM_KEYS)} "
                     f"(missing {missing}, unknown {unknown}) — every identity fact comes from the "
                     "PIN, so there is no staged_id / bank / sha field to name a different rung with"]
    pins_rel = str(mapping["pins"])
    pin_ref = str(mapping["pin"])
    rec["pins"] = pins_rel
    rec["pin"] = pin_ref
    pins_path = repo / pins_rel      # an absolute `pins:` stands as given (pathlib's `/`)
    if not pins_path.is_file():
        return rec, [f"{cell_id}: no pins file at {pins_path} (roster `pins: {pins_rel}`)"]
    try:
        doc = json.loads(pins_path.read_text(encoding="utf-8"))
    except ValueError as e:
        return rec, [f"{cell_id}: unreadable pins file {pins_path}: {e}"]
    parts = pin_ref.split(".")
    if len(parts) != 2 or not all(parts):
        return rec, [f"{cell_id}: pin reference {pin_ref!r} is not `<model key>.<cell>` "
                     f"(e.g. gemma4e4b.arditi) in {pins_rel}"]
    pin = (doc.get(parts[0]) or {}).get(parts[1])
    if not isinstance(pin, dict):
        return rec, [f"{cell_id}: {pins_rel} has no {pin_ref} entry"]
    absent = [k for k in PIN_ENTRY_KEYS if pin.get(k) in (None, "")]
    if absent:
        return rec, [f"{cell_id}: pin {pin_ref} of {pins_rel} lacks {absent} — the builder reads "
                     f"every identity fact ({list(PIN_ENTRY_KEYS)}) from the pin"]
    staged_id = str(pin["staged_id"])
    rec.update(staged_id=staged_id, source_bank=str(pin["bank"]), pin_source=str(pin["source"]),
               kind=str(pin["kind"]), served_layer=int(pin["served_layer"]),
               identity_sidecar=str(pin["identity_sidecar"]))
    if pin.get("view"):
        rec["source_view"] = str(pin["view"])
    # the arm's kind, from the harness's own table — the roster names the CELL, the registry says
    # what kind of object that arm is, and the pin must agree with both
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS
    from sisteer.eval.defense_cell import arm_of

    arm = arm_of(cell_id)
    arm_kind = ARM_OBJECT_KINDS.get(arm or "")
    if arm_kind is None:
        problems.append(f"{cell_id}: arm {arm!r} has no ARM_OBJECT_KINDS mapping — the pin form "
                        "serves a registered arm, never an unmapped name")
    elif arm_kind != str(pin["kind"]):
        problems.append(f"{cell_id}: ARM_OBJECT_KINDS[{arm!r}] is {arm_kind!r} but pin {pin_ref} "
                        f"says kind {pin['kind']!r} — the roster and the pin name different objects")
    # the SOURCE BANK ROOT, from the pin's own `source` (…/cells_view/<view>/<cell>)
    src_parts = Path(str(pin["source"])).parts
    if len(src_parts) < 4 or src_parts[-3] != "cells_view":
        return rec, problems + [
            f"{cell_id}: pin {pin_ref} source {pin['source']!r} is not a "
            "`<bank>/…/cells_view/<view>/<cell>` view path"]
    if src_parts[0] != str(pin["bank"]):
        problems.append(f"{cell_id}: pin {pin_ref} source {pin['source']!r} is not under its own "
                        f"bank {pin['bank']!r}")
    if pin.get("view") and src_parts[-2] != str(pin["view"]):
        problems.append(f"{cell_id}: pin {pin_ref} source names view {src_parts[-2]!r} but the pin "
                        f"records view {pin['view']!r}")
    try:
        objects_root = objects_root_of(bank, roster, target)
    except SystemExit as e:
        return rec, problems + [f"{cell_id}: {e}"]
    src_bank_root = objects_root.joinpath(*src_parts[:-3])
    src = src_bank_root / "cells" / staged_id
    rec["source"] = str(src)
    view_path = objects_root / Path(str(pin["source"]))
    rec["source_view_path"] = str(view_path)
    obj = src / "object.pt"
    if not obj.is_file():
        return rec, problems + [f"{staged_id}: no object.pt at {obj} (cross-bank pin {pin_ref})"]
    sha = sha256_file(obj)
    rec["object_sha256"] = sha
    if pin["object_sha256"] != sha:
        problems.append(f"{staged_id}: object.pt sha {sha[:12]} != pinned "
                        f"{str(pin['object_sha256'])[:12]} ({pins_rel} {pin_ref}) — a substituted "
                        "rung or a drifted object; never served")
    if not view_path.exists():
        problems.append(f"{staged_id}: the pin's source view {view_path} does not exist")
    elif os.path.realpath(view_path) != os.path.realpath(src):
        problems.append(f"{staged_id}: the pin's source view {view_path} resolves to "
                        f"{os.path.realpath(view_path)}, not the staged copy {src} — the owning "
                        "lane's view names other bytes than the pin's staged id")
    if not (src / "staged_from.json").is_file():
        problems.append(f"{staged_id}: no staged_from.json (not a staged copy)")
    present = [s for s in IDENTITY_SIDECARS if (src / s).is_file()]
    if present != [str(pin["identity_sidecar"])]:
        problems.append(f"{staged_id}: declares identity sidecar(s) {present} but pin {pin_ref} "
                        f"records {pin['identity_sidecar']!r} — exactly one, and the pin's")
    problems.extend(identity_entry_problems(cell_id, staged_id, src, obj, sha, rec,
                                            derived_id=(str(pin["derived_id"]) if pin.get("derived_id") else None)))
    problems.extend(blob_identity_problems(staged_id, obj, pin))
    return rec, problems


def blob_identity_problems(staged_id: str, obj: Path, pin: dict) -> list[str]:
    """The BLOB's own stored facts against the pin (the cross-bank form's last gate): its `kind`
    — the string `methods.load_object` dispatches on and the serving adapters read — and the
    block layer the edit is SERVED at, which for a `site='pre'` object is one block earlier than
    the layer it stores (`lens_adapters._block_output_layer`; the residual slot's exported
    `boundaries`). A rung staged at another layer, or an object of another class under this
    cell's name, refuses here even when its bytes somehow hash to the pin."""
    import torch  # the builder's light path stays torch-free; the pin form pays this import

    problems: list[str] = []
    try:
        blob = torch.load(obj, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 — every failure is a problem, never a pass
        return [f"{staged_id}: {obj} does not load as a weights-only torch blob: "
                f"{type(e).__name__}: {e}"]
    if not isinstance(blob, dict):
        return [f"{staged_id}: {obj} is not a torch blob dict"]
    if blob.get("kind") != pin["kind"]:
        problems.append(f"{staged_id}: the blob stores kind {blob.get('kind')!r} but the pin says "
                        f"{pin['kind']!r} — a different object class under this cell's name")
        return problems
    # THREE spellings of the stored layers, one per blob class: `layers` (psr / psr_paired_write / lora_all), the keys of
    # `rs` (static_r: one direction per layer), the keys of `vectors` (vector_all — extraction/multilayer
    # .MultiLayerVectorObject.save writes no `layers` list; its layers ARE the `vectors` keys, stored at the SERVED block
    # layer, site post — the all-positions Arditi rungs the padded steered wave pins, 2026-09-23), else a bare `layer`
    layers = sorted(int(l) for l in (blob.get("layers") or list(blob.get("rs") or {}) or list(blob.get("vectors") or {})))
    if not layers and blob.get("layer") is not None:
        layers = [int(blob["layer"])]
    if not layers:
        problems.append(f"{staged_id}: the blob declares no layers — its served layer cannot be "
                        f"held to the pin's {pin['served_layer']}")
        return problems
    shift = 1 if str(blob.get("site") or "post") == "pre" else 0
    served = [l - shift for l in layers]
    if int(pin["served_layer"]) not in served:
        problems.append(f"{staged_id}: the blob's served block layer(s) {served} (stored {layers} "
                        f"at site {str(blob.get('site') or 'post')!r}) do not include the pin's "
                        f"served_layer {pin['served_layer']} — the pin describes another rung")
    return problems


def _rm_scratch(p: Path) -> None:
    """Remove one of THIS builder's own `.tmp-` / `.old-` scratch paths, unconditionally.

    `_clear_entry` deliberately refuses to delete a directory that is not a recognised view entry —
    a guard for the DESTINATION path, where an unexpected directory means someone else's data. Our
    scratch paths are ours by construction (name-prefixed, pid-suffixed) and a half-built one has no
    `full22_view.json` yet, so it must not go through that guard."""
    name = p.name
    if not (name.startswith(".tmp-") or name.startswith(".old-")):
        raise SystemExit(f"_rm_scratch refuses {p} — not a builder scratch path")
    if p.is_symlink():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def _clear_entry(dst: Path) -> None:
    if dst.is_symlink():
        dst.unlink()
    elif dst.is_dir():
        if not (dst / "full22_view.json").is_file():
            raise SystemExit(f"{dst} exists and is not a full22 view entry — refusing to remove it")
        shutil.rmtree(dst)
    elif dst.exists():
        raise SystemExit(f"{dst} exists and is not a symlink or view directory — refusing")


def write_entry(rec: dict, view_dir: Path) -> None:
    """Build the entry OUT OF PLACE and rename it in (2026-09-07).

    The old shape — `_clear_entry(dst)`, `mkdir`, then populate file by file — left the entry
    missing or half-populated for as long as the build took, and a concurrent reader that verified
    it in that window failed on whichever file had not landed yet (job 423389: FileNotFoundError on
    a stamped `train_budget_meta.json` that existed seconds later). Now the entry is assembled at
    `.tmp-<cell>-<pid>` and swapped in with `os.rename`, so a reader sees the OLD complete entry or
    the NEW complete one, with only the microsecond gap between the two renames instead of the whole
    build. `common.sh` layers a per-view `flock` and a verify-first path on top; the wave's
    submission builds every view once up front, before any job that consumes it."""
    src = Path(rec["source"])
    dst = view_dir / rec["cell_id"]
    tmp = view_dir / f".tmp-{rec['cell_id']}-{os.getpid()}"
    old_aside = view_dir / f".old-{rec['cell_id']}-{os.getpid()}"
    _rm_scratch(tmp)
    _rm_scratch(old_aside)
    try:
        _populate_entry(rec, src, tmp, view_dir)
        had_old = tmp_exists(dst)
        if had_old:
            os.rename(dst, old_aside)
        try:
            os.rename(tmp, dst)
        except OSError:
            if had_old:                      # put the old entry back rather than leaving a hole
                os.rename(old_aside, dst)
            raise
        if had_old:
            _rm_scratch(old_aside)
    finally:
        _rm_scratch(tmp)
        _rm_scratch(old_aside)


def tmp_exists(p: Path) -> bool:
    """`Path.exists()` follows symlinks, so a dangling one would read as absent and never be
    replaced; a view entry is a symlink far more often than not."""
    return p.is_symlink() or p.exists()


def _populate_entry(rec: dict, src: Path, dst: Path, view_dir: Path) -> None:
    rel = Path(os.path.relpath(src, view_dir))
    if rec["mode"] == "symlink":
        dst.symlink_to(rel, target_is_directory=True)
        return
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.name in ("train_budget_meta.json", "COMPLETE"):
            continue
        # relative to where the entry will LIVE (view_dir/<cell_id>), not to the tmp dir it is
        # assembled in — the rename moves the directory, and a link relative to tmp would dangle
        final = view_dir / rec["cell_id"]
        (dst / f.name).symlink_to(Path(os.path.relpath(f, final)), target_is_directory=f.is_dir())
    meta_p = src / "train_budget_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["object_sha256"] = rec["object_sha256"]
    meta["full22_view_stamp"] = {
        "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_sidecar_sha256": sha256_file(meta_p),
        "original_sidecar": str(meta_p),
        "reason": rec["stamp_reason"]}
    (dst / "train_budget_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (dst / "full22_view.json").write_text(json.dumps(
        {**rec, "complete_marker_omitted": (src / "COMPLETE").is_file(),
         "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n",
        encoding="utf-8")


def verify_entry(rec: dict, view_dir: Path) -> list[str]:
    dst = view_dir / rec["cell_id"]
    if not (dst / "object.pt").is_file():
        return [f"view entry {dst} missing (or no object.pt)"]
    if sha256_file(dst / "object.pt") != rec["object_sha256"]:
        return [f"view entry {dst}: object.pt does not hash to the pin"]
    if rec.get("identity") == "derived":
        # a derived entry is a plain symlink to the staged dir: its sidecar
        # is derived_meta.json, which records the sha itself (nothing to
        # stamp), and the shared verdict already ran in plan_entry
        meta = json.loads((dst / "derived_meta.json").read_text(encoding="utf-8"))
        if meta.get("object_sha256") != rec["object_sha256"]:
            return [f"view entry {dst}: derived sidecar object_sha256 != pin"]
        if (dst / "train_budget_meta.json").is_file():
            return [f"view entry {dst}: a derived entry must carry no budget sidecar"]
        return []
    if rec.get("identity") == "fit_provenance":
        # likewise a plain symlink: the fit record carries no sha (the PIN
        # binds the bytes, checked above) and is never translated into a
        # budget sidecar; exactly one identity file may be present
        others = [s for s in IDENTITY_SIDECARS if s != "fit_provenance.json" and (dst / s).is_file()]
        if not (dst / "fit_provenance.json").is_file():
            return [f"view entry {dst}: fit_provenance.json missing"]
        if others:
            return [f"view entry {dst}: a fit-record entry must carry no other identity sidecar, "
                    f"found {others}"]
        return []
    meta = json.loads((dst / "train_budget_meta.json").read_text(encoding="utf-8"))
    if meta.get("object_sha256") != rec["object_sha256"]:
        return [f"view entry {dst}: sidecar object_sha256 != pin"]
    if rec["mode"] == "stamped" and not (dst / "full22_view.json").is_file():
        return [f"view entry {dst}: stamped entry without its full22_view.json record"]
    return []


def view_completeness_problems(view_dir: Path, cells: dict) -> list:
    """THE SHORT-VIEW GUARD (2026-09-07). A view that holds fewer cells than the roster names is an
    ERROR, not a log line: at the first launch attempt one target's view built 7 entries and the
    other's 5, both exited 0, and the wave would have produced a results table with no safety-SI arm
    on one of two targets — a shortfall nobody sees unless they count. Extras are refused too: a
    stale entry from a narrower or wider roster could be served under a cell id the roster no longer
    maps."""
    want = set(cells)
    have = {p.name for p in view_dir.iterdir()
            if not p.name.startswith(("VIEW_MANIFEST", "full22_"))} if view_dir.is_dir() else set()
    probs = []
    missing, extra = sorted(want - have), sorted(have - want)
    if missing:
        probs.append(f"view {view_dir} is SHORT: the roster names {len(want)} cells, {len(have)} are "
                     f"present; missing {missing} — a view with fewer cells than the roster names is an "
                     "error (the results table would silently lose those arms)")
    if extra:
        probs.append(f"view {view_dir} carries {len(extra)} entry/entries the roster does not name "
                     f"({extra}) — stale from an earlier roster; remove them rather than serving them")
    return probs


def derived_quality(recs: list, bank: Path) -> dict:
    """Per DERIVED cell, the fit-quality facts a cross-target comparison must carry, read from the
    staged ``derived_meta.json``: held-out loss, best epoch / early-stop, and gate liveness
    (min / mean / count below 0.5 over the target layers). The two targets' distilled residuals are
    NOT of comparable quality (README section 4a), so the numbers travel with the view instead of
    living only in a doc."""
    out = {}
    for rec in recs:
        # the entry's OWN source dir (this bank's `cells/<staged_id>` for a string entry, the
        # source bank's for a cross-bank pin entry) — never re-derived from the local bank
        d = Path(rec.get("source") or (bank / "cells" / rec["staged_id"]))
        meta_p = d / "derived_meta.json"
        if not meta_p.is_file():
            continue
        try:
            m = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 — recorded as a problem-shaped entry, never swallowed
            out[rec["cell_id"]] = {"error": f"unreadable derived_meta.json: {e!r}"}
            continue
        tr = m.get("trainer") or {}
        gl = m.get("gate_liveness") or {}
        # the distill's held-out loss lives in the build's report.json (the sidecar keeps the trainer
        # hyperparameters + the stop reason); read it from there and record which files it came from
        rep = {}
        rep_p = d / "report.json"
        if rep_p.is_file():
            try:
                rep = json.loads(rep_p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                rep = {}
        q = {"staged_id": rec["staged_id"], "kind": m.get("kind"),
             "recovery_verdict": m.get("recovery_verdict"),
             "best_heldout_loss": rep.get("best_heldout_loss"),
             "heldout_init": rep.get("heldout_init"),
             "best_epoch": tr.get("best_epoch", rep.get("best_epoch")),
             "epochs_run": tr.get("epochs_run", rep.get("epochs_run")),
             "epochs_cap": tr.get("epochs", rep.get("epochs_cap")),
             "stopped_early": tr.get("stopped_early"), "stop_reason": tr.get("stop_reason"),
             "quality_source": "report.json + derived_meta.json" if rep else "derived_meta.json only"}
        try:
            vals = [float(v) for v in gl.values()]
        except Exception:  # noqa: BLE001
            vals = []
        if vals:
            q.update(liveness_n_layers=len(vals), liveness_min=round(min(vals), 4),
                     liveness_mean=round(sum(vals) / len(vals), 4),
                     liveness_n_below_0p5=sum(v < 0.5 for v in vals))
        out[rec["cell_id"]] = q
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--target", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--bank", type=Path, default=None,
                    help="the wave bank root (default $SHARED/exp/objects/sdfbase/full22/<target>, $SHARED being the "
                         "cluster's shared root; on the grading host e.g. $STAGE/sdfbase/full22_objects/<target>)")
    ap.add_argument("--roster", type=Path, default=ROSTER)
    ap.add_argument("--pins", type=Path, default=PINS)
    ap.add_argument("--check", action="store_true", help="verify only; write nothing; exit 2 on a problem")
    args = ap.parse_args(argv)
    if not args.roster.is_file():
        raise SystemExit(f"no roster at {args.roster}")
    roster = yaml.safe_load(args.roster.read_text(encoding="utf-8"))
    pins = json.loads(args.pins.read_text(encoding="utf-8")).get(args.target, {})
    t = (roster.get("targets") or {}).get(args.target)
    if t is None or args.view not in (t.get("views") or {}):
        # by name, never a KeyError: a certification-only view read through the
        # production roster is the expected mistake, and the fix is the roster, not the view
        raise SystemExit(f"roster {args.roster} names no view {args.target}/{args.view} (targets: "
                         f"{sorted((roster.get('targets') or {}))}; views of {args.target}: "
                         f"{sorted((t or {}).get('views') or {})}) — pass --roster <the roster that names it>")
    # view cells + every ENABLED lane's cells that carry a STAGED OBJECT — the generator's
    # explicit_cells minus its prompted baselines (prompted__<si>: the SI as the system prompt, no
    # object, so no view entry; the job-side guard compares the view against this count and the
    # config against the full served count — roster_cell_count.py with and without --staged)
    try:
        cells = staged_cells(roster, args.target, args.view)
    except ValueError as e:
        # a malformed roster cell (f22_roster._check_cell: a prompted cell mapped to a pin, a
        # mapping that is not exactly {pins, pin}, a rung named beside the pin) is a PROBLEM the
        # caller reads on stdout with exit 2 — the same shape as every other refusal here, never a
        # traceback (the job-side guard and the tests read the exit code and the message)
        print(f"PROBLEM roster {args.roster}: {e}")
        return 2
    bank = args.bank or (default_shared() / roster["bank_root"] / args.target)
    view_dir = bank / "cells_view" / args.view
    recs, problems = [], []
    for cid, sid in cells.items():
        # TWO roster forms per cell: a STRING names a staged id of THIS
        # wave's bank, pinned in configs/sdfbase_full22_pins.json — unchanged, byte for byte; a
        # MAPPING names a PIN in another authority (`{pins: …, pin: <model>.<cell>}`) and every
        # identity fact, the source bank included, is read from that pin (plan_pin_entry).
        if isinstance(sid, dict):
            rec, probs = plan_pin_entry(cid, sid, bank, roster, args.target)
        else:
            rec, probs = plan_entry(cid, sid, bank, pins.get(sid))
        recs.append(rec); problems.extend(probs)
    if problems:
        print("\n".join(f"PROBLEM {p}" for p in problems))
        return 2
    if args.check:
        for rec in recs:
            problems.extend(verify_entry(rec, view_dir))
        problems.extend(view_completeness_problems(view_dir, cells))
        print("\n".join(f"PROBLEM {p}" for p in problems) or f"ok: {view_dir} ({len(recs)} entries verified)")
        return 2 if problems else 0
    view_dir.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        write_entry(rec, view_dir)
        problems.extend(verify_entry(rec, view_dir))
    if problems:
        print("\n".join(f"PROBLEM {p}" for p in problems)); return 2
    problems.extend(view_completeness_problems(view_dir, cells))
    if problems:
        print("\n".join(f"PROBLEM {p}" for p in problems)); return 2
    quality = derived_quality(recs, bank)
    if quality:
        (view_dir / "full22_derived_quality.json").write_text(
            json.dumps({"target": args.target, "view": args.view, "cells": quality,
                        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                       indent=2) + "\n", encoding="utf-8")
        for cid, q in sorted(quality.items()):
            print(f"derived {args.target}/{args.view}: {cid:<48} "
                  + (f"heldout={q['best_heldout_loss']} best_epoch={q['best_epoch']} "
                     f"early_stopped={q['stopped_early']} liveness min={q['liveness_min']} "
                     f"mean={q['liveness_mean']} below_0.5={q['liveness_n_below_0p5']}/{q['liveness_n_layers']}"
                     if q.get("liveness_n_layers") else f"kind={q.get('kind')} (structural; no fit record)"))
    (view_dir / "VIEW_MANIFEST.json").write_text(json.dumps(
        {"wave": roster["wave"], "target": args.target, "view": args.view, "bank": str(bank),
         "entries": recs, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2) + "\n", encoding="utf-8")
    for rec in recs:
        print(f"view {args.target}/{args.view}: {rec['cell_id']:<32} <- {rec['staged_id']} "
              f"[{rec['mode']}] sha {rec['object_sha256'][:12]}")
    print(f"ok: {view_dir} ({len(recs)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
