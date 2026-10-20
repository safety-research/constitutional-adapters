#!/usr/bin/env python3
"""The INSTRUCT-DIRECT ("it-doc") batch-64 re-fit objects — their ADMISSION RECORD and their BANK BINDING
(math-eight-arms-extension E3; decisions-e3 E3-2 / E3-3). STDLIB ONLY: the sbatch reads it through the two matrix
readers before any venv exists, and the grading host dry run reads it without torch.

THE PROBLEM THIS FILE EXISTS FOR. A cell id is NOT unique: `constv2-mix-full__lora_r4_ce` is a BASE-lane cell
(`base__…`, the `main` views), a superseded direct-fit cell (`itdoc__…`, the `it` views, sha 0018b32b… on qwen35_4b),
five dose bakes of that superseded parent (`d0p1it__…` …), and — this record — the batch-64 re-fit (sha 80b75aae…).
A staged id is not unique either: `base__constv2-mix-full__lora_r4_ce` exists under all five targets. A run that
pinned the re-fit but resolved the cell through today's `it` bank view would SERVE the superseded bytes while REPORTING
the re-fit (the serving bank's manifest still names the old staged id). So:

  * every pin and every hold is keyed by the FULL tuple (lane, model key, view, cell id) -> object sha256, never by a
    cell id or a staged id alone; a pins-file lookup is keyed by (target, staged id), never by staged id alone;
  * a row BINDS to a bank entry by SHA: the bank view's VIEW_MANIFEST entry for the cell, the object's bytes on disk,
    the record's pin, and — when the full22 pin file carries (target, staged id) — its entry must ALL agree; a
    disagreement REFUSES, names the lane and the target, and names a KNOWN STALE object by label ("stale dose bake",
    "superseded direct-fit object") when the served bytes are one;
  * the sidecar read is the one OF THE KIND: `train_budget_meta.json` for a fitted cell (the lane's fit-time sidecar,
    which `sweep.resolve_training_mirror` admits as a DOCUMENT fit), `derived_meta.json` for a derived object (the
    subtraction the lane builds under `derived/lora_subtraction/<model key>/`; never in `cells_all/`), and a derived
    object's recorded parents must be EXACTLY the record's two it-doc parents by sha;
  * a row whose record carries no launchable binding (no sha; a sha but no bank id; a derived object not built) is
    HELD with the record's reason — never a placeholder sha, never a silent launch.

`bind_problems` is the CPU check both launchers run at the bank door (`st_itdoc_bank_check`, before the standup), the
dry runs print per row (`ITDOC_BIND:`), and `audit` runs over every entry of the record against a shared root — PASS
per cell explicitly, not only refusals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

REPO = Path(__file__).resolve().parents[3]

SCHEMA = "itdoc_admission/1"
ADMISSION_REL = Path("configs/sdfbase/itdoc_admission.json")
ADMISSION_PATH = REPO / ADMISSION_REL
#: the row's `object_source` spelling for a re-fit row (the launchers dispatch the bank door on it; `external` is another wave's)
OBJECT_SOURCE_ITDOC = "itdoc"
#: the two object kinds the record admits and the ONE sidecar each is proven through
OBJECT_KIND_FITTED, OBJECT_KIND_DERIVED = "fitted", "derived"
SIDECAR_BY_KIND = {OBJECT_KIND_FITTED: "train_budget_meta.json", OBJECT_KIND_DERIVED: "derived_meta.json"}
OBJECT_KINDS = tuple(SIDECAR_BY_KIND)
VIEW_MANIFEST = "VIEW_MANIFEST.json"
OBJECT_FILE = "object.pt"
#: the refusal labels the tests pin
STALE_DOSE_BAKE = "stale dose bake"
SUPERSEDED_OBJECT = "superseded direct-fit object"
#: verdict words of `bind_verdict` / `audit`
VERDICT_PASS, VERDICT_HELD, VERDICT_REFUSED = "PASS", "HELD", "REFUSED"
#: the STAGED-ID RULE: the re-fit's ids carry the lane token `itdoc64` (the effective batch that
#: changed) — parents `itdoc64__<cell>`, the derived object `derived_itdoc64__<cell>` (the label's lineage equals the sha's, the
#: llama precedent), dose re-bakes `d<dose>it64__<cell>` — never overwriting the batch-8 `itdoc__` / `derived_itdoc__` / `*it__`
#: cells whose shas the historical stores carry. A resolution landing on one of THOSE is refused by lane name.
LANE_TOKEN = "itdoc64"
STAGED_PREFIX_BY_KIND = {OBJECT_KIND_FITTED: f"{LANE_TOKEN}__", OBJECT_KIND_DERIVED: f"derived_{LANE_TOKEN}__"}
#: other lanes' staged-id spellings of the SAME cell ids, by prefix -> the lane they belong to (named in a refusal)
OTHER_LANE_BY_PREFIX = (
    ("derived_itdoc__", "the batch-8 direct-fit lane's derived object (itdoc__ parents)"),
    ("itdoc__", "the batch-8 direct-fit lane (the superseded parent, D8)"),
    ("derived__", "the BASE lane's derived object (base__ parents; a same-named cell of another lane)"),
    ("base__", "the BASE lane (the main views' object; a same-named cell of another lane)"),
)


def other_lane_of(staged_id: object) -> Optional[str]:
    """The lane a staged id belongs to when it is NOT this lane's spelling (`d0p25it__…` is a batch-8 dose bake, `itdoc__…` the
    superseded parent, `base__…` / `derived__…` the base lane); None for this lane's own ids."""
    sid = str(staged_id or "")
    if not sid:
        return None
    if sid.startswith(tuple(STAGED_PREFIX_BY_KIND.values())) or "it64__" in sid:      # this lane's parents / derived / dose re-bakes
        return None
    for prefix, lane in OTHER_LANE_BY_PREFIX:
        if sid.startswith(prefix):
            return lane
    if "it__" in sid:
        return "a batch-8 dose bake of the superseded parent (a `*it__` staged id) — a stale dose bake"
    return "an unrecognised lane (not this lane's `itdoc64__` / `derived_itdoc64__` spelling)"


_HEX = set("0123456789abcdef")


def is_sha256(s: object) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in _HEX for c in s)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tuple_of(entry: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """(lane, model key, view, cell) — THE key of every pin and hold."""
    return (str(entry["lane"]), str(entry["model_key"]), str(entry["view"]), str(entry["cell"]))


def tuple_label(lane: str, model_key: str, view: str, cell: str) -> str:
    return f"(lane {lane}, target {model_key}, view {view}, cell {cell})"


# --------------------------------------------------------------------------- the record
_ENTRY_REQUIRED = ("lane", "model_key", "view", "cell", "object_kind", "staged_id", "object_sha256", "bank", "hold_reason", "qualification")
_BANK_REQUIRED = ("view", "bank_view_rel", "bank_cell_dir", "cluster_constraint")


def bank_complete(bank: Any) -> bool:
    """A bank binding a launcher can dispatch on: the bank view's name, the view dir RELATIVE to the shared root for the
    primary role (the overflow role's, or null), the cell dir inside it, and the placement the staging implies (the staged
    id is the ENTRY's: ruled before the path exists)."""
    if not isinstance(bank, Mapping):
        return False
    if any(bank.get(f) in (None, "") for f in _BANK_REQUIRED if f != "bank_view_rel"):
        return False
    rel = bank.get("bank_view_rel")
    return isinstance(rel, Mapping) and set(rel) == {"primary", "overflow"} and bool(rel.get("primary"))


def entry_launchable(entry: Mapping[str, Any]) -> bool:
    """A row LAUNCHES iff its record carries a 64-hex pin AND a complete bank binding AND no hold reason; anything less
    is HELD (the loader requires the reason then)."""
    return (is_sha256(entry.get("object_sha256")) and bool(entry.get("staged_id")) and bank_complete(entry.get("bank"))
            and not str(entry.get("hold_reason") or "").strip())


def entry_problem(entry: Mapping[str, Any], *, where: str, lane: str, key: str, view: str, cell: str,
                  parents: Mapping[str, Any]) -> Optional[str]:
    """Why one admission entry is malformed (refused BY NAME at load, never at the batch boundary of a job holding a GPU)."""
    missing = [f for f in _ENTRY_REQUIRED if f not in entry]
    if missing:
        return f"{where}: the entry lacks {missing}"
    if (entry["lane"], entry["model_key"], entry["view"], entry["cell"]) != (lane, key, view, cell):
        return (f"{where}: the entry names {tuple_label(entry['lane'], entry['model_key'], entry['view'], entry['cell'])} but sits under "
                f"{tuple_label(lane, key, view, cell)} — an entry is keyed by (lane, target, view, cell) and must say so")
    kind = entry["object_kind"]
    if kind not in OBJECT_KINDS:
        return f"{where}: object_kind {kind!r} is not one of {OBJECT_KINDS}"
    if entry.get("sidecar", SIDECAR_BY_KIND[kind]) != SIDECAR_BY_KIND[kind]:
        return f"{where}: a {kind} object is proven through {SIDECAR_BY_KIND[kind]!r}, not {entry.get('sidecar')!r}"
    sid = entry["staged_id"]
    if not isinstance(sid, str) or not sid.startswith(STAGED_PREFIX_BY_KIND[kind]) or sid != STAGED_PREFIX_BY_KIND[kind] + cell:
        return (f"{where}: staged_id {sid!r} is not this lane's spelling {STAGED_PREFIX_BY_KIND[kind] + cell!r} (the staged-id ruling: the lane "
                f"token {LANE_TOKEN!r} in the id; never a batch-8 `itdoc__` / `*it__` / `derived_itdoc__` id, whose shas the historical stores carry)")
    sha = entry["object_sha256"]
    if sha is not None and not is_sha256(sha):
        return f"{where}: object_sha256 {sha!r} is neither null (not yet verified on disk) nor a 64-hex sha — never a placeholder (D5)"
    bank = entry["bank"]
    if bank is not None and not bank_complete(bank):
        return f"{where}: bank {bank!r} is neither null (not staged) nor a complete binding {_BANK_REQUIRED} with bank_view_rel.primary"
    if bank is not None and sha is None:
        return f"{where}: a bank binding without a verified sha — a staged object is hashed before it is bound"
    pe = entry.get("parents_expected")
    if kind == OBJECT_KIND_DERIVED:
        if pe is not None and not (isinstance(pe, Mapping) and len(pe) == 2 and all(is_sha256(v) for v in pe.values())):
            return f"{where}: a derived object names EXACTLY its two parents (cell -> 64-hex sha), got {pe!r}"
        if pe is not None:
            want = {str(p["cell"]): str(p["object_sha256"]) for p in parents.values() if is_sha256(p.get("object_sha256"))}
            if dict(pe) != want:
                return (f"{where}: parents_expected {dict(pe)!r} != the record's it-doc parents of {key} {want!r} — a derived row "
                        "binds to THIS lane's parents, never to a same-named derived object of another lane")
    elif pe is not None:
        return f"{where}: parents_expected is a derived object's field, not a {kind} object's"
    launchable = entry_launchable(entry)
    if not launchable and not str(entry.get("hold_reason") or "").strip():
        return f"{where}: the entry is not launchable (sha {sha!r}, bank {'complete' if bank_complete(bank) else bank!r}) and carries no hold_reason"
    if sha is not None and (not str(entry.get("qualification") or "").strip() or not str(entry.get("qualification_pattern") or "").strip()):
        return f"{where}: a verified object carries the pair's QUALIFICATION and the cross-model qualification PATTERN verbatim, or it is not admitted"
    stale = entry.get("stale_objects") or {}
    if not isinstance(stale, Mapping) or any(not is_sha256(k) or not str(v).strip() for k, v in stale.items()):
        return f"{where}: stale_objects is a {{64-hex sha: label}} map, got {stale!r}"
    if sha is not None and sha in stale:
        return f"{where}: the pin {sha[:12]}… is itself listed as a stale object ({stale[sha]})"
    return None


def load_admission(path: Path = ADMISSION_PATH) -> dict[str, Any]:
    """The record: ``{doc, lane, entries: {(lane, key, view, cell): entry}, parents: {key: {role: {...}}}}`` — every entry
    validated (:func:`entry_problem`)."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"no it-doc admission record at {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise SystemExit(f"{p}: schema {doc.get('schema')!r} is not {SCHEMA!r}")
    lane = str(doc.get("lane") or "")
    if not lane:
        raise SystemExit(f"{p}: the record names its lane")
    parents = doc.get("parents") or {}
    entries: dict[tuple[str, str, str, str], dict] = {}
    for key, by_view in (doc.get("entries") or {}).items():
        for view, by_cell in by_view.items():
            for cell, entry in by_cell.items():
                where = f"{p.name}: entries.{key}.{view}.{cell}"
                problem = entry_problem(entry, where=where, lane=lane, key=key, view=view, cell=cell, parents=parents.get(key) or {})
                if problem:
                    raise SystemExit(problem)
                entries[(lane, key, view, cell)] = entry
    return {"doc": doc, "lane": lane, "entries": entries, "parents": parents, "path": str(p),
            "qualification": doc.get("qualification"), "sha256": sha256_of(p)}


def entry_for(admission: Mapping[str, Any], lane: str, key: str, view: str, cell: str) -> Optional[Mapping[str, Any]]:
    return admission["entries"].get((lane, key, view, cell))


def instrument_hold(admission: Mapping[str, Any], instrument: str) -> Optional[str]:
    """The record's INSTRUMENT-level hold: a non-empty reason means every row of that instrument generated from this record is
    HELD whatever its bank binding says (the SUBMIT authorization of 2026-09-13 covers the GSM8K wave alone; the MATH-500 legs wait for
    their own go). None when the instrument is not held."""
    holds = admission["doc"].get("instrument_holds") or {}
    reason = holds.get(instrument) if isinstance(holds, Mapping) else None
    return (str(reason).strip() or None) if reason else None


def pins_file_sha(pins_doc: Mapping[str, Any], *, target: str, staged_id: str) -> Optional[str]:
    """The full22 pin file's object sha for (TARGET, staged id) — keyed by BOTH (a staged id repeats across targets:
    `base__constv2-mix-full__lora_r4_ce` sits under every target), None when the pair is not pinned."""
    block = pins_doc.get(target)
    if not isinstance(block, Mapping):
        return None
    entry = block.get(staged_id)
    sha = entry.get("object_sha256") if isinstance(entry, Mapping) else entry
    return str(sha) if is_sha256(sha) else None


# --------------------------------------------------------------------------- the binding check
def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stale_label(sha: Optional[str], stale: Mapping[str, str]) -> str:
    return f" — {stale[sha]}" if sha in stale else ""


def bind_problems(view_dir: Path, *, lane: str, model_key: str, view: str, cell: str, pin: Optional[str],
                  object_kind: str, staged_id: Optional[str] = None, cell_dir: Optional[str] = None,
                  parents_expected: Optional[Mapping[str, str]] = None, stale: Optional[Mapping[str, str]] = None,
                  pins_doc: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The four-way binding of ONE row to the bank at ``view_dir`` — returns ``{verdict, problems, hashes, ...}``:

    1. the bank view's VIEW_MANIFEST names an entry for the cell whose ``object_sha256`` is the pin (and whose
       ``staged_id`` is the row's, when the row carries one);
    2. the object's BYTES hash to the pin — a mismatch names the lane and the target and, when the served bytes are a
       known stale object (the superseded direct-fit parent, a dose bake of it), says so by label;
    3. the sidecar OF THE KIND sits beside the object (and the other kind's does not), records the pin as its
       ``object_sha256`` and, for a derived object, ``derived_id == cell`` and parents EXACTLY ``parents_expected``;
    4. when the pin file carries (target, staged id), its sha is the pin too.
    Disk == manifest != pin is reported as a PIN / record disagreement (the bank is self-consistent; the comparator's
    inputs are what differ), disk != manifest as a torn bank — so the reader of the refusal knows which side to fix."""
    stale = dict(stale or {})
    label = tuple_label(lane, model_key, view, cell)
    rec: dict[str, Any] = {"tuple": [lane, model_key, view, cell], "pin": pin, "view_dir": str(view_dir),
                           "hashes": {"pin": pin, "manifest": None, "disk": None, "sidecar": None, "pins_file": None},
                           "problems": [], "verdict": None}
    problems: list[str] = rec["problems"]
    if pin is None:
        rec["verdict"] = VERDICT_HELD
        problems.append(f"{label}: no pin — the row is HELD; nothing binds")
        return rec
    cell_dir = cell_dir or cell
    d = Path(view_dir) / cell_dir
    man_path = Path(view_dir) / VIEW_MANIFEST
    man = _load_json(man_path) if man_path.is_file() else None
    entry = None
    if not man_path.is_file():
        problems.append(f"{label}: {man_path} is absent — not a built bank view (a raw fits store `cells_all/` is not a serving bank)")
    elif not isinstance(man, Mapping) or not isinstance(man.get("entries"), list):
        problems.append(f"{label}: {man_path} is not a VIEW_MANIFEST document")
    else:
        matches = [e for e in man["entries"] if isinstance(e, Mapping) and str(e.get("cell_id")) in (cell, cell_dir)]
        if not matches:
            problems.append(f"{label}: the bank view's manifest names no entry for cell {cell!r} (entries: "
                            f"{[str(e.get('cell_id')) for e in man['entries'] if isinstance(e, Mapping)]})")
        else:
            entry = matches[0]
            msha = entry.get("object_sha256")
            rec["hashes"]["manifest"] = msha
            rec["manifest_staged_id"] = entry.get("staged_id")
            if staged_id and entry.get("staged_id") != staged_id:
                other = other_lane_of(entry.get("staged_id"))
                problems.append(f"REFUSED {label}: the bank entry names staged id {entry.get('staged_id')!r}, the row binds {staged_id!r}"
                                + (f" — that id is {other}" if other else "") + f"{_stale_label(msha, stale)}")
    # the cell dir of a full22 view is a symlink `../../cells/<staged id>`: its target's name is a second reading of the staged id
    if staged_id and d.is_symlink():
        target = Path(os.readlink(d)).name
        if target != staged_id:
            other = other_lane_of(target)
            problems.append(f"REFUSED {label}: the cell dir resolves to staged id {target!r}, the row binds {staged_id!r}"
                            + (f" — that id is {other}" if other else ""))
    obj = d / OBJECT_FILE
    if not obj.is_file():
        problems.append(f"{label}: {obj} is absent")
    else:
        disk = sha256_of(obj)
        rec["hashes"]["disk"] = disk
        msha = rec["hashes"]["manifest"]
        if disk != pin:
            if msha is not None and msha == disk:
                problems.append(f"REFUSED {label}: the bank is self-consistent on ANOTHER object — manifest and disk both say "
                                f"{disk[:12]}…, the pin is {pin[:12]}…{_stale_label(disk, stale)}; a leg would SERVE {disk[:12]}… while "
                                "reporting the pin. Not a bank defect: the record's pin and the bank's entry disagree — re-stage or re-pin")
            else:
                problems.append(f"REFUSED {label}: {obj.name} hashes to {disk[:12]}…, the pin is {pin[:12]}…{_stale_label(disk, stale)}"
                                + (f"; the manifest says {str(msha)[:12]}… (a TORN bank: manifest and disk disagree)" if msha is not None and msha != disk else ""))
        elif msha is not None and msha != pin:
            problems.append(f"REFUSED {label}: disk == pin {pin[:12]}… but the manifest entry says {str(msha)[:12]}…{_stale_label(msha, stale)}"
                            " — a torn bank (a re-pointed cell dir under a stale manifest); rebuild the view")
    want = SIDECAR_BY_KIND[object_kind]
    others = [n for k, n in SIDECAR_BY_KIND.items() if k != object_kind and (d / n).is_file()]
    if d.is_dir():
        if not (d / want).is_file():
            problems.append(f"{label}: no {want} beside the object — a {object_kind} object is proven through its {want} (the mirror's rule)")
        if others:
            problems.append(f"{label}: the dir carries {others} beside a {object_kind} object's {want} — a dir declares ONE identity")
        side = _load_json(d / want) if (d / want).is_file() else None
        if isinstance(side, Mapping):
            ssha = side.get("object_sha256")
            rec["hashes"]["sidecar"] = ssha
            if ssha != pin:
                problems.append(f"{label}: {want} records object_sha256 {str(ssha)[:12]}…, the pin is {pin[:12]}…{_stale_label(ssha, stale)}"
                                " — the sidecar describes another object than the record pins")
            disk_sha = rec["hashes"]["disk"]
            if disk_sha is not None and ssha != disk_sha:
                # the sidecar describes bytes that are not beside it (a re-pointed cell dir, a re-bake in place): torn on ITS axis
                problems.append(f"{label}: {want} records object_sha256 {str(ssha)[:12]}… but the bytes beside it hash {disk_sha[:12]}… — a stale "
                                "object/sidecar pair (the sidecar describes bytes that are not there)")
            if object_kind == OBJECT_KIND_DERIVED:
                if side.get("derived_id") != cell:
                    problems.append(f"{label}: {want} records derived_id {side.get('derived_id')!r}, not this cell — a derived object is bound to the cell it was built AS")
                got = {str(b.get("cell_id")): str(b.get("object_sha256")) for k, b in side.items() if k.startswith("parent_") and isinstance(b, Mapping)}
                rec["derived_parents"] = got
                if parents_expected is None:
                    problems.append(f"{label}: a derived row names its two expected parents; the record carries none")
                elif got != dict(parents_expected):
                    stale_parents = [f"{c}={s[:12]}…{_stale_label(s, stale)}" for c, s in got.items() if s in stale]
                    problems.append(f"REFUSED {label}: the derived object's parents {({c: s[:12] + '…' for c, s in got.items()})} are not exactly "
                                    f"the lane's two it-doc parents {({c: s[:12] + '…' for c, s in parents_expected.items()})} — a same-named derived "
                                    "object of ANOTHER lane (the base-lane subtraction) or a stale bake" + (f" ({'; '.join(stale_parents)})" if stale_parents else ""))
    if pins_doc is not None and staged_id:
        psha = pins_file_sha(pins_doc, target=model_key, staged_id=staged_id)
        rec["hashes"]["pins_file"] = psha
        if psha is None:
            problems.append(f"{label}: the pin file carries no entry for (target {model_key}, staged id {staged_id}) — a served object is pinned there before its view builds")
        elif psha != pin:
            problems.append(f"REFUSED {label}: the pin file's (target {model_key}, staged id {staged_id}) sha {psha[:12]}… != the record's pin {pin[:12]}…"
                            f"{_stale_label(psha, stale)} — two pins for one object; the record and the pin file disagree")
    rec["verdict"] = VERDICT_REFUSED if problems else VERDICT_PASS
    return rec


def bind_verdict_line(rec: Mapping[str, Any]) -> str:
    lane, key, view, cell = rec["tuple"]
    h = rec["hashes"]
    short = lambda s: (str(s)[:12] + "…") if s else "none"
    head = (f"ITDOC_BIND: {rec['verdict']} {tuple_label(lane, key, view, cell)} pin={short(h['pin'])} manifest={short(h['manifest'])} "
            f"disk={short(h['disk'])} sidecar={short(h['sidecar'])} pins_file={short(h['pins_file'])}")
    return head if not rec["problems"] else head + "\n" + "\n".join(f"  - {p}" for p in rec["problems"])


def row_bind(row: Mapping[str, Any], shared_root: Path, *, cluster_role: str, pins_doc: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """:func:`bind_problems` for a MATRIX ROW (either instrument's shape) against ``<shared root>/<bank_view_rel[role]>``;
    a held row (no pin) or a row with no view for the role reads HELD / REFUSED without touching the disk."""
    lane, key, view, cell = str(row.get("lane")), str(row["model_key"]), str(row["view"]), str(row["cell"])
    pin = row.get("object_sha256")
    held = str(row.get("status") or "") == "held" or bool(str(row.get("held_reason") or "").strip())
    if pin is None or held:
        # a HELD row (the record's hold, or no pin yet) is a design status, not a refusal: nothing binds, nothing is hashed
        reason = str(row.get("held_reason") or "").strip() or "no pin — the row is HELD; nothing binds"
        return {"tuple": [lane, key, view, cell], "pin": pin, "view_dir": None,
                "hashes": {"pin": pin, "manifest": None, "disk": None, "sidecar": None, "pins_file": None},
                "problems": [f"HELD: {reason}"], "verdict": VERDICT_HELD}
    rel = (row.get("bank_view_rel") or {}).get(cluster_role)
    if not rel:
        rec = {"tuple": [lane, key, view, cell], "pin": pin, "view_dir": None, "hashes": {"pin": pin, "manifest": None, "disk": None, "sidecar": None, "pins_file": None},
               "problems": [f"{tuple_label(lane, key, view, cell)}: no bank view for the {cluster_role} cluster role — the object is staged elsewhere; run it where it is staged"],
               "verdict": VERDICT_REFUSED}
        return rec
    return bind_problems(Path(shared_root) / rel, lane=lane, model_key=key, view=view, cell=cell, pin=pin, object_kind=str(row["object_kind"]),
                         staged_id=row.get("staged_id"), cell_dir=row.get("bank_cell_dir"), parents_expected=row.get("parents_expected"),
                         stale=row.get("stale_objects") or {}, pins_doc=pins_doc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bind-check", help="ONE row's binding at the bank door (the launchers call this before the standup)")
    p.add_argument("--admission", type=Path, default=ADMISSION_PATH)
    for f in ("lane", "key", "view", "cell"):
        p.add_argument(f"--{f}", required=True)
    p.add_argument("--view-dir", type=Path, required=True, help="<shared root>/<bank_view_rel[role]>")
    p.add_argument("--cell-dir", default=None)
    p.add_argument("--pin", required=True, help="the row's object sha256 (must equal the record's)")
    p.add_argument("--pins", type=Path, default=None, help="the full22 pin file (its (target, staged id) entry is the fourth hash)")
    p = sub.add_parser("audit", help="every entry of the record against a shared root: PASS / HELD / REFUSED per cell")
    p.add_argument("--admission", type=Path, default=ADMISSION_PATH)
    p.add_argument("--shared-root", type=Path, required=True)
    p.add_argument("--cluster-role", choices=("primary", "overflow"), required=True)
    p.add_argument("--pins", type=Path, default=None)
    a = ap.parse_args(argv)
    adm = load_admission(a.admission)
    pins_doc = _load_json(a.pins) if a.pins else None
    if a.cmd == "bind-check":
        entry = entry_for(adm, a.lane, a.key, a.view, a.cell)
        if entry is None:
            print(f"REFUSED {tuple_label(a.lane, a.key, a.view, a.cell)}: no entry in {a.admission.name} — a re-fit row never exists without its record")
            return 3
        if entry["object_sha256"] != a.pin:
            print(f"REFUSED {tuple_label(a.lane, a.key, a.view, a.cell)}: the row's pin {a.pin[:12]}… != the record's {str(entry['object_sha256'])[:12]}…")
            return 3
        bank = entry.get("bank") or {}
        rec = bind_problems(a.view_dir, lane=a.lane, model_key=a.key, view=a.view, cell=a.cell, pin=a.pin, object_kind=entry["object_kind"],
                            staged_id=entry.get("staged_id"), cell_dir=a.cell_dir or bank.get("bank_cell_dir"), parents_expected=entry.get("parents_expected"),
                            stale=entry.get("stale_objects") or {}, pins_doc=pins_doc)
        print(bind_verdict_line(rec))
        return 0 if rec["verdict"] == VERDICT_PASS else 3
    if a.cmd == "audit":
        worst = 0
        for (lane, key, view, cell), entry in adm["entries"].items():
            launch = entry_launchable(entry)          # the record's status: launchable, else HELD with its reason (never REFUSED for being unstaged)
            row = {"lane": lane, "model_key": key, "view": view, "cell": cell, "object_sha256": entry["object_sha256"], "object_kind": entry["object_kind"],
                   "bank_view_rel": (entry.get("bank") or {}).get("bank_view_rel"), "staged_id": entry.get("staged_id"),
                   "bank_cell_dir": (entry.get("bank") or {}).get("bank_cell_dir"), "parents_expected": entry.get("parents_expected"),
                   "stale_objects": entry.get("stale_objects") or {},
                   "status": "launch" if launch else "held", "held_reason": None if launch else str(entry.get("hold_reason") or "")}
            rec = row_bind(row, a.shared_root, cluster_role=a.cluster_role, pins_doc=pins_doc)
            print(bind_verdict_line(rec))
            worst = max(worst, 0 if rec["verdict"] == VERDICT_PASS else 1 if rec["verdict"] == VERDICT_HELD else 3)
        return worst
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
