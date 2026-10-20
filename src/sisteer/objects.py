"""A human-readable address for the committed paper objects (`objects/MANIFEST.json`).

An object is addressed as ``(model_id, paper_label[, dose][, seed][, kind][, derivation])`` instead of its 64-hex
``object_id``::

    from sisteer.objects import resolve
    resolve("qwen35_4b_it", "difference_lora")                   # the LoRA constitutional adapter, full dose
    resolve("qwen35_4b_it", "difference_lora", dose=0.5)         # its dose-scaled copy at 0.5
    resolve("qwen35_4b_base", "constitution_lora", seed=2)       # the effective-batch-64 training-seed replicate

`dose` and `seed` are lifted from each object's sidecars, never from the manifest alone: the dose coefficient from
``dose_scaled_from.scale`` (``train_budget_meta.json`` on fits and their copies, ``derived_meta.json`` on differences
and theirs), or the refusal direction's raw coefficient ``c`` (``fit_provenance.json``); a fit or derivation that is
not a dose-scaled copy is dose 1.0, and a manifest-only (virtual) row has none. The seed is the fit's own
``train_budget_meta.json`` ``seed`` (else ``fit_record.realized_batch_check.seed``); a difference takes the seed of
its parents through ``derived_meta.json`` (``parent_pos`` / ``parent_neg`` must agree), and a dose-scaled copy its
parent's. The canonical fit of a label (on ``qwen35_4b_base`` the effective-batch-8 fit; elsewhere the only one)
records no seed, and resolves when ``seed`` is left out.

The address is well-defined because ``(model_id, paper_label, object_kind, derivation, dose, seed)`` is unique over
the manifest (``tests/test_objects_resolve.py`` pins that); `resolve` refuses by name, listing the candidates with
their dose, seed and effective batch, whenever a request still matches several objects or none.

`serve_checkpoint` and `fit_checkpoint` give the ``(hf_repo, revision)`` of the checkpoint the paper serves a model's
objects on, and of the one they were fit on, read from ``reproduce/models.yaml`` through its one reader
(``reproduce/lib/models.py``, loaded by path). The model is a table key (``qwen35_4b``) or the registry ``model_id``
of either of its checkpoints (``qwen35_4b_base``, ``qwen35_4b_it``: the names ``objects/`` uses); a ``_base`` row's
objects are fit on the base checkpoint and served on the post-trained one (``transfer_kind: base_to_instruct``), so
``serve_checkpoint("qwen35_4b_base")`` is the pinned Qwen/Qwen3.5-4B. An unknown model, a row without a pinned 40-hex
revision, and a checkout without the table or its reader are refused by name.

The address side is pure stdlib and reads the manifest and sidecars once per repository root (`load_index`); the
checkpoint side reads the models table once per root (PyYAML, through the reader); neither opens the registry. The
root is an explicit argument, else the checkout this package is imported from (``<root>/src/sisteer/objects.py``),
else the working directory.

A SPARSE checkout (the README's lighter checkouts hold a few object directories under the whole manifest) indexes
every manifest row but marks the rows whose directory is absent ``present=False``, with no dose or seed (their sidecars
are not here to read; the listing shows ``?``); `resolve` never returns one. A refusal names what is missing: when only
absent objects answer an address, or a dose or seed matches none of the present ones, the absent rows are listed with
the ``git sparse-checkout add`` that fetches one. A difference whose parents' directories are absent has no readable
seed either (the seed is the parents'); its listing names those directories, and it resolves when ``seed`` is left out
or by ``object_id`` alone.

A WITHHELD object is one the paper evaluates that this repository does not ship (``objects/WITHHELD.json``, which the
manifest names as ``withheld_list``): it has no directory and no manifest row, the records that name it keep its id, and
every lookup that reaches it refuses by name with the reason the list records (`WithheldObjectError`, a ValueError):
`resolve` at an address only withheld objects answer, `ObjectIndex.get` by id, and the committed-object loader
(``sisteer.extraction.committed_objects``). `find` lists shipped objects only; `withheld` lists the withheld rows, and the
CLI's ``find`` names the ones its filters match on stderr. A manifest that names a withheld list absent from the checkout
is refused by name (a cone-mode sparse checkout brings it beside ``MANIFEST.json``).

CLI::

    python -m sisteer.objects find --model qwen35_4b_it --label difference_lora [--dose 0.5] [--seed 2] [--kind lora]
    python -m sisteer.objects resolve qwen35_4b_it difference_lora [--dose 0.5] [--seed 2] [--kind lora]
    python -m sisteer.objects checkpoint qwen35_4b_base [--fit]       # "<hf_repo> <revision>": served on (default) / fit on
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MANIFEST_REL = Path("objects") / "MANIFEST.json"
#: the paper objects this repository does not ship, beside the manifest (which names it as ``withheld_list``)
WITHHELD_REL = Path("objects") / "WITHHELD.json"
MODELS_TABLE_REL = Path("reproduce") / "models.yaml"
#: the table's ONE reader (strict: duplicate keys refused), loaded by path so this module never sys.path-inserts reproduce/lib
MODELS_READER_REL = Path("reproduce") / "lib" / "models.py"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
#: Dose coefficients are decimal literals in the sidecars (0.1, 0.25, 0.75, 1.25, -0.5 ...); this only guards float noise.
_DOSE_TOL = 1e-9
_UNSET = object()


class WithheldObjectError(ValueError):
    """A paper object this repository does not ship (``objects/WITHHELD.json``): refused by name, with the reason the list
    records, wherever a lookup or a load reaches it."""


def read_withheld(path: str | Path) -> dict[str, dict]:
    """``object_id -> row`` of a withheld list (``objects/WITHHELD.json``: ``{objects: [{object_id, model_id, paper_label,
    reason, ...}]}``); empty when the file is absent. A malformed list, a row missing one of those four fields, or an id
    listed twice is refused by name."""
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    rows = doc.get("objects") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected {{objects: [...]}}")
    out: dict[str, dict] = {}
    for r in rows:
        missing = [k for k in ("object_id", "model_id", "paper_label", "reason") if not (isinstance(r, dict) and r.get(k))]
        if missing:
            raise ValueError(f"{path}: a withheld row lacks {', '.join(missing)}: {r!r}")
        if str(r["object_id"]) in out:
            raise ValueError(f"{path}: object {r['object_id']} is listed twice")
        out[str(r["object_id"])] = r
    return out


def withheld_message(row: dict) -> str:
    """The refusal every reader of ``objects/`` gives for a withheld object: its id, its address and the recorded reason."""
    return (f"object {row['object_id']} ({row['model_id']} {row['paper_label']}) is withheld from this repository: "
            f"{row['reason']} ({WITHHELD_REL.as_posix()})")


@dataclass(frozen=True)
class ObjectEntry:
    """One manifest row with its sidecar-lifted dose and seed."""

    object_id: str
    model_id: str | None
    label: str
    kind: str
    role: str
    derivation: str
    form: str
    dir: str | None
    campaign: str | None
    dose: float | None
    seed: int | None
    effective_batch: int | None
    #: The object's directory is in this checkout (a virtual row has none and counts as present). False in a sparse
    #: checkout for a row whose directory was not checked out: its dose and seed are then None, unread rather than absent.
    present: bool = True
    #: The seed was read from sidecars present here. False where it could not be: the row is absent, or it is a
    #: difference (or a dose-scaled copy) whose parents' directories are not in this checkout.
    seed_read: bool = True
    #: The directories under ``objects/`` the seed read needed and did not find (the row's own, or its parents'): what
    #: `git sparse-checkout add` would fetch to make the seed readable. Empty when the seed was read.
    unread_dirs: tuple[str, ...] = ()

    @property
    def key(self) -> tuple:
        """The address this entry answers to; unique over the manifest (the contract the tests pin)."""
        return (self.model_id, self.label, self.kind, self.derivation, self.dose, self.seed)

    def describe(self) -> str:
        if not self.present:
            note = "  (not in this checkout)"
        elif not self.seed_read:
            note = "  (seed unreadable: not in this checkout: " + ", ".join(f"objects/{d}" for d in self.unread_dirs) + ")"
        else:
            note = ""
        return (f"{self.object_id[:12]}  kind={self.kind} derivation={self.derivation} "
                f"dose={_fmt(self.dose) if self.present else '?'} seed={_fmt(self.seed) if self.seed_read else '?'} "
                f"effective_batch={_fmt(self.effective_batch)}{note}")


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and v == int(v):
        return f"{v:.1f}"
    return str(v)


def _get(d: Any, *path: str) -> Any:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def find_repo_root(root: str | Path | None = None) -> Path:
    """The checkout holding ``objects/MANIFEST.json``: `root` when given, else this package's checkout, else the cwd."""
    if root is not None:
        root = Path(root).expanduser().resolve()
        if not (root / MANIFEST_REL).exists():
            raise FileNotFoundError(f"no {MANIFEST_REL} under {root}")
        return root
    for cand in (Path(__file__).resolve().parents[2], Path.cwd().resolve()):
        if (cand / MANIFEST_REL).exists():
            return cand
    raise FileNotFoundError(
        f"{MANIFEST_REL} not found beside this package ({Path(__file__).resolve().parents[2]}) or under the working "
        "directory; pass repo_root= (the clone's root)")


class ObjectIndex:
    """Every manifest row with dose and seed lifted from its sidecars. Build it through `load_index`."""

    def __init__(self, root: Path):
        self.root = root
        self.manifest_path = root / MANIFEST_REL
        with open(self.manifest_path) as f:
            manifest = json.load(f)
        self._rows: dict[str, dict] = {row["object_id"]: row for row in manifest["objects"]}
        declared = manifest.get("withheld_list")
        if declared is not None and (not isinstance(declared, str) or Path(declared).name != declared):
            raise ValueError(f"{self.manifest_path}: withheld_list {declared!r} is not a file name beside the manifest")
        self.withheld_path = root / "objects" / (declared or WITHHELD_REL.name)
        if declared and not self.withheld_path.is_file():
            raise FileNotFoundError(f"{self.manifest_path} names its withheld list {declared}, which is not in this checkout "
                                    f"({self.withheld_path}); a sparse checkout brings it with objects/MANIFEST.json")
        #: object_id -> row of the paper objects this repository does not ship (objects/WITHHELD.json)
        self.withheld: dict[str, dict] = read_withheld(self.withheld_path)
        both = sorted(set(self.withheld) & set(self._rows))
        if both:
            raise ValueError(f"{self.manifest_path} lists {', '.join(o[:12] for o in both)}, which {self.withheld_path.name} "
                             "withholds: an object is shipped or withheld, never both")
        self._sidecars: dict[str, dict[str, Any]] = {}
        self._seed_cache: dict[str, tuple[int | None, int | None, bool, tuple[str, ...]]] = {}
        self.entries: list[ObjectEntry] = [self._build(row) for row in manifest["objects"]]
        self._by_id: dict[str, ObjectEntry] = {e.object_id: e for e in self.entries}

    # ---- sidecars -------------------------------------------------------------------------------------------------
    def _sidecar(self, row: dict, name: str) -> Any:
        if not row.get("dir"):
            return None
        oid = row["object_id"]
        cache = self._sidecars.setdefault(oid, {})
        if name not in cache:
            p = self.root / "objects" / row["dir"] / name
            cache[name] = json.load(open(p)) if p.exists() else None
        return cache[name]

    def _present(self, row: dict) -> bool:
        """The row's directory is in this checkout (a virtual row has none and is always present)."""
        return not row.get("dir") or (self.root / "objects" / row["dir"]).is_dir()

    def _dose(self, row: dict) -> float | None:
        if row.get("form") == "virtual" or not self._present(row):
            return None
        for name in ("train_budget_meta.json", "derived_meta.json"):
            scale = _get(self._sidecar(row, name), "dose_scaled_from", "scale")
            if scale is not None:
                return float(scale)
        prov = self._sidecar(row, "fit_provenance.json")
        c = _get(prov, "c")
        if c is None:
            c = _get(prov, "arditi_allpos", "c")
        if c is not None:
            return float(c)
        if row["derivation"] == "dose_scale":
            raise ValueError(f"object {row['object_id']} is a dose_scale derivation but no sidecar records its scale")
        return 1.0

    def _seed_and_batch(self, oid: str, _stack: tuple[str, ...] = ()) -> tuple[int | None, int | None, bool, tuple[str, ...]]:
        """(seed, effective_batch, read, unread_dirs) of an object, through its parents when it is derived or dose-scaled;
        `read` is False when a directory the seed lives in (the object's own, or a parent's) is not in this checkout, and
        `unread_dirs` names those directories."""
        if oid in self._seed_cache:
            return self._seed_cache[oid]
        if oid in _stack or oid not in self._rows:
            return (None, None, True, ())
        row = self._rows[oid]
        if not self._present(row):
            self._seed_cache[oid] = (None, None, False, (row["dir"],))
            return self._seed_cache[oid]
        read, unread = True, []
        tb = self._sidecar(row, "train_budget_meta.json")
        seed = _get(tb, "seed")
        if seed is None:
            seed = _get(tb, "fit_record", "realized_batch_check", "seed")
        batch = _get(tb, "effective_batch")
        if batch is None:
            batch = _get(self._sidecar(row, "summary.json"), "effective_batch")
        dm = self._sidecar(row, "derived_meta.json")
        if seed is None and isinstance(dm, dict) and (dm.get("parent_pos") or dm.get("parent_neg")):
            parents = [_get(dm, side, "object_sha256") for side in ("parent_pos", "parent_neg")]
            lifted = [self._seed_and_batch(p, _stack + (oid,)) for p in parents if p]
            for _, _, r, u in lifted:
                read = read and r
                unread.extend(u)
            seeds = {s for s, _, r, _ in lifted if r}          # a parent whose directory is absent has nothing to say
            if len(seeds) > 1:
                raise ValueError(f"object {oid}: its parents record different training seeds {sorted(seeds, key=str)}")
            if seeds:
                seed = next(iter(seeds))
                batch = batch if batch is not None else next((b for _, b, r, _ in lifted if r and b is not None), None)
        if seed is None:
            parent = _get(tb or dm, "dose_scaled_from", "parent_object_sha256")
            if parent and parent != oid:
                pseed, pbatch, pread, punread = self._seed_and_batch(parent, _stack + (oid,))
                seed = pseed
                batch = batch if batch is not None else pbatch
                read = read and pread
                unread.extend(punread)
        out = (int(seed) if seed is not None else None, int(batch) if batch is not None else None, read,
               tuple(dict.fromkeys(unread)))
        self._seed_cache[oid] = out
        return out

    def _build(self, row: dict) -> ObjectEntry:
        seed, batch, read, unread = self._seed_and_batch(row["object_id"])
        return ObjectEntry(
            object_id=row["object_id"], model_id=row.get("model_id"), label=row["paper_label"], kind=row["object_kind"],
            role=row.get("object_role", ""), derivation=row["derivation"], form=row.get("form", "whole"),
            dir=row.get("dir"), campaign=row.get("campaign"), dose=self._dose(row), seed=seed, effective_batch=batch,
            present=self._present(row), seed_read=read, unread_dirs=unread)

    # ---- queries --------------------------------------------------------------------------------------------------
    def get(self, object_id: str) -> ObjectEntry:
        """The entry of a shipped object; a withheld one is refused by name (`WithheldObjectError`), an unknown id is a KeyError."""
        if object_id in self.withheld:
            raise WithheldObjectError(withheld_message(self.withheld[object_id]))
        return self._by_id[object_id]

    def find_withheld(self, *, model_id: Any = _UNSET, label: Any = _UNSET, kind: Any = _UNSET,
                      derivation: Any = _UNSET) -> list[dict]:
        """The withheld rows matching every filter given (their dose and seed are not recorded here, so neither filters)."""
        out = []
        for r in self.withheld.values():
            if model_id is not _UNSET and r.get("model_id") != model_id:
                continue
            if label is not _UNSET and r.get("paper_label") != label:
                continue
            if kind is not _UNSET and r.get("object_kind") != kind:
                continue
            if derivation is not _UNSET and r.get("derivation") != derivation:
                continue
            out.append(r)
        return sorted(out, key=lambda r: (str(r.get("model_id")), str(r.get("paper_label")), str(r["object_id"])))

    def path(self, entry: ObjectEntry) -> Path | None:
        """The object's directory under ``objects/`` (None for a virtual row)."""
        return self.root / "objects" / entry.dir if entry.dir else None

    def find(self, *, model_id: Any = _UNSET, label: Any = _UNSET, kind: Any = _UNSET, derivation: Any = _UNSET,
             dose: Any = _UNSET, seed: Any = _UNSET) -> list[ObjectEntry]:
        """Entries matching every filter given (a filter left out matches anything; ``model_id=None`` is the virtual rows)."""
        out = []
        for e in self.entries:
            if model_id is not _UNSET and e.model_id != model_id:
                continue
            if label is not _UNSET and e.label != label:
                continue
            if kind is not _UNSET and e.kind != kind:
                continue
            if derivation is not _UNSET and e.derivation != derivation:
                continue
            if dose is not _UNSET and not _dose_eq(e.dose, dose):
                continue
            if seed is not _UNSET and e.seed != (int(seed) if seed is not None else None):
                continue
            out.append(e)
        return sorted(out, key=_sort_key)

    def duplicate_keys(self) -> dict[tuple, list[ObjectEntry]]:
        """Addresses answered by more than one object (empty when the address is well-defined)."""
        by_key: dict[tuple, list[ObjectEntry]] = {}
        for e in self.entries:
            by_key.setdefault(e.key, []).append(e)
        return {k: v for k, v in by_key.items() if len(v) > 1}

    def resolve(self, model_id: str | None, label: str, *, dose: float | None = None, seed: int | None = None,
                kind: str | None = None, derivation: str | None = None) -> str:
        """The one object_id at an address; ValueError, naming the candidates, when several or none match."""
        filters: dict[str, Any] = {"model_id": model_id, "label": label}
        if kind is not None:
            filters["kind"] = kind
        if derivation is not None:
            filters["derivation"] = derivation
        cands = self.find(**filters)
        where = ", ".join(f"{k}={v!r}" for k, v in filters.items())
        if not cands:
            held = self.find_withheld(**filters)
            if held:
                raise WithheldObjectError("; ".join(withheld_message(r) for r in held))
        absent = [e for e in cands if not e.present]
        cands = [e for e in cands if e.present]
        if not cands and absent:
            raise ValueError(f"no object with {where} is in this checkout; the manifest's {len(absent)} at that address "
                             f"(a sparse checkout: `git sparse-checkout add objects/<dir>` fetches one):\n" + _listing(absent))
        if not cands:
            labels = sorted({e.label for e in self.entries if e.model_id == model_id})
            if not labels:
                models = sorted({str(e.model_id) for e in self.entries})
                raise ValueError(f"unknown model_id {model_id!r}; the manifest's model_ids: {', '.join(models)}")
            held = sorted({str(r["paper_label"]) for r in self.find_withheld(model_id=model_id)})
            raise ValueError(f"no object with {where}; paper_labels of {model_id!r}: {', '.join(labels)}"
                             + (f"; withheld from this repository ({WITHHELD_REL.as_posix()}): {', '.join(held)}" if held else ""))
        if dose is None:
            full = [e for e in cands if _dose_eq(e.dose, 1.0)]
            if full:
                cands = full
            elif len({e.dose for e in cands}) > 1:
                raise ValueError(f"no full-dose (1.0) object with {where}; pass dose= to pick one of:\n"
                                 + _listing(cands))
        else:
            at = [e for e in cands if _dose_eq(e.dose, dose)]
            if not at:
                doses = sorted({e.dose for e in cands if e.dose is not None})
                raise ValueError(f"no object with {where} at dose {dose}; doses present: "
                                 f"{', '.join(_fmt(d) for d in doses) or 'none'}:\n" + _listing(cands) + _sparse_note(absent))
            cands = at
        if seed is None:
            if len(cands) > 1:
                canonical = [e for e in cands if e.seed is None]
                if len(canonical) != 1:
                    canonical = [e for e in cands if e.effective_batch == 8]
                if len(canonical) == 1:
                    cands = canonical
        else:
            at = [e for e in cands if e.seed == int(seed)]
            if not at:
                raise ValueError(f"no object with {where} at seed {seed}; pass one of:\n" + _listing(cands)
                                 + _sparse_note(absent, unread=[e for e in cands if not e.seed_read]))
            cands = at
        if len(cands) != 1:
            raise ValueError(f"ambiguous: {len(cands)} objects match {where}"
                             + (f" dose={_fmt(dose)}" if dose is not None else "")
                             + "; pass dose=, seed=, kind= or derivation= to pick one of:\n" + _listing(cands))
        return cands[0].object_id


def _dose_eq(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) < _DOSE_TOL


def _sort_key(e: ObjectEntry) -> tuple:
    return (e.model_id or "", e.label, e.kind, e.derivation, e.dose if e.dose is not None else float("-inf"),
            e.seed if e.seed is not None else -1, e.object_id)


def _listing(entries: Iterable[ObjectEntry]) -> str:
    return "\n".join("  " + e.describe() for e in entries)


def _sparse_note(absent: list[ObjectEntry], unread: list[ObjectEntry] = ()) -> str:
    """The lines a refusal adds in a sparse checkout (empty in a full one): the present candidates whose seed could not be
    read, with the directories that would make it readable, and the manifest's rows at the address whose directories are
    not here, each with the `git sparse-checkout add` that fetches what is missing."""
    out = []
    if unread:
        dirs = list(dict.fromkeys(d for e in unread for d in e.unread_dirs))
        out.append(f"  the seed of {', '.join(e.object_id[:12] for e in unread)} is unreadable in this checkout: leave seed= "
                   f"out, or fetch it with `git sparse-checkout add {' '.join('objects/' + d for d in dirs)}`")
    if absent:
        out.append("  not in this checkout (dose and seed unread until fetched; `git sparse-checkout add objects/<dir>`):")
        out.append(_listing(absent))
    return "\n" + "\n".join(out) if out else ""


_INDEX: dict[Path, ObjectIndex] = {}


def load_index(repo_root: str | Path | None = None) -> ObjectIndex:
    """The index of one checkout, built once per root and cached."""
    root = find_repo_root(repo_root)
    if root not in _INDEX:
        _INDEX[root] = ObjectIndex(root)
    return _INDEX[root]


def resolve(model_id: str | None, label: str, *, dose: float | None = None, seed: int | None = None,
            kind: str | None = None, derivation: str | None = None, repo_root: str | Path | None = None) -> str:
    """The object_id at ``(model_id, label[, dose][, seed][, kind][, derivation])``; see the module docstring."""
    return load_index(repo_root).resolve(model_id, label, dose=dose, seed=seed, kind=kind, derivation=derivation)


def find(repo_root: str | Path | None = None, **filters: Any) -> list[ObjectEntry]:
    """Entries matching the filters (``model_id``, ``label``, ``kind``, ``derivation``, ``dose``, ``seed``)."""
    return load_index(repo_root).find(**filters)


def withheld(repo_root: str | Path | None = None, **filters: Any) -> list[dict]:
    """The rows of ``objects/WITHHELD.json`` matching the filters (``model_id``, ``label``, ``kind``, ``derivation``): the paper
    objects this repository does not ship, each with the reason."""
    return load_index(repo_root).find_withheld(**filters)


# ---- checkpoints (reproduce/models.yaml) -------------------------------------------------------------------------------
#: per root: every name a table row answers to (its key, its fit checkpoint's model_id, its serve checkpoint's) -> (key, row)
_TABLE: dict[Path, dict[str, tuple[str, dict]]] = {}


def _models_table(root: Path) -> dict[str, dict]:
    """``reproduce/models.yaml`` through ``reproduce/lib/models.py`` loaded by path; a checkout missing either is refused by name."""
    reader, table = root / MODELS_READER_REL, root / MODELS_TABLE_REL
    for path, what in ((reader, "the models table's reader"), (table, "the models table")):
        if not path.is_file():
            raise FileNotFoundError(f"{path}: {what} is absent from this checkout; the checkpoints the paper fits and serves "
                                    "a model's objects on cannot be read")
    spec = importlib.util.spec_from_file_location("sisteer_objects_models_table", str(reader))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.load_models(str(table))


def _table_rows(root: Path) -> dict[str, tuple[str, dict]]:
    if root not in _TABLE:
        names: dict[str, tuple[str, dict]] = {}
        for key, row in _models_table(root).items():
            for name in {str(key), *(str(_get(row, side, "checkpoint", "model_id") or "") for side in ("fit", "serve"))} - {""}:
                if name in names and names[name][0] != key:
                    raise ValueError(f"{MODELS_TABLE_REL}: {name!r} names both rows {names[name][0]!r} and {key!r}")
                names[name] = (key, row)
        _TABLE[root] = names
    return _TABLE[root]


def _checkpoint(model: str, side: str, repo_root: str | Path | None) -> tuple[str, str]:
    root = find_repo_root(repo_root)
    rows = _table_rows(root)
    if model not in rows:
        raise ValueError(f"unknown model {model!r}; {MODELS_TABLE_REL} names (table keys and the model_ids of their fit and "
                         f"serve checkpoints): {', '.join(sorted(rows))}")
    key, row = rows[model]
    ck = _get(row, side, "checkpoint")
    hf_repo, revision = _get(ck, "hf_repo"), _get(ck, "revision")
    if not hf_repo or not isinstance(hf_repo, str):
        raise ValueError(f"{MODELS_TABLE_REL}: {key}.{side}.checkpoint.hf_repo is missing")
    revision = str(revision) if revision is not None else ""      # as the reader's own `check` reads it
    if not _HEX40.match(revision):
        raise ValueError(f"{MODELS_TABLE_REL}: {key}.{side}.checkpoint.revision {revision!r} is not a pinned 40-hex sha")
    return hf_repo, revision


def serve_checkpoint(model: str, repo_root: str | Path | None = None) -> tuple[str, str]:
    """``(hf_repo, revision)`` of the checkpoint the paper SERVES this model's objects on (``reproduce/models.yaml``
    ``serve.checkpoint``). ``model`` is a table key or the registry ``model_id`` of either of the row's checkpoints: a
    ``_base`` row's adapters are fit on the base checkpoint and served on the post-trained one, so
    ``serve_checkpoint("qwen35_4b_base")`` is Qwen/Qwen3.5-4B at its pin. Refuses an unknown model by name."""
    return _checkpoint(model, "serve", repo_root)


def fit_checkpoint(model: str, repo_root: str | Path | None = None) -> tuple[str, str]:
    """``(hf_repo, revision)`` of the checkpoint this model's objects were FIT on (``fit.checkpoint``); the same as
    `serve_checkpoint` for a row fit and served on one checkpoint (an ``_it`` row's objects, Nemotron)."""
    return _checkpoint(model, "fit", repo_root)


# ---- CLI --------------------------------------------------------------------------------------------------------------
def _table(entries: list[ObjectEntry]) -> str:
    cols = ("object_id", "label", "kind", "derivation", "dose", "seed", "dir")
    rows = [(e.object_id, e.label, e.kind, e.derivation, _fmt(e.dose) if e.present else "?",
             _fmt(e.seed) if e.seed_read else "?", (e.dir or "-") + ("" if e.present else " (absent)")) for e in entries]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    return "\n".join([fmt.format(*cols)] + [fmt.format(*r) for r in rows])


def _add_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dose", type=float, default=None, help="dose coefficient (1.0 is the fit itself)")
    p.add_argument("--seed", type=int, default=None, help="training seed of a replicate")
    p.add_argument("--kind", default=None, help="object_kind (lora, lora_subtraction, relu_steer, composed, ...)")
    p.add_argument("--derivation", default=None, help="derivation (none, dose_scale, lora_subtraction, composition)")
    p.add_argument("--root", default=None, help="checkout root (default: this package's checkout, else the cwd)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sisteer.objects", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find", help="list the objects matching the filters as a table")
    f.add_argument("--model", default=_UNSET, help="model_id (qwen35_4b_it, gemma4e4b_base, ...); 'none' = virtual rows")
    f.add_argument("--label", default=_UNSET, help="paper_label (difference_lora, constitution_relu, ...)")
    _add_filters(f)
    r = sub.add_parser("resolve", help="print the one object_id at an address")
    r.add_argument("model", help="model_id; 'none' for the virtual rows")
    r.add_argument("label", help="paper_label")
    _add_filters(r)
    c = sub.add_parser("checkpoint", help="print '<hf_repo> <revision>' of the checkpoint the paper serves a model's objects on")
    c.add_argument("model", help="reproduce/models.yaml key (qwen35_4b) or a checkpoint's model_id (qwen35_4b_base, qwen35_4b_it)")
    c.add_argument("--fit", action="store_true", help="the checkpoint the objects were fit on instead of the served one")
    c.add_argument("--root", default=None, help="checkout root (default: this package's checkout, else the cwd)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "checkpoint":
            hf_repo, revision = (fit_checkpoint if a.fit else serve_checkpoint)(a.model, repo_root=a.root)
            print(f"{hf_repo} {revision}")
            return 0
        idx = load_index(a.root)
        if a.cmd == "find":
            filters: dict[str, Any] = {}
            if a.model is not _UNSET:
                filters["model_id"] = None if a.model == "none" else a.model
            if a.label is not _UNSET:
                filters["label"] = a.label
            for k in ("dose", "seed", "kind", "derivation"):
                if getattr(a, k) is not None:
                    filters[k] = getattr(a, k)
            entries = idx.find(**filters)
            print(_table(entries))
            if a.dose is None and a.seed is None:                # a withheld row's dose and seed are not recorded here
                for r in idx.find_withheld(**{k: v for k, v in filters.items() if k in ("model_id", "label", "kind", "derivation")}):
                    print(f"withheld: {withheld_message(r)}", file=sys.stderr)
            return 0 if entries else 1
        oid = idx.resolve(None if a.model == "none" else a.model, a.label, dose=a.dose, seed=a.seed, kind=a.kind,
                          derivation=a.derivation)
        print(oid)
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
