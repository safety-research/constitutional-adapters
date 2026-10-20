"""The object banks → ``object`` rows.

ONE ``object`` row per distinct ``artifact_sha256``: ``discover()`` hashes every weights file it finds under the
declared roots, the comparison GROUP of every input is that sha (a directory without a local blob groups
under the sha its sidecars record), so every directory holding the same bytes lands in one group and
``artifact_locations_json`` accumulates every location — a bank listing is a location, never a second object.
The richest sidecar set (a fit record over a promotion record over a bare blob) is the primary source; the
twins contribute locations and their sidecar files. A dose ladder's 1.0 ANCHOR (``d1__<subtraction>``: the subtraction's own
bytes re-served, ``dose_scaled_from.anchor`` true, parent sha == own sha) is one such twin of its ``derived__`` original and
wins primary by sidecar count; it reads as the SUBTRACTION (parents = the two fits), never as a dose copy of itself
(``fit_sidecars._subtraction_anchor``), and its ``fit_effective_batch`` resolves through the parents' fit records in the
banks (``fit_records_by_sha``). A dose COPY's ``fit_effective_batch`` is its PARENT's effective batch, stated ``derived`` from ONE
source: the parent's fit record copied beside the copy, the copy's own ``train_budget_meta.json``
(``fit_sidecars._dose_batch_from_parent``, switched on here with ``dose_batch_from_parent``), and NULL by name when that record names no
integer effective batch or the copy carries none -- never the bare NULL beside the parent's block that the mirror used to leave. The
record is a file of the copy's own group, so the copy's revision covers it as it covers every other sidecar: nothing of another group's
directory is read for it, and whether the copied record agrees with the parent's own registry row
is ``reconcile fit-effective-batch``'s read-only cross-check, which lists a disagreeing copy by name and writes nothing for it.

Only what is on THIS machine is read (C19: cluster and Modal banks need a CPU srun per cluster and are
recorded as locations from the staging records — ``_records/*.json``, ``staging_manifest.json``,
``configs/sdfbase_full22_pins.json`` — without being read). Weights are opened with
``torch.load(weights_only=True)`` on the CPU; a blob that is not local leaves its blob-only fields
``unverified`` with reason ``blob_not_local``.

``status`` is NOT the bank's to decide. The bank is a MIRROR of the objects; whether an object
is ``approved`` or the paper's ``reference`` is its producer's or a ruling's statement, and the ``object`` fold is per-column
latest-statement, so a bank record stating a default overwrote whatever the producer said (three paper reference objects
were demoted by a re-read). The adapter therefore files a ``status`` only when the bundle STATES one
(``fit_sidecars.bank_status_statement``: a directory under ``_withdrawn/`` or ``_superseded_by_it64/``, a ``superseded_by.json``
naming this object's fit fingerprint) and OMITS the column otherwise: an ``object`` record may omit ``status``; the fold keeps the
folded value and its stamp when a row
exists, and the harvester dependency-blocks the record on its own row -- never a refusal -- when none does, so the bank's mirror
waits for the producer's statement instead of guessing it. ``extra_json.status_unstated`` says why the bank states none. No
registry read is needed for it, so a read with no connection publishes the same records.

The same rule holds for EVERY producer-owned column: the bank states nothing it did not
read verbatim. ``fit_code_sha`` is omitted whenever its value would be the bank default (``unverified``: no sidecar records the fit code
sha) and ``derivation_params_json`` unless it is a sidecar's own ``params`` block verbatim (``derived_meta.json:params``, the field the
producer-side writer copies; the all-positions dose record's ``params``) -- the dicts ``fit_sidecars.build_object_payload`` assembles from a
``dose_scaled_from`` stamp or a ``derived_meta.json`` record are abbreviations of the producer's record (five blocked bank
revisions for new dose copies stated ``fit_code_sha='unverified'``, and eight of the stamp's twenty-one keys are keys where the producer's native
records state a real sha and thirteen keys of their own), and a bank revision read AFTER those native records fold would overwrite them
column by column. ``extra_json.producer_owned_unstated`` names each omitted column and why; an anchor's stamp, evidence the assembled
params carried, rides ``extra_json.dose_anchor.stamp`` instead.

Roots reachable from the grading host (defaults; every one optional and skipped when absent):

* ``$STAGE/sdfbase/full22_objects/<model>/cells/<staged_id>/`` (+ ``_superseded_by_it64/``, ``_withdrawn/``)
* ``$STAGE/fits/mirror/<model>/<si>__<arm>/``
* ``$GRADE/objects/<bank>/<model>/<si>__<arm>/``
* ``$HOME/wave-objects/<si_dir>/objects_{a,b,c}/<si>__<method>/``, ``diffmean_*/<…>/``, ``_controls/<…>/``, ``classB/objects_b/<method>/`` (the 2026-08 stage objects; layout ``stage``)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .. import identity
from .. import site_aliases as SA
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, store_listdir, require_store_access
from .declared_common import resolve_seeds_dir
from .fit_sidecars import (BLOB_NAMES, OBJECT_DIR_MARKERS, SCOPE_IN_SCOPE, SCOPE_PARKED, SIDECAR_NAMES, OutOfScopeObject, ScopeVerdict, Seeds,
                           SidecarBundle, SidecarRefusal, build_object_payload, load_seeds, read_bundle, read_json, scope_verdict)
from .base import site_root

HOME = os.path.expanduser("~")

#: (bank label, root path, layout) — layout names how object directories sit under the root.
DEFAULT_BANKS: Tuple[Tuple[str, str, str], ...] = (
    ("full22_objects", site_root("$STAGE/sdfbase/full22_objects"), "full22"),
    ("fits_mirror", site_root("$STAGE/fits/mirror"), "model/obj"),
    ("grade_objects", site_root("$GRADE/objects"), "bank/model/obj"),
    ("stage_objects", os.path.join(HOME, "si-steering-wave-objects"), "stage"),
)
_CAMPAIGN_BY_BANK = {"full22_objects": "sdfbase_full22", "fits_mirror": "two_si", "grade_objects": "two_si", "stage_objects": "two_si"}
#: how a verified cluster location's `bytes` was decided: the record attests the remote copy hashes to the object's sha, and this
#: machine's copy of those same bytes was measured here -- a sha256-identical file has the same length
STAGED_LOCATION_BYTES_SOURCE = "the size measured on this machine's copy of the same bytes (the record verifies the remote copy's sha256 equals the object's)"
_ROLE_FOR_FILE = {"object.pt": "object_weights", "object.safetensors": "object_weights", "COMPLETE": "completion", "metrics.jsonl": "log"}


def _is_object_dir(path: str) -> bool:
    return os.path.isdir(path) and any(os.path.isfile(os.path.join(path, m)) for m in OBJECT_DIR_MARKERS)


def _listdir(path: str) -> List[str]:
    try:
        return sorted(store_listdir(path))
    except OSError:
        return []


@dataclass
class ObjectDir:
    path: str
    bank: str
    layout: str
    target: Optional[str] = None          # the bank's model directory name
    status_hint: Optional[str] = None     # superseded_twin | withdrawn
    twin_staged_id: Optional[str] = None
    sub_bank: Optional[str] = None        # grade_objects: scale-12 | scale-12-x2 | norejects


def enumerate_object_dirs(banks: Sequence[Tuple[str, str, str]] = DEFAULT_BANKS) -> List[ObjectDir]:
    """Every object directory under the declared roots — bounded listing (never a recursive walk)."""
    out: List[ObjectDir] = []
    for bank, root, layout in banks:
        if not os.path.isdir(root):
            continue
        if layout == "full22":
            for target in _listdir(root):
                cells = os.path.join(root, target, "cells")
                if not os.path.isdir(cells):
                    continue
                for sid in _listdir(cells):
                    p = os.path.join(cells, sid)
                    if sid == "_superseded_by_it64":
                        for twin in _listdir(p):
                            tp = os.path.join(p, twin)
                            if _is_object_dir(tp):
                                # d0p5it__X → its it64 twin d0p5it64__X in the same bank
                                head, _, tail = twin.partition("__")
                                twin_id = f"{head}64__{tail}" if head.endswith("it") else None
                                out.append(ObjectDir(tp, bank, layout, target, "superseded_twin", twin_id))
                        continue
                    if sid.startswith("_"):
                        continue
                    if _is_object_dir(p):
                        out.append(ObjectDir(p, bank, layout, target))
            withdrawn = os.path.join(root, "_withdrawn")
            for sid in _listdir(withdrawn):
                p = os.path.join(withdrawn, sid)
                if _is_object_dir(p):
                    target = sid.split("__", 1)[0] if "__" in sid else None
                    out.append(ObjectDir(p, bank, layout, target, "withdrawn"))
        elif layout == "model/obj":
            for target in _listdir(root):
                tdir = os.path.join(root, target)
                if not os.path.isdir(tdir):
                    continue
                for sid in _listdir(tdir):
                    p = os.path.join(tdir, sid)
                    if _is_object_dir(p):
                        out.append(ObjectDir(p, bank, layout, target))
        elif layout == "bank/model/obj":
            for sub in _listdir(root):
                sdir = os.path.join(root, sub)
                if not os.path.isdir(sdir):
                    continue
                for target in _listdir(sdir):
                    tdir = os.path.join(sdir, target)
                    if not os.path.isdir(tdir):
                        continue
                    for sid in _listdir(tdir):
                        p = os.path.join(tdir, sid)
                        if _is_object_dir(p):
                            out.append(ObjectDir(p, bank, layout, target, sub_bank=sub))
        elif layout == "stage":
            for si_dir in _listdir(root):
                sdir = os.path.join(root, si_dir)
                if not os.path.isdir(sdir):
                    continue
                for grp in _listdir(sdir):
                    gdir = os.path.join(sdir, grp)
                    if _is_object_dir(gdir):
                        out.append(ObjectDir(gdir, bank, layout, None, sub_bank=si_dir))   # diffmean_*/<obj>, _controls/<obj>
                        continue
                    if not os.path.isdir(gdir) or not grp.startswith("objects_"):
                        continue
                    for sid in _listdir(gdir):
                        p = os.path.join(gdir, sid)
                        if not sid.startswith("_") and _is_object_dir(p):
                            out.append(ObjectDir(p, bank, layout, None, sub_bank=f"{si_dir}/{grp}"))
    return out


# --------------------------------------------------------------------------------------------
# cluster / Modal locations recorded from the staging records (never read on the clusters)
# --------------------------------------------------------------------------------------------


def staging_locations(full22_root: str, pins_path: Optional[str]) -> Dict[str, List[Dict[str, object]]]:
    """``{object_sha256: [location, …]}`` from ``_records/*.json`` and ``configs/sdfbase_full22_pins.json``.

    A staging record's ``bytes`` is the staged DIRECTORY's total (``stage_objects.py`` sums every file it moved: the weights and every
    sidecar), not the object's size (9 of 9 live records equal the directory total, 0 equal object.pt), so it rides each cluster
    location as ``staged_dir_bytes`` and ``bytes`` is left unset here; :meth:`ObjectsBankAdapter.payload_for_group` fills ``bytes`` for a
    VERIFIED copy (the record's remote sha equals the object's) from the size measured on this machine's copy of the same bytes, and leaves
    it ``None`` otherwise -- never a figure the adapter did not measure."""
    out: Dict[str, List[Dict[str, object]]] = {}
    rec_dir = os.path.join(full22_root, "_records")
    for fn in _listdir(rec_dir):
        if not fn.endswith(".json"):
            continue
        try:
            r = read_json(os.path.join(rec_dir, fn))
        except (OSError, ValueError):
            continue
        sha = r.get("object_sha256")
        if not isinstance(sha, str):
            continue
        for host in ("cluster_a", "cluster_b"):
            # a staging record written with a site's own spelling of the cluster in its keys reads through the alias table
            path = SA.get_compat(r, f"{host}_path")
            if isinstance(path, str):
                remote = SA.get_compat(r, f"{host}_remote_sha256")
                verified = SA.get_compat(r, f"{host}_verified") is True and remote == sha
                out.setdefault(sha, []).append({"uri": f"{host}:{path}", "verified_utc": r.get("pulled_at") if verified else None,
                                                "bytes": None, "staged_dir_bytes": r.get("bytes"), "source": f"_records/{fn}",
                                                "remote_sha256": remote})
        vol = r.get("volume_path")
        if isinstance(vol, str):
            out.setdefault(sha, []).append({"uri": vol, "verified_utc": None, "bytes": None, "source": f"_records/{fn}"})
    if pins_path and os.path.isfile(pins_path):
        try:
            pins = read_json(pins_path)
        except (OSError, ValueError):
            pins = {}
        for target, cells in pins.items():
            if not isinstance(cells, Mapping):
                continue
            for sid, ent in cells.items():
                if not isinstance(ent, Mapping) or not isinstance(ent.get("object_sha256"), str):
                    continue
                sha = str(ent["object_sha256"])
                src = ent.get("source")
                if isinstance(src, str):
                    out.setdefault(sha, []).append({"uri": src, "verified_utc": None, "bytes": None, "source": f"{os.path.basename(pins_path)}:{target}.{sid}",
                                                    "verified_cluster_a": ent.get("verified_cluster_a"), "verified_cluster_b": ent.get("verified_cluster_b")})
    return out


# --------------------------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------------------------


class ObjectsBankAdapter:
    name = "objects_bank"
    covers_sources: Tuple[str, ...] = ()   # objects feed no lane_watermarks source; the runs' adapters reference these rows
    #: 2 = the distilled students' paper label (difference_relu_distilled). 3 = the harvester's instance reads
    #: the checkout's seeds like every other adapter (`legacy_spellings.yaml` labels, object-kind overrides, fit-target overrides — the
    #: registered factory used to load none, so every fitted object read `other` and every paired write was refused by name), the paper
    #: label is spelled from (object_kind, campaign STEM) with the paper's arm table as the second spelling, and the stem / fit-variant
    #: prefix are parked on the object (`extra_json.campaign_stem_legacy` / `fit_variant_prefix_legacy`). 4 = a dose ladder's 1.0 ANCHOR of
    #: a LoRA subtraction (derived_meta.json from lora_subtract.py + dose_scaled_from.anchor=true, parent sha == own sha) reads as the
    #: subtraction — derivation lora_subtraction, parents = the two fits, fit_effective_batch through both parents' fit records, the
    #: stamp kept as evidence — where it used to read as a dose_scale copy whose only parent was itself (four objects, 2026-09-25).
    #: 5 = `status` is filed only where the bank states one: a bundle with no supersession or withdrawal evidence
    #: RESTATED the registry's folded status for the id, marked unstated (`extra.unstated_columns`), where 4 filed `approved` over the
    #: producers' `reference` and demoted three paper reference objects. 6 = the same bundle OMITS `status` (the rule that supersedes the
    #: restatement, 2026-10-04): the record carries no status column, no `unstated_columns` marker and no
    #: `status_origin`; `extra_json.status_unstated` names why the bank states none, and the fold keeps the producer's folded status.
    #: Accepted rows' bytes move (every unstated status leaves the record), hence the bump; the re-emission is harmless: an omitted
    #: column is no statement, so no folded status moves. 7 = (a) every producer-owned column follows the same omission rule:
    #: `fit_code_sha` is OMITTED whenever its value would be the bank default (`unverified`) and `derivation_params_json` unless it is a
    #: sidecar's own `params` block verbatim (the assembled dicts were abbreviations of the producer's record); `extra_json.producer_owned_unstated`
    #: names each omitted column, an anchor's stamp rides `extra_json.dose_anchor.stamp`; (b) a cluster location's `bytes` is the OBJECT's
    #: size there: the staging record's figure was the staged directory's total and rode the object.pt location beside its sha
    #: (9 of 9 live records); it is now `staged_dir_bytes`, `bytes` is the size measured here when the record verified the remote copy's
    #: sha (`bytes_source` says so) and None otherwise. Every bank row re-emits; no folded producer value moves (an omitted column is no
    #: statement) and `artifact_bytes` was right all along.
    #: 8 = a dose copy's `fit_writer` / `fit_finished_utc` are the copy's own dose_scaled_from stamp's `producer` /
    #: `written_at`, omitted when the stamp records neither -- never the parent's builder / built_at / writer / COMPLETE time copied beside it.
    #: 9 = a bundle of a kind ruled out of scope (`legacy_spellings.yaml: out_of_scope_object_kinds` -- the 40 synthetic Gaussian control
    #: directions under `_controls/`, the lofit / alphakl / token_appended / bidpo fits) is PARKED by name instead of refused:
    #: an accepted revision with no object record and a `source_status` fact (`extra.refuse` + `extra.parked`) citing the table, so the 48
    #: standing refusals end, the adapter stops reading REFUSED in every window and tick, and nothing enters the frame (no run references
    #: these bundles). A bump re-mints every bank id; the accepted groups re-emit byte-identically, as 6 / 7 / 8 did. The park is the
    #: GROUP's verdict: every directory holding the artifact's bytes is classified first
    #: (`fit_sidecars.scope_verdict`); all ruled out -> parked, each listed under its own spelling (`ParkedGroup`); a ruled-out twin beside an
    #: in-scope or undecided one -> a refusal naming both (`twin_scope_conflict_why`), where the richest twin's ruling used to park the group
    #: and drop the in-scope twin's object record silently. No bump of its own: status facts are not records, and no accepted record's
    #: bytes move (9 is unpublished besides).
    #: 10 = a dose COPY states `fit_effective_batch` as its PARENT's effective batch, `derived` (ruling (a) on the 92
    #: rows `reconcile fit-effective-batch` listed as ambiguous: the mirror's NULL -- "nothing was fit" -- beside the parent's recorded
    #: block, 75 `scale_lora_dose.py` and 17 `scale_psr_dose.py` copies at effective batch 8 or 64). ONE source:
    #: the parent's fit record COPIED beside the copy -- the copy's own `train_budget_meta.json`, a file of
    #: the copy's group, so its revision already covers the value's evidence and no other group's directory is read -- stated when that
    #: record names an integer in its own type, else NULL by name (`extra_json.fit_effective_batch_unresolved`: the copy carries no such
    #: record, or the record present names no `effective_batch` or a non-integer one -- a record present and naming none is a reading,
    #: never skipped); the confidence entry names the parent and the rule (`fit_sidecars._dose_batch_from_parent`,
    #: `dose_batch_from_parent=True` below). The bump is the correction: the bank's own re-emission restates the rows, and no correction
    #: record is written for them (ruling (b), keeping the NULL and having the hook's dose path omit the column too, was rejected). Every
    #: bank row re-emits; only a dose copy's `fit_effective_batch` (and its confidence entry and `extra`) moves -- from unstated to the
    #: parent's value, which the fold's per-column latest statement takes. The NULL is an explicit statement (the column present as
    #: null), never an omission, so a re-emission that cannot resolve writes NULL over an earlier derived value instead of leaving it
    #: beside a block it no longer agrees with -- and the reason's top-level key is restated every time too (`null` when the column
    #: resolved), so a copy re-emitted after its copied record was repaired does not keep "left NULL" beside the derived integer (the fold
    #: merges `extra` by key). Whether the copied record agrees with the parent's OWN registry row is the registry's
    #: read-only cross-check (`reconcile fit-effective-batch`), never this adapter's reading of another group's files.
    EMISSION_VERSION = 10
    rng_scope = "none"
    allow_symlinks = False
    priority = 0                           # The only adapter measuring `object` keys from bank directories; no tie to break

    def __init__(self, banks: Sequence[Tuple[str, str, str]] = DEFAULT_BANKS, seeds: Optional[Seeds] = None, seeds_dir: Optional[str] = None,
                 read_blobs: bool = True, pins_path: Optional[str] = None):
        self.banks = tuple(banks)
        self.roots = tuple(r for _, r, _ in self.banks)
        self.seeds = seeds
        self.seeds_dir = seeds_dir
        self.read_blobs = read_blobs
        self.pins_path = pins_path
        self._dirs: Dict[str, ObjectDir] = {}
        self._group_of_dir: Dict[str, str] = {}
        self._staging: Optional[Dict[str, List[Dict[str, object]]]] = None
        self._fit_records: Optional[Dict[str, Dict[str, object]]] = None
        #: group -> the status facts of a bundle PARKED by name: the `refuse` line that explains the empty read and the
        #: structured `parked` entry the status page renders; read by the runner through :meth:`status_extra` after the group's read
        self._parked: Dict[str, Dict[str, object]] = {}

    # -- discovery ---------------------------------------------------------------------------
    def discover(self) -> List[Input]:
        inputs: List[Input] = []
        self._dirs = {}
        self._group_of_dir = {}
        self._fit_records = None
        for od in enumerate_object_dirs(self.banks):
            self._dirs[od.path] = od
            blob_sha = None
            dir_inputs: List[Input] = []
            for name in BLOB_NAMES + SIDECAR_NAMES + ("metrics.jsonl",):
                p = os.path.join(od.path, name)
                if os.path.lexists(p):
                    role = _ROLE_FOR_FILE.get(name, "report")
                    inp = check_input(p, role=role, allow_symlink=self.allow_symlinks)
                    dir_inputs.append(inp)
                    if name in BLOB_NAMES and blob_sha is None:
                        blob_sha = inp.sha256
            if blob_sha is None:
                blob_sha = _recorded_sha(od.path) or ("nosha:" + identity.sha16(od.path))
            self._group_of_dir[od.path] = blob_sha
            inputs.extend(dir_inputs)
        return inputs

    def comparison_group(self, inp: Input) -> str:
        d = os.path.dirname(inp.uri)
        g = self._group_of_dir.get(d)
        if g is None:
            # discover() not run for this input (a direct read): group by the directory's recorded sha
            g = _recorded_sha(d) or ("nosha:" + identity.sha16(d))
            self._group_of_dir[d] = g
        return g

    # -- reading -------------------------------------------------------------------------------
    def _seeds(self, ctx: AdapterContext) -> Seeds:
        if self.seeds is not None:
            return self.seeds
        # the seeds directory is resolved the way every other adapter resolves it (an explicit argument, else $SISTEER_REGISTRY_SEEDS, else
        # this checkout's configs/registry — refusing loudly when a seed file is missing). The registered factory passes nothing, so
        # `load_seeds(None)` used to leave every label / kind / fit-target table EMPTY: 337 fitted objects read `other` and the paired
        # writes were refused for a kind the seeds do spell.
        seeds = load_seeds(resolve_seeds_dir(self.seeds_dir), conn=ctx.conn)
        legacy = ctx.legacy_spellings or {}
        if isinstance(legacy.get("si_texts"), Mapping):
            seeds.si_sha_by_id.update({str(k): str(v) for k, v in legacy["si_texts"].items()})
        if isinstance(legacy.get("model_ids"), Mapping):
            seeds.model_id_by_hf_repo.update({str(k): str(v) for k, v in legacy["model_ids"].items()})
        return seeds

    def _staging_locations(self) -> Dict[str, List[Dict[str, object]]]:
        if self._staging is None:
            full22 = next((r for b, r, _ in self.banks if b == "full22_objects"), None)
            self._staging = staging_locations(full22, self.pins_path) if full22 else {}
        return self._staging

    def fit_records_by_sha(self) -> Dict[str, Dict[str, object]]:
        """``{object sha256: {"dir", "train_budget_meta", "cell_id_legacy", "model_id", "target"}}`` for every directory in the banks
        that carries a ``train_budget_meta.json`` — the index a derived record reaches through to its parents (a subtraction anchor's
        ``fit_effective_batch``; a parent named by spelling alone). Keyed by the directory's artifact sha (``discover()``'s grouping,
        else the sidecars' recorded sha), so a fit staged twice resolves once; where twins differ in sidecars the ORIGINAL fit record
        (no ``dose_scaled_from`` stamp) is the one kept. Built once per discovery, read-only."""
        if self._fit_records is not None:
            return self._fit_records
        out: Dict[str, Dict[str, object]] = {}
        dirs = list(self._dirs.values()) or enumerate_object_dirs(self.banks)
        for od in dirs:
            path = os.path.join(od.path, "train_budget_meta.json")
            if not os.path.isfile(path):
                continue
            try:
                tbm = read_json(path)
            except (OSError, ValueError):
                continue
            sha = self._group_of_dir.get(od.path) or _recorded_sha(od.path)
            if not isinstance(sha, str) or len(sha) != 64:
                continue
            rec = {"dir": od.path, "train_budget_meta": tbm, "cell_id_legacy": tbm.get("cell_id"), "model_id": tbm.get("model_id"), "target": od.target}
            prior = out.get(sha)
            if prior is None or (isinstance(prior["train_budget_meta"].get("dose_scaled_from"), Mapping) and not isinstance(tbm.get("dose_scaled_from"), Mapping)):
                out[sha] = rec
        self._fit_records = out
        return out

    def bundles_for(self, dirs: Sequence[str]) -> List[SidecarBundle]:
        out = []
        for d in dirs:
            od = self._dirs.get(d) or ObjectDir(d, "unknown", "unknown")
            b = read_bundle(d, read_blob=self.read_blobs, bank=od.bank, campaign_hint=_CAMPAIGN_BY_BANK.get(od.bank), target_hint=od.target)
            b.status_hint = od.status_hint
            b.twin_staged_id = od.twin_staged_id
            if od.sub_bank:
                b.bank = f"{od.bank}/{od.sub_bank}"
                b.campaign_hint = _CAMPAIGN_BY_BANK.get(od.bank)
                if od.sub_bank == "norejects":
                    b.campaign_hint = "norejects"
            out.append(b)
        return out

    def payload_for_group(self, bundles: Sequence[SidecarBundle], seeds: Seeds, now: str) -> Dict[str, object]:
        """Merge the twins of one artifact: the richest bundle is primary; the others add locations and sidecars. ``status`` is the
        primary bundle's own statement or absent (:func:`fit_sidecars.build_object_payload`), and so are ``fit_code_sha`` and
        ``derivation_params_json`` (never the bank default, never an abbreviation); no registry read is involved."""
        def richness(b: SidecarBundle) -> Tuple[int, int, int]:
            sc = b.sidecars
            return (2 if "train_budget_meta.json" in sc or "derived_meta.json" in sc else (1 if "provenance.json" in sc or "fit_meta.json" in sc or "meta.json" in sc else 0),
                    0 if b.status_hint else 1, len(sc))
        ordered = sorted(bundles, key=richness, reverse=True)
        primary = ordered[0]
        # The park is a verdict on the GROUP, so every twin is classified first. All twins
        # ruled out of scope -> the group parks, each directory under its own spelling (:class:`ParkedGroup`); a ruled-out twin beside an
        # in-scope or undecided one -> a refusal naming every directory and what it is (:func:`twin_scope_conflict_why`), never a park that
        # omits the in-scope twin's object record silently and never a read that files a ruled-out kind as an object. Before this the
        # richest twin alone decided: identical bytes under an in-scope fitted name and under a richer directory declaring a ruled-out kind
        # parked as that kind -- two directories, no object row, no refusal.
        verdicts = [(b, scope_verdict(b, seeds)) for b in ordered]
        parked = [v for _, v in verdicts if v.state == SCOPE_PARKED]
        if parked:
            if len(parked) < len(verdicts):
                raise SidecarRefusal(primary.dir, twin_scope_conflict_why(verdicts))
            raise ParkedGroup([v.ruling for v in parked if v.ruling is not None])
        for twin in ordered[1:]:
            if twin.blob_path:
                primary.extra_locations.append({"uri": twin.blob_path, "verified_utc": now, "bytes": twin.blob_bytes, "twin_dir": twin.dir})
            else:
                primary.extra_locations.append({"uri": twin.dir, "verified_utc": None, "bytes": None, "note": "sidecars only"})
            stf = twin.sidecars.get("staged_from.json")
            if isinstance(stf, Mapping) and isinstance(stf.get("source"), str):
                primary.extra_locations.append({"uri": str(stf["source"]), "verified_utc": None, "bytes": None, "source": f"{twin.dir}/staged_from.json:source"})
        payload = build_object_payload(primary, seeds, now, blob_reads_enabled=self.read_blobs, parents_in_bank=self.fit_records_by_sha(),
                                       omit_producer_owned=True,       # No bank default, no abbreviation of a producer-owned column
                                       dose_batch_from_parent=True)    # a dose copy's batch is its parent's, derived (EMISSION_VERSION 10)
        sidecars = list(payload["object_sidecars_json"])
        for twin in ordered[1:]:
            sidecars.extend(twin.sidecar_files)
        payload["object_sidecars_json"] = sidecars
        extra = dict(payload.get("extra") or {})
        extra["bank_dirs"] = [b.dir for b in ordered]
        extra["staged_ids"] = sorted({b.staged_id for b in ordered})
        # twin supersession: an it64 twin's sha names the successor
        if primary.status_hint == "superseded_twin" and primary.twin_staged_id:
            twin_sha = _twin_sha(primary.dir, primary.twin_staged_id)
            if twin_sha:
                payload["superseded_by_object_id"] = twin_sha
                payload["field_confidence_json"]["superseded_by_object_id"] = {"confidence": "recorded", "source": f"the it64 twin {primary.twin_staged_id}'s recorded object sha"}
            else:
                payload["field_confidence_json"]["superseded_by_object_id"] = {"confidence": "unverified", "source": f"twin {primary.twin_staged_id} not found beside the bank",
                                                                                "reason": f"twin {primary.twin_staged_id} not found beside the bank"}
        # cluster / Modal locations from the staging records. A location's `bytes` is the OBJECT's size there: the record's own
        # figure is the staged directory's total (kept as `staged_dir_bytes`); a copy the record VERIFIED (remote sha == the object's sha)
        # holds the same bytes as the copy measured on this machine, so its size is `artifact_bytes`; anything else stays None
        sha = str(payload["artifact_sha256"])
        measured = payload.get("artifact_bytes") if isinstance(payload.get("artifact_bytes"), int) else None
        staged = []
        for loc in self._staging_locations().get(sha, []):
            loc = dict(loc)
            if "staged_dir_bytes" in loc and loc.get("verified_utc") and loc.get("remote_sha256") == sha and measured is not None:
                loc["bytes"] = measured
                loc["bytes_source"] = STAGED_LOCATION_BYTES_SOURCE
            staged.append(loc)
        if staged:
            payload["artifact_locations_json"] = list(payload["artifact_locations_json"]) + staged
        payload["extra"] = extra
        return payload

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        seeds = self._seeds(ctx)
        dirs = sorted({os.path.dirname(i.uri) for i in inputs})
        bundles = self.bundles_for(dirs)
        self._parked.pop(group, None)
        try:
            payload = self.payload_for_group(bundles, seeds, ctx.now)
        except OutOfScopeObject as exc:
            # A kind ruled out of scope is PARKED by name -- an accepted revision with no object record and the
            # ruling on the status row (a non-empty `refuse` list is an explanation, so the empty read is never a silent one; `ok = 0`) --
            # never a refusal of the group, which read the whole adapter REFUSED and aborted every window on a by-design condition. The
            # verdict is the GROUP's (every twin ruled out: `ParkedGroup`); twins that disagree are a SidecarRefusal below, naming them
            self._parked[group] = parked_status_facts(exc)
            yield RecordBatch(f"{self.name}:{group}", group)
            return
        except SidecarRefusal as exc:
            raise AdapterRefusal(exc.where, exc.why)
        batch = RecordBatch(f"{self.name}:{group}", group)
        batch.add("object", payload, ctx.producer, ctx.now)
        yield batch

    def status_extra(self, group: str) -> Dict[str, object]:
        """The facts of the group's last read: for a bundle parked by name, the `refuse` line and the `parked` entry citing the
        out-of-scope table (:func:`parked_status_facts`); otherwise none (the bank's product is the object record itself)."""
        return dict(self._parked.get(group) or {})

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        return []


#: the ruling every parked bundle cites, spelled once
PARKED_RULING = "results-registry D37 (2026-09-16): legacy_spellings.yaml out_of_scope_object_kinds"


class ParkedGroup(OutOfScopeObject):
    """The park of a whole artifact group: EVERY directory holding the artifact's bytes is a bundle the out-of-scope table rules out
    (:func:`ObjectsBankAdapter.payload_for_group`, after :func:`fit_sidecars.scope_verdict` on each). ``rulings`` is each directory's own
    :class:`OutOfScopeObject`, the richest bundle's first; that one's ``where`` / ``why`` / ``kind_legacy`` / ``entry`` are the group's, so
    every caller that reads an ``OutOfScopeObject`` reads this one unchanged, and :func:`parked_status_facts` lists each directory under
    the spelling that parked IT (twins of one artifact may be named differently -- ``<si>__lofit`` beside a directory whose blob declares
    ``lofit``)."""

    def __init__(self, rulings: Sequence[OutOfScopeObject]):
        first = rulings[0]
        super().__init__(first.where, first.why, first.kind_legacy, first.entry)
        self.rulings: List[OutOfScopeObject] = list(rulings)


def parked_rulings(exc: OutOfScopeObject) -> List[OutOfScopeObject]:
    """The per-directory rulings of a park: a :class:`ParkedGroup`'s list, or the one ruling of a single-bundle park."""
    return list(getattr(exc, "rulings", None) or [exc])


def twin_scope_conflict_why(verdicts: Sequence[Tuple[SidecarBundle, ScopeVerdict]]) -> str:
    """The refusal of an artifact group whose twins (the directories holding the same bytes) disagree on its scope: at least one is a bundle
    the out-of-scope table rules out and at least one is not -- an in-scope bundle whose kind resolves, or an undecided one the builder refuses
    plainly. Names every directory with its verdict, so the refusal reads as the conflict it is and the holder can rule on it (rename or
    remove the twin that is wrong, or add its spelling to the table). NOT a park: parking would accept an empty revision for the group and
    omit the in-scope twin's object record under a ruling that does not name it; NOT a read of the in-scope twin: the same bytes would then
    stand in the registry as an object while a directory beside them says the kind is ruled out."""
    parts = []
    for b, v in verdicts:
        if v.state == SCOPE_PARKED and v.ruling is not None:
            parts.append(f"{b.dir} parks as {v.ruling.kind_legacy!r} (legacy_spellings.yaml out_of_scope_object_kinds[{v.ruling.kind_legacy}])")
        elif v.state == SCOPE_IN_SCOPE:
            parts.append(f"{b.dir} is an in-scope {v.object_kind!r} bundle")
        else:
            parts.append(f"{b.dir} is undecided ({str(v.refusal.why if v.refusal is not None else '')[:160]})")
    return ("twins of one artifact disagree on its scope: " + "; ".join(parts) + " -- the same bytes cannot be both a kind "
            "results-registry D37 rules out of scope and an object under the vocabulary; NOT parked: a park would omit the in-scope twin's "
            "object record silently under a ruling that does not name it; NOT read: the in-scope twin's bytes would stand as an object beside "
            "a directory that says the kind is ruled out; rule on the group (rename or remove the twin that is wrong, or add its spelling to "
            "the table row it belongs to) before it is read")


def parked_status_facts(exc: OutOfScopeObject) -> Dict[str, object]:
    """The `source_status` facts of a group parked by name: ``refuse`` -- one readable line PER PARKED DIRECTORY (the runner's `detail`
    renders them; a non-empty list is an explanation under `EXPLANATION_KEYS`, so the read files `ok = 0` and is never judged silent) -- and
    ``parked`` -- one structured entry per directory for the status page (the directory, the spelling that matched IT, the table row's
    status / role / reason, the ruling). A single-bundle park lists one of each; a :class:`ParkedGroup` one per twin."""
    refuse: List[str] = []
    parked: List[Dict[str, object]] = []
    for r in parked_rulings(exc):
        ent = r.entry
        refuse.append(f"parked by name: {r.where} is a {r.kind_legacy!r} bundle, a kind ruled out of scope ({PARKED_RULING}); no object record "
                      f"-- {str(ent.get('reason', ''))[:160]}")
        parked.append({"dir": r.where, "kind_legacy": r.kind_legacy, "legacy": ent.get("legacy"), "status_ruled": ent.get("status"),
                       "object_role_ruled": ent.get("object_role"), "reason": ent.get("reason"), "ruling": PARKED_RULING,
                       "ruling_source": ent.get("source")})
    return {"refuse": refuse, "parked": parked}


def _recorded_sha(dir_path: str) -> Optional[str]:
    for name in ("train_budget_meta.json", "COMPLETE", "derived_meta.json", "staged_from.json", "provenance.json"):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            try:
                d = read_json(p)
            except (OSError, ValueError):
                continue
            sha = d.get("object_sha256")
            if isinstance(sha, str) and len(sha) == 64:
                return sha
    return None


def _twin_sha(twin_dir: str, twin_staged_id: str) -> Optional[str]:
    cells = os.path.dirname(os.path.dirname(twin_dir))   # …/cells/_superseded_by_it64/<id> → …/cells
    p = os.path.join(cells, twin_staged_id)
    if os.path.isdir(p):
        blob = os.path.join(p, "object.pt")
        if os.path.isfile(blob):
            from .fit_sidecars import sha256_file
            return sha256_file(blob)[0]
        return _recorded_sha(p)
    return None


# --------------------------------------------------------------------------------------------
# scan (the CLI's report; no registry writes)
# --------------------------------------------------------------------------------------------


@dataclass
class BankCounts:
    bank: str
    dirs: int = 0
    objects: int = 0              # distinct artifact shas that produced a row
    with_sidecars: int = 0        # dirs holding at least one fit record / promotion record / derived record
    blobs_local: int = 0          # dirs whose weights file is on this machine
    blobs_read: int = 0           # weights files whose metadata was read with torch
    refused: int = 0
    parked: int = 0               # bundles of a kind ruled out of scope, parked by name (no row, no refusal)
    unverified_fields: int = 0
    unverified_by_field: Dict[str, int] = field(default_factory=dict)
    by_kind: Dict[str, int] = field(default_factory=dict)
    by_target_kind: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    status_unstated: int = 0          # payloads with no status statement: the bank carries no evidence
    refusals: List[Dict[str, str]] = field(default_factory=list)
    parked_dirs: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ScanReport:
    banks: Dict[str, BankCounts] = field(default_factory=dict)
    objects_total: int = 0
    payloads: List[Dict[str, object]] = field(default_factory=list)
    refusals: List[Dict[str, str]] = field(default_factory=list)
    parked: List[Dict[str, object]] = field(default_factory=list)      # one entry per parked group
    torch_available: bool = False

    def as_dict(self, with_payloads: bool = False) -> Dict[str, object]:
        d = {"objects_total": self.objects_total, "torch_available": self.torch_available,
             "banks": {k: {kk: vv for kk, vv in v.__dict__.items() if kk not in ("refusals", "parked_dirs")} for k, v in self.banks.items()},
             "refusals": self.refusals, "parked": self.parked}
        if with_payloads:
            d["payloads"] = self.payloads
        return d


def scan(banks: Sequence[Tuple[str, str, str]] = DEFAULT_BANKS, seeds: Optional[Seeds] = None, seeds_dir: Optional[str] = None,
         read_blobs: bool = True, pins_path: Optional[str] = None, now: Optional[str] = None, conn=None) -> ScanReport:
    """Enumerate the banks reachable from this machine and build every object payload in memory."""
    require_store_access("objects_bank.scan: enumerating the object banks")      # the S2 dispatch-site guard (C6)
    from .. import records as R
    now = now or R.utc_now()
    rep = ScanReport()
    try:
        import torch  # noqa: F401
        rep.torch_available = True
    except Exception:
        rep.torch_available = False
        read_blobs = False
    ad = ObjectsBankAdapter(banks, seeds=seeds, seeds_dir=seeds_dir, read_blobs=read_blobs, pins_path=pins_path)
    ctx = AdapterContext(root="", now=now, producer={}, conn=conn)
    sd = ad._seeds(ctx)
    inputs = ad.discover()
    groups: Dict[str, List[str]] = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(os.path.dirname(i.uri))
    for g, dirs in sorted(groups.items()):
        dirs = sorted(set(dirs))
        bundles = ad.bundles_for(dirs)
        for b in bundles:
            bc = rep.banks.setdefault(b.bank, BankCounts(b.bank))
            bc.dirs += 1
            if any(k in b.sidecars for k in ("train_budget_meta.json", "derived_meta.json", "fit_provenance.json", "provenance.json", "fit_meta.json", "meta.json")):
                bc.with_sidecars += 1
            if b.blob_path:
                bc.blobs_local += 1
            if b.blob_meta is not None:
                bc.blobs_read += 1
        primary_bank = bundles[0].bank
        try:
            payload = ad.payload_for_group(bundles, sd, now)
        except OutOfScopeObject as exc:
            facts = parked_status_facts(exc)
            kind_by_dir = {r.where: r.kind_legacy for r in parked_rulings(exc)}      # every twin parks under ITS OWN spelling (ParkedGroup)
            for b in bundles:
                rep.banks[b.bank].parked += 1
                rep.banks[b.bank].parked_dirs.append({"dir": b.dir, "kind_legacy": kind_by_dir.get(b.dir, exc.kind_legacy)})
            rep.parked.append({"group": g, "dirs": ";".join(dirs), "kind_legacy": exc.kind_legacy, "why": exc.why, "parked": facts["parked"][0],
                               "twins": facts["parked"]})
            continue
        except SidecarRefusal as exc:
            for b in bundles:
                rep.banks[b.bank].refused += 1
                rep.banks[b.bank].refusals.append({"dir": b.dir, "why": exc.why})
            rep.refusals.append({"group": g, "dirs": ";".join(dirs), "why": exc.why})
            continue
        rep.payloads.append(payload)
        rep.objects_total += 1
        bc = rep.banks[primary_bank]
        bc.objects += 1
        fc = payload.get("field_confidence_json") or {}
        n_unv = 0
        for col, ent in fc.items():
            if ent.get("confidence") == "unverified":
                n_unv += 1
                bc.unverified_by_field[col] = bc.unverified_by_field.get(col, 0) + 1
        bc.unverified_fields += n_unv
        bc.by_kind[str(payload["object_kind"])] = bc.by_kind.get(str(payload["object_kind"]), 0) + 1
        bc.by_target_kind[str(payload["fit_target_kind"])] = bc.by_target_kind.get(str(payload["fit_target_kind"]), 0) + 1
        if "status" in payload:
            bc.by_status[str(payload["status"])] = bc.by_status.get(str(payload["status"]), 0) + 1
        else:
            bc.status_unstated += 1          # the bank carries no evidence: the producer's folded status stands
    return rep


def format_scan(rep: ScanReport, top_fields: int = 8) -> str:
    lines = [f"objects: {rep.objects_total} distinct artifacts across {len(rep.banks)} banks (torch {'available' if rep.torch_available else 'UNAVAILABLE: blob metadata not read'})"]
    for name, bc in sorted(rep.banks.items()):
        lines.append(f"- {name}: dirs {bc.dirs}, objects {bc.objects}, with sidecars {bc.with_sidecars}, weights local {bc.blobs_local} (read {bc.blobs_read}), "
                     f"refused {bc.refused}, parked {bc.parked} (D37 out-of-scope kinds, by name), unverified fields {bc.unverified_fields}")
        if bc.by_kind:
            lines.append("    kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(bc.by_kind.items())))
        if bc.by_target_kind:
            lines.append("    fit_target_kind: " + ", ".join(f"{k}={v}" for k, v in sorted(bc.by_target_kind.items())))
        if bc.by_status or bc.status_unstated:
            lines.append("    status: " + ", ".join(f"{k}={v}" for k, v in sorted(bc.by_status.items()))
                         + (f"{', ' if bc.by_status else ''}unstated={bc.status_unstated} (the bank states none; the producer's folded status stands, D721)"
                            if bc.status_unstated else ""))
        if bc.unverified_by_field:
            top = sorted(bc.unverified_by_field.items(), key=lambda kv: (-kv[1], kv[0]))[:top_fields]
            lines.append("    most unverified: " + ", ".join(f"{k}={v}" for k, v in top))
        for r in bc.refusals:
            lines.append(f"    refused {r['dir']}: {r['why'][:220]}")
        for r in bc.parked_dirs:
            lines.append(f"    parked {r['dir']}: {r['kind_legacy']!r} ({PARKED_RULING})")
    return "\n".join(lines)


def register() -> None:
    from .base import register as _register
    _register(ObjectsBankAdapter.name, lambda: ObjectsBankAdapter())
