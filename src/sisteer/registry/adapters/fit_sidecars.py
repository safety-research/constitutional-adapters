"""Fit sidecars → the ``object`` table's ``fit_*`` columns.

Every fitted object on disk is described by one or more SIDECARS beside its weights file:

* ``train_budget_meta.json`` — the two-SI / sdfbase trainer's Class A record (``si_id``, ``data_mode``,
  ``teacher_forward``, ``positions``, ``target_layers``, ``lr``, ``fit_record`` …). A dose-scaled copy carries
  its PARENT's record plus a ``dose_scaled_from`` block.
* ``derived_meta.json`` — a derived object (LoRA subtraction, ReLU-steer residual distillation, paired write). A dose
  ladder's 1.0 ANCHOR re-serves a subtraction's own bytes under a dose id with the builder's record plus a
  ``dose_scaled_from`` block whose ``anchor`` is true (parent sha == own sha): NOT a dose scaling — the object IS the
  subtraction, its parents are the two fits the record names (:func:`_subtraction_anchor`).
* ``fit_provenance.json`` + ``provenance.json`` + ``summary.json`` — a promoted HPO trial.
* ``fit_meta.json`` (``alphasteer_fit_meta/1``), ``meta.json`` + ``object.json`` (antidote / CAA) — contrastive directions.
* ``object.pt.provenance.json``, ``COMPLETE``, ``superseded_by.json``, ``staged_from.json`` — identity and status.
* the weights blob itself (``object.pt``: ``kind``, ``layers``, ``rank``, ``positions``, ``response_header_len``),
  read with ``torch.load(weights_only=True)`` ONLY when the file is on this machine.

:func:`build_object_payload` turns a :class:`SidecarBundle` into the ``object`` record payload with
``field_confidence_json`` naming the source key of every recorded field and the reason for every
``unverified`` one; it REFUSES (:class:`SidecarRefusal`) when the paired-nullness CHECK could not be
satisfied honestly (an unmapped arm, an ambiguous ``fit_target_kind``, a teacher instruction with no
``si_text`` row, a sha that disagrees with the bytes). Nothing here defaults a provenance field.

A dose copy's ``fit_effective_batch``: a dose-scaled copy was not fit, and this builder used to leave every
``fit_*`` scalar unstated for one -- a NULL that sat beside the PARENT's hyperparameter block (``fit_hyperparameters_json``, the
parent's record copied beside the copy, verbatim) and that ``reconcile fit-effective-batch`` could only list as ambiguous (92 rows,
the ``scale_lora_dose.py`` / ``scale_psr_dose.py`` ladders). The ruling: the copy STATES the column as its parent's effective batch,
``derived`` from ONE source -- the parent's fit record copied beside the copy, the copy's own ``train_budget_meta.json``
(:func:`_dose_batch_from_parent`; the registry owner's design decision, 2026-10-13: an input of the copy's own group, which its revision
already covers, so nothing of another group's directory is read) -- with the parent and the rule in the confidence entry
(:func:`dose_batch_derived_source`); a copy whose copied record names no integer effective batch, or which carries no such record, is
stated NULL by name -- the column PRESENT as null in the payload, so the fold writes NULL over an earlier derived statement rather than
keeping that value beside a block it no longer agrees with. The objects bank opts in (``dose_batch_from_parent``; its EMISSION_VERSION 10
re-emits the rows); no correction record is written for them, and whether the copied record agrees with the parent's own registry row is
``reconcile fit-effective-batch``'s read-only cross-check. The other caller keeps the legacy reading until its own switch.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Set, Tuple

from .. import derive as D
from .. import identity
from .. import vocab as V
from . import _items_common as IC
from .declared_common import ARMS as _PAPER_ARMS   # the paper's arm table (the readers' contract), keyed by campaign stem

SIDECAR_NAMES: Tuple[str, ...] = (
    "train_budget_meta.json", "derived_meta.json", "fit_provenance.json", "provenance.json", "summary.json",
    "fit_meta.json", "meta.json", "object.json", "object.pt.provenance.json", "COMPLETE", "superseded_by.json",
    "staged_from.json", "object.safetensors.json",
)
BLOB_NAMES: Tuple[str, ...] = ("object.pt", "object.safetensors")
#: Sidecar files whose presence marks a directory as an object directory (C25: a bank holds non-object files too).
OBJECT_DIR_MARKERS: Tuple[str, ...] = BLOB_NAMES + ("train_budget_meta.json", "derived_meta.json", "fit_provenance.json", "fit_meta.json")

_SI_TEACHERS = ("safety_refusal", "si_constitution", "si_constitution_full", "si_irrelevant")
_BLOB_KIND_TO_OBJECT_KIND = {"lora_all": "lora", "lora": "lora", "psr": "relu_steer", "lora_subtraction": "lora_subtraction",
                             "diffmean": "diffmean", "vector": "vector", "vector_all": "vector_all", "alphasteer": "alphasteer",
                             "antidote": "antidote", "static_r": "static_r_arditi", "caa_static_r": "caa_static_r"}
_DERIVED_DERIVATION = {"lora_subtraction": ("lora_subtraction", "lora_subtraction"),
                       "psr_residual_mse": ("residual_distillation", "relu_steer_distilled_difference")}
_ROLE_BY_SI = {"safety_refusal": "safety", "si_constitution": "constitution", "si_constitution_full": "constitution",
               "si_irrelevant": "irrelevant"}
_ROLE_BY_CORPUS = {"constv2-mix-full": "constitution", "constv2-mix-10m": "constitution", "constv2ctl2-full": "control"}
_ROLE_BY_SI_KIND = {"safety_instruction": "safety", "constitution_short": "constitution", "constitution_full": "constitution", "irrelevant_control": "irrelevant"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class SidecarRefusal(Exception):
    """The sidecars cannot be turned into an honest ``object`` row; ``why`` names the keys seen."""

    def __init__(self, where: str, why: str):
        super().__init__(f"{where}: {why}")
        self.where = where
        self.why = why


class OutOfScopeObject(SidecarRefusal):
    """A bundle of a kind the vocabulary puts OUT OF SCOPE by name (the table
    ``legacy_spellings.yaml: out_of_scope_object_kinds``): for example a synthetic Gaussian control direction, or a LoFiT /
    token-appended / BiDPO fit. No ``object_kind`` spells it, so no ``object`` row can exist under the vocabulary -- and that is the
    rule, not a defect of the bundle or the reader. A :class:`SidecarRefusal` (so every caller that refuses on the base class still
    does), distinguished so the objects bank can PARK the bundle -- an accepted revision with no object record and a status fact citing
    the table -- instead of refusing its group on every read (48 of the bank's 53 standing refusals were these, re-refused every
    window and tick, and a per-group refusal reads the whole adapter REFUSED and aborts the window). ``kind_legacy`` is the spelling that
    matched, ``entry`` the table row (``legacy``, ``status``, ``object_role``, ``reason``, ``source``)."""

    def __init__(self, where: str, why: str, kind_legacy: str, entry: Mapping[str, object]):
        super().__init__(where, why)
        self.kind_legacy = kind_legacy
        self.entry = dict(entry)


# --------------------------------------------------------------------------------------------
# seeds (read-only: configs/registry/*.yaml on the seeds branch, or a caller-built mapping)
# --------------------------------------------------------------------------------------------


@dataclass
class Seeds:
    """The legacy → controlled mappings the fit adapters consult (``legacy_spellings.yaml``), the ``si_text``
    shas (``si_texts.yaml`` or the ``si_text`` table), and the model closure (``models.yaml`` or ``model``)."""
    object_kind_by_spelling: Dict[str, Dict[str, object]] = field(default_factory=dict)   # arm / cell spelling → {object_kind, fit_rank, derivation, paper_label}
    paper_label_by_spelling: Dict[str, Dict[str, object]] = field(default_factory=dict)   # cell spelling → {paper_label, object_role, fit_corpus_id, si_id}
    corpus_ids: Dict[str, str] = field(default_factory=dict)                               # producer corpus spelling → fit_corpus_id
    si_sha_by_id: Dict[str, str] = field(default_factory=dict)                             # si_id → si_text_sha256
    si_kind_by_id: Dict[str, str] = field(default_factory=dict)                            # si_id → si_kind (the object_role rule for SI distillations)
    model_id_by_hf_repo: Dict[str, str] = field(default_factory=dict)                      # hf repo → model_id
    model_revision_by_id: Dict[str, str] = field(default_factory=dict)
    campaign_by_spelling: Dict[str, str] = field(default_factory=dict)
    fit_target_overrides: Dict[str, Dict[str, object]] = field(default_factory=dict)       # si_id spelling → {fit_target_kind, fit_teacher_si_id | fit_corpus_id, citation}
    #: The out-of-scope object kinds (`legacy_spellings.yaml: out_of_scope_object_kinds`): exact spelling → the table row, and the
    #: spellings ending in `_` (today `rand_`) as PREFIXES. A bundle matching either is parked by name, never refused (:class:`OutOfScopeObject`).
    out_of_scope_by_spelling: Dict[str, Dict[str, object]] = field(default_factory=dict)
    out_of_scope_prefixes: Tuple[str, ...] = ()
    source: str = "caller"

    def object_kind_for(self, spelling: str) -> Optional[Dict[str, object]]:
        return _lookup_spelling(self.object_kind_by_spelling, spelling)

    def paper_label_for(self, spelling: str) -> Optional[Dict[str, object]]:
        return _lookup_spelling(self.paper_label_by_spelling, spelling)

    def out_of_scope_for(self, *spellings: Optional[str]) -> Optional[Tuple[str, Dict[str, object]]]:
        """``(the spelling that matched, the out-of-scope table row)`` for the first of ``spellings`` the out-of-scope table names -- by exact
        spelling, by the tail after the first ``__`` (``safety_refusal__lofit`` → ``lofit``), or by a declared prefix on either
        (``rand_recipeB_s31`` → ``rand_``); None when no spelling is in the table."""
        for sp in spellings:
            if not sp:
                continue
            for cand in (sp, sp.split("__", 1)[1] if "__" in sp else None):
                if not cand:
                    continue
                if cand in self.out_of_scope_by_spelling:
                    return cand, self.out_of_scope_by_spelling[cand]
                for pre in self.out_of_scope_prefixes:
                    if cand.startswith(pre):
                        return pre, self.out_of_scope_by_spelling[pre]
        return None


def _split_legacy(text: str) -> List[str]:
    """``"psr | psr_kl_all (e.g. x)"`` → ``["psr", "psr_kl_all"]`` (parenthetical examples dropped)."""
    out = []
    for part in str(text).split("|"):
        p = part.strip()
        p = re.sub(r"\s*\(.*$", "", p).strip()
        if p:
            out.append(p)
    return out


def out_of_scope_tables(entries: Sequence[Mapping[str, object]]) -> Tuple[Dict[str, Dict[str, object]], Tuple[str, ...]]:
    """The ``out_of_scope_object_kinds`` entries → ``(spelling → the table row, the prefixes)``: each ``legacy`` is ``a | b | c``, a
    spelling is the token before any space or parenthetical, and a spelling ending in ``_`` (today ``rand_``) is a PREFIX."""
    by_spelling: Dict[str, Dict[str, object]] = {}
    prefixes: List[str] = []
    for ent in entries:
        for sp in _split_legacy(str(ent.get("legacy", ""))):
            sp = sp.split(" ")[0]
            if not sp:
                continue
            by_spelling[sp] = dict(ent)
            if sp.endswith("_") and sp not in prefixes:
                prefixes.append(sp)
    return by_spelling, tuple(prefixes)


def _lookup_spelling(table: Mapping[str, Dict[str, object]], spelling: str) -> Optional[Dict[str, object]]:
    if spelling in table:
        return table[spelling]
    # ``<si>__X`` patterns match a cell id by its suffix; a bare arm matches the cell id's tail after the first ``__``
    for key, val in table.items():
        if key.startswith("<si>__") and spelling.endswith(key[len("<si>"):]):
            return val
    tail = spelling.split("__", 1)[1] if "__" in spelling else None
    if tail and tail in table:
        return table[tail]
    return None


def paper_label_entry_for(object_kind: Optional[str], spellings: Tuple[Optional[str], ...], seeds: Seeds,
                          extra: Optional[Dict[str, object]] = None) -> Tuple[Optional[Dict[str, object]], Optional[str], Optional[str], Optional[str]]:
    """``(label entry, source, campaign stem, fit-variant prefix)`` for a fitted object from its cell spellings (the sidecar's cell id, the
    staged directory name, the arm), tried in that order. Each spelling is reduced to its campaign STEM (``derive.campaign_stem_legacy``:
    ``itdoc64__constv2-mix-full__lora_r4_ce`` → ``constv2-mix-full__lora_r4_ce``, the ``itdoc64`` prefix being fit provenance, never the
    arm) and spelled by, in order, the seeded ``legacy_spellings.yaml paper_labels`` (the spelling itself, then its stem) and the paper's
    arm table (``declared_common.ARMS``) when that arm's ``object_kind`` is this object's — a stem the paper spells for ANOTHER kind is
    parked in ``extra['paper_label_kind_disagrees']`` and never borrowed. The entry carries ``paper_label`` and ``object_role``; None
    when no table spells any stem (the caller files ``other`` and names both tables). The stem returned is the matched spelling's (else
    the first spelling's); the prefix is the first fit-variant prefix ANY spelling carries — the trainer records the bare stem as the
    cell id and the staged directory name carries the variant — so the caller can park both whether or not a label was found."""
    first_stem: Optional[str] = None
    variant_prefix: Optional[str] = None
    parsed = []
    for spelling in spellings:
        if not spelling:
            continue
        stem, prefix = D.campaign_stem_legacy(str(spelling))
        parsed.append((str(spelling), stem, prefix))
        if first_stem is None:
            first_stem = stem
        if variant_prefix is None and prefix:
            variant_prefix = prefix
    for spelling, stem, _prefix in parsed:
        ent = seeds.paper_label_for(spelling)
        if not ent and stem and stem != spelling:
            ent = seeds.paper_label_for(stem)
        if ent:
            return ent, f"legacy_spellings.yaml paper_labels[{stem}] (spelling {spelling!r})", stem, variant_prefix
        paper_row = _PAPER_ARMS.get(str(stem))
        if paper_row is not None and paper_row.paper_label in V.PAPER_LABELS and paper_row.object_role in V.OBJECT_ROLES:
            if paper_row.object_kind == object_kind:
                return ({"paper_label": paper_row.paper_label, "object_role": paper_row.object_role, "source": "declared_common.ARMS"},
                        f"declared_common.ARMS[{stem}] (the paper's own table: object_kind {object_kind}; spelling {spelling!r})", stem, variant_prefix)
            if extra is not None:
                extra["paper_label_kind_disagrees"] = {"stem": stem, "spelling": spelling, "object_kind": object_kind,
                                                       "paper_object_kind": paper_row.object_kind, "paper_label_of_stem": paper_row.paper_label}
    return None, None, first_stem, variant_prefix


def load_seeds(seeds_dir: Optional[str], conn=None) -> Seeds:
    """Read ``legacy_spellings.yaml``, ``si_texts.yaml``, ``models.yaml`` from ``seeds_dir`` (read-only) and/or the
    ``si_text`` / ``model`` tables of ``conn``. Missing files leave their mapping empty (refusals then name it)."""
    seeds = Seeds(source=seeds_dir or "conn")
    if seeds_dir:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - yaml is in the shared venv
            raise SidecarRefusal(seeds_dir, f"PyYAML unavailable ({exc}); pass a Seeds object instead")
        ls_path = os.path.join(seeds_dir, "legacy_spellings.yaml")
        if os.path.isfile(ls_path):
            with open(ls_path, "r", encoding="utf-8") as fh:
                ls = yaml.safe_load(fh) or {}
            for ent in ls.get("object_kinds") or []:
                kinds = [k.strip() for k in str(ent.get("object_kind", "")).split("|")]
                spellings = _split_legacy(ent.get("legacy", ""))
                for i, sp in enumerate(spellings):
                    kind = kinds[i] if len(kinds) == len(spellings) else kinds[0]
                    if kind not in V.OBJECT_KINDS:
                        continue   # "OUT OF SCOPE …" and context-dependent entries are not mappings
                    seeds.object_kind_by_spelling[sp] = {"object_kind": kind, "fit_rank": ent.get("fit_rank"),
                                                         "derivation": ent.get("derivation"), "paper_label": ent.get("paper_label"),
                                                         "fit_objective": ent.get("fit_objective"), "source": ent.get("source")}
            for ent in ls.get("paper_labels") or []:
                label = str(ent.get("paper_label", ""))
                role = str(ent.get("object_role", ""))
                if label not in V.PAPER_LABELS or role not in V.OBJECT_ROLES:
                    continue
                for sp in _split_legacy(ent.get("legacy", "")):
                    seeds.paper_label_by_spelling[sp] = {"paper_label": label, "object_role": role, "fit_corpus_id": ent.get("fit_corpus_id"),
                                                         "si_id": ent.get("si_id"), "source": ent.get("source")}
                    if ent.get("fit_corpus_id"):
                        seeds.corpus_ids[str(ent["fit_corpus_id"])] = str(ent["fit_corpus_id"])
            for ent in ls.get("campaigns") or []:
                camp = str(ent.get("campaign", ""))
                if " " in camp or not camp:
                    continue
                for sp in _split_legacy(ent.get("legacy", "")):
                    seeds.campaign_by_spelling[sp] = camp
            for ent in ls.get("fit_target_overrides") or []:
                for sp in _split_legacy(ent.get("legacy", "")):
                    seeds.fit_target_overrides[sp] = dict(ent)
            seeds.out_of_scope_by_spelling, seeds.out_of_scope_prefixes = out_of_scope_tables(ls.get("out_of_scope_object_kinds") or [])
            for ent in ls.get("corpus_ids") or []:
                for sp in _split_legacy(ent.get("legacy", "")):
                    seeds.corpus_ids[sp] = str(ent.get("fit_corpus_id", sp))
        si_path = os.path.join(seeds_dir, "si_texts.yaml")
        if os.path.isfile(si_path):
            with open(si_path, "r", encoding="utf-8") as fh:
                st = yaml.safe_load(fh) or {}
            for ent in st.get("si_texts") or []:
                sha = str(ent.get("si_text_sha256", ""))
                if ent.get("si_id") and _SHA256_RE.match(sha):
                    seeds.si_sha_by_id[str(ent["si_id"])] = sha
                    if ent.get("si_kind"):
                        seeds.si_kind_by_id[str(ent["si_id"])] = str(ent["si_kind"])
        m_path = os.path.join(seeds_dir, "models.yaml")
        if os.path.isfile(m_path):
            with open(m_path, "r", encoding="utf-8") as fh:
                ms = yaml.safe_load(fh) or {}
            # hf_repo -> model_id through the shared ownership rule (a repo two rows pin names the paper's row), never the
            # file order; the revision map is per model_id and needs no rule
            for repo, mid in IC.model_map_by_hf_repo(ms.get("models") or []).items():
                seeds.model_id_by_hf_repo.setdefault(repo, mid)
            for ent in ms.get("models") or []:
                if ent.get("hf_repo") and ent.get("model_id"):
                    seeds.model_revision_by_id[str(ent["model_id"])] = str(ent.get("hf_revision", ""))
    if conn is not None:
        try:
            for si_id, sha, kind in conn.execute("SELECT si_id, si_text_sha256, si_kind FROM si_text").fetchall():
                seeds.si_sha_by_id.setdefault(si_id, sha)
                seeds.si_kind_by_id.setdefault(si_id, kind)
            table = [{"model_id": mid, "hf_repo": repo, "hf_revision": rev, "paper_status": status}
                     for mid, repo, rev, status in conn.execute("SELECT model_id, hf_repo, hf_revision, paper_status FROM model").fetchall()]
            for repo, mid in IC.model_map_by_hf_repo(table).items():
                seeds.model_id_by_hf_repo.setdefault(repo, mid)
            for m in table:
                seeds.model_revision_by_id.setdefault(m["model_id"], m["hf_revision"])
        except Exception:
            pass
    return seeds


# --------------------------------------------------------------------------------------------
# reading sidecars and the blob
# --------------------------------------------------------------------------------------------


@dataclass
class SidecarBundle:
    """Everything read from ONE object directory."""
    dir: str
    staged_id: str                                    # the directory name (the bank's key for the object)
    sidecars: Dict[str, Dict[str, object]] = field(default_factory=dict)   # name → parsed JSON
    sidecar_files: List[Dict[str, object]] = field(default_factory=list)   # [{kind, uri, sha256}]
    blob_path: Optional[str] = None
    blob_sha256: Optional[str] = None
    blob_bytes: Optional[int] = None
    blob_meta: Optional[Dict[str, object]] = None     # the blob's non-tensor keys when read
    blob_read_error: Optional[str] = None
    bank: str = ""                                    # a short bank label (full22_objects | fits_mirror | grade_objects | wave_objects)
    campaign_hint: Optional[str] = None
    target_hint: Optional[str] = None                 # the bank's model directory name when one exists
    status_hint: Optional[str] = None                 # superseded_twin | withdrawn
    twin_staged_id: Optional[str] = None
    extra_locations: List[Dict[str, object]] = field(default_factory=list)


def sha256_file(path: str) -> Tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # COMPLETE markers are sometimes one line of prose
        return {"_text": text.strip()}
    return obj if isinstance(obj, dict) else {"_value": obj}


def read_blob_meta(path: str) -> Dict[str, object]:
    """The blob's non-tensor keys via ``torch.load(weights_only=True)`` (CPU; the file must be local). Tensor
    shapes are summarised, never the values. Raises on an unreadable file or a missing torch."""
    import torch  # noqa: F401  (import error propagates: the caller marks blob_not_local / torch_unavailable)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    meta: Dict[str, object] = {}
    if not isinstance(obj, dict):
        return {"_type": type(obj).__name__}
    tensor_groups: Dict[str, Dict[str, object]] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            tensor_groups[str(k)] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        elif isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values()):
            shapes = sorted({tuple(x.shape) for x in v.values()})
            tensor_groups[str(k)] = {"n": len(v), "shapes": [list(s) for s in shapes], "keys": sorted(str(kk) for kk in v)[:64]}
        elif isinstance(v, (str, int, float, bool)) or v is None or isinstance(v, (list, dict)):
            meta[str(k)] = v
        else:
            meta[str(k)] = repr(v)[:200]
    meta["_tensors"] = tensor_groups
    return meta


def read_bundle(dir_path: str, read_blob: bool = True, bank: str = "", campaign_hint: Optional[str] = None,
                target_hint: Optional[str] = None) -> SidecarBundle:
    """Read every known sidecar in ``dir_path`` (named files only, one level) and the blob's metadata."""
    b = SidecarBundle(dir=dir_path, staged_id=os.path.basename(dir_path.rstrip("/")), bank=bank,
                      campaign_hint=campaign_hint, target_hint=target_hint)
    for name in SIDECAR_NAMES:
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            try:
                b.sidecars[name] = read_json(p)
            except (OSError, UnicodeDecodeError) as exc:
                raise SidecarRefusal(p, f"unreadable sidecar ({exc})")
            sha, _ = sha256_file(p)
            b.sidecar_files.append({"kind": name, "uri": p, "sha256": sha})
    for name in BLOB_NAMES:
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            b.blob_path = p
            b.blob_sha256, b.blob_bytes = sha256_file(p)
            if read_blob and name == "object.pt":
                try:
                    b.blob_meta = read_blob_meta(p)
                except Exception as exc:  # noqa: BLE001 — recorded, never fatal
                    b.blob_read_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            break
    metrics = os.path.join(dir_path, "metrics.jsonl")
    if os.path.isfile(metrics):
        b.sidecar_files.append({"kind": "metrics.jsonl", "uri": metrics, "sha256": sha256_file(metrics)[0]})
    return b


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


class _Conf:
    """Collects ``field_confidence_json`` entries and the ``unverified`` reasons as columns are decided."""

    def __init__(self):
        self.entries: Dict[str, Dict[str, str]] = {}

    def recorded(self, col: str, source: str):
        self.entries[col] = {"confidence": "recorded", "source": source}

    def derived(self, col: str, source: str):
        self.entries[col] = {"confidence": "derived", "source": source}

    def assumed(self, col: str, source: str):
        self.entries[col] = {"confidence": "assumed_default", "source": source}

    def unverified(self, col: str, reason: str):
        self.entries[col] = {"confidence": "unverified", "source": reason, "reason": reason}

    def set(self, col: str, value, confidence_: str, source: str):
        if confidence_ == "unverified":
            self.unverified(col, source)
        else:
            self.entries[col] = {"confidence": confidence_, "source": source}


def _first(*pairs: Tuple[Optional[Mapping[str, object]], str]) -> Tuple[Optional[object], Optional[str]]:
    """``((dict, "key"), …)`` → the first present, non-null value and ``"<sidecar>:<key>"``; key may be dotted."""
    for d, key in pairs:
        if not d:
            continue
        cur: object = d
        ok = True
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur and cur[part] is not None:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur, key
    return None, None


def _utc(text: object) -> Optional[str]:
    """Normalise a producer timestamp to the registry's ISO-8601-with-milliseconds ``…Z`` form."""
    if not isinstance(text, str) or not text:
        return None
    t = text.strip().replace("Z", "+00:00") if text.endswith("Z") else text.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(t, fmt)
            break
        except ValueError:
            dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(t)
        except ValueError:
            if re.match(r"^\d{4}-\d{2}-\d{2}$", t):
                dt = datetime.strptime(t, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _int(v) -> Optional[int]:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _num(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


_LORA_RANK_RE = re.compile(r"(?:^|_)lora_r(\d+)(?:_|$)")
_DOSE_PREFIX_RE = re.compile(r"^(d(?P<d>[0-9p]+)|dm(?P<m>[0-9p]+)|halfdose)(?P<it>it64|it|itdoc64|itdoc)?__")
_REPLICATE_RE = re.compile(r"^(?:derived_)?base64s(?P<s>\d+)__")


def _hyperparameter_block(tbm: Mapping[str, object]) -> Dict[str, object]:
    """Every scalar top-level key of ``train_budget_meta.json`` that is a hyperparameter (verbatim), plus the
    nested blocks that are (``fit_record.early_stop``, ``continued_from.lr_schedule``)."""
    skip = {"cell_id", "arm", "si_id", "object_sha256", "model_id", "model_revision", "model_revision_source", "template_hash",
            "target_layers", "positions", "positions_at_fit", "site", "response_header_len", "fit_record", "continued_from",
            "teacher_topk", "dose_scaled_from", "sample_manifest_sha", "heldout_pool_sha", "revision_source", "parallel", "contrast",
            "extraction", "provenance", "control_note", "expected_r_norm_job187890"}
    out: Dict[str, object] = {}
    for k, v in tbm.items():
        if k in skip:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list) and len(v) <= 64:
            out[k] = v
    fr = tbm.get("fit_record")
    if isinstance(fr, Mapping):
        if isinstance(fr.get("early_stop"), Mapping):
            out["early_stop"] = dict(fr["early_stop"])
        for k in ("epochs", "patience", "seed"):
            if k in fr and k not in out:
                out[k] = fr[k]
    cf = tbm.get("continued_from")
    if isinstance(cf, Mapping):
        out["continuation"] = {k: cf.get(k) for k in ("new_cap", "prior_cap", "lr_mult", "prior_lr_mult", "stopper_rule", "prior_stopper_rule",
                                                       "resume_epoch", "resume_step", "replay_exempt_epoch", "stop_reason") if k in cf}
        if isinstance(cf.get("lr_schedule"), Mapping):
            out["continuation"]["lr_schedule"] = {k: v for k, v in cf["lr_schedule"].items() if k != "note"}
    if isinstance(tbm.get("teacher_topk"), Mapping):
        tk = tbm["teacher_topk"]
        out["teacher_topk"] = {k: tk.get(k) for k in ("top_k", "n_examples", "n_positions") if k in tk}
    return out


# --------------------------------------------------------------------------------------------
# the payload
# --------------------------------------------------------------------------------------------


def _object_kind(b: SidecarBundle, seeds: Seeds, conf: _Conf) -> Tuple[str, Dict[str, object]]:
    """Decide ``object_kind`` from the blob's own ``kind`` first, then the sidecars, then the arm spelling."""
    sc = b.sidecars
    dm = sc.get("derived_meta.json")
    prov = sc.get("object.pt.provenance.json")
    blob = b.blob_meta or {}
    tbm = sc.get("train_budget_meta.json")
    arm_legacy = None
    for d in (tbm, sc.get("provenance.json"), sc.get("summary.json"), sc.get("meta.json"), sc.get("COMPLETE")):
        if isinstance(d, Mapping) and d.get("arm"):
            arm_legacy = str(d["arm"])
            break
    cell_id_legacy = None
    for d in (tbm, sc.get("COMPLETE"), sc.get("staged_from.json"), dm, sc.get("meta.json")):
        if isinstance(d, Mapping) and (d.get("cell_id") or d.get("derived_id")):
            cell_id_legacy = str(d.get("cell_id") or d.get("derived_id"))
            break
    info: Dict[str, object] = {"arm_legacy": arm_legacy, "cell_id_legacy": cell_id_legacy}
    # derived objects: the derivation names the kind
    if dm and dm.get("kind"):
        dkind = str(dm["kind"])
        derivation = str(dm.get("derivation", ""))
        if dkind == "lora_subtraction":
            conf.recorded("object_kind", "derived_meta.json:kind")
            return "lora_subtraction", info
        if dkind == "psr" and derivation == "psr_residual_mse":
            conf.recorded("object_kind", "derived_meta.json:kind + derivation")
            return "relu_steer_distilled_difference", info
        # a derived kind the vocabulary does not spell: the seeded override (legacy_spellings.yaml object_kinds) decides BEFORE the
        # refusal — the table rules `psr_paired_write` a paired-write COMPOSITION of two ReLU-steer parents (composed / composition), and the
        # refusal below, whose own message tells the operator to add that override, used to fire before it was ever consulted
        for spelling in (dkind, derivation, arm_legacy, cell_id_legacy, b.staged_id):
            m = seeds.object_kind_for(spelling) if spelling else None
            if m:
                conf.derived("object_kind", f"legacy_spellings.yaml object_kinds[{spelling}] → {m['object_kind']} (derived_meta.json kind={dkind!r})")
                info["legacy_object_kind_entry"] = m
                return str(m["object_kind"]), info
        raise SidecarRefusal(b.dir, f"derived_meta.json kind={dkind!r} derivation={derivation!r} maps to no object_kind in the vocabulary "
                                    f"(known: lora_subtraction → lora_subtraction, psr + psr_residual_mse → relu_steer_distilled_difference); "
                                    f"add a legacy_spellings.yaml object_kinds override; keys seen: {sorted(dm)}")
    for src_name, src in (("blob", blob), ("object.pt.provenance.json", prov)):
        if isinstance(src, Mapping) and src.get("kind"):
            k = str(src["kind"])
            if k in _BLOB_KIND_TO_OBJECT_KIND:
                conf.recorded("object_kind", f"{src_name}:kind={k}")
                return _BLOB_KIND_TO_OBJECT_KIND[k], info
            why = f"{src_name} kind={k!r} maps to no object_kind in the vocabulary; keys seen: {sorted(src)}"
            # a kind ruled out of scope is parked by name, never a defect; an UNKNOWN kind still refuses loudly -- and the declared kind
            # alone decides which (a ruled-out directory / arm_legacy spelling beside it is a conflict, not proof: :func:`_declared_kind_refusal`)
            raise _declared_kind_refusal(b, seeds, f"{src_name}:kind", k, why, arm_legacy, cell_id_legacy)
    fm = sc.get("fit_meta.json")
    if fm and str(fm.get("schema", "")).startswith("alphasteer_fit_meta"):
        conf.recorded("object_kind", "fit_meta.json:schema")
        return "alphasteer", info
    oj = sc.get("object.json")
    if oj and oj.get("method") in _BLOB_KIND_TO_OBJECT_KIND:
        conf.recorded("object_kind", "object.json:method")
        return _BLOB_KIND_TO_OBJECT_KIND[str(oj["method"])], info
    for spelling in (arm_legacy, cell_id_legacy, b.staged_id):
        if not spelling:
            continue
        m = seeds.object_kind_for(spelling)
        if m:
            conf.derived("object_kind", f"legacy_spellings.yaml object_kinds[{spelling}] → {m['object_kind']}")
            info["legacy_object_kind_entry"] = m
            return str(m["object_kind"]), info
    if arm_legacy and arm_legacy.startswith("psr"):
        conf.derived("object_kind", f"arm spelling {arm_legacy!r} starts with psr")
        return "relu_steer", info
    if arm_legacy and _LORA_RANK_RE.search(arm_legacy):
        conf.derived("object_kind", f"arm spelling {arm_legacy!r} names a LoRA rank")
        return "lora", info
    why = (f"object_kind undecidable: arm={arm_legacy!r} cell_id={cell_id_legacy!r} staged_id={b.staged_id!r} blob_kind="
           f"{blob.get('kind')!r}; sidecars seen: {sorted(sc)}; add a legacy_spellings.yaml object_kinds entry")
    if oj and oj.get("method"):
        # `object.json:method` is the bundle's own statement of its kind (the bidpo fits): it alone decides park vs refuse, as above
        raise _declared_kind_refusal(b, seeds, "object.json:method", str(oj["method"]), why, arm_legacy, cell_id_legacy)
    # no sidecar declares the contents' kind (the blob unread or kind-less, no provenance, no object.json): the NAME is the only evidence,
    # and a name the table rules out of scope parks the bundle (the live lofit / alphakl_soft bundles when blob reads are off)
    oos = seeds.out_of_scope_for(arm_legacy, cell_id_legacy, b.staged_id)
    if oos:
        raise OutOfScopeObject(b.dir, _out_of_scope_why(why, oos), oos[0], oos[1])
    raise SidecarRefusal(b.dir, why)


def _declared_kind_refusal(b: SidecarBundle, seeds: Seeds, declared_by: str, declared: str, why: str,
                           arm_legacy: Optional[str], cell_id_legacy: Optional[str]) -> SidecarRefusal:
    """The refusal for a kind the bundle DECLARES (the blob's or ``object.pt.provenance.json``'s ``kind``, ``object.json``'s ``method``)
    that the vocabulary does not spell. :class:`OutOfScopeObject` (parked by name) only when the declared spelling ITSELF is in the
    out-of-scope table -- its exact spelling, or an alias the table maps (a blob's own ``kind`` beside the table's canonical
    spelling); otherwise a plain :class:`SidecarRefusal`. The directory name and the ``arm_legacy`` / ``cell_id_legacy`` spellings are NOT consulted for a bundle that
    declares its kind: an out-of-scope name beside a declared kind the table does not list is a CONFLICT the refusal names, never proof
    of the contents' kind. Before this rule a ``<si>__lofit`` directory rewritten by its producer with
    ``kind='new_unruled_method'`` parked as ``lofit``, the bank accepted an empty revision for contents nobody had classified, and the
    status attributed the new contents to the old out-of-scope entry."""
    oos = seeds.out_of_scope_for(declared)
    if oos:
        return OutOfScopeObject(b.dir, _out_of_scope_why(why, oos), oos[0], oos[1])
    by_name = seeds.out_of_scope_for(arm_legacy, cell_id_legacy, b.staged_id)
    if by_name:
        why += (f"; NOT parked: the bundle's name spells {by_name[0]!r} (arm_legacy={arm_legacy!r} cell_id_legacy={cell_id_legacy!r} staged_id="
                f"{b.staged_id!r}), a kind legacy_spellings.yaml out_of_scope_object_kinds rules out of scope (results-registry D37), but the "
                f"declared {declared_by}={declared!r} is not that kind nor a spelling the table maps to it -- a name is not proof of the "
                f"contents' kind; rule on {declared!r} (add it to the table row it belongs to, or to the vocabulary) before this bundle is read")
    return SidecarRefusal(b.dir, why)


def _synthetic_control_refusal(b: SidecarBundle, seeds: Seeds, why: str) -> SidecarRefusal:
    """The refusal for a bundle whose ``train_budget_meta.json`` declares ``synthetic_control: true`` -- a control of the fitting pipeline,
    never an object under test; no ``fit_target_kind`` admits it. The synthetic controls are ruled out of scope BY NAME
    (``diffmean_random | rand_ | …``), so the bundle parks (:class:`OutOfScopeObject`) only when a spelling of ITS OWN is in the table: the
    control's ``arm`` / ``cell_id`` as the sidecar spells them, its directory, or a kind it declares (the blob's / ``object.pt.provenance.json``'s
    ``kind``, ``object.json``'s ``method``) -- ``kind_legacy`` is that spelling (``rand_recipeB_s31``, not the table's ``rand_`` prefix; a
    directory matched through its ``<si>__<kind>`` tail reads as the tail). A control none of whose spellings the table names refuses
    plainly, naming every spelling tried: a ruling on the Gaussian controls is not a ruling on every control a producer may write, and a
    bundle parked under a ruling that does not name it is omitted silently under the wrong citation. Before this rule the lookup ended on
    an unconditional ``diffmean_random``, so ``synthetic_control: true`` under ``arm='other_method'`` parked as the Gaussian control. The
    declared kinds are further spellings to MATCH, never a veto the way :func:`_declared_kind_refusal` reads them: the flag is the
    producer's own statement that the bundle is a control, and the live controls' blobs carry a ``kind`` of their own beside a ruled
    ``arm`` (the bank reads blobs by default). With no table in the seeds (a caller-built Seeds) the plain refusal stands, as before."""
    sc = b.sidecars
    tbm = sc.get("train_budget_meta.json") or {}
    blob = b.blob_meta if isinstance(b.blob_meta, Mapping) else {}
    prov = sc.get("object.pt.provenance.json")
    oj = sc.get("object.json")
    declared = [("train_budget_meta.json:arm", tbm.get("arm")), ("train_budget_meta.json:cell_id", tbm.get("cell_id")), ("directory", b.staged_id),
                ("blob:kind", blob.get("kind")), ("object.pt.provenance.json:kind", prov.get("kind") if isinstance(prov, Mapping) else None),
                ("object.json:method", oj.get("method") if isinstance(oj, Mapping) else None)]
    spellings = [(by, str(sp)) for by, sp in declared if sp]
    for by, sp in spellings:
        oos = seeds.out_of_scope_for(sp)
        if oos:
            tail = sp.split("__", 1)[1] if "__" in sp else None
            if sp == oos[0] or sp.startswith(oos[0]):
                kind_legacy = sp
            elif tail and (tail == oos[0] or tail.startswith(oos[0])):
                kind_legacy = tail
            else:
                kind_legacy = oos[0]
            return OutOfScopeObject(b.dir, _out_of_scope_why(why, oos), kind_legacy, oos[1])
    if seeds.out_of_scope_by_spelling or seeds.out_of_scope_prefixes:
        why += ("; NOT parked: none of this control's spellings is in legacy_spellings.yaml out_of_scope_object_kinds ("
                + ", ".join(f"{by}={sp!r}" for by, sp in spellings) + ") -- results-registry D37 rules the synthetic controls out of scope "
                "BY NAME (`diffmean_random | rand_ | …`), and a spelling the table does not name inherits no ruling; rule on this control "
                "(add its spelling to the table row it belongs to) before this bundle is read")
    return SidecarRefusal(b.dir, why)


def _out_of_scope_why(why: str, oos: Tuple[str, Dict[str, object]]) -> str:
    """The refusal text of a parked bundle: the reader's own reason, then the ruling it stands on."""
    sp, ent = oos
    return (f"{why}; out of scope per legacy_spellings.yaml out_of_scope_object_kinds[{sp}] (results-registry D37, 2026-09-16): "
            f"{str(ent.get('reason', ''))[:200]}")


def _kind_or_ruling(b: SidecarBundle, seeds: Seeds, conf: "_Conf") -> Tuple[str, Dict[str, object]]:
    """The ``object_kind`` of a bundle, or the refusal that stops :func:`build_object_payload` before any column is read: the two steps that
    can rule a bundle OUT OF SCOPE (:class:`OutOfScopeObject`), in the builder's order -- a ``train_budget_meta.json`` declaring
    ``synthetic_control: true`` (:func:`_synthetic_control_refusal`), then the kind (:func:`_object_kind`). Nothing after these two raises an
    out-of-scope ruling, so a bundle this function resolves builds a payload or refuses plainly, never parks."""
    tbm = b.sidecars.get("train_budget_meta.json")
    if tbm and tbm.get("synthetic_control") is True:
        why = ("train_budget_meta.json:synthetic_control=true — a synthetic control direction, not a fitted object "
               f"({str(tbm.get('control_note', ''))[:120]!r}); no fit_target_kind admits it (a vocabulary / ruling decision, not an adapter default)")
        # the synthetic controls are ruled out of scope BY NAME (`diffmean_random | rand_ | …`): a control whose own spelling the table names
        # parks, never a defect; one it does not name refuses, naming the spellings tried (:func:`_synthetic_control_refusal`)
        raise _synthetic_control_refusal(b, seeds, why)
    return _object_kind(b, seeds, conf)


#: the three things the out-of-scope table can say about one bundle (:class:`ScopeVerdict.state`)
SCOPE_PARKED, SCOPE_IN_SCOPE, SCOPE_UNDECIDED = "parked", "in_scope", "undecided"


@dataclass
class ScopeVerdict:
    """What the vocabulary and the out-of-scope table say about ONE bundle before its payload is built (:func:`scope_verdict`): ``parked`` -- its kind is
    one the table rules out of scope (``ruling`` is the :class:`OutOfScopeObject` the builder raises for it); ``in_scope`` -- its
    ``object_kind`` resolves (``object_kind``); ``undecided`` -- the builder refuses it plainly (``refusal``): a kind nobody has ruled on, an
    undecidable bundle."""
    state: str
    ruling: Optional[OutOfScopeObject] = None
    object_kind: Optional[str] = None
    refusal: Optional[SidecarRefusal] = None


def scope_verdict(b: SidecarBundle, seeds: Seeds) -> ScopeVerdict:
    """Classify one bundle without building its payload: the same two steps :func:`build_object_payload` runs first
    (:func:`_kind_or_ruling`), their outcome returned instead of raised. The objects bank runs this on EVERY directory of an artifact
    group before it parks the group (round 2 of the review of registry-hardening-2b, 2026-10-07): identical bytes staged under an in-scope
    fitted name and under a richer directory whose sidecars declared a ruled-out kind made the ruled-out bundle the group's primary, and
    its ruling parked the whole group -- the in-scope twin lost its object record with no refusal naming the conflict."""
    try:
        kind, _ = _kind_or_ruling(b, seeds, _Conf())
    except OutOfScopeObject as exc:
        return ScopeVerdict(SCOPE_PARKED, ruling=exc)
    except SidecarRefusal as exc:
        return ScopeVerdict(SCOPE_UNDECIDED, refusal=exc)
    return ScopeVerdict(SCOPE_IN_SCOPE, object_kind=kind)


def _dose_copy(b: SidecarBundle) -> Optional[Dict[str, object]]:
    """The dose-scale record of a scaled copy, in either spelling the banks use (mirrors :func:`derive._dose_stamp`):

    * the two-SI ladders' ``object.pt.provenance.json:dose_scale`` / ``train_budget_meta.json:dose_scaled_from`` (scale + parent sha);
    * the Arditi all-positions lane's ``derivation: dose_scale`` + ``parent_object_ids: [<sha>]`` + ``params: {base_scale: <k>}``, written
      into ``staged_from.json`` / ``fit_provenance.json`` / ``object.pt.provenance.json`` alike (``derive.dose_scale_derivation``); the
      provenance's ``derived`` block (how, parent view, the verification against the ladder invariant) is the scaling method.
    ``spelling`` names the key the record was read from, so the confidence entries cite what was actually read."""
    prov = b.sidecars.get("object.pt.provenance.json") or {}
    tbm = b.sidecars.get("train_budget_meta.json") or {}
    dsf = tbm.get("dose_scaled_from") if isinstance(tbm.get("dose_scaled_from"), Mapping) else None
    if prov.get("dose_scale") is not None or dsf is not None:
        scale = prov.get("dose_scale", dsf.get("scale") if dsf else None)
        parent = prov.get("parent_object_sha256") or (dsf or {}).get("parent_object_sha256")
        out = {"scale": scale, "parent_object_sha256": parent, "block": dsf or {}, "provenance": prov,
                "spelling": ("train_budget_meta.json:dose_scaled_from" if dsf else "object.pt.provenance.json:dose_scale"), "lane": None}
        out["params"] = None          # the two-SI spelling carries no params block of its own
        return out
    spelled = D.dose_scale_derivation({k: v for k, v in b.sidecars.items() if isinstance(v, Mapping)})
    if spelled is None:
        return None
    name, rec = spelled
    params = rec.get("params") if isinstance(rec.get("params"), Mapping) else {}
    derived = prov.get("derived") if isinstance(prov.get("derived"), Mapping) else {}
    derived_record = next((s["derived_rung"] for s in (b.sidecars.get(n) for n in D.DOSE_SCALE_DERIVATION_SIDECARS)
                           if isinstance(s, Mapping) and isinstance(s.get("derived_rung"), Mapping)), {})
    block = {"scaling_method": derived.get("how") or "the parent direction multiplied by params.base_scale",
             "parent_staged_id": (f"{derived_record.get('from')}__{rec.get('cell_id')}" if derived_record.get("from") and rec.get("cell_id") else None),
             "parent_identity": {"parent_view": derived.get("parent_view") or derived_record.get("from"),
                                 "parent_object_sha256": derived.get("parent_object_sha256")},
             "verification": derived.get("verified"), "why": derived.get("why"), "built_at": derived.get("built_at"), "built_by": derived.get("built_by"),
             "c": rec.get("c"), "dose_unit": rec.get("dose_unit")}
    stf = b.sidecars.get("staged_from.json") or {}
    out = {"scale": params.get("base_scale", params.get("scale")), "parent_object_sha256": rec["parent_object_ids"][0], "block": block,
            "provenance": prov, "spelling": f"{name}:derivation=dose_scale+parent_object_ids+params.base_scale",
            "lane": (stf.get("lane") if isinstance(stf.get("lane"), str) and stf.get("lane") else None)}
    out["params"] = dict(rec["params"]) if isinstance(rec.get("params"), Mapping) else None     # the record's own params block, verbatim
    return out


def _dose_copy_stamp(dm: Optional[Mapping[str, object]], tbm: Optional[Mapping[str, object]],
                     dose: Optional[Dict[str, object]]) -> Tuple[Optional[Mapping[str, object]], str]:
    """The dose COPY's own record of its making -- the ``dose_scaled_from`` block ``scale_lora_dose.py`` writes beside the copy -- with
    the key it was read from: ``derived_meta.json``'s first (a LoRA subtraction's dose child carries the parent's
    ``derived_meta.json`` plus the stamp), else ``train_budget_meta.json``'s (the two-SI ladders: the parent's fit record plus the stamp),
    else the all-positions derivation's assembled block (:func:`_dose_copy`: its ``built_by`` / ``built_at`` are the derivation's own).
    ``(None, "")`` when the copy carries none -- the columns the stamp would state are then OMITTED by the bank, never read from the
    parent's record copied beside the copy."""
    for src, key in ((dm, "derived_meta.json:dose_scaled_from"), (tbm, "train_budget_meta.json:dose_scaled_from")):
        if isinstance(src, Mapping) and isinstance(src.get("dose_scaled_from"), Mapping):
            return src["dose_scaled_from"], key
    block = (dose or {}).get("block") if dose else None
    if isinstance(block, Mapping) and (block.get("built_by") or block.get("built_at")):
        return block, str((dose or {}).get("spelling") or "derivation")
    return None, ""


def _subtraction_anchor(b: SidecarBundle, dm: Optional[Mapping[str, object]], dose: Optional[Dict[str, object]],
                        artifact_sha: str) -> Optional[Dict[str, object]]:
    """The dose ladder's 1.0 ANCHOR of a LoRA subtraction, or None.

    ``scripts/sdfbase/scale_lora_dose.py --anchor`` re-serves the subtraction's OWN bytes under the ladder's dose-1 staged id,
    copying ``lora_subtract.py``'s ``derived_meta.json`` and adding a ``dose_scaled_from`` block with ``anchor: true``,
    ``scale: 1.0`` and ``parent_object_sha256`` == the object's own sha (plus an ``object.pt.provenance.json`` with
    ``dose_scale: 1.0`` naming the same sha). Read as a dose copy, such an object records ITSELF as its only parent, so the
    anchor is what :func:`_dose_copy` must NOT decide: the object is the subtraction, its parents are the two fits the record
    names, and the stamp is evidence that nothing was scaled. Returns ``{"stamp": <the block, norms dropped>, "spelled_by":
    <which keys made it an anchor>}``; refuses a stamp that says anchor while naming another object's sha."""
    if not isinstance(dm, Mapping) or str(dm.get("kind")) != "lora_subtraction":
        return None
    dsf = dm.get("dose_scaled_from") if isinstance(dm.get("dose_scaled_from"), Mapping) else None
    spelled: List[str] = []
    if dsf is not None and dsf.get("anchor") is True:
        spelled.append("derived_meta.json:dose_scaled_from.anchor=true")
    if dsf is not None and _num(dsf.get("scale")) == 1.0 and dsf.get("parent_object_sha256") == artifact_sha:
        spelled.append("derived_meta.json:dose_scaled_from.scale=1.0 + parent_object_sha256 == own sha")
    if dose is not None and _num(dose.get("scale")) == 1.0 and dose.get("parent_object_sha256") == artifact_sha:
        spelled.append(f"{dose.get('spelling')}=1.0 + parent_object_sha256 == own sha")
    if not spelled:
        return None
    named = (dsf or {}).get("parent_object_sha256") if dsf is not None else (dose or {}).get("parent_object_sha256")
    if isinstance(named, str) and named != artifact_sha:
        raise SidecarRefusal(b.dir, f"dose_scaled_from says anchor (the object's own bytes re-served) but names parent_object_sha256 "
                                    f"{named[:12]}… while the object is {artifact_sha[:12]}…; an anchor's parent is itself — which record is wrong?")
    stamp: Dict[str, object] = {k: v for k, v in (dsf or {}).items() if k != "norms_by_tensor_and_layer"}
    if isinstance(stamp.get("verification"), Mapping):
        ver = stamp["verification"]
        stamp["verification"] = {k: v for k, v in ver.items() if k != "checks"}
        if isinstance(ver.get("checks"), list):
            stamp["verification"]["checks"] = [{k: c.get(k) for k in ("check", "status") if isinstance(c, Mapping) and k in c} for c in ver["checks"]]
    if dsf is None and dose is not None:
        stamp = {"scale": dose.get("scale"), "parent_object_sha256": dose.get("parent_object_sha256"), "spelling": dose.get("spelling")}
    return {"stamp": stamp, "spelled_by": spelled}


def _model_id(hf_repo: Optional[str], seeds: Seeds, b: SidecarBundle) -> str:
    if not hf_repo:
        raise SidecarRefusal(b.dir, "no sidecar names the fit model (train_budget_meta.json:model_id / fit_provenance.json:resolved_model_id / "
                                    "derived_meta.json:model_id / provenance.json:model / summary.json:model)")
    mid = seeds.model_id_by_hf_repo.get(hf_repo)
    if not mid:
        raise SidecarRefusal(b.dir, f"fit model {hf_repo!r} has no `model` row / models.yaml entry (the model closure must cover it, plan F15); "
                                    f"known repos: {sorted(seeds.model_id_by_hf_repo)[:20]}")
    return mid


def _si_sha(si_id: str, seeds: Seeds, b: SidecarBundle) -> str:
    sha = seeds.si_sha_by_id.get(si_id)
    if not sha:
        raise SidecarRefusal(b.dir, f"teacher instruction {si_id!r} has no si_text row / si_texts.yaml entry (fit_teacher_si_text_sha256 is NOT NULL "
                                    f"under the fit_target_kind CHECK); known: {sorted(seeds.si_sha_by_id)}")
    return sha


def _layers(b: SidecarBundle, conf: _Conf, kind: str) -> Tuple[Optional[List[int]], Optional[str]]:
    sc = b.sidecars
    blob = b.blob_meta or {}
    val, key = _first((sc.get("train_budget_meta.json"), "target_layers"), (blob, "layers"), (sc.get("object.pt.provenance.json"), "layers"),
                      (sc.get("derived_meta.json"), "layers"), (sc.get("derived_meta.json"), "target_layers"), (sc.get("meta.json"), "layers"),
                      (sc.get("object.json"), "layers"), (sc.get("fit_meta.json"), "layers"))
    src_name = None
    if val is not None:
        for name, d in (("train_budget_meta.json", sc.get("train_budget_meta.json")), ("blob", blob),
                        ("object.pt.provenance.json", sc.get("object.pt.provenance.json")), ("derived_meta.json", sc.get("derived_meta.json")),
                        ("meta.json", sc.get("meta.json")), ("object.json", sc.get("object.json")), ("fit_meta.json", sc.get("fit_meta.json"))):
            if d and key in d and d[key] is not None and d[key] == val:
                src_name = name
                break
    if isinstance(val, list) and all(isinstance(x, int) for x in val):
        conf.recorded("fit_layers_json", f"{src_name}:{key}")
        return list(val), f"{src_name}:{key}"
    tbm = sc.get("train_budget_meta.json") or {}
    for name, d in (("train_budget_meta.json", tbm), ("blob", blob), ("meta.json", sc.get("meta.json") or {})):
        lay = d.get("layer") if isinstance(d, Mapping) else None
        if isinstance(lay, int) and not isinstance(lay, bool):
            conf.recorded("fit_layers_json", f"{name}:layer")
            return [lay], f"{name}:layer"
    md = blob.get("metadata") if isinstance(blob.get("metadata"), Mapping) else None
    if md:
        if isinstance(md.get("layers"), list) and all(isinstance(x, int) for x in md["layers"]):
            conf.recorded("fit_layers_json", "blob:metadata.layers")
            return list(md["layers"]), "blob:metadata.layers"
        for key in ("r_norms", "per_layer_r_norm", "projection_targets"):
            if isinstance(md.get(key), Mapping) and md[key]:
                try:
                    ls = sorted(int(k) for k in md[key])
                except (TypeError, ValueError):
                    continue
                conf.recorded("fit_layers_json", f"blob:metadata.{key} keys")
                return ls, f"blob:metadata.{key} keys"
    return None, None


def _parent_spelling(blk: object) -> Optional[str]:
    if not isinstance(blk, Mapping):
        return None
    return str(blk.get("object_sha256") or blk.get("cell_id") or "") or None


def _parent_sha_by_spelling(b: SidecarBundle, role: str, blk: Mapping[str, object], dm: Mapping[str, object],
                            parents_in_bank: Optional[Mapping[str, Mapping[str, object]]]) -> str:
    """The sha of a subtraction parent that ``derived_meta.json`` names only by its staged spelling (``cell_id``): the ONE fit
    record in the banks on this machine with that spelling on the same checkpoint (``model_id``); refused by name otherwise."""
    spelling = str(blk.get("cell_id"))
    model = blk.get("model_id") or dm.get("model_id")
    if not parents_in_bank:
        raise SidecarRefusal(b.dir, f"derived_meta.json:{role} names no object_sha256 (only cell_id={spelling!r}) and no bank index was supplied "
                                    f"to resolve it (objects_bank reads the banks' fit records; a direct read cannot)")
    hits = sorted({sha for sha, rec in parents_in_bank.items()
                   if rec.get("cell_id_legacy") == spelling and (model is None or rec.get("model_id") == model)})
    if len(hits) != 1:
        raise SidecarRefusal(b.dir, f"derived_meta.json:{role} names no object_sha256 (only cell_id={spelling!r}, model_id={model!r}); "
                                    f"the banks on this machine hold {len(hits)} fit(s) of that spelling on that checkpoint "
                                    f"({[h[:12] for h in hits]}) — one is needed to resolve the parent, never a guess from the name")
    return hits[0]


def _batch_through_parents(p: Dict[str, object], conf: "_Conf", extra: Dict[str, object], anchor_parents: Mapping[str, str],
                           parents_in_bank: Optional[Mapping[str, Mapping[str, object]]]) -> None:
    """A subtraction has no batch of its own; the anchor's ``fit_effective_batch`` is its parents' when both fit records agree.
    Both parents' values (and where each was read) are recorded in ``extra_json.parent_fit_effective_batch`` whatever happens;
    the column resolves only on agreement, else stays NULL with the reason in ``extra_json.fit_effective_batch_unresolved``."""
    seen: Dict[str, Dict[str, object]] = {}
    for role in ("parent_pos", "parent_neg"):
        sha = anchor_parents.get(role)
        rec = (parents_in_bank or {}).get(sha) if sha else None
        tbm = rec.get("train_budget_meta") if isinstance(rec, Mapping) and isinstance(rec.get("train_budget_meta"), Mapping) else None
        seen[role] = {"object_sha256": sha,
                      "effective_batch": _int(tbm.get("effective_batch")) if tbm else None,
                      "batch_size": _int(tbm.get("batch_size")) if tbm else None,
                      "accum_steps": _int(tbm.get("accum_steps")) if tbm else None,
                      "resolved_batch_size": _int(tbm.get("resolved_batch_size")) if tbm else None,
                      "source": (os.path.join(str(rec["dir"]), "train_budget_meta.json") + ":effective_batch") if rec else None}
    extra["parent_fit_effective_batch"] = seen
    values = {role: v["effective_batch"] for role, v in seen.items()}
    missing = [role for role, v in values.items() if v is None]
    if missing:
        why = ("a subtraction's effective batch resolves through its parents' fit records, and "
               + ", ".join(f"{r} ({str(anchor_parents.get(r))[:12]}…)" for r in missing)
               + (" is not staged in a bank on this machine" if len(missing) == 1 else " are not staged in a bank on this machine")
               + " (or its record names no effective_batch)")
        extra["fit_effective_batch_unresolved"] = why
        conf.unverified("fit_effective_batch", why)
        return
    if len(set(values.values())) != 1:
        why = ("a subtraction's effective batch resolves through its parents' fit records, which DISAGREE: "
               + ", ".join(f"{r}={v}" for r, v in values.items()) + "; left NULL rather than picking one")
        extra["fit_effective_batch_unresolved"] = why
        conf.unverified("fit_effective_batch", why)
        return
    p["fit_effective_batch"] = values["parent_pos"]
    conf.derived("fit_effective_batch", "both parents' train_budget_meta.json:effective_batch agree "
                                        f"(parent_pos={values['parent_pos']}, parent_neg={values['parent_neg']}); a subtraction has no batch of its own")


def _integral(v) -> bool:
    """A batch value in the record's own type: an ``int`` that is not a ``bool``. A float (an integral 64.0 included), a numeric string, a
    boolean or an absent key is NOT one -- the trainers record integers, and a value coerced into one would be a reading the record never
    made. The same rule ``reconcile fit-effective-batch`` reads the folded row by, so the adapter and the pass agree about what an integer is."""
    return isinstance(v, int) and not isinstance(v, bool)


#: the rule a dose copy's ``fit_effective_batch`` is derived by, as the confidence entry and the unresolved reason spell it
_DOSE_BATCH_RULE = "a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13)"
#: where the one reading is taken: the copy's own ``train_budget_meta.json``, which is the PARENT's fit record copied beside the copy
_DOSE_BATCH_COPIED_SOURCE = "train_budget_meta.json:effective_batch (the PARENT's fit record copied beside the copy)"


def dose_batch_derived_source(parent_sha: str) -> str:
    """The ``source`` of a dose copy's ``derived`` confidence entry for ``fit_effective_batch``, word for word -- the parent named by its
    sha's first twelve hex digits, the one record the value is read from, and the rule. ``reconcile fit-effective-batch`` recognises the
    adapter's statement by the entry's ``confidence`` (``derived``), never by this text or by its writer: the sidecar hook's dose path
    writes the same words for the copies it files, and the pass must read both alike."""
    return f"the parent's effective batch ({parent_sha[:12]}…), read from {_DOSE_BATCH_COPIED_SOURCE}; {_DOSE_BATCH_RULE}"


def _dose_batch_from_parent(p: Dict[str, object], conf: "_Conf", extra: Dict[str, object], dose: Mapping[str, object],
                            tbm: Optional[Mapping[str, object]]) -> None:
    """A dose copy's ``fit_effective_batch`` is its PARENT's effective batch, stated ``derived``: the 92 ``object``
    rows whose column read NULL -- this adapter's "nothing was fit" convention -- beside the parent's recorded hyperparameter block, which
    ``reconcile fit-effective-batch`` listed as ambiguous; the alternative, keeping the NULL and having the sidecar hook's dose path omit
    the column too, was rejected. ONE source: the parent's fit record COPIED beside
    the copy -- the copy's own ``train_budget_meta.json``, which ``scale_lora_dose.py`` / ``scale_psr_dose.py`` write as the parent's record
    plus the ``dose_scaled_from`` stamp, and which ``fit_hyperparameters_json`` carries verbatim, so the value stated here is the block's
    own ``effective_batch`` and a copy stated from it is consistent by construction. The record is an input of the copy's OWN group, so
    the copy's revision already covers it: nothing of another group's directory is read, and no dependency machinery is needed. Whether
    the copied record agrees with the parent's OWN registry row is the registry's to say, read-only (``reconcile fit-effective-batch``
    cross-checks every derived copy against its parent row and lists a disagreement by name, never writing for it).

    The reading and where it was taken ride ``extra_json.dose_parent_fit_effective_batch`` whatever happens, with ``unresolved`` -- the
    reason the column is NULL, or None when it resolved -- so the LATEST statement's dict says what the row's column is, whatever an
    earlier emission parked under ``extra`` (the fold merges ``extra`` by key, so a top-level marker outlives the emission that wrote it;
    the dict is rewritten whole). The top-level ``extra_json.fit_effective_batch_unresolved`` is stated on EVERY emission for the same
    reason: the reason when the column is NULL, and explicitly ``None`` when it resolved, so a copy once emitted unresolved and re-emitted
    after its copied record was repaired folds with the derived integer and a null reason, never "left NULL" beside a value (review round
    sol-2, 2026-10-13: a key the successful path did not restate kept the first emission's reason for good). A record PRESENT is a reading
    -- its value, or its lack of one -- never skipped: the column resolves when
    the copied record names an ``effective_batch`` that is an integer in its own type (:func:`_integral`: an ``int`` that is not a
    ``bool``; ``64.0`` and ``"8"`` are refused by name with the value and its type, nothing coerced), and it is stated NULL -- the column
    PRESENT in the payload as ``None``, the reason by name in ``extra_json.fit_effective_batch_unresolved`` and an ``unverified`` confidence
    entry -- when the copy carries no ``train_budget_meta.json`` at all (a dose copy of a derived object's bytes: ``derived_meta.json``
    beside it, and a parent with no fit record of its own), when the record present names no ``effective_batch`` (the key absent or
    null), or when the value it names is not an integer.

    The NULL is an explicit statement, never an omission: an omitted column keeps its folded value, so a copy first emitted ``derived``
    8 and re-emitted after its copied record moved to a value this function refuses would have kept the 8 beside a block it no longer
    agrees with while the ``unverified`` entry replaced the ``derived`` one. Stated as null, the fold writes NULL and the pass lists the
    row as left NULL by name."""
    parent = str(dose.get("parent_object_sha256"))
    copied: Optional[Mapping[str, object]] = tbm if isinstance(tbm, Mapping) else None
    value = copied.get("effective_batch") if copied is not None else None
    evidence: Dict[str, object] = {
        "parent_object_sha256": parent,
        "copied_record": ({"effective_batch": value, "source": _DOSE_BATCH_COPIED_SOURCE} if copied is not None else None),
        "unresolved": None}
    extra["dose_parent_fit_effective_batch"] = evidence

    def unresolved(why: str) -> None:
        # an explicit NULL: the column is present as None so the fold writes NULL over any earlier statement (never an omission, which
        # would keep a retained derived value beside a block it no longer agrees with)
        p["fit_effective_batch"] = None
        evidence["unresolved"] = why
        extra["fit_effective_batch_unresolved"] = why
        conf.unverified("fit_effective_batch", why)

    if copied is None:
        unresolved(f"{_DOSE_BATCH_RULE}, and the copy carries no train_budget_meta.json (the parent {parent[:12]}… is a derived object with "
                   "no fit record of its own to be copied beside the copy); left NULL")
        return
    if value is None:
        unresolved(f"{_DOSE_BATCH_RULE}, and the parent's fit record copied beside the copy (train_budget_meta.json, parent {parent[:12]}…) "
                   "names no effective_batch; a record present and naming none is read as naming none; left NULL")
        return
    if not _integral(value):
        unresolved(f"{_DOSE_BATCH_RULE}, and the parent's fit record copied beside the copy (train_budget_meta.json, parent {parent[:12]}…) "
                   f"names effective_batch {value!r} ({type(value).__name__}), not an integer; nothing is coerced; left NULL")
        return
    p["fit_effective_batch"] = value
    # the current state of the top-level reason too, never left to an earlier emission: the fold merges `extra` by key, so a copy once
    # emitted unresolved and re-emitted after its copied record was repaired would otherwise keep "left NULL" beside the derived integer
    # (review round sol-2, 2026-10-13); stated null, the fold writes null over it
    extra["fit_effective_batch_unresolved"] = None
    conf.derived("fit_effective_batch", dose_batch_derived_source(parent))


#: The ``status`` a bank object reads as when the bundle states none (see :func:`bank_status_statement`): a bank directory with no
#: supersession or withdrawal evidence beside it. This is a DEFAULT, not a statement the bank makes -- the bank is a mirror of the
#: objects, and whether an object is ``approved`` or the paper's ``reference`` is its producer's or a ruling's to say. On 2026-10-03
#: the objects_bank adapter re-read three directories a producer had just staged and filed this default over the producers'
#: ``reference``; the per-column latest-statement fold took it and demoted three paper reference objects. Since then
#: :func:`build_object_payload` files NO ``status`` in that case (the record omits the column; the fold keeps the folded value, the
#: harvester blocks a brand-new object on its own row until its producer states one), so this value is never written by the bank;
#: it names what the producers state for a plain complete fit and what the tests expect the fold to read.
BANK_DEFAULT_STATUS = "approved"

#: The branches of :func:`bank_status_statement`: the two the bank LAYOUT states, the one a matching ``superseded_by.json``
#: states, and the three that state nothing (the default applies).
STATUS_BRANCHES: Tuple[str, ...] = ("withdrawn", "superseded_twin", "superseded_by_json", "refit_of_superseded",
                                     "superseded_by_json_unresolved", "none")


def own_fit_fingerprints(b: SidecarBundle) -> Set[str]:
    """The fit fingerprints the bundle's own sidecars record for THIS object (``COMPLETE`` / ``summary.json`` / ``provenance.json``)."""
    out: Set[str] = set()
    for name in ("COMPLETE", "summary.json", "provenance.json"):
        v, _ = _first((b.sidecars.get(name), "fit_fingerprint"))
        if isinstance(v, str):
            out.add(v)
    return out


def bank_status_statement(b: SidecarBundle) -> Tuple[Optional[str], str, str]:
    """``(status, source, branch)``: the ``status`` the bundle's OWN evidence states, or ``None`` when it states none.

    A bank directory states a status in exactly three ways, and :func:`build_object_payload` files no other: where it SITS
    (``_withdrawn/`` → ``withdrawn``; ``_superseded_by_it64/`` → ``superseded``; both ride :attr:`SidecarBundle.status_hint`,
    set by the bank's enumeration) and a ``superseded_by.json`` whose ``fit_fingerprint`` is this object's own (→ ``superseded``).
    Everything else -- a complete fit with no such evidence, the expected REFIT of a superseded snapshot, a ``superseded_by.json``
    that names neither of this object's fingerprints -- is ``None``: the bank has nothing to say, and the ``source`` names why.
    The ``branch`` is one of :data:`STATUS_BRANCHES`, so a caller can park the matching evidence without re-deriving it."""
    own_fps = own_fit_fingerprints(b)
    sb = b.sidecars.get("superseded_by.json")
    if b.status_hint == "withdrawn":
        return "withdrawn", f"{b.dir} sits under _withdrawn/ (WITHDRAWN.txt)", "withdrawn"
    if b.status_hint == "superseded_twin":
        return "superseded", f"{b.dir} sits under _superseded_by_it64/ (README.txt: re-staged under the it64 ids)", "superseded_twin"
    if sb:
        sfp = sb.get("fit_fingerprint")
        exp = (sb.get("superseded_by") or {}).get("expected_refit_fingerprint") if isinstance(sb.get("superseded_by"), Mapping) else None
        if isinstance(sfp, str) and sfp in own_fps:
            return "superseded", f"superseded_by.json:fit_fingerprint == this object's fit fingerprint ({sfp[:12]}…)", "superseded_by_json"
        if isinstance(exp, str) and exp in own_fps:
            return (None, f"superseded_by.json describes the PRIOR snapshot ({str(sfp)[:12]}…); this object is its expected refit ({exp[:12]}…)",
                    "refit_of_superseded")
        return (None, f"superseded_by.json present but neither its fit_fingerprint nor expected_refit_fingerprint is this object's ({sorted(own_fps)})",
                "superseded_by_json_unresolved")
    return None, "no supersession or withdrawal evidence beside the object", "none"


def build_object_payload(b: SidecarBundle, seeds: Seeds, now: str, blob_reads_enabled: bool = True,
                         parents_in_bank: Optional[Mapping[str, Mapping[str, object]]] = None, omit_producer_owned: bool = False,
                         dose_batch_from_parent: bool = False) -> Dict[str, object]:
    """The ``object`` record payload for one bundle (locations from this directory only; the caller merges twins).

    ``parents_in_bank`` — ``{object sha256: {"dir", "train_budget_meta", "cell_id_legacy", "model_id", "target"}}`` for every FIT
    record staged in the banks on this machine (the objects_bank adapter's index) — is consulted only where a record must
    reach through its parents: a subtraction anchor's ``fit_effective_batch`` (both parents' values, resolved when they agree)
    and a subtraction parent named by its staged spelling alone (resolved to the one fit on the same checkpoint, else refused
    by name). Every other record reads the same with or without it -- a dose copy's ``fit_effective_batch`` (below) included, which
    reads the copy's OWN files alone.

    ``dose_batch_from_parent`` (the objects bank): a dose COPY states ``fit_effective_batch`` as its PARENT's
    effective batch, ``derived``, read from ONE source -- the parent's fit record copied beside the copy, the copy's own
    ``train_budget_meta.json`` (:func:`_dose_batch_from_parent`; that record is an input of the copy's own group, so its
    revision covers it and no other group's directory is read) -- when that record names an integer in its own type; otherwise the
    column is STATED NULL (present as ``None``, so the fold writes NULL over an earlier derived statement) with the reason by name
    (``extra.fit_effective_batch_unresolved``, an ``unverified`` entry): the copy carries no such record, or the record present names
    no ``effective_batch`` or a non-integer one (a record present and naming none is a reading, never skipped). ``False`` (the default) keeps
    the legacy reading -- no statement and no confidence entry, the "nothing was fit" convention -- for the other caller of this
    builder (``relu_paired``, whose records would otherwise change under its unchanged emission version; its own switch is a
    follow-up with its own bump).

    ``omit_producer_owned`` (the objects bank; the no-``status`` rule above generalized): a MIRROR states nothing of a producer-owned column
    that it did not read verbatim. ``fit_code_sha`` is OMITTED when its value would be the bank default (``unverified``: no sidecar
    records the fit code sha) and ``derivation_params_json`` unless it is a sidecar's own ``params`` block verbatim (``derived_meta.json:params``,
    the field the producer-side writer ``hooks.object_record`` copies; the dose-scale derivation record's ``params`` in the all-positions
    spelling) -- the dicts this builder ASSEMBLES from a ``dose_scaled_from`` stamp or a ``derived_meta.json`` record are abbreviations of
    the producer's record (eight of the stamp's twenty-one keys; a producer's dose record names thirteen of its own), and the ``object`` fold
    is per-column latest statement, so a later bank revision stating either would overwrite the producer's native record. The omitted
    columns and why are parked under ``extra.producer_owned_unstated``; an anchor's stamp, evidence the assembled params used to carry,
    moves to ``extra.dose_anchor.stamp``. For a dose COPY, ``fit_writer`` and ``fit_finished_utc`` are the copy's own stamp's ``producer``
    / ``written_at`` (:func:`_dose_copy_stamp`) and are omitted when the stamp records neither -- never the parent's ``builder`` /
    ``built_at`` / ``writer`` / ``COMPLETE.published_at`` copied beside the copy. ``False`` (the default) keeps the legacy statement for the other caller of this builder
    (``relu_paired``, whose own switch is an open follow-up and needs its own emission bump)."""
    conf = _Conf()
    extra: Dict[str, object] = {}
    producer_owned_unstated: Dict[str, str] = {}
    sc = b.sidecars
    tbm = sc.get("train_budget_meta.json")
    dm = sc.get("derived_meta.json")
    fp = sc.get("fit_provenance.json")
    pv = sc.get("provenance.json")
    sm = sc.get("summary.json")
    fm = sc.get("fit_meta.json")
    meta = sc.get("meta.json")
    oj = sc.get("object.json")
    complete = sc.get("COMPLETE")
    sb = sc.get("superseded_by.json")
    stf = sc.get("staged_from.json")
    blob = b.blob_meta or {}
    if b.blob_meta is None and b.blob_path and blob_reads_enabled:
        blob_reason = f"blob_not_readable: {b.blob_read_error}" if b.blob_read_error else "blob_read_disabled"
    elif b.blob_meta is None:
        blob_reason = "blob_not_local" if not b.blob_path else "blob_read_disabled"
    else:
        blob_reason = None

    # the two steps that can rule the bundle out of scope (the synthetic-control check, then the kind), in one place: the objects bank runs
    # the same two on EVERY twin of an artifact before it parks a group (:func:`scope_verdict`), so a park is never one directory's verdict
    kind, kind_info = _kind_or_ruling(b, seeds, conf)
    cell_id_legacy = kind_info.get("cell_id_legacy") or b.staged_id
    arm_legacy = kind_info.get("arm_legacy")

    # ---- artifact identity -----------------------------------------------------------------
    sidecar_sha, sidecar_sha_key = _first((tbm, "object_sha256"), (complete, "object_sha256"), (dm, "object_sha256"), (stf, "object_sha256"),
                                          (pv, "object_sha256"), (fm, "object_sha256"))
    if b.blob_sha256:
        artifact_sha = b.blob_sha256
        conf.recorded("artifact_sha256", f"sha256({os.path.basename(b.blob_path or 'object.pt')}) computed on this machine")
        if isinstance(sidecar_sha, str) and _SHA256_RE.match(sidecar_sha) and sidecar_sha != artifact_sha:
            raise SidecarRefusal(b.dir, f"{sidecar_sha_key} records object sha {sidecar_sha[:12]}… but the bytes on disk hash to "
                                        f"{artifact_sha[:12]}… (which copy did you hash?)")
    elif isinstance(sidecar_sha, str) and _SHA256_RE.match(sidecar_sha):
        artifact_sha = sidecar_sha
        conf.set("artifact_sha256", artifact_sha, "recorded", f"{sidecar_sha_key} (blob_not_local: bytes not re-hashed here)")
    else:
        raise SidecarRefusal(b.dir, f"no weights file on this machine and no sidecar records object_sha256; sidecars seen: {sorted(sc)}")
    object_id = identity.object_id(kind, artifact_sha256=artifact_sha)

    p: Dict[str, object] = {"object_id": object_id, "object_kind": kind, "artifact_sha256": artifact_sha}
    if b.blob_bytes is not None:
        p["artifact_bytes"] = b.blob_bytes
        conf.recorded("artifact_bytes", "stat(object.pt)")
    else:
        conf.unverified("artifact_bytes", blob_reason or "blob_not_local")

    # ---- campaign / legacy id ---------------------------------------------------------------
    p["campaign_cell_id_legacy"] = b.staged_id if b.staged_id != cell_id_legacy else cell_id_legacy
    if b.staged_id != cell_id_legacy:
        extra["cell_id_legacy"] = cell_id_legacy
    if arm_legacy:
        extra["arm_legacy"] = arm_legacy
    camp = b.campaign_hint or seeds.campaign_by_spelling.get(b.bank) or b.bank or "unknown"
    if _REPLICATE_RE.match(b.staged_id):
        camp = "reseed"
    p["campaign"] = camp
    conf.derived("campaign", f"bank {b.bank!r} / staged id prefix")

    # ---- fit_target_kind and its paired columns --------------------------------------------
    dose = _dose_copy(b)
    # the ladder's 1.0 anchor of a subtraction is the subtraction itself (parent sha == own sha): never a dose copy of itself
    anchor = _subtraction_anchor(b, dm, dose, artifact_sha)
    if anchor is not None:
        dose = None
    anchor_parents: Dict[str, str] = {}
    si_id = None
    for d, k in ((tbm, "si_id"), (pv, "si"), (sm, "si"), (dm, "parent_pos.data_scale.si"), (meta, "si_id"), (complete, "si_id")):
        v, _ = _first((d, k))
        if isinstance(v, str) and v:
            si_id = v
            break
    corpus_ids = dict(seeds.corpus_ids)
    corpus_ids.update({c: c for c in _ROLE_BY_CORPUS})
    ftk_sidecars = {k: v for k, v in sc.items() if isinstance(v, Mapping)}
    if dose is not None:
        # the derivation rule reads the dose stamp itself: a dose-scaled copy's train_budget_meta.json is the PARENT's fit record
        ftk, ftk_conf, ftk_src = D.fit_target_kind(kind, ftk_sidecars, corpus_ids, teacher_ids=seeds.si_sha_by_id)
        if ftk != "derived":
            raise SidecarRefusal(b.dir, f"a dose-scaled copy read as fit_target_kind={ftk!r} by derive.fit_target_kind; the dose stamp must decide it")
        conf.set("fit_target_kind", ftk, ftk_conf, ftk_src)
        derivation, derivation_src = "dose_scale", "fit_target_kind / dose stamp"
        parent = dose.get("parent_object_sha256")
        if not (isinstance(parent, str) and _SHA256_RE.match(parent)):
            raise SidecarRefusal(b.dir, "a dose-scaled copy names no parent_object_sha256")
        spelling = str(dose.get("spelling") or "object.pt.provenance.json:dose_scale")
        arditi_spelling = spelling.startswith(D.DOSE_SCALE_DERIVATION_SIDECARS) and "parent_object_ids" in spelling
        p["parent_object_ids_json"] = [parent]
        conf.recorded("parent_object_ids_json", f"{spelling.split('=')[0]}" if arditi_spelling else
                      "object.pt.provenance.json:parent_object_sha256 / dose_scaled_from.parent_object_sha256")
        blk = dose["block"]
        p["derivation_params_json"] = {"scale": dose.get("scale"), "scaling_method": blk.get("scaling_method"), "tensors_scaled": blk.get("tensors_scaled"),
                                       "tensors_unchanged": blk.get("tensors_unchanged"), "anchor": blk.get("anchor"), "parent_staged_id": blk.get("parent_staged_id"),
                                       "parent_identity": blk.get("parent_identity"), "verification": blk.get("verification")}
        if arditi_spelling:
            p["derivation_params_json"].update({k: blk.get(k) for k in ("why", "built_at", "built_by", "c", "dose_unit") if blk.get(k) is not None})
            p["derivation_params_json"]["base_scale"] = dose.get("scale")
        conf.recorded("derivation_params_json", spelling if arditi_spelling else ("train_budget_meta.json:dose_scaled_from" if blk else "object.pt.provenance.json"))
        p["base_scale"] = _num(dose.get("scale"))
        conf.recorded("base_scale", f"{spelling.split(':')[0]}:params.base_scale" if arditi_spelling else "object.pt.provenance.json:dose_scale")
        p["fit_objective"] = "not_applicable"
        conf.derived("fit_objective", "dose-scaled copy: not fit")
        if arditi_spelling:
            extra["parent_record_note"] = ("fit_provenance.json is the PARENT's fit record, copied and re-stamped with this copy's sha "
                                           "(derived_rung.fit_is_unchanged)")
            if isinstance(dose.get("lane"), str):
                # the staging lane names the campaign of a scaled copy (`staged_from.json:lane`), so the derived rung's row sits beside
                # its fitted siblings' rows (the Arditi objects' own reader mints those under the same campaign) rather than under the bank's
                p["campaign"] = str(dose["lane"])
                conf.derived("campaign", "staged_from.json:lane (the staging lane of the dose-scaled copy; its fitted siblings' rows carry the same)")
        else:
            extra["parent_record_note"] = "train_budget_meta.json is the PARENT's fit record (dose_scaled_from.note)"
    else:
        promo_src = None
        if tbm is None and si_id and (pv or sm):
            # a promoted HPO trial: the fit record is summary.json + provenance.json (tune_arm.py), not a train_budget_meta.json
            tf, tf_key = _first((sm, "teacher_forward"), (pv, "teacher_forward"))
            ftk_sidecars["train_budget_meta.json"] = {"si_id": si_id, "teacher_forward": tf, "data_mode": (sm or {}).get("data_mode")}
            promo_src = f"provenance.json:si={si_id} + {'summary.json' if sm and tf_key == 'teacher_forward' and sm.get('teacher_forward') == tf else 'provenance.json'}:teacher_forward={tf}"
        ov = seeds.fit_target_overrides.get(str(si_id)) if si_id else None
        ov_kind = ov.get("fit_target_kind") if ov and ov.get("fit_target_kind") in V.FIT_TARGET_KINDS else None
        try:
            # teacher ids come from the seeds (the si_text rows), never a roster in derive.py
            ftk, ftk_conf, ftk_src = D.fit_target_kind(kind, ftk_sidecars, corpus_ids, teacher_ids=seeds.si_sha_by_id)
            if promo_src:
                ftk_src = promo_src + " (promotion record read as the fit record)"
            if ov_kind is not None and ov_kind != ftk:
                # the sidecar classifies itself one way and a ruling says another: two authorities in conflict, never resolved quietly
                raise SidecarRefusal(b.dir, f"fit_target_kind conflict: the sidecars read as {ftk!r} ({ftk_src}) but legacy_spellings.yaml "
                                            f"fit_target_overrides[{si_id}] rules {ov_kind!r} ({ov.get('citation') or ov.get('source')}); fix one")
            if ov_kind is not None:
                ftk_src += f" (legacy_spellings.yaml fit_target_overrides[{si_id}] agrees: {ov.get('citation') or ov.get('source')})"
        except D.UnclassifiableSidecar as exc:
            if ov_kind is not None:
                ftk, ftk_conf, ftk_src = ov_kind, "derived", f"legacy_spellings.yaml fit_target_overrides[{si_id}]: {ov.get('citation') or ov.get('source')}"
            else:
                raise SidecarRefusal(b.dir, f"fit_target_kind undecidable ({exc}); si_id={si_id!r} arm={arm_legacy!r}; "
                                            f"add a legacy_spellings.yaml fit_target_overrides entry with a citation")
        conf.set("fit_target_kind", ftk, ftk_conf, ftk_src)
        derivation = "none"
        derivation_src = "fit_target_kind / derived_meta.json kind"
        if ftk == "derived":
            if not dm:
                raise SidecarRefusal(b.dir, "fit_target_kind=derived without derived_meta.json")
            dkind = str(dm.get("kind"))
            dder = str(dm.get("derivation", ""))
            key = dkind if dkind in _DERIVED_DERIVATION else dder
            legacy_kind_entry = kind_info.get("legacy_object_kind_entry") or {}
            if key in _DERIVED_DERIVATION:
                derivation = _DERIVED_DERIVATION[key][0]
            elif legacy_kind_entry.get("derivation") in V.DERIVATIONS:
                # the object_kinds override that decided the kind names the derivation too (psr_paired_write → composition)
                derivation = str(legacy_kind_entry["derivation"])
                derivation_src = f"legacy_spellings.yaml object_kinds[{dkind}].derivation"
            else:
                raise SidecarRefusal(b.dir, f"derived_meta.json kind={dkind!r} derivation={dder!r} names no derivation in the vocabulary"
                                            + (f"; the legacy_spellings.yaml object_kinds entry says derivation={legacy_kind_entry.get('derivation')!r}, "
                                               f"not one of {list(V.DERIVATIONS)}" if legacy_kind_entry else
                                               "; add a legacy_spellings.yaml object_kinds entry carrying a `derivation`"))
            parents = []
            resolved_by_spelling: List[str] = []
            for role in ("parent_pos", "parent_neg"):
                blk = dm.get(role)
                if isinstance(blk, Mapping) and isinstance(blk.get("object_sha256"), str):
                    parents.append(blk["object_sha256"])
                    if anchor is not None:
                        anchor_parents[role] = str(blk["object_sha256"])
                elif derivation == "lora_subtraction" and isinstance(blk, Mapping) and blk.get("cell_id"):
                    # a parent named by its staged spelling alone: the one fit of that spelling on the same checkpoint in the banks
                    # on this machine decides its sha — never a guess from the name, refused by name when none or several match
                    sha = _parent_sha_by_spelling(b, role, blk, dm, parents_in_bank)
                    parents.append(sha)
                    resolved_by_spelling.append(f"{role}={blk.get('cell_id')}→{sha[:12]}…")
                    if anchor is not None:
                        anchor_parents[role] = sha
            if not parents:
                raise SidecarRefusal(b.dir, "derived_meta.json names no parent object_sha256 (parent_pos / parent_neg)")
            if derivation == "lora_subtraction" and len(parents) != 2:
                raise SidecarRefusal(b.dir, f"a LoRA subtraction has two parents; derived_meta.json resolves {len(parents)} "
                                            f"(parent_pos={_parent_spelling(dm.get('parent_pos'))!r}, parent_neg={_parent_spelling(dm.get('parent_neg'))!r})")
            p["parent_object_ids_json"] = parents
            if resolved_by_spelling:
                conf.derived("parent_object_ids_json", "derived_meta.json:parent_pos / parent_neg — object_sha256 where recorded; "
                                                       + "; ".join(resolved_by_spelling) + " resolved through the bank's fit of that spelling on the same checkpoint")
            else:
                conf.recorded("parent_object_ids_json", "derived_meta.json:parent_pos.object_sha256, parent_neg.object_sha256")
            params: Dict[str, object] = {"derivation_text": dm.get("derivation"), "scale_matched": dm.get("scale_matched"), "comparison": dm.get("comparison"),
                                         "builder": dm.get("builder"), "built_at": dm.get("built_at"),
                                         "parents": {r: {k: (dm.get(r) or {}).get(k) for k in ("cell_id", "object_sha256", "fit_fingerprint", "sample_manifest_sha", "data_scale")}
                                                     for r in ("parent_pos", "parent_neg") if isinstance(dm.get(r), Mapping)}}
            if isinstance(dm.get("teacher"), Mapping):
                params["teacher"] = dict(dm["teacher"])
            if isinstance(dm.get("trainer"), Mapping):
                tr = dm["trainer"]
                params["lr_mult"] = tr.get("lr_mult")
                params["lr_mult_selected_by"] = tr.get("lr_mult_selected_by")
            if "recovery_verdict" in dm:
                params["recovery_verdict"] = dm.get("recovery_verdict")
            if isinstance(dm.get("calibration"), Mapping):
                cal = dm["calibration"]
                params["calibration"] = {k: cal.get(k) for k in ("lr_mult", "recovery", "budget", "production_stop_reason") if k in cal}
            if isinstance(dm.get("gate_liveness"), Mapping):
                gl = dm["gate_liveness"]
                params["gate_liveness_summary"] = {"n_layers": len(gl)}
            if anchor is not None:
                params["dose_scaled_from"] = anchor["stamp"]
                extra["dose_anchor"] = {"spelled_by": anchor["spelled_by"],
                                        "note": "the dose ladder's 1.0 anchor: the subtraction's own bytes re-served under this staged id; "
                                                "nothing was scaled, so the record is the subtraction's (parents = the two fits), never a dose copy of itself"}
            p["derivation_params_json"] = params
            conf.recorded("derivation_params_json", "derived_meta.json" + (" (+ dose_scaled_from kept as evidence: the ladder's 1.0 anchor, nothing scaled)"
                                                                          if anchor is not None else ""))
            p["fit_objective"] = "not_applicable"
            conf.derived("fit_objective", "derived object (the CHECK admits only not_applicable)")
        elif ftk == "system_instruction_distillation":
            teacher = str(si_id)
            p["fit_teacher_si_id"] = teacher
            conf.recorded("fit_teacher_si_id", "train_budget_meta.json:si_id" if tbm else "provenance.json:si / summary.json:si")
            p["fit_teacher_si_text_sha256"] = _si_sha(teacher, seeds, b)
            conf.recorded("fit_teacher_si_text_sha256", f"si_text[{teacher}].si_text_sha256 (seeds)")
            tk = (tbm or {}).get("teacher_topk") if tbm else None
            if isinstance(tk, Mapping) and tk.get("path"):
                path = str(tk["path"])
                stem = os.path.basename(path).split(".jsonl")[0]
                pool_dir = os.path.basename(os.path.dirname(path))
                p["fit_prompt_pool_id"] = f"{pool_dir}/{stem}"
                conf.recorded("fit_prompt_pool_id", "train_budget_meta.json:teacher_topk.path")
                extra["teacher_topk_path"] = path
                if isinstance(tk.get("pool_content_sha256"), str) and _SHA256_RE.match(str(tk["pool_content_sha256"])):
                    p["fit_prompt_pool_sha256"] = str(tk["pool_content_sha256"])
                    conf.recorded("fit_prompt_pool_sha256", "train_budget_meta.json:teacher_topk.pool_content_sha256")
                else:
                    p["fit_prompt_pool_sha256"] = "unverified"
                    conf.unverified("fit_prompt_pool_sha256", "teacher_topk names no pool_content_sha256")
            else:
                src = None
                for d, k in ((pv, "source"), (sm, "heldout_pool"), (fp, "note")):
                    v, _ = _first((d, k))
                    if v:
                        src = f"{k}={str(v)[:80]}"
                        break
                p["fit_prompt_pool_id"] = "unverified"
                conf.unverified("fit_prompt_pool_id", "no sidecar names the distillation prompt pool (train_budget_meta.json:teacher_topk.path absent"
                                + (f"; promotion record only says {src})" if src else ")"))
                p["fit_prompt_pool_sha256"] = "unverified"
                conf.unverified("fit_prompt_pool_sha256", "no sidecar names the prompt pool sha (train_budget_meta.json:sample_manifest_sha / teacher_topk.pool_content_sha256 absent)")
            if tbm and isinstance(tbm.get("sample_manifest_sha"), str):
                p["fit_sample_manifest_sha256"] = str(tbm["sample_manifest_sha"])
                conf.recorded("fit_sample_manifest_sha256", "train_budget_meta.json:sample_manifest_sha")
            # objective from the arm spelling / loss
            loss = (tbm or {}).get("loss_impl") if tbm else None
            fr_loss = ((tbm or {}).get("fit_record") or {}).get("loss") if tbm else None
            if arm_legacy and "_kl" in arm_legacy or fr_loss == "soft":
                p["fit_objective"] = "kl"
                conf.derived("fit_objective", f"arm spelling {arm_legacy!r} / fit_record.loss={fr_loss!r} (soft = KL to the teacher's top-k)")
            elif arm_legacy and "_ce" in arm_legacy:
                p["fit_objective"] = "ce"
                conf.derived("fit_objective", f"arm spelling {arm_legacy!r}")
            elif kind in ("lora", "relu_steer", "vector", "vector_all") and (
                    ((tbm or {}).get("teacher_forward") or (sm or {}).get("teacher_forward")) in ("precomputed_topk", "shared_lockstep", "solo")):
                tf_src = "train_budget_meta.json" if (tbm or {}).get("teacher_forward") else "summary.json"
                p["fit_objective"] = "kl"
                conf.derived("fit_objective", f"{tf_src}:teacher_forward={((tbm or {}).get('teacher_forward') or (sm or {}).get('teacher_forward'))!r}: "
                                              "a teacher-distribution distillation (KL)")
            else:
                raise SidecarRefusal(b.dir, f"fit_objective undecidable for an SI distillation: arm={arm_legacy!r} loss_impl={loss!r} fit_record.loss={fr_loss!r}")
        elif ftk == "corpus_document_fit":
            corpus = corpus_ids.get(str(si_id), str(si_id))
            p["fit_corpus_id"] = corpus
            conf.recorded("fit_corpus_id", "train_budget_meta.json:si_id (corpus label as the two-SI pipeline spells it)")
            smsha = (tbm or {}).get("sample_manifest_sha")
            if isinstance(smsha, str) and _SHA256_RE.match(smsha):
                p["fit_corpus_sha256"] = smsha
                p["fit_sample_manifest_sha256"] = smsha
                conf.recorded("fit_corpus_sha256", "train_budget_meta.json:sample_manifest_sha")
                conf.recorded("fit_sample_manifest_sha256", "train_budget_meta.json:sample_manifest_sha")
            else:
                p["fit_corpus_sha256"] = "unverified"
                conf.unverified("fit_corpus_sha256", "train_budget_meta.json has no sample_manifest_sha")
            hp = (tbm or {}).get("heldout_pool_sha")
            if isinstance(hp, str) and _SHA256_RE.match(hp):
                p["fit_pool_sha256"] = hp
                conf.recorded("fit_pool_sha256", "train_budget_meta.json:heldout_pool_sha")
            else:
                conf.unverified("fit_pool_sha256", "train_budget_meta.json has no 64-hex heldout_pool_sha")
            conf.unverified("fit_corpus_n_documents", "no sidecar records the document count")
            ds_tokens = None
            if dm and isinstance(dm.get("parent_pos"), Mapping):
                ds_tokens = ((dm["parent_pos"].get("data_scale") or {}).get("official_tokens"))
            if isinstance(ds_tokens, int):
                p["fit_corpus_n_tokens"] = ds_tokens
                conf.recorded("fit_corpus_n_tokens", "derived_meta.json:parent_pos.data_scale.official_tokens")
            else:
                conf.unverified("fit_corpus_n_tokens", "train_budget_meta.json records no corpus token count (training_budget_kind="
                                f"{(tbm or {}).get('training_budget_kind')!r})")
            p["fit_corpus_description"] = f"data_mode={(tbm or {}).get('data_mode')!r} training_budget_kind={(tbm or {}).get('training_budget_kind')!r}"
            conf.recorded("fit_corpus_description", "train_budget_meta.json:data_mode, training_budget_kind")
            if arm_legacy and "_ce" in arm_legacy or (tbm or {}).get("loss_impl") == "full":
                p["fit_objective"] = "ce"
                conf.derived("fit_objective", f"arm spelling {arm_legacy!r} / loss_impl={(tbm or {}).get('loss_impl')!r} (document CE)")
            else:
                raise SidecarRefusal(b.dir, f"fit_objective undecidable for a corpus fit: arm={arm_legacy!r} loss_impl={(tbm or {}).get('loss_impl')!r}")
        elif ftk == "contrastive_direction":
            sets: Dict[str, object] = {}
            if isinstance(meta, Mapping) and isinstance(meta.get("content_meta"), Mapping):
                cm = meta["content_meta"]
                sets = {"positive": {"id": "harmful", "sha256": cm.get("harmful_sha256"), "n": cm.get("n_harmful"), "source": cm.get("harmful_source")},
                        "negative": {"id": "benign", "sha256": cm.get("benign_sha256"), "n": cm.get("n_benign"), "source": cm.get("benign_sources")}}
                conf.recorded("fit_contrast_sets_json", "meta.json:content_meta")
            elif tbm and isinstance(tbm.get("contrast"), Mapping):
                ct = tbm["contrast"]
                sets = {"positive": {"id": ct.get("pos_si"), "sha256": None, "n": tbm.get("n_pairs")},
                        "negative": {"id": ct.get("neg_si") or "no_si", "sha256": None, "n": tbm.get("n_pairs")},
                        "completions": ct.get("completions_both_sides")}
                conf.recorded("fit_contrast_sets_json", "train_budget_meta.json:contrast, n_pairs (set shas not recorded)")
            elif fm:
                sets = {k: fm.get(k) for k in ("harmful", "benign", "positive", "negative", "capture") if k in fm}
                conf.recorded("fit_contrast_sets_json", "fit_meta.json")
            else:
                sets = {"positive": None, "negative": None}
                conf.unverified("fit_contrast_sets_json", "no sidecar names the contrast sets (meta.json:content_meta / train_budget_meta.json:contrast / fit_meta.json)")
            p["fit_contrast_sets_json"] = sets
            obj_by_kind = {"diffmean": "diffmean", "caa_static_r": "diffmean", "static_r_arditi": "arditi_search", "alphasteer": "alphasteer",
                           "antidote": "antidote"}
            if kind not in obj_by_kind:
                raise SidecarRefusal(b.dir, f"object_kind {kind!r} is contrastive by sidecar shape but names no fit_objective")
            p["fit_objective"] = obj_by_kind[kind]
            conf.derived("fit_objective", f"object_kind={kind}")
        elif ftk in ("none", "declared_prompt"):
            raise SidecarRefusal(b.dir, f"a bank directory produced fit_target_kind={ftk}; none/prompt objects are seeded by the dimensions adapter")
    p["fit_target_kind"] = ftk
    p["derivation"] = derivation
    conf.derived("derivation", derivation_src if derivation != "none" else "not a derived object")

    # ---- model ------------------------------------------------------------------------------
    hf_repo, hf_key = _first((tbm, "model_id"), (fp, "resolved_model_id"), (dm, "model_id"), (pv, "model"), (sm, "model"), (oj, "model_name"),
                             (fm, "model_id"))
    p["fit_model_id"] = _model_id(hf_repo if isinstance(hf_repo, str) else None, seeds, b)
    conf.recorded("fit_model_id", f"{hf_key}={hf_repo}")
    rev, rev_key = _first((fp, "resolved_revision"), (tbm, "model_revision"), (dm, "model_revision"), (pv, "model_revision"), (sm, "model_revision"))
    if isinstance(rev, str) and _SHA1_RE.match(rev):
        extra["fit_model_revision"] = rev
        conf.recorded("fit_model_revision", str(rev_key))
    else:
        extra["fit_model_revision"] = None
        conf.unverified("fit_model_revision", "no sidecar records a 40-hex fit model revision" + (f" ({rev_key}={rev!r})" if rev_key else ""))
    th, th_key = _first((tbm, "template_hash"), (fp, "chat_template_hash"), (dm, "template_hash"), (sm, "template_hash"), (pv, "template_hash"))
    if isinstance(th, str) and th:
        p["fit_chat_template_sha256"] = "unverified"
        extra["fit_chat_template_sha16"] = th
        conf.unverified("fit_chat_template_sha256", f"{th_key} records the 16-hex prefix {th} (capeval sha256_hex16), not the full sha256")
    else:
        p["fit_chat_template_sha256"] = "unverified"
        conf.unverified("fit_chat_template_sha256", "no sidecar records the chat template hash")
    if tbm and tbm.get("chat_template_bypassed") is True:
        p["fit_thinking_render"] = "not_applicable"
        conf.recorded("fit_thinking_render", "train_budget_meta.json:chat_template_bypassed=true (documents, no chat render)")
    elif ftk == "derived":
        p["fit_thinking_render"] = "not_applicable"
        conf.derived("fit_thinking_render", "derived object: inherits its parents' render")
    else:
        p["fit_thinking_render"] = "unverified"
        conf.unverified("fit_thinking_render", "no sidecar records whether the teacher / fit render enabled thinking")

    # ---- positions / layers / site / rank ----------------------------------------------------
    sidecars_for_pos = {k: v for k, v in sc.items() if isinstance(v, Mapping)}
    if blob:
        sidecars_for_pos["blob"] = blob
    pos, pos_conf, pos_src = D.fit_positions(kind, sidecars_for_pos)
    if pos == "unverified" and blob_reason and not blob:
        pos_src = f"{pos_src}; {blob_reason}"
    p["fit_positions"] = pos
    conf.set("fit_positions", pos, pos_conf, pos_src)
    rhl, rhl_key = _first((tbm, "response_header_len"), (blob, "response_header_len"), (pv, "response_header_len"), (dm, "response_header_len"))
    if pos == "response_incl_assistant_header":
        if _int(rhl) is not None:
            p["fit_response_header_len"] = _int(rhl)
            conf.recorded("fit_response_header_len", f"{'blob' if rhl_key and blob.get('response_header_len') == rhl and not (tbm or {}).get('response_header_len') else 'train_budget_meta.json'}:{rhl_key}")
        else:
            conf.unverified("fit_response_header_len", "positions=response but no sidecar or blob records response_header_len")
    layers, layers_src = _layers(b, conf, kind)
    if ftk in ("system_instruction_distillation", "corpus_document_fit", "contrastive_direction"):
        if layers is None:
            raise SidecarRefusal(b.dir, "no sidecar or blob names the exact fit layers (fit_layers_json is NOT NULL for a fitted object; never a count alone)"
                                        + (f"; {blob_reason}" if blob_reason else ""))
    if layers is not None:
        p["fit_layers_json"] = layers
        p["fit_layer_count"] = len(layers)
        p["fit_layer_index_convention"] = "zero_based_decoder_block"
        conf.recorded("fit_layer_count", f"len({layers_src})")
        conf.derived("fit_layer_index_convention", "the trainers index decoder blocks from 0 (model.model.layers[i]); the blob keys its tensors by that index")
        if ftk == "derived":
            p["fit_layer_selection_rule"] = "inherited"
            conf.derived("fit_layer_selection_rule", "derived object")
        elif kind in ("alphasteer", "static_r_arditi", "caa_static_r", "antidote"):
            p["fit_layer_selection_rule"] = "published_set"
            conf.derived("fit_layer_selection_rule", f"object_kind={kind}: layer set from the published method")
        else:
            p["fit_layer_selection_rule"] = "listed_in_config"
            conf.derived("fit_layer_selection_rule", f"{layers_src} is the trainer's resolved list from its config")
    site = (tbm or {}).get("site") if tbm else None
    if site in V.FIT_CAPTURE_SITES:
        p["fit_capture_site"] = site
        conf.recorded("fit_capture_site", "train_budget_meta.json:site")
    elif site:
        extra["write_site"] = site
        p["fit_target_modules_json"] = {"site": site}
        conf.recorded("fit_target_modules_json", "train_budget_meta.json:site")
    if kind in ("lora", "lora_subtraction"):
        rank, rank_key = _first((blob, "rank"), (tbm, "lora_rank"), (pv, "lora_rank"), (sm, "trial.lora_rank"), (sm, "lora_rank"), (dm, "rank"))
        if _int(rank) is not None:
            p["fit_rank"] = _int(rank)
            src = "blob:rank" if blob.get("rank") == rank else str(rank_key)
            conf.recorded("fit_rank", src)
            legacy = kind_info.get("legacy_object_kind_entry") or (seeds.object_kind_for(str(arm_legacy)) if arm_legacy else None)
            if legacy and legacy.get("fit_rank") not in (None, _int(rank)):
                extra["fit_rank_legacy_alias_disagrees"] = {"legacy_spellings": legacy.get("fit_rank"), "recorded": _int(rank)}
            if arm_legacy and _LORA_RANK_RE.search(arm_legacy) and int(_LORA_RANK_RE.search(arm_legacy).group(1)) != _int(rank):
                extra["fit_rank_arm_spelling_disagrees"] = {"arm": arm_legacy, "recorded": _int(rank)}
        else:
            legacy = kind_info.get("legacy_object_kind_entry") or (seeds.object_kind_for(str(arm_legacy)) if arm_legacy else None)
            if legacy and _int(legacy.get("fit_rank")) is not None:
                p["fit_rank"] = _int(legacy["fit_rank"])
                conf.derived("fit_rank", f"legacy_spellings.yaml object_kinds[{arm_legacy}].fit_rank ({legacy.get('source')})")
            elif arm_legacy and _LORA_RANK_RE.search(arm_legacy):
                p["fit_rank"] = int(_LORA_RANK_RE.search(arm_legacy).group(1))
                conf.derived("fit_rank", f"arm spelling {arm_legacy!r}" + (f"; {blob_reason}" if blob_reason else ""))
            else:
                conf.unverified("fit_rank", "no blob rank, sidecar lora_rank or arm spelling names the rank" + (f"; {blob_reason}" if blob_reason else ""))
        conf.unverified("fit_alpha", "no sidecar or blob key names a LoRA alpha")

    # ---- hyperparameters -----------------------------------------------------------------------
    hp: Dict[str, object] = {}
    hp_src = []
    if tbm:
        hp.update(_hyperparameter_block(tbm))
        hp_src.append("train_budget_meta.json")
    if dm and isinstance(dm.get("trainer"), Mapping):
        hp["trainer"] = {k: v for k, v in dm["trainer"].items() if k not in ("lr_mult_selected_by", "position_classes")}
        hp_src.append("derived_meta.json:trainer")
    if sm:
        hp["summary"] = {k: sm.get(k) for k in ("trial", "epochs", "early_stop_patience", "batch_size", "accum_steps", "effective_batch", "max_len",
                                                   "warmup_frac", "temperature", "weight_decay", "lora_rank", "total_steps", "length_sorted", "fit_subsample",
                                                   "teacher_forward", "box", "warmup_levels", "heldout_objective", "n_fit") if k in sm}
        hp_src.append("summary.json")
    if pv:
        hp["promotion"] = {k: pv.get(k) for k in ("lr", "clip", "lora_rank", "effective_batch", "epochs", "epochs_run", "best_epoch") if k in pv}
        hp_src.append("provenance.json")
    if meta and isinstance(meta.get("content_meta"), Mapping):
        hp["content_meta"] = dict(meta["content_meta"])
        hp["fit"] = meta.get("fit")
        hp_src.append("meta.json")
    if fm:
        hp["fit_meta"] = {k: v for k, v in fm.items() if isinstance(v, (str, int, float, bool)) or v is None}
        hp_src.append("fit_meta.json")
    if hp:
        p["fit_hyperparameters_json"] = hp
        conf.recorded("fit_hyperparameters_json", " + ".join(hp_src) + " (verbatim)")
    elif ftk != "derived" or dose is None:
        conf.unverified("fit_hyperparameters_json", f"no sidecar carries hyperparameters; sidecars seen: {sorted(sc)}")

    fr = (tbm or {}).get("fit_record") if tbm and isinstance(tbm.get("fit_record"), Mapping) else {}
    tr = dm.get("trainer") if dm and isinstance(dm.get("trainer"), Mapping) else {}
    cf = tbm.get("continued_from") if tbm and isinstance(tbm.get("continued_from"), Mapping) else {}

    def take(col: str, pairs, cast, missing_reason: str, source_prefixes=None):
        v, k = _first(*pairs)
        val = cast(v) if v is not None else None
        if val is not None:
            p[col] = val
            name = None
            for (d, key), label in zip(pairs, source_prefixes or [None] * len(pairs)):
                if d and key == k:
                    vv, _ = _first((d, key))
                    if vv == v:
                        name = label
                        break
            conf.recorded(col, f"{name}:{k}" if name else str(k))
        elif v is not None:
            conf.unverified(col, f"{k}={v!r} is not a usable value ({missing_reason})")
        else:
            conf.unverified(col, missing_reason)

    if dose is None:
        take("fit_lr", ((tbm, "lr"), (pv, "lr"), (sm, "trial.lr"), (tr, "lr")), _num,
             "no sidecar records the learning rate (train_budget_meta.json:lr / provenance.json:lr / summary.json:trial.lr / derived_meta.json:trainer.lr)",
             ["train_budget_meta.json", "provenance.json", "summary.json", "derived_meta.json:trainer"])
        take("fit_lr_schedule", ((tbm, "lr_schedule"), (cf, "lr_schedule.lr_schedule"), (tr, "lr_schedule")), lambda v: str(v) if isinstance(v, str) else None,
             "no sidecar names the lr schedule", ["train_budget_meta.json", "train_budget_meta.json:continued_from", "derived_meta.json:trainer"])
        take("fit_effective_batch", ((tbm, "effective_batch"), (pv, "effective_batch"), (sm, "effective_batch")), _int, "no sidecar records the effective batch",
             ["train_budget_meta.json", "provenance.json", "summary.json"])
        rb = {k: (tbm or {}).get(k) for k in ("batch_size", "accum_steps", "resolved_batch_size") if tbm and k in tbm}
        for k in ("realized_eff_batch", "realized_batch_check"):
            if k in fr:
                rb[k] = fr[k]
        if tr:
            rb.update({k: tr.get(k) for k in ("batch_size", "accum_steps") if k in tr})
        if sm:
            rb.update({k: sm.get(k) for k in ("batch_size", "accum_steps") if k in sm})
        if rb:
            p["fit_realized_batch_json"] = rb
            conf.recorded("fit_realized_batch_json", "train_budget_meta.json:batch_size, accum_steps, resolved_batch_size, fit_record.realized_eff_batch")
        else:
            conf.unverified("fit_realized_batch_json", "no sidecar records batch_size / accum_steps")
        take("fit_epochs_cap", ((cf, "new_cap"), (sm, "epochs"), (tr, "epochs"), (pv, "epochs"), (fr, "epochs")), _int,
             "no sidecar names the epoch cap (train_budget_meta.json carries none; the cap lives in configs/*_fits.yaml train_kwargs_by_si)",
             ["train_budget_meta.json:continued_from", "summary.json", "derived_meta.json:trainer", "provenance.json", "train_budget_meta.json:fit_record"])
        take("fit_epochs_seen", ((fr, "epochs_run"), (tr, "epochs_run"), (pv, "epochs_run")), _int, "no sidecar records epochs_run",
             ["train_budget_meta.json:fit_record", "derived_meta.json:trainer", "provenance.json"])
        take("fit_best_epoch", ((fr, "best_epoch"), (sm, "best_epoch"), (tr, "best_epoch"), (pv, "best_epoch")), _int, "no sidecar records best_epoch",
             ["train_budget_meta.json:fit_record", "summary.json", "derived_meta.json:trainer", "provenance.json"])
        take("fit_patience", ((fr, "early_stop.patience"), (sm, "early_stop_patience"), (tr, "patience")), _int, "no sidecar records the early-stopping patience",
             ["train_budget_meta.json:fit_record", "summary.json", "derived_meta.json:trainer"])
        es, es_key = _first((fr, "stopped_early"), (sm, "stopped_early"), (tr, "stopped_early"))
        if isinstance(es, bool):
            p["fit_early_stopped"] = int(es)
            conf.recorded("fit_early_stopped", str(es_key))
            cap = p.get("fit_epochs_cap")
            seen = p.get("fit_epochs_seen")
            if not es and cap is None and seen is not None:
                # the loop completed without an early stop: the epochs it ran ARE its cap
                p["fit_epochs_cap"] = cap = seen
                conf.derived("fit_epochs_cap", f"{es_key}=false with epochs_run={seen}: the run reached its cap")
            elif not es and seen is None and cap is not None:
                p["fit_epochs_seen"] = seen = cap
                conf.derived("fit_epochs_seen", f"{es_key}=false with an epoch cap of {cap}: the run reached it")
            if es:
                p["fit_stopped_on"] = "patience"
                conf.derived("fit_stopped_on", f"{es_key}=true")
            elif cap is not None and seen is not None and seen >= cap:
                p["fit_stopped_on"] = "epoch_cap"
                conf.derived("fit_stopped_on", f"{es_key}=false and epochs_seen {seen} >= cap {cap}")
            elif isinstance(cf.get("stop_reason"), str) or isinstance(tr.get("stop_reason"), str):
                reason = cf.get("stop_reason") or tr.get("stop_reason")
                p["fit_stopped_on"] = "epoch_cap" if "cap" in str(reason) else ("patience" if "early" in str(reason) else "unverified")
                conf.derived("fit_stopped_on", f"stop_reason={reason!r}")
                if p["fit_stopped_on"] == "unverified":
                    conf.unverified("fit_stopped_on", f"stop_reason={reason!r} names neither patience nor the epoch cap")
            else:
                p["fit_stopped_on"] = "unverified"
                conf.unverified("fit_stopped_on", f"{es_key}=false but no cap / stop_reason recorded to tell epoch_cap from another stop")
        else:
            conf.unverified("fit_early_stopped", "no sidecar records stopped_early")
            p["fit_stopped_on"] = "unverified"
            conf.unverified("fit_stopped_on", "no sidecar records how the fit stopped")
        take("fit_steps", ((fr, "optimizer_steps_run"), (fr, "optimizer_steps"), (tr, "optimizer_steps_run")), _int,
             "no sidecar records the realised optimizer-step count (summary.json's steps_run counts micro-steps and total_steps is the plan)",
             ["train_budget_meta.json:fit_record", "train_budget_meta.json:fit_record", "derived_meta.json:trainer"])
        conf.unverified("fit_clip_fraction", "no trial log or sidecar records the clipped-step fraction for this object")
        conf.unverified("fit_final_train_loss", "sidecars record held-out / probe losses, never the final train loss")
        curve = fr.get("heldout_curve") if isinstance(fr.get("heldout_curve"), list) else (sm or {}).get("heldout_curve")
        if isinstance(curve, list) and curve and _num(curve[-1]) is not None:
            p["fit_final_heldout_loss"] = _num(curve[-1])
            conf.recorded("fit_final_heldout_loss", "train_budget_meta.json:fit_record.heldout_curve[-1]" if fr.get("heldout_curve") else "summary.json:heldout_curve[-1]")
        else:
            conf.unverified("fit_final_heldout_loss", "no heldout_curve recorded")
        take("fit_best_heldout_loss", ((fr, "best_heldout_loss"), (fr, "best_heldout_ce"), (sm, "best_heldout_loss")), _num, "no sidecar records the best held-out loss",
             ["train_budget_meta.json:fit_record", "train_budget_meta.json:fit_record", "summary.json"])
        metrics = next((f for f in b.sidecar_files if f["kind"] == "metrics.jsonl"), None)
        if metrics:
            p["fit_heldout_curve_uri"] = str(metrics["uri"])
            conf.recorded("fit_heldout_curve_uri", "metrics.jsonl beside the object")
        elif isinstance(curve, list) and curve:
            p["fit_heldout_curve_uri"] = os.path.join(b.dir, "train_budget_meta.json") + "#$.fit_record.heldout_curve" if fr.get("heldout_curve") else os.path.join(b.dir, "summary.json") + "#$.heldout_curve"
            conf.recorded("fit_heldout_curve_uri", "the sidecar's heldout_curve (no metrics.jsonl on this machine)")
        else:
            conf.unverified("fit_heldout_curve_uri", "no metrics.jsonl or heldout_curve on this machine")
        seed, seed_key = _first((tbm, "seed"), (tr, "seed"), (sm, "seed"), (fr, "seed"))
        if _int(seed) is not None:
            p["fit_seed"] = _int(seed)
            p["fit_seed_confidence"] = "recorded"
            conf.recorded("fit_seed", {"seed": "train_budget_meta.json:seed", "trial.seed": "summary.json"}.get(str(seed_key), f"{seed_key}"))
        else:
            p["fit_seed_confidence"] = "unverified"
            seed_reason = ("no sidecar key names the seed (train_budget_meta.json:seed / derived_meta.json:trainer.seed / summary.json:seed); "
                           "configs/lora_all_recipe.json documents no default seed")
            conf.unverified("fit_seed", seed_reason)
            conf.unverified("fit_seed_confidence", seed_reason)
        conf.unverified("fit_heldout_fraction", "no sidecar records the held-out fraction (train_budget_meta.json:fit_record.early_stop.heldout_n is a count)")
        if isinstance(fr.get("early_stop"), Mapping) and _int(fr["early_stop"].get("heldout_n")) is not None:
            extra["heldout_n"] = _int(fr["early_stop"]["heldout_n"])
    else:
        p["fit_seed_confidence"] = "not_applicable"
        conf.derived("fit_seed_confidence", "dose-scaled copy: nothing was fit")
    if anchor is not None:
        _batch_through_parents(p, conf, extra, anchor_parents, parents_in_bank)
    elif dose is not None and dose_batch_from_parent:
        # a dose copy's effective batch is its parent's, stated derived from the parent's record copied beside the
        # copy (the copy's own train_budget_meta.json) -- never the NULL beside the parent's block
        _dose_batch_from_parent(p, conf, extra, dose, tbm)

    # replicate label / split salt
    m = _REPLICATE_RE.match(b.staged_id)
    if m:
        p["fit_replicate_label"] = f"s{m.group('s')}"
        conf.recorded("fit_replicate_label", "staged id prefix base64s<s>")
        extra["split_salt"] = f"sdfbase-heldout-v1-s{m.group('s')}"
        conf.unverified("fit_split_seed", "the split seed is the carve salt (extra_json.split_salt), not an integer; salt per D16 (data/sdfbase/heldout_split.s<s>.provenance.json on branch reseed)")

    # ---- where and when ---------------------------------------------------------------------
    code, code_key = _first(((dose or {}).get("block") if dose else None, "producer_commit"), (dm, "builder_commit"), (fp, "code_sha"), (fm, "code_sha"))
    if isinstance(code, str) and _SHA1_RE.match(code):
        p["fit_code_sha"] = code
        conf.recorded("fit_code_sha", f"{'train_budget_meta.json:dose_scaled_from' if dose else 'derived_meta.json'}:{code_key}")
    elif omit_producer_owned:
        # `unverified` is the bank DEFAULT for a producer-owned column -- omitted, so the fold keeps whatever the producer stated
        producer_owned_unstated["fit_code_sha"] = ("no sidecar records the fit code sha" + (f" ({code_key}={code!r})" if code_key else "")
                                                   + "; `unverified` is the bank default for a producer-owned column, which a mirror never states (D874)")
    else:
        p["fit_code_sha"] = "unverified"
        conf.unverified("fit_code_sha", "no sidecar records the fit code sha" + (f" ({code_key}={code!r})" if code_key else ""))
    # the producer-owned-stamp rule: a dose COPY's `fit_writer` / `fit_finished_utc` are the copy's OWN
    # dose_scaled_from stamp's `producer` / `written_at` (the all-positions derivation's `built_by` / `built_at`); the records copied beside
    # the copy -- derived_meta.json:builder / built_at, fit_provenance.json:writer, summary.json:writer, COMPLETE.published_at,
    # provenance.json:promoted -- are the PARENT's (or the bank's) and a mirror never restates them for the copy; `unverified` is the
    # bank default; so with no stamp the bank OMITS the column (extra.producer_owned_unstated says why) and the fold keeps whatever the
    # producer's native record stated. The builder's default keeps the legacy reading for its other caller.
    copy_stamp, copy_key = _dose_copy_stamp(dm, tbm, dose) if (omit_producer_owned and dose is not None) else (None, "")
    if omit_producer_owned and dose is not None:
        writer, writer_key = _first((copy_stamp, "producer"), (copy_stamp, "built_by"))
        if isinstance(writer, str) and writer:
            p["fit_writer"] = writer
            conf.recorded("fit_writer", f"{copy_key}:{writer_key}")
        else:
            producer_owned_unstated["fit_writer"] = (
                "a dose copy's writer is its own dose_scaled_from stamp's `producer`, and "
                + (f"the stamp ({copy_key}) names none" if copy_stamp is not None else "the copy carries no stamp")
                + "; derived_meta.json:builder / fit_provenance.json:writer / summary.json:writer beside it name the PARENT's writer, which a "
                  "mirror never restates for the copy, and `unverified` is the bank default (D874 / D917)")
    else:
        writer, writer_key = _first(((dose or {}).get("block") if dose else None, "producer"), (dm, "builder"), (fp, "writer"), (sm, "writer"))
        if isinstance(writer, str) and writer:
            p["fit_writer"] = writer
            conf.recorded("fit_writer", str(writer_key))
        else:
            conf.unverified("fit_writer", "no sidecar names the writing script")
    cfg, cfg_key = _first((tr, "config"),)
    if isinstance(cfg, str):
        p["fit_config_path"] = cfg
        conf.recorded("fit_config_path", "derived_meta.json:trainer[config]")
        csha = tr.get("config_sha256")
        if isinstance(csha, str) and _SHA256_RE.match(csha):
            p["fit_config_sha256"] = csha
            conf.recorded("fit_config_sha256", "derived_meta.json:trainer[config_sha256]")
    else:
        conf.unverified("fit_config_path", "no sidecar names the fit config file")
        conf.unverified("fit_config_sha256", "no sidecar records the fit config sha")
    if omit_producer_owned and dose is not None:
        finished, fin_key = _first((copy_stamp, "written_at"), (copy_stamp, "built_at"))
        fu = _utc(finished)
        if fu:
            p["fit_finished_utc"] = fu
            conf.recorded("fit_finished_utc", f"{copy_key}:{fin_key}")
        else:
            producer_owned_unstated["fit_finished_utc"] = (
                "a dose copy's finish time is its own dose_scaled_from stamp's `written_at`, and "
                + (f"the stamp ({copy_key}) records none" if copy_stamp is not None else "the copy carries no stamp")
                + "; COMPLETE.published_at / derived_meta.json:built_at / provenance.json:promoted beside it are the parent's or the bank's "
                  "times, which a mirror never restates for the copy (D874 / D917)")
    else:
        finished, fin_key = _first((complete, "published_at"), (dm, "built_at"), ((dose or {}).get("block") if dose else None, "written_at"), (pv, "promoted"))
        fu = _utc(finished)
        if fu:
            p["fit_finished_utc"] = fu
            conf.set("fit_finished_utc", fu, "recorded" if fin_key != "promoted" else "derived", f"{fin_key} ({'date only' if fin_key == 'promoted' else 'timestamp'})")
        else:
            conf.unverified("fit_finished_utc", "no sidecar records when the fit finished (COMPLETE.published_at / derived_meta.built_at / dose_scaled_from.written_at)")
    conf.unverified("fit_started_utc", "no sidecar records when the fit started")
    wall, wall_key = _first((sm, "wall_s"),)
    if _num(wall) is not None:
        p["fit_walltime_s"] = _num(wall)
        conf.recorded("fit_walltime_s", "summary.json:wall_s")
    else:
        conf.unverified("fit_walltime_s", "no sidecar records the fit walltime")
    sub_path = None
    for d, k in (((dose or {}).get("block") if dose else None, "parent_dir"), (stf, "source"), (dm, "parent_pos.object_path"), (pv, "source"),
                 (cf, "checkpoint"), (tbm, "teacher_topk.path")):
        v, _ = _first((d, k))
        if isinstance(v, str) and v:
            sub_path = (k, v)
            break
    if dose is not None and sub_path and sub_path[0] == "parent_dir":
        sub, sub_conf, sub_src = D.substrate_from_path(sub_path[1])
    elif sub_path:
        sub, sub_conf, sub_src = D.substrate_from_path(sub_path[1])
    else:
        sub, sub_conf, sub_src = ("unverified", "unverified", "no sidecar names a path on the producing substrate")
    if sub in V.SUBSTRATES:
        p["fit_substrate"] = sub
        conf.set("fit_substrate", sub, sub_conf, f"{sub_path[0]}={sub_path[1][:80]} → {sub_src}" if sub_path else sub_src)
    else:
        conf.unverified("fit_substrate", sub_src)
    conf.unverified("fit_job_id", "no sidecar records a job / Modal call id")
    conf.unverified("fit_node_hostname", "no sidecar records the fit host")
    conf.unverified("fit_gpu_type", "no sidecar records the fit GPU type (teacher_topk.build.device is the top-k build's device)")
    gpus, _ = _first((tbm, "parallel.job_gpus"),)
    if _int(gpus) is not None:
        p["fit_gpu_count"] = _int(gpus)
        conf.recorded("fit_gpu_count", "train_budget_meta.json:parallel.job_gpus")
    else:
        conf.unverified("fit_gpu_count", "no sidecar records the GPU count")
    wb, wb_key = _first((pv, "wandb_run"), (sm, "wandb_run"), (tbm, "wandb_run"))
    if isinstance(wb, str) and wb:
        p["fit_wandb_run_id"] = wb
        conf.recorded("fit_wandb_run_id", str(wb_key))
        conf.unverified("fit_wandb_url", "W&B run id named; the URL resolves through the W&B API (wandb_runs) when reachable")
    else:
        conf.unverified("fit_wandb_run_id", "no sidecar names a W&B run")
        conf.unverified("fit_wandb_url", "no sidecar names a W&B run")
    conf.unverified("fit_producer_session", "legacy fit: no session recorded")

    # lr selection (linked to a search by hpo_logs when one matches)
    if dose is not None:
        p["fit_lr_selected_by"] = "inherited_from:" + str((dose or {}).get("parent_object_sha256"))
        conf.derived("fit_lr_selected_by", "dose-scaled copy inherits the parent's fit")
    elif ftk == "derived":
        p["fit_lr_selected_by"] = "inherited_from:" + str(p["parent_object_ids_json"][0])
        conf.derived("fit_lr_selected_by", "derived object: hyperparameters inherit from the parents (the distillation trainer's lr_mult is in derivation_params_json)")
    elif pv and isinstance(pv.get("source"), str) and "hpo" in str(pv["source"]):
        p["fit_lr_selected_by"] = "search"
        conf.recorded("fit_lr_selected_by", f"provenance.json:source={pv['source']} (a promoted HPO trial)")
        extra["hpo_trial_source"] = pv["source"]
    else:
        p["fit_lr_selected_by"] = "unverified"
        conf.unverified("fit_lr_selected_by", "no sidecar names how the learning rate was chosen (a fit_search link, when one matches, sets it)")
    conf.unverified("fit_argmin_interior", "only a linked fit_search records interiority")

    # ---- role / label -----------------------------------------------------------------------
    # The paper label is keyed on the campaign STEM of the object's cell spelling (`<si>__<arm>`), never on the fit-variant prefix a
    # staged id carries before it (`base__` / `itdoc64__` / `d0p5_eb64s1__` / `derived_itdoc64__` / `si12__` …): the prefix is fit
    # provenance and dose, which ride fit_model_id / base_scale / fit_replicate_label / fit_effective_batch, and the roster then
    # selects the variant the paper draws (model × label × transfer_kind, plus the operating-point and object rulings). Two tables
    # spell a stem, in order: the seeded `legacy_spellings.yaml paper_labels`, then the paper's own arm table (declared_common.ARMS,
    # the readers' contract) when its object_kind is this object's. The stem and the prefix are parked on the object.
    label_entry, label_src, campaign_stem, variant_prefix = paper_label_entry_for(kind, (cell_id_legacy, b.staged_id, arm_legacy), seeds, extra)
    if campaign_stem and campaign_stem != p["campaign_cell_id_legacy"]:
        extra["campaign_stem_legacy"] = campaign_stem
    if variant_prefix:
        extra["fit_variant_prefix_legacy"] = variant_prefix
    override_entry = seeds.fit_target_overrides.get(str(si_id)) if si_id else None
    if override_entry and override_entry.get("object_role") in V.OBJECT_ROLES:
        p["object_role"] = str(override_entry["object_role"])
        p["paper_label"] = str(override_entry.get("paper_label")) if override_entry.get("paper_label") in V.PAPER_LABELS else "other"
        conf.derived("object_role", f"legacy_spellings.yaml fit_target_overrides[{si_id}].object_role: {override_entry.get('citation') or override_entry.get('source')}")
        conf.derived("paper_label", f"legacy_spellings.yaml fit_target_overrides[{si_id}]")
    elif label_entry:
        p["paper_label"] = str(label_entry["paper_label"])
        p["object_role"] = str(label_entry["object_role"])
        conf.derived("paper_label", str(label_src))
        conf.derived("object_role", str(label_src))
    else:
        role = None
        role_src = f"si_id / corpus {si_id!r}"
        if ftk == "system_instruction_distillation":
            role = _ROLE_BY_SI.get(str(si_id)) or _ROLE_BY_SI_KIND.get(str(seeds.si_kind_by_id.get(str(si_id))))
            if role and str(si_id) not in _ROLE_BY_SI:
                role_src = f"si_text[{si_id}].si_kind={seeds.si_kind_by_id.get(str(si_id))}"
        elif ftk == "corpus_document_fit":
            role = _ROLE_BY_CORPUS.get(str(p.get("fit_corpus_id")))
        elif ftk == "derived" and dose is None:
            role = "difference"
        elif dose is not None:
            role = _ROLE_BY_CORPUS.get(str(si_id)) or _ROLE_BY_SI.get(str(si_id))
            if role is None and "minus_ctl" in b.staged_id:
                role = "difference"
        elif ftk == "contrastive_direction":
            if isinstance(meta, Mapping) and meta.get("si_id") is None and meta.get("fit", "").startswith("si_agnostic"):
                role = "baseline_published"
            else:
                role = _ROLE_BY_SI.get(str(si_id))
        if role is None:
            raise SidecarRefusal(b.dir, f"object_role undecidable: no legacy_spellings.yaml paper_labels entry and no paper arm (declared_common.ARMS) "
                                        f"for stem {campaign_stem!r} (cell {cell_id_legacy!r}) and si_id={si_id!r} maps to no role; add a paper_labels entry")
        p["object_role"] = role
        conf.derived("object_role", f"{role_src} → {role} (no paper_labels entry / paper arm for stem {campaign_stem!r})")
        p["paper_label"] = "other"
        conf.derived("paper_label", f"no legacy_spellings.yaml paper_labels entry and no paper arm (declared_common.ARMS) for stem {campaign_stem!r} "
                                    f"(cell {cell_id_legacy!r}, object_kind {kind})")
    if dose is not None and "minus_ctl" in b.staged_id and p.get("object_role") != "difference":
        p["object_role"] = "difference"

    # ---- status -------------------------------------------------------------------------------
    # `status` is the PRODUCER's statement. The bank states it only where the bank itself carries the evidence
    # (:func:`bank_status_statement`, shared with the objects_bank adapter: a directory under _withdrawn/ or _superseded_by_it64/, a
    # superseded_by.json naming this object's fit fingerprint); the "approved" default was a RESTATEMENT -- it overwrote a producer
    # statement stamped before the pass but folded after it, and demoted reference objects to the bank default -- so it is
    # OMITTED: the fold keeps the folded status when a row exists, and the harvester dependency-blocks the record on its own row (never
    # a refusal) when none does, until the producer's statement folds. The evidence the default rested on rides `extra.status_unstated`;
    # the evidence beside a statement, or beside a superseded_by.json that states nothing, stays parked where it was.
    own_fps = own_fit_fingerprints(b)                # the object's own fit fingerprints, parked under extra below
    stated, status_src, branch = bank_status_statement(b)
    if branch == "superseded_by_json":
        extra["superseded_by_json"] = sb
        conf.unverified("superseded_by_object_id", "superseded_by.json names the successor's expected fit fingerprint, not its object sha")
    elif branch == "refit_of_superseded":
        extra["supersedes_fit_fingerprint"] = sb.get("fit_fingerprint")
        extra["supersedes_snapshot"] = sb.get("snapshot")
    elif branch == "superseded_by_json_unresolved":
        extra["superseded_by_json_unresolved"] = sb
    if stated is not None:
        p["status"] = stated
        if "status" not in conf.entries:
            conf.derived("status", status_src)
    else:
        conf.entries.pop("status", None)
        extra["status_unstated"] = f"the bank states no status for this object (D721): {status_src}"

    # ---- locations / sidecars ------------------------------------------------------------------
    locs: List[Dict[str, object]] = []
    if b.blob_path:
        locs.append({"uri": b.blob_path, "verified_utc": now, "bytes": b.blob_bytes})
    else:
        locs.append({"uri": b.dir, "verified_utc": None, "bytes": None, "note": "sidecars only; weights not on this machine"})
    if stf and isinstance(stf.get("source"), str):
        # a location only when the field IS one: the two-SI banks write the staging path there, the Arditi objects a sentence about the rung
        if re.search(r"[/:]", str(stf["source"])) and " " not in str(stf["source"]).strip():
            locs.append({"uri": str(stf["source"]), "verified_utc": None, "bytes": None, "source": "staged_from.json:source"})
        else:
            extra["staged_from_source_note"] = str(stf["source"])
    locs.extend(b.extra_locations)
    p["artifact_locations_json"] = locs
    p["object_sidecars_json"] = list(b.sidecar_files)
    if b.twin_staged_id:
        extra["twin_staged_id"] = b.twin_staged_id
    if blob_reason:
        extra["blob_read"] = blob_reason
    elif blob:
        extra["blob_keys"] = sorted(k for k in blob if k != "_tensors")
        extra["blob_tensors"] = blob.get("_tensors")
    if own_fps:
        extra["fit_fingerprint"] = sorted(own_fps)[0]
    if omit_producer_owned and "derivation_params_json" in p:
        # the assembled dict is an ABBREVIATION of the producer's record (a dose stamp's eight keys of twenty-one; a derived_meta.json
        # record restructured); the bank states the column only as a sidecar's own `params` block, verbatim, and omits it otherwise
        assembled = p.pop("derivation_params_json")
        conf.entries.pop("derivation_params_json", None)
        verbatim, where = _verbatim_derivation_params(dm, dose)
        if verbatim is not None:
            p["derivation_params_json"] = dict(verbatim)
            conf.recorded("derivation_params_json", f"{where} (verbatim; D874)")
        else:
            producer_owned_unstated["derivation_params_json"] = (f"assembled from {_derivation_params_assembly_source(dm, dose, anchor)} -- an abbreviation of the "
                                                                 "producer's record; a mirror states a producer-owned column only verbatim from a sidecar's own "
                                                                 "params block, and no sidecar carries one (D874)")
            if anchor is not None and isinstance(assembled, Mapping) and isinstance(assembled.get("dose_scaled_from"), Mapping) and isinstance(extra.get("dose_anchor"), dict):
                extra["dose_anchor"]["stamp"] = assembled["dose_scaled_from"]      # the evidence the assembled params carried stays on the record
    if producer_owned_unstated:
        extra["producer_owned_unstated"] = producer_owned_unstated
    if dm and dm.get("recovery_verdict") is not None and "recovery_verdict" not in json.dumps(p.get("derivation_params_json") or {}):
        extra["recovery_verdict"] = dm.get("recovery_verdict")
    p["field_confidence_json"] = conf.entries
    p["extra"] = extra
    return p


def _verbatim_derivation_params(dm: Optional[Mapping[str, object]], dose: Optional[Dict[str, object]]) -> Tuple[Optional[Mapping[str, object]], Optional[str]]:
    """The one value of ``derivation_params_json`` a mirror may state: a sidecar's own ``params`` mapping, verbatim -- the dose-scale
    derivation record's ``params`` in the all-positions spelling (``_dose_copy``'s ``params``), else ``derived_meta.json:params`` (the field the
    producer-side writer copies); ``(None, None)`` when no sidecar carries one."""
    if dose is not None and isinstance(dose.get("params"), Mapping):
        return dose["params"], f"{str(dose.get('spelling') or '').split(':')[0]}:params"
    if dose is None and isinstance(dm, Mapping) and isinstance(dm.get("params"), Mapping):
        return dm["params"], "derived_meta.json:params"
    return None, None


def _derivation_params_assembly_source(dm: Optional[Mapping[str, object]], dose: Optional[Dict[str, object]], anchor: Optional[Dict[str, object]]) -> str:
    if dose is not None:
        return str(dose.get("spelling") or "object.pt.provenance.json:dose_scale")
    return "derived_meta.json" + (" + the dose_scaled_from anchor stamp" if anchor is not None else "")
