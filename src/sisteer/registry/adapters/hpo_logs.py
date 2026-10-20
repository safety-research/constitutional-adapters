"""HPO logs → ``fit_search`` and ``fit_trial`` rows.

One SWEEP directory = one ``fit_search``: a directory holding ``r<k>_t<NN>/summary.json`` trials (the
``tune_arm.py`` layout) with, when the analysis ran, ``r1_fit.json`` / ``r1+2_fit.json`` (the quadratic fit
over log loss: ``fit.coef``, ``fit.hessian_eigs``, ``fit.positive_definite``, ``fit.sigma_log_y``, ``selection``
with ``on_edge`` / ``selected_from`` / ``alarms``, ``winner``), and for the ``reseed`` lineage
``confirm_check.json`` (``confirmation_verdict`` in_interval | outside_above | outside_below, ``d_over_sigma``,
``selection_action``, ``fit_reliability``), the Sobol designs under ``trials/*.json``, the spawn manifests
under ``manifests/`` and the grad-clip reconnaissance under ``recon/``.

Roots reachable from the grading host (defaults; every one optional):

* ``$STAGE/sdfbase/reseed_eb64/hpo/results/<key>/<arm>/`` (+ ``../trials``, ``../manifests``, ``../hpo_fold.json``, ``../../recon``)
* ``~/hpo-llama/<arm>_<positions>/``, ``~/hpo-rehpo/<model>/<arm>_<positions>/``, ``~/hpo-sync/<arm>_<positions>/``,
  ``~/hpo-r1/<arm>_<positions>/`` (+ ``~/hpo-r1/e4b_hpo_winners*.json`` and ``PODS.md``, the RunPod evidence)

Every value is copied verbatim from the record that holds it and named in ``field_confidence_json``; a
search whose ``method`` / ``model`` cannot be read off a record is refused, never guessed. A sweep with no
substrate record at all files ``substrate = unverified`` naming the evidence that is missing (eight
``~/hpo-rehpo`` sweeps carry none; ``fit_search.substrate`` alone admits the value).
The link from an ``object`` to its search (``fit_search_id``, ``fit_lr_selected_by = search``,
``fit_argmin_interior``) is made by :func:`link_objects` on the (model, object_kind, si, positions, winner
lr == object lr) key; an unmatched object keeps ``unverified``.

A sweep whose intervention is a kind ruled out of scope (``token_faithful`` / ``bidpo``: the EXACT spellings of
``_OUT_OF_SCOPE_SPELLINGS_BY_KIND``, the spellings ``legacy_spellings.yaml`` ``object_kinds`` declares OUT OF SCOPE and
``out_of_scope_object_kinds`` rules on) is PARKED by name, never refused (park-by-name, 2026-10-07; the objects bank's
rule), on the kind alone and before any model-closure check: the group reads to an accepted revision with no ``fit_search`` / ``fit_trial`` record and a ``source_status`` fact
(``extra.refuse`` + ``extra.parked``, :func:`parked_status_facts`, the objects bank's shape) citing the table, so the status
page lists it under "Parked by name" and no window or tick re-reads it. A spelling the table does not name -- a kind nobody has ruled on,
a spelling that only BEGINS with a parked one (``token_faithful_v2``, ``bidpo_control``) included -- still refuses by name
(:class:`HpoRefusal`); parking is never inferred: the table declares spellings, not prefixes (only a spelling ending in ``_`` is a prefix there,
``fit_sidecars.out_of_scope_tables``), and the in-scope prefixes of ``_KIND_BY_SWEEP_PREFIX`` (``psr`` / ``lora``, which do name families of
spellings) are a separate resolver. The kind is ONE per sweep and is read off every record
that claims a spelling (the fit record's and each trial summary's) before it is decided: records that resolve to different entries of
the table refuse the sweep by name, citing each other (:func:`_check_one_kind`), so a mixed directory is never parked whole with its
in-scope trials filed as nothing, nor read as one search with an out-of-scope trial among its trials. The decisive spelling (the one
the search or the fact records) is drawn from that same claim set -- the fit record's, else the first summary that names one -- and
the directory's name stands in only when no record claims a spelling at all (a claimless first summary would otherwise let
the directory's name decide against the later summaries' claim).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .. import identity
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, store_listdir, store_open, require_store_access, site_root
from .fit_sidecars import Seeds, _utc, load_seeds, read_json
from .objects_bank import PARKED_RULING

HOME = os.path.expanduser("~")
DEFAULT_HPO_ROOTS: Tuple[Tuple[str, str, str], ...] = (
    # (label, root, layout): reseed → results/<key>/<arm>; flat → <arm_pos>; model → <model>/<arm_pos>
    ("reseed_eb64", site_root("$STAGE/sdfbase/reseed_eb64/hpo"), "reseed"),
    ("hpo-llama", os.path.join(HOME, "hpo-llama"), "flat"),
    ("hpo-rehpo", os.path.join(HOME, "hpo-rehpo"), "model"),
    ("hpo-sync", os.path.join(HOME, "hpo-sync"), "flat"),
    ("hpo-r1", os.path.join(HOME, "hpo-r1"), "flat"),
)
_TRIAL_RE = re.compile(r"^r(?P<round>\d+)_t(?P<idx>\d+)$")
_FIT_RE = re.compile(r"^r[0-9+]+_fit\.json$")
_SWEEP_DIR_RE = re.compile(r"^(?P<arm>[a-z0-9_]+?)_(?P<pos>all|response)$")
#: (intervention-spelling prefix, object_kind) -- the IN-SCOPE half of hpo_logs's kind resolver (:func:`_kind_entry`): a spelling that
#: begins with the prefix is a sweep over that object kind (``psr_ce_all`` / ``psr_kl_all_response`` → relu_steer, ``lora_r1_all`` → lora);
#: these two families genuinely spell themselves by prefix. Nothing in this table parks.
_KIND_BY_SWEEP_PREFIX = (("psr", "relu_steer"), ("lora", "lora"))
#: kind (the table entry, the kind's own spelling) -> the EXACT intervention spellings of it that PARK by name (:class:`HpoParked`) --
#: the OUT-OF-SCOPE half of the resolver, the spellings ``legacy_spellings.yaml`` ``object_kinds`` declares OUT OF SCOPE and
#: ``out_of_scope_object_kinds`` rules on (``token_faithful | token_faithful_all``, ``bidpo``) that an HPO sweep's records can carry. Exact
#: spellings, never prefixes: the table declares spellings (only one ending in ``_`` is a prefix there, ``fit_sidecars.out_of_scope_tables``), so
#: ``token_faithful_v2`` or ``bidpo_control`` is a spelling nobody has ruled on and refuses by name (:class:`HpoRefusal`): a prefix match
#: here would park any such sweep whole, an accepted empty revision under a ruling that never named
#: it. The set that parks is exactly what this table declares, never inferred from the seeds (the registered factory's seeds carry no
#: yaml table at all), and never widened by a suffix. The row's other spellings (``token_appended``, ``did``, ``cast``,
#: ``circuit_breaker``) are not here: no HPO layout has ever spelled them, and a declared spelling the adapter does not list refuses loudly
#: rather than parking -- widening this set is a decision, not a fix.
_OUT_OF_SCOPE_SPELLINGS_BY_KIND: Mapping[str, Tuple[str, ...]] = {"token_faithful": ("token_faithful", "token_faithful_all"), "bidpo": ("bidpo",)}
#: Sobol design evidence for the tune_arm.py sweeps (scripts/tune_arm.py docstring: "Sobol design (spec Fixed 7 …)").
_TUNE_SOBOL_CITATION = "scripts/tune_arm.py docstring l.32 ('Sobol design (spec Fixed 7; review P1-2 …)'): every tune_arm.py round is a Sobol design per warmup level"


def _kind_entry(spelling: object) -> Tuple[Optional[str], Optional[str]]:
    """``(entry, object_kind)`` of the kind-table entry an intervention spelling resolves to: ``(kind, None)`` when the spelling is
    EXACTLY one :data:`_OUT_OF_SCOPE_SPELLINGS_BY_KIND` declares (a kind ruled out of scope; the entry is the kind's own spelling,
    ``token_faithful`` for ``token_faithful_all``), ``(prefix, object_kind)`` of the :data:`_KIND_BY_SWEEP_PREFIX` prefix it begins with
    (in scope), ``(None, None)`` when neither names it (a spelling nobody has ruled on -- ``token_faithful_v2`` as much as
    ``novel_method``: beginning with a parked spelling is not being one)."""
    s = str(spelling)
    for kind, spellings in _OUT_OF_SCOPE_SPELLINGS_BY_KIND.items():
        if s in spellings:
            return kind, None
    for prefix, k in _KIND_BY_SWEEP_PREFIX:
        if s.startswith(prefix):
            return prefix, k
    return None, None


def _parked_spellings() -> List[str]:
    """Every intervention spelling hpo_logs parks by name, in table order (the refusal of an unlisted spelling names them)."""
    return [s for spellings in _OUT_OF_SCOPE_SPELLINGS_BY_KIND.values() for s in spellings]


def _describe_kind_entry(entry: Optional[str]) -> str:
    if entry is None:
        return "no entry of the kind table"
    if entry in _OUT_OF_SCOPE_SPELLINGS_BY_KIND:
        return f"{entry!r}, a kind legacy_spellings.yaml object_kinds declares OUT OF SCOPE"
    return f"object_kind {dict(_KIND_BY_SWEEP_PREFIX)[entry]}"


def _check_one_kind(where: str, claims: Sequence[Tuple[str, str]]) -> None:
    """Refuse (:class:`HpoRefusal`) a sweep whose records claim spellings that resolve to DIFFERENT entries of the kind table -- a
    ``token_faithful`` summary beside a ``psr_ce_all`` one, a fit record that disagrees with the summaries. ``claims`` are ``(spelling,
    record)`` for every record that carries a spelling; a record with none claims nothing. Spellings within one entry agree (the kind is
    the entry, and the fact or the search records the decisive record's spelling). Review round 1 of aafa76c51 (2026-10-08): the park
    was decided off the fit record or the first summary alone, so a mixed sweep whose first record was out of scope PARKED whole -- an
    accepted empty revision, its in-scope trials filed as nothing, a fact calling the whole directory out of scope -- and the same sweep
    the other way round read as one search with the out-of-scope trial among its trials. Neither is a reading of the records; the
    refusal names the records so a person splits the directory or corrects them. The caller resolves the kind from the FIRST of these
    same claims (the fit record's comes first), never from a record outside them, so what the check certifies is what gets decided."""
    by_entry: Dict[Optional[str], List[Tuple[str, str]]] = {}
    for spelling, record in claims:
        by_entry.setdefault(_kind_entry(spelling)[0], []).append((spelling, record))
    if len(by_entry) < 2:
        return
    sides = []
    for prefix, items in by_entry.items():
        spellings = " / ".join(repr(s) for s in sorted({s for s, _ in items}))
        sides.append(f"{', '.join(r for _, r in items)} name{'s' if len(items) == 1 else ''} {spellings} ({_describe_kind_entry(prefix)})")
    raise HpoRefusal(where, "the sweep's records disagree on the intervention's kind: " + "; ".join(sides)
                            + " -- a mixed sweep is neither parked as one kind nor read as one search; split the directory or correct the "
                              "records before it is read")


def _listdir(p: str) -> List[str]:
    try:
        return sorted(store_listdir(p))
    except OSError:
        return []


@dataclass
class SweepDir:
    path: str
    label: str
    layout: str
    key: Optional[str] = None       # reseed model key (qwen35_4b_base_eb64_s1) or the model dir
    arm_dir_legacy: str = ""
    substrate_evidence: Optional[Tuple[str, str]] = None   # (substrate, evidence)


def enumerate_sweeps(roots: Sequence[Tuple[str, str, str]] = DEFAULT_HPO_ROOTS) -> List[SweepDir]:
    out: List[SweepDir] = []
    for label, root, layout in roots:
        if not os.path.isdir(root):
            continue
        pods = os.path.join(root, "PODS.md")
        sub_ev = ("runpod", f"{pods} (pod endpoints table)") if os.path.isfile(pods) else None
        if sub_ev is None:
            # ~/hpo-sync is the pods' summary mirror that ~/hpo-r1/hpo-sync.sh maintains (MIRROR=<this root>)
            sync_sh = os.path.join(os.path.dirname(root.rstrip("/")), "hpo-r1", "hpo-sync.sh")
            if os.path.isfile(sync_sh):
                try:
                    with store_open(sync_sh, "r", encoding="utf-8") as fh:
                        head = fh.read(4000)
                except OSError:
                    head = ""
                if f"MIRROR={root.rstrip('/')}" in head and os.path.isfile(os.path.join(os.path.dirname(sync_sh), "PODS.md")):
                    sub_ev = ("runpod", f"{sync_sh} (MIRROR={root}: the pods' summary mirror) + {os.path.join(os.path.dirname(sync_sh), 'PODS.md')}")
        if layout == "reseed":
            res = os.path.join(root, "results")
            for key in _listdir(res):
                kdir = os.path.join(res, key)
                for arm in _listdir(kdir):
                    adir = os.path.join(kdir, arm)
                    if os.path.isdir(adir) and any(_TRIAL_RE.match(t) for t in _listdir(adir)):
                        out.append(SweepDir(adir, label, layout, key, arm, ("modal", "manifests/hpo_calls_*.json:app (Modal function calls)")))
        elif layout == "flat":
            for arm in _listdir(root):
                adir = os.path.join(root, arm)
                if os.path.isdir(adir) and any(_TRIAL_RE.match(t) for t in _listdir(adir)):
                    out.append(SweepDir(adir, label, layout, None, arm, sub_ev or _trace_substrate(adir)))
        elif layout == "model":
            for model in _listdir(root):
                mdir = os.path.join(root, model)
                if not os.path.isdir(mdir):
                    continue
                for arm in _listdir(mdir):
                    adir = os.path.join(mdir, arm)
                    if os.path.isdir(adir) and any(_TRIAL_RE.match(t) for t in _listdir(adir)):
                        out.append(SweepDir(adir, label, layout, model, arm, sub_ev or _trace_substrate(adir)))
    return out


def _trace_substrate(sweep_dir: str) -> Optional[Tuple[str, str]]:
    """A trial's ``tb/`` profiler trace named ``modal_<n>.<id>.pt.trace.json`` is the Modal worker's own stamp."""
    for fn in _listdir(sweep_dir):
        if not _TRIAL_RE.match(fn):
            continue
        tb = os.path.join(sweep_dir, fn, "tb")
        for tf in _listdir(tb):
            if tf.startswith("modal_") and tf.endswith(".pt.trace.json"):
                return ("modal", f"{os.path.join(tb, tf)} (profiler trace written by the Modal worker)")
    return None


def _sweep_files(sd: SweepDir) -> List[Tuple[str, str]]:
    """(path, role) of every record file the sweep directory carries (bounded: named files, trial dirs one level)."""
    files: List[Tuple[str, str]] = []
    for fn in _listdir(sd.path):
        p = os.path.join(sd.path, fn)
        if _FIT_RE.match(fn) or fn in ("confirm_check.json", "r1_next_round.json") or fn.startswith("pull_check_"):
            files.append((p, "report"))
        elif _TRIAL_RE.match(fn) and os.path.isdir(p):
            s = os.path.join(p, "summary.json")
            if os.path.isfile(s):
                files.append((s, "report"))
            m = os.path.join(p, "metrics.jsonl")
            if os.path.isfile(m):
                files.append((m, "log"))
    if sd.layout == "reseed":
        hpo_root = os.path.dirname(os.path.dirname(os.path.dirname(sd.path)))   # …/hpo
        for fn in _listdir(os.path.join(hpo_root, "trials")):
            if sd.key and sd.arm_dir_legacy and fn.endswith(f"{sd.key}_{sd.arm_dir_legacy}_all.json") or (sd.key and sd.arm_dir_legacy and f"_{sd.key}_{sd.arm_dir_legacy}" in fn):
                files.append((os.path.join(hpo_root, "trials", fn), "report"))
        for fn in _listdir(os.path.join(hpo_root, "manifests")):
            if sd.key and sd.arm_dir_legacy and f"_{sd.key}_{sd.arm_dir_legacy}" in fn:
                files.append((os.path.join(hpo_root, "manifests", fn), "report"))
        for fn in ("hpo_fold.json", "search_result.json"):
            p = os.path.join(hpo_root, fn)
            if os.path.isfile(p):
                files.append((p, "report"))
        recon = os.path.join(os.path.dirname(hpo_root), "recon", f"recon_clip_{sd.key}_{sd.arm_dir_legacy}.json")
        if os.path.isfile(recon):
            files.append((recon, "report"))
    return files


# --------------------------------------------------------------------------------------------
# building the rows
# --------------------------------------------------------------------------------------------


class HpoRefusal(Exception):
    def __init__(self, where: str, why: str):
        super().__init__(f"{where}: {why}")
        self.where = where
        self.why = why


class HpoParked(HpoRefusal):
    """A sweep directory whose intervention is a kind ruled OUT OF SCOPE by name
    (``legacy_spellings.yaml`` ``object_kinds`` declares it so, ``out_of_scope_object_kinds`` carries the ruling per kind) -- the EXACT
    spellings of :data:`_OUT_OF_SCOPE_SPELLINGS_BY_KIND` (``token_faithful`` / ``token_faithful_all``, ``bidpo``). No ``object_kind`` spells it, so no ``fit_search`` can name a
    target object class under the vocabulary -- and that is the ruling, not a defect of the sweep or the reader. An :class:`HpoRefusal`
    (so every caller that refuses on the base class still does), distinguished so the adapter can PARK the group -- an accepted revision
    with no record and a status fact citing the table (:func:`parked_status_facts`) -- instead of refusing it on every read (the three
    ``token_faithful`` sweeps were refused in every window since 2026-10-07, and a per-group refusal reads the whole adapter REFUSED).
    ``kind_legacy`` is the intervention spelling as the sweep records it (always one the table names exactly), ``kind_legacy_source`` the
    record it was read off (the fit record's, else the first summary that names one; the sweep directory's name only when no record
    claims a spelling), ``prefix`` the table ENTRY it belongs to -- the kind's own spelling, ``token_faithful`` for a
    ``token_faithful_all`` sweep (the name predates the exact-spelling table; it was a prefix once) -- ``entry`` the table row when the seeds carry it (``legacy``, ``status``, ``object_role``,
    ``reason``, ``source``) and ``{}`` when they do not (the registered factory loads seeds from the registry connection alone)."""

    def __init__(self, where: str, why: str, kind_legacy: str, kind_legacy_source: str, prefix: str, entry: Optional[Mapping[str, object]],
                 root_label: str, seeds_source: str, n_trial_summaries: int):
        super().__init__(where, why)
        self.kind_legacy = kind_legacy
        self.kind_legacy_source = kind_legacy_source
        self.prefix = prefix
        self.entry: Dict[str, object] = dict(entry or {})
        self.root_label = root_label
        self.seeds_source = seeds_source
        self.n_trial_summaries = n_trial_summaries


#: the record types a parked sweep files none of
_PARKED_WITHHELD: Tuple[str, ...] = ("fit_search", "fit_trial")


def parked_status_facts(exc: HpoParked) -> Dict[str, object]:
    """The ``source_status`` facts of a sweep parked by name, in the objects bank's shape (``objects_bank.parked_status_facts``), so the
    status page's "Parked by name" section renders both adapters' parks through one query: ``refuse`` -- one readable line (the runner's
    ``detail`` renders it; a non-empty list is an explanation under ``EXPLANATION_KEYS``, so the read files ``ok = 0`` and is never judged
    silent) -- and ``parked`` -- one structured entry: the directory, the intervention spelling and the record it was read off, the table
    row's status / role / reason and source when the seeds carried the row (``table_row_read``), the ruling, what was withheld."""
    ent = exc.entry
    row_read = bool(ent)
    line = (f"parked by name: {exc.where} is a {exc.kind_legacy!r} sweep ({exc.kind_legacy_source}), a kind ruled out of scope ({PARKED_RULING}; "
            f"legacy_spellings.yaml object_kinds declares {exc.prefix!r} OUT OF SCOPE); no fit_search / fit_trial record -- "
            + (str(ent.get("reason", ""))[:160] if row_read else f"the table row is not in the seeds (source: {exc.seeds_source}); the park stands on hpo_logs's kind table"))
    entry = {"dir": exc.where, "kind_legacy": exc.kind_legacy, "kind_legacy_source": exc.kind_legacy_source, "legacy": ent.get("legacy"),
             "status_ruled": ent.get("status"), "object_role_ruled": ent.get("object_role"), "reason": ent.get("reason"), "ruling": PARKED_RULING,
             "ruling_source": ent.get("source"), "table_row_read": row_read, "seeds_source": exc.seeds_source, "root_label": exc.root_label,
             "withheld": list(_PARKED_WITHHELD), "n_trial_summaries": exc.n_trial_summaries}
    return {"refuse": [line], "parked": [entry]}


def search_id_for(label: str, model: str, arm: str, position: str, si: str, key: Optional[str]) -> str:
    return "hpo:" + identity.sha16("|".join([label, model, arm, position, si, key or ""]))


def _conf(entries: Dict[str, Dict[str, str]], col: str, confidence: str, source: str) -> None:
    e = {"confidence": confidence, "source": source}
    if confidence == "unverified":
        e["reason"] = source
    entries[col] = e


def build_search(sd: SweepDir, seeds: Seeds, now: str, mtime_utc: Optional[str] = None) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """(fit_search payload, [fit_trial payloads]) for one sweep directory."""
    fc: Dict[str, Dict[str, str]] = {}
    extra: Dict[str, object] = {}
    trials_raw: List[Tuple[str, Dict[str, object]]] = []
    for fn in _listdir(sd.path):
        m = _TRIAL_RE.match(fn)
        if m and os.path.isfile(os.path.join(sd.path, fn, "summary.json")):
            try:
                trials_raw.append((fn, read_json(os.path.join(sd.path, fn, "summary.json"))))
            except (OSError, ValueError) as exc:
                raise HpoRefusal(os.path.join(sd.path, fn, "summary.json"), f"unreadable trial summary ({exc})")
    if not trials_raw:
        raise HpoRefusal(sd.path, "no r<k>_t<NN>/summary.json trial records")
    fits = [fn for fn in _listdir(sd.path) if _FIT_RE.match(fn)]
    fit = None
    fit_name = None
    if fits:
        fit_name = sorted(fits, key=lambda s: (len(s), s))[-1]   # r1+2_fit.json over r1_fit.json when both exist
        fit = read_json(os.path.join(sd.path, fit_name))
    confirm = read_json(os.path.join(sd.path, "confirm_check.json")) if os.path.isfile(os.path.join(sd.path, "confirm_check.json")) else None
    t0 = trials_raw[0][1]
    model = (fit or {}).get("model") or t0.get("model")
    position = (fit or {}).get("position") or t0.get("position")
    si = (fit or {}).get("si") or t0.get("si")
    # ONE kind per sweep, read off every record that claims a spelling before anything is decided on it (:func:`_check_one_kind`): a summary
    # that records no spelling claims nothing (the failed trials of several live sweeps carry none). The DECISIVE spelling -- the one the
    # kind is resolved from and the search or the fact records -- is drawn from the SAME claim set the check reads: the fit record's when
    # it claims one, else the first summary that names one; the directory's name stands in only when no record claims a spelling at all.
    # Review round 3 of 125f4194a (2026-10-08): the decisive spelling was the fit record's, else the FIRST summary's, else the directory's
    # name, so a sweep whose first summary recorded no spelling was classified by its directory name against the later summaries' claim
    # (the check saw only the claims and found them consistent): an in-scope claim under a `token_faithful_all` directory parked, a
    # declared out-of-scope claim under an unmapped directory name refused, and one under a `psr_…` directory read as a relu_steer search
    # with the out-of-scope trial among its trials. Records outrank the directory's name, as they always did when the first one spoke.
    claims: List[Tuple[str, str]] = ([(str(fit["arm"]), str(fit_name))] if (fit or {}).get("arm") else []) \
        + [(str(t["arm"]), f"{fn}/summary.json") for fn, t in trials_raw if t.get("arm")]
    _check_one_kind(sd.path, claims)
    if claims:
        arm_legacy, arm_legacy_source = claims[0][0], f"{claims[0][1]}:arm"
    else:
        arm_legacy, arm_legacy_source = sd.arm_dir_legacy, "the sweep directory's name"
    # the kind resolver runs BEFORE the model-closure checks: a spelling the table declares OUT OF SCOPE parks on the kind alone -- parking
    # needs no model to apply, and the parked path files no fit_search that would carry the model's key (the objects bank classifies
    # scope before it builds a payload, likewise). An archived token_faithful sweep on a model with no `model` row would otherwise refuse
    # at the model check and never reach the park, leaving the adapter not read whole on the very condition the park replaces. A spelling
    # the table does not name still refuses AFTER the model checks, in the order it always did -- and the table names EXACT spellings:
    # `token_faithful_v2` is no more parked than `novel_method` is.
    matched_prefix, kind = _kind_entry(arm_legacy)
    if matched_prefix is not None and kind is None:
        # park-by-name: the prefix is one the table declares OUT OF SCOPE; the table row (status / role / reason) rides the fact when
        # the seeds carry it (`out_of_scope_for` matches the spelling, its directory, or the prefix itself) and is marked unread when not
        oos = seeds.out_of_scope_for(str(arm_legacy), sd.arm_dir_legacy, matched_prefix)
        entry = oos[1] if oos else None
        why = (f"intervention {arm_legacy!r} ({arm_legacy_source}) is a kind ruled out of scope ({PARKED_RULING}; legacy_spellings.yaml "
               f"object_kinds declares {matched_prefix!r} OUT OF SCOPE): parked by name, no fit_search / fit_trial record"
               + (f" -- {str(entry.get('reason', ''))[:200]}" if entry else ""))
        raise HpoParked(sd.path, why, str(arm_legacy), arm_legacy_source, matched_prefix, entry, sd.label, seeds.source, len(trials_raw))
    if not (isinstance(model, str) and model):
        raise HpoRefusal(sd.path, "no trial summary or fit record names the model")
    if not isinstance(position, str):
        mm = _SWEEP_DIR_RE.match(sd.arm_dir_legacy)
        position = mm.group("pos") if mm else None
    model_id = seeds.model_id_by_hf_repo.get(model)
    if not model_id:
        raise HpoRefusal(sd.path, f"model {model!r} has no `model` row / models.yaml entry (plan F15 closure); known: {sorted(seeds.model_id_by_hf_repo)[:12]}")
    if matched_prefix is None:
        # a kind nobody has ruled on: a refusal naming the spelling, never a park (parking is the declared set alone, and the
        # declared set is exact spellings: one that merely begins with a parked spelling is not in it)
        raise HpoRefusal(sd.path, f"arm {arm_legacy!r} maps to no object_kind in the vocabulary (the D37 out-of-scope spellings hpo_logs parks "
                                  f"by name are exactly {', '.join(repr(s) for s in _parked_spellings())}; a spelling that only begins with one "
                                  "of them is not declared and refuses here; add this spelling to the table row it belongs to, or to the "
                                  "vocabulary, before the sweep is read)")
    sid = search_id_for(sd.label, model_id, str(arm_legacy), str(position), str(si), sd.key)
    p: Dict[str, object] = {"search_id": sid,
                            "campaign": "reseed" if sd.layout == "reseed" else "two_si",
                            "target_object_class_json": {"model_id": model_id, "model_hf_repo": model, "object_kind": kind, "si_id": si, "positions": position,
                                                         "arm_legacy": arm_legacy, "model_key_legacy": sd.key}}
    _conf(fc, "campaign", "derived", "reseed_eb64 sweep root → reseed; the tune_arm sweeps are the two-SI campaign's")
    # method: the reseed designs say sobol; tune_arm.py rounds are Sobol by its own docstring
    design = None
    if sd.layout == "reseed":
        hpo_root = os.path.dirname(os.path.dirname(os.path.dirname(sd.path)))
        for fn in _listdir(os.path.join(hpo_root, "trials")):
            if fn.startswith("r1_") and sd.key and sd.arm_dir_legacy and f"_{sd.key}_{sd.arm_dir_legacy}" in fn:
                design = read_json(os.path.join(hpo_root, "trials", fn))
                p["method"] = "sobol"
                _conf(fc, "method", "recorded", f"trials/{fn}:sobol_m/sobol_d")
                break
    if "method" not in p:
        if all(("box" in t and "warmup_levels" in t) for _, t in trials_raw):
            p["method"] = "sobol"
            _conf(fc, "method", "derived", _TUNE_SOBOL_CITATION)
        else:
            raise HpoRefusal(sd.path, "search method undecidable: no Sobol design file and the trial summaries lack tune_arm's box/warmup_levels shape")
    completed = [(fn, t) for fn, t in trials_raw if t.get("status") == "ok"]
    p["n_trials"] = len(trials_raw)
    p["n_trials_completed"] = len(completed)
    _conf(fc, "n_trials", "recorded", "count of r<k>_t<NN>/summary.json")
    _conf(fc, "n_trials_completed", "recorded", "count of trial summaries with status ok")
    box = (fit or {}).get("box") or t0.get("box") or {}
    space: Dict[str, object] = {}
    for axis, rng in (box or {}).items():
        if isinstance(rng, list) and len(rng) == 2:
            space[axis] = {"kind": "log_uniform", "low": rng[0], "high": rng[1]}
    levels = (fit or {}).get("levels") or (fit or {}).get("fit", {}).get("levels") if fit else None
    wl = t0.get("warmup_levels") or (design or {}).get("warmup_levels") or levels
    if isinstance(wl, list):
        space["warmup_frac"] = {"kind": "choice", "values": wl}
    if design and isinstance(design.get("constant_axes"), Mapping):
        for axis, val in design["constant_axes"].items():
            space[axis] = {"kind": "choice", "values": [val]}
    p["search_space_json"] = space
    _conf(fc, "search_space_json", "recorded", ("trials design box/warmup_levels/constant_axes" if design else f"{fit_name or 'summary.json'}:box, warmup_levels"))
    p["objective"] = (fit or {}).get("y_key") or t0.get("heldout_objective") or "best_heldout_loss"
    p["objective_direction"] = "minimize"
    _conf(fc, "objective", "recorded" if fit else "derived", f"{fit_name}:y_key" if fit else "summary.json:heldout_objective (best held-out loss)")
    _conf(fc, "objective_direction", "derived", "held-out loss is minimised")
    # winner / selection
    sel = (fit or {}).get("selection") if fit else None
    guard = (fit or {}).get("selection_guard") if fit else None
    winner = (fit or {}).get("winner") if fit else None
    fold_entry = None
    if sd.layout == "reseed":
        hpo_root = os.path.dirname(os.path.dirname(os.path.dirname(sd.path)))
        fold_path = os.path.join(hpo_root, "hpo_fold.json")
        if os.path.isfile(fold_path):
            fold = read_json(fold_path)
            fold_entry = ((fold.get(sd.key) or {}).get(sd.arm_dir_legacy)) if isinstance(fold.get(sd.key), Mapping) else None
    if fold_entry:
        p["winner_params_json"] = {k: fold_entry.get(k) for k in ("lr", "max_grad_norm", "warmup_frac")}
        _conf(fc, "winner_params_json", "recorded", f"hpo_fold.json[{sd.key}][{sd.arm_dir_legacy}] (the folded winner)")
        g = fold_entry.get("selection_guard") or {}
        if g.get("selected_trial"):
            p["winner_trial_id"] = str(g["selected_trial"])
            _conf(fc, "winner_trial_id", "recorded", "hpo_fold.json selection_guard.selected_trial (raw best adopted)")
        else:
            _conf(fc, "winner_trial_id", "unverified", "the winner is the fitted argmin (a point, not a trial): selection_guard.selected_trial is null")
        extra["selection_guard"] = {k: g.get(k) for k in ("selected_from", "alarmed", "alarms", "fall_through") if k in g}
        extra["selection_action"] = (confirm or {}).get("selection_action")
    elif winner:
        p["winner_params_json"] = {k: winner.get(k) for k in ("lr", "max_grad_norm", "warmup_frac") if k in winner}
        _conf(fc, "winner_params_json", "recorded", f"{fit_name}:winner")
        # no hpo_fold.json (the tune_arm layouts): the fit record's own selection_guard says whether a raw trial was adopted
        g = guard if isinstance(guard, Mapping) else {}
        if g.get("selected_trial"):
            p["winner_trial_id"] = str(g["selected_trial"])
            _conf(fc, "winner_trial_id", "recorded", f"{fit_name}:selection_guard.selected_trial (raw best adopted)")
        else:
            _conf(fc, "winner_trial_id", "unverified", "the winner is the fitted argmin (a point, not a trial)"
                                                       + (": selection_guard.selected_trial is null" if g else "; no selection_guard in the fit record"))
        if g:
            extra["selection_guard"] = {k: g.get(k) for k in ("selected_from", "alarmed", "alarms", "fall_through") if k in g}
    elif sel:
        # a selection with no adopted winner (the argmin sat on the box edge, or the lane never concluded): NOT a winner
        _conf(fc, "winner_params_json", "unverified", f"{fit_name}:winner is null (selection on_edge={sel.get('on_edge')}); the fitted argmin is kept in extra_json.fitted_argmin, not adopted")
        _conf(fc, "winner_trial_id", "unverified", f"{fit_name}:winner is null")
    else:
        _conf(fc, "winner_params_json", "unverified", "no fit record (r*_fit.json) names a winner; the sweep may not have concluded")
        _conf(fc, "winner_trial_id", "unverified", "no fit record names a winner")
    best = None
    for fn, t in completed:
        v = t.get("best_heldout_loss")
        if isinstance(v, (int, float)) and (best is None or v < best[1]):
            best = (fn, float(v))
    if best:
        p["winner_objective"] = best[1]
        _conf(fc, "winner_objective", "recorded", f"min best_heldout_loss over completed trials ({best[0]})")
        extra["best_raw_trial"] = best[0]
    if sel is not None and "on_edge" in sel:
        p["argmin_interior"] = 0 if sel.get("on_edge") else 1
        _conf(fc, "argmin_interior", "recorded", f"{fit_name}:selection.on_edge")
        extra["edge_pos"] = sel.get("edge_pos")
        extra["fitted_argmin"] = {k: sel.get(k) for k in ("lr", "max_grad_norm", "warmup_frac")}
        if sel.get("alarms"):
            extra["selection_alarms"] = sel.get("alarms")
    else:
        _conf(fc, "argmin_interior", "unverified", "no fit record evaluates interiority (selection.on_edge absent)")
    fitblk = (fit or {}).get("fit") if fit else None
    if isinstance(fitblk, Mapping) and "positive_definite" in fitblk:
        p["quadratic_fit_positive_definite"] = 1 if fitblk.get("positive_definite") else 0
        _conf(fc, "quadratic_fit_positive_definite", "recorded", f"{fit_name}:fit.positive_definite")
        p["quadratic_fit_params_json"] = {k: fitblk.get(k) for k in ("names", "coef", "hessian_eigs", "argmin", "fitted_min_log_y", "n_rows", "design_rank",
                                                                       "design_cond", "levels", "sigma_log_y", "rms_log_y", "resid_log_y", "fixed_axes", "cont_axes") if k in fitblk}
        _conf(fc, "quadratic_fit_params_json", "recorded", f"{fit_name}:fit")
        if isinstance(fitblk.get("sigma_log_y"), (int, float)):
            p["noise_estimate"] = float(fitblk["sigma_log_y"])
            _conf(fc, "noise_estimate", "recorded", f"{fit_name}:fit.sigma_log_y")
        else:
            _conf(fc, "noise_estimate", "unverified", f"{fit_name}:fit records no sigma_log_y")
    else:
        _conf(fc, "quadratic_fit_positive_definite", "unverified", "no quadratic fit record")
        _conf(fc, "noise_estimate", "unverified", "no quadratic fit record")
    p["recentre_rounds"] = max(0, len((fit or {}).get("rounds") or [1]) - 1)
    _conf(fc, "recentre_rounds", "recorded" if fit else "derived", f"{fit_name}:rounds" if fit else "one round of trials on disk")
    r9 = [fn for fn, _ in trials_raw if fn.startswith("r9_")]
    if confirm:
        p["confirmation_trial_id"] = r9[0] if r9 else "r9_t00"
        _conf(fc, "confirmation_trial_id", "recorded", "confirm_check.json (the r9 confirmation trial)")
        verdict = confirm.get("confirmation_verdict")
        p["confirmation_within_noise"] = 1 if verdict == "in_interval" else 0
        _conf(fc, "confirmation_within_noise", "recorded", f"confirm_check.json:confirmation_verdict={verdict}")
        extra["confirmation"] = {k: confirm.get(k) for k in ("confirmation_verdict", "d_over_sigma", "fit_reliability", "selection_action", "criterion_3_met", "pass")}
    else:
        _conf(fc, "confirmation_trial_id", "unverified", "no confirm_check.json (no confirmation trial)")
        _conf(fc, "confirmation_within_noise", "unverified", "no confirmation trial")
    if sd.layout == "reseed":
        recon = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sd.path)))), "recon", f"recon_clip_{sd.key}_{sd.arm_dir_legacy}.json")
        if os.path.isfile(recon):
            p["grad_norm_reconnaissance_uri"] = recon
            _conf(fc, "grad_norm_reconnaissance_uri", "recorded", "recon/recon_clip_<key>_<arm>.json")
            rc = read_json(recon)
            extra["recon_clip"] = {k: rc.get(k) for k in ("clip", "clip_raw", "rule", "percentile") if k in rc}
        else:
            _conf(fc, "grad_norm_reconnaissance_uri", "unverified", "no recon_clip record beside the sweep")
    else:
        _conf(fc, "grad_norm_reconnaissance_uri", "unverified", "the tune_arm sweeps ran no grad-norm reconnaissance pass")
    if sd.substrate_evidence:
        p["substrate"] = sd.substrate_evidence[0]
        _conf(fc, "substrate", "recorded" if sd.layout == "reseed" else "derived", sd.substrate_evidence[1])
    else:
        # not a refusal — `fit_search.substrate` admits `unverified` for exactly this; the missing evidence is the reason
        p["substrate"] = "unverified"
        _conf(fc, "substrate", "unverified", "no record names the substrate the sweep ran on (no PODS.md beside the root, "
                                             "no Modal profiler trace beside a trial, no Modal manifest)")
    p["job_ids_json"] = []
    if sd.layout == "reseed":
        hpo_root = os.path.dirname(os.path.dirname(os.path.dirname(sd.path)))
        ids = []
        for fn in _listdir(os.path.join(hpo_root, "manifests")):
            if sd.key and sd.arm_dir_legacy and f"_{sd.key}_{sd.arm_dir_legacy}" in fn:
                man = read_json(os.path.join(hpo_root, "manifests", fn))
                for t in man.get("trials") or []:
                    if isinstance(t, Mapping) and t.get("call_id"):
                        ids.append(str(t["call_id"]))
                if isinstance(man.get("written_at"), str) and "started_utc" not in p:
                    p["started_utc"] = _utc(man["written_at"]) or now
                    _conf(fc, "started_utc", "recorded", f"manifests/{fn}:written_at")
        p["job_ids_json"] = ids
        _conf(fc, "job_ids_json", "recorded", "manifests/hpo_calls_*.json trials[].call_id")
    if "started_utc" not in p:
        if mtime_utc:
            p["started_utc"] = mtime_utc
            _conf(fc, "started_utc", "derived", "earliest trial summary mtime on this machine (a mirror time; the sweep's own start is not recorded)")
        else:
            p["started_utc"] = now
            _conf(fc, "started_utc", "unverified", "no record and no file time names the start")
    p["code_sha"] = "unverified"
    sha_src = (design or {}).get("config_sha") if design else None
    _conf(fc, "code_sha", "unverified", "the trial records carry no 40-hex code sha" + (f" (design config_sha {sha_src} is a 16-hex config hash)" if sha_src else ""))
    if sd.layout == "reseed":
        sr = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sd.path))), "search_result.json")
        if os.path.isfile(sr):
            srd = read_json(sr)
            if isinstance(srd.get("sha"), str) and re.match(r"^[0-9a-f]{40}$", srd["sha"]):
                p["code_sha"] = srd["sha"]
                _conf(fc, "code_sha", "recorded", "search_result.json:sha (the worktree the search ran from)")
    p["trials_uri"] = sd.path
    _conf(fc, "trials_uri", "recorded", "the sweep directory on this machine")
    if fit_name:
        extra["fit_record"] = os.path.join(sd.path, fit_name)
    extra["arm_dir"] = sd.arm_dir_legacy
    p["field_confidence_json"] = fc
    p["extra"] = extra
    # trials
    trials: List[Dict[str, object]] = []
    rows_by_index = {}
    if isinstance((fit or {}).get("rows"), list):
        for r in fit["rows"]:
            if isinstance(r, Mapping) and "index" in r:
                rows_by_index[(r.get("round", 1), r["index"])] = r
    for fn, t in trials_raw:
        m = _TRIAL_RE.match(fn)
        tfc: Dict[str, Dict[str, str]] = {}
        params = dict(t.get("trial") or {})
        tp: Dict[str, object] = {"search_id": sid, "trial_id": fn, "params_json": params,
                                 "status": "complete" if t.get("status") == "ok" else "failed"}
        _conf(tfc, "params_json", "recorded", "summary.json:trial")
        _conf(tfc, "status", "recorded", f"summary.json:status={t.get('status')!r}")
        if isinstance(t.get("best_heldout_loss"), (int, float)):
            tp["objective_value"] = float(t["best_heldout_loss"])
            _conf(tfc, "objective_value", "recorded", "summary.json:best_heldout_loss")
        row = rows_by_index.get((int(m.group("round")), int(m.group("idx")))) if m else None
        if row and isinstance(row.get("clip_activation_rate"), (int, float)):
            tp["clip_fraction"] = float(row["clip_activation_rate"])
            _conf(tfc, "clip_fraction", "recorded", f"{fit_name}:rows[].clip_activation_rate")
        else:
            _conf(tfc, "clip_fraction", "unverified", "no fit-record row carries clip_activation_rate for this trial")
        curve = t.get("heldout_curve")
        if isinstance(curve, list) and curve:
            tp["epochs_seen"] = max(0, len(curve) - 1)
            _conf(tfc, "epochs_seen", "derived", "len(summary.json:heldout_curve) - 1 (epoch 0 is the untrained probe)")
        elif isinstance(t.get("best_epoch"), int) and t.get("stopped_early") is False and isinstance(t.get("epochs"), int):
            tp["epochs_seen"] = int(t["epochs"])
            _conf(tfc, "epochs_seen", "derived", "summary.json:epochs with stopped_early=false")
        if isinstance(t.get("wandb_run"), str) and t["wandb_run"]:
            tp["wandb_run_id"] = t["wandb_run"]
            _conf(tfc, "wandb_run_id", "recorded", "summary.json:wandb_run")
        extra_t = {k: t.get(k) for k in ("fit_fingerprint", "seed", "wall_s", "steps_run", "total_steps", "best_epoch", "stopped_early", "floor_ratio", "attempt") if k in t}
        extra_t["field_confidence"] = tfc
        tp["extra"] = extra_t
        trials.append(tp)
    return p, trials


# --------------------------------------------------------------------------------------------
# linking objects to searches
# --------------------------------------------------------------------------------------------


def link_objects(objects: Sequence[Dict[str, object]], searches: Sequence[Dict[str, object]], rel_tol: float = 1e-9,
                 allow_class_link: bool = False) -> int:
    """Set ``fit_search_id`` / ``fit_lr_selected_by = search`` / ``fit_argmin_interior`` on every object payload whose
    (fit_model_id, object_kind, teacher-or-corpus si, positions) matches a search WITH an adopted winner and whose
    own ``fit_lr`` equals that winner's lr; returns the number linked. An object that records no lr is left
    unlinked (``allow_class_link`` — off by default — admits a ``derived`` link when exactly one such search matches;
    a fit config can move after a search, so this is never the default)."""
    by_key: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for s in searches:
        if not isinstance(s.get("winner_params_json"), Mapping) or s["winner_params_json"].get("lr") is None:
            continue   # a search with no adopted winner links nothing
        cls = s.get("target_object_class_json") or {}
        pos = {"all": "all", "response": "response_incl_assistant_header"}.get(str(cls.get("positions")), str(cls.get("positions")))
        by_key.setdefault((str(cls.get("model_id")), str(cls.get("object_kind")), str(cls.get("si_id")), pos), []).append(s)
    n = 0
    for o in objects:
        if o.get("fit_target_kind") not in ("system_instruction_distillation", "corpus_document_fit"):
            continue
        si = o.get("fit_teacher_si_id") or o.get("fit_corpus_id")
        key = (str(o.get("fit_model_id")), str(o.get("object_kind")), str(si), str(o.get("fit_positions")))
        cands = by_key.get(key, [])
        fc = o.setdefault("field_confidence_json", {})
        lr = o.get("fit_lr")
        matched = None
        if lr is not None:
            for s in cands:
                w = (s.get("winner_params_json") or {}).get("lr")
                if isinstance(w, (int, float)) and abs(float(w) - float(lr)) <= rel_tol * max(1.0, abs(float(lr))):
                    matched = s
                    break
            if matched is None:
                continue
            conf = "recorded"
            why = f"winner lr == object lr ({lr})"
        else:
            if not allow_class_link or len(cands) != 1:
                continue
            matched = cands[0]
            conf = "derived"
            why = "the only search on this machine for (model, kind, si, positions); the object records no lr of its own"
            w = (matched.get("winner_params_json") or {}).get("lr")
            if isinstance(w, (int, float)):
                o["fit_lr"] = float(w)
                fc["fit_lr"] = {"confidence": "derived", "source": f"fit_search {matched['search_id']} winner lr ({why})"}
        o["fit_search_id"] = matched["search_id"]
        fc["fit_search_id"] = {"confidence": conf, "source": f"hpo_logs match on (model, kind, si, positions): {why}"}
        o["fit_lr_selected_by"] = "search"
        fc["fit_lr_selected_by"] = {"confidence": conf, "source": f"fit_search {matched['search_id']}"}
        if matched.get("argmin_interior") is not None:
            o["fit_argmin_interior"] = matched["argmin_interior"]
            fc["fit_argmin_interior"] = {"confidence": "recorded", "source": f"fit_search {matched['search_id']}.argmin_interior"}
        n += 1
    return n


# --------------------------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------------------------


class HpoLogsAdapter:
    name = "hpo_logs"
    covers_sources: Tuple[str, ...] = ()
    rng_scope = "none"
    allow_symlinks = False
    priority = 0                           # The sole source of fit_search / fit_trial keys; no tie to break
    #: 2 = a sweep directory whose intervention is a kind ruled out of scope (`token_faithful` / `bidpo`: the `None` entries of
    #: `_KIND_BY_SWEEP_PREFIX`, the spellings `legacy_spellings.yaml` `object_kinds` declares OUT OF SCOPE and `out_of_scope_object_kinds`
    #: rules on) is PARKED by name instead of refused (park-by-name, 2026-10-07; the objects bank's rule since its version 9): an
    #: accepted revision with no `fit_search` / `fit_trial` record and a `source_status` fact (`extra.refuse` + `extra.parked`,
    #: :func:`parked_status_facts`, the bank's shape) citing the table, so the three standing refusals of the cluster's reads
    #: (`~/hpo-r1/token_faithful_all`, `~/hpo-r1/llama_token_faithful`, `~/hpo-sync/token_faithful_all`, each spelling `arm: token_faithful`
    #: in its trial summaries; every pass since 2026-10-07 listed hpo_logs as not read whole over them) end, and nothing enters the frame
    #: (a parked sweep is no search, so `link_objects` links nothing to it). The park is decided on the kind alone, before the model-closure
    #: checks (a sweep on an unregistered model would otherwise refuse at the model check and never park), and on ONE kind read off
    #: every record that claims a spelling: a sweep whose records disagree on the kind refuses by name citing them (deciding off the first
    #: record alone would park a mixed directory whole, its in-scope trials filed as nothing). The parked set is EXACT
    #: spellings (`_OUT_OF_SCOPE_SPELLINGS_BY_KIND`; a prefix match would park `token_faithful_v2` / `bidpo_control`
    #: sweeps the table never names); a spelling the table does not name still refuses by name, unchanged. The decisive spelling is drawn
    #: from the same claims the consistency check reads (a claimless first summary would otherwise let the directory's NAME decide
    #: against the later summaries' claim -- an in-scope claim parked, a declared one refused, an out-of-scope trial read as a search).
    #: The bump re-mints every hpo revision id (the runner keys the re-read on the version the head's marker records, `runner.plan_groups`):
    #: the accepted groups re-emit byte-identically (no payload moves), the refused ones publish their first accepted, empty revision.
    EMISSION_VERSION = 2

    def __init__(self, roots: Sequence[Tuple[str, str, str]] = DEFAULT_HPO_ROOTS, seeds: Optional[Seeds] = None, seeds_dir: Optional[str] = None):
        self.hpo_roots = tuple(roots)
        self.roots = tuple(r for _, r, _ in self.hpo_roots)
        self.seeds = seeds
        self.seeds_dir = seeds_dir
        self._sweeps: Dict[str, SweepDir] = {}
        #: group (the sweep directory) -> the status facts of a sweep PARKED by name: the `refuse` line that explains the empty read
        #: and the structured `parked` entry the status page renders; read by the runner through :meth:`status_extra` after the group's read
        self._parked: Dict[str, Dict[str, object]] = {}

    def discover(self) -> List[Input]:
        out: List[Input] = []
        self._sweeps = {}
        for sd in enumerate_sweeps(self.hpo_roots):
            self._sweeps[sd.path] = sd
            for p, role in _sweep_files(sd):
                out.append(check_input(p, role=role, allow_symlink=False))
        return out

    def comparison_group(self, inp: Input) -> str:
        d = inp.uri
        while d and d != "/":
            if d in self._sweeps:
                return d
            d = os.path.dirname(d)
        # shared files (hpo_fold.json, trials/*.json, recon) belong to every sweep that names them; group by the file's own dir
        return os.path.dirname(inp.uri)

    def _seeds(self, ctx: AdapterContext) -> Seeds:
        return self.seeds if self.seeds is not None else load_seeds(self.seeds_dir, conn=ctx.conn)

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        sd = self._sweeps.get(group)
        if sd is None:
            return
        self._parked.pop(group, None)
        seeds = self._seeds(ctx)
        mt = min((i.mtime_utc for i in inputs if i.uri.endswith("summary.json")), default=None)
        try:
            search, trials = build_search(sd, seeds, ctx.now, mtime_utc=mt)
        except HpoParked as exc:
            # park-by-name: a kind declared out of scope is PARKED -- an accepted revision with no record and the ruling on the
            # status row (a non-empty `refuse` list is an explanation, so the empty read is never a silent one; `ok = 0`) -- never a
            # refusal of the group, which read the whole adapter REFUSED in every pass on a by-design condition
            self._parked[group] = parked_status_facts(exc)
            yield RecordBatch(f"{self.name}:{group}", group)
            return
        except HpoRefusal as exc:
            raise AdapterRefusal(exc.where, exc.why)
        batch = RecordBatch(f"{self.name}:{group}", group)
        batch.add("fit_search", search, ctx.producer, ctx.now)
        for t in trials:
            batch.add("fit_trial", t, ctx.producer, ctx.now)
        yield batch

    def status_extra(self, group: str) -> Dict[str, object]:
        """The facts of the group's last read: for a sweep parked by name, the `refuse` line and the `parked` entry citing the table
        (:func:`parked_status_facts`); otherwise none (the adapter's product is the `fit_search` / `fit_trial` records themselves)."""
        return dict(self._parked.get(group) or {})

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        return []


@dataclass
class HpoScan:
    searches: List[Dict[str, object]] = field(default_factory=list)
    trials: List[Dict[str, object]] = field(default_factory=list)
    refusals: List[Dict[str, str]] = field(default_factory=list)
    parked: List[Dict[str, object]] = field(default_factory=list)      # one entry per sweep parked by name: dir, kind_legacy, why, the status entry
    by_root: Dict[str, Dict[str, int]] = field(default_factory=dict)


def scan(roots: Sequence[Tuple[str, str, str]] = DEFAULT_HPO_ROOTS, seeds: Optional[Seeds] = None, seeds_dir: Optional[str] = None,
         now: Optional[str] = None, conn=None) -> HpoScan:
    require_store_access("hpo_logs.scan: enumerating the hyperparameter-search logs")      # the S2 dispatch-site guard (C6)
    from .. import records as R
    now = now or R.utc_now()
    rep = HpoScan()
    sd_seeds = seeds if seeds is not None else load_seeds(seeds_dir, conn=conn)
    for sd in enumerate_sweeps(roots):
        counts = rep.by_root.setdefault(sd.label, {"sweeps": 0, "searches": 0, "trials": 0, "refused": 0, "parked": 0})
        counts["sweeps"] += 1
        try:
            s, ts = build_search(sd, sd_seeds, now)
        except HpoParked as exc:
            counts["parked"] += 1
            rep.parked.append({"dir": sd.path, "kind_legacy": exc.kind_legacy, "why": exc.why, "parked": parked_status_facts(exc)["parked"][0]})
            continue
        except HpoRefusal as exc:
            counts["refused"] += 1
            rep.refusals.append({"dir": sd.path, "why": exc.why})
            continue
        rep.searches.append(s)
        rep.trials.extend(ts)
        counts["searches"] += 1
        counts["trials"] += len(ts)
    return rep


def register() -> None:
    from .base import register as _register
    _register(HpoLogsAdapter.name, lambda: HpoLogsAdapter())
