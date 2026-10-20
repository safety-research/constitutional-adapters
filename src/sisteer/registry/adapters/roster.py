"""The roster / rulings loader: ``configs/registry/roster_<round>_v<n>.yaml`` +
``rulings_<round>_v<n>.jsonl`` → ``paper_round`` / ``roster_entry`` / ``ruling`` / ``disposition`` records under
``<root>/rulings/<round>_v<n>/`` (part of the record log, mirrored, replayed by ``registry rebuild``).

The roster is DECLARATIVE (roster_transcription_notes.md §4): each ``comparisons[]`` block is one figure call of
the paper notebook at the transcribed sha, and its entries are the cross product ``models × objects × evals ×
strength_rules`` of the block (plus its ``sub_blocks``), annotated by its ``exceptions``. Every expanded entry gets
``roster_entry_id = uuid5(NAMESPACE_ROSTER, canonical JSON of the identity fields)`` so a re-load is idempotent.

NAMED INSTRUMENTS (roster v6): an ``evals[]`` item may carry ``instrument_ids: [<instrument_id>, ...]`` (or one
``instrument_id``); the item then expands to ONE entry per named instrument, ``instrument_id`` set and ``instrument_selector_json``
null (the DDL's ``roster_entry_instrument_xor_selector``), and ``v_roster_match`` admits a run on ``re.instrument_id =
r.instrument_id`` alone. This pins a comparison to instruments a selector cannot single out: the padded static instruments differ
from the unpadded static instrument of the same eval, and from each other, in ``padding_tokens`` (migration 0012), a field
:func:`_instrument_selector` does not carry, so a selector-shaped entry would also match the unpadded runs of the same model / eval and
could never tell the two pad levels apart. The converse held too until migration 0013: a padded instrument satisfies every selector
field of the eval's ordinary entries, so the 24 padded runs also matched the ordinary static comparisons by selector (live,
2026-09-23). Since 0013 ``v_roster_match``'s selector branch requires ``padding_tokens IS NULL`` -- a selector never matches a padded
instrument, which is reached only by name. An id must be a 64-hex sha256 (``identity.instrument_id``);
an item naming both ids and a selector override refuses by name. Against a database the loader reports every named instrument id the
``instrument`` table does not carry (``unknown_to_db.instrument_ids``): such an entry dependency-blocks at the fold on its
``instrument`` parent (``validate.parent_references`` over the DDL foreign key; the harvester retries the blocked queue every pass)
until the reader that mints the instrument lands, and is never refused here.

NAMED OBJECTS (roster v10, the appendix transfer figure): an ``objects[]`` item may carry ``object_id: <id>`` or
``object_ids: [<id>, ...]`` beside its ``object`` key; the item then expands to ONE entry per named object, ``object_id`` set and
``object_selector_json`` null (the DDL's object xor selector), and ``v_roster_match`` admits a run on ``re.object_id = r.object_id`` alone.
This pins a comparison to fits a selector cannot single out: two instruct-direct constitution_lora fits of one model differ only in their
training batch (itdoc batch 8 beside itdoc64 batch 64), a field no selector carries, and the three base-checkpoint seed fits of one recipe
share every selector field with the retired batch-8 original. An id must be a 64-hex sha256 (``identity.object_id``: the artifact's sha, so
the unsteered object and a system instruction cannot be named); an id listed twice, or a named object on a block that carries
``fitted_objects_use_transfer_fit`` (a named object is not re-shaped), refuses by name. The entry's ``extra`` carries the item's
``paper_label`` and ``object_kind`` so a consumer reading a named entry still knows what it names; against a database the loader reports a
named id the ``object`` table does not carry (``unknown_to_db.object_ids``; the entry dependency-blocks at the fold until the object lands)
and REFUSES a named id whose object row carries another paper_label or object_kind than the item (:func:`named_object_disagreements`).

THE TRANSFER FIT (roster v9): the base-model fit served on the instruct checkpoint is used wherever the option
exists -- a comparison block may carry ``fitted_objects_use_transfer_fit: true`` (or ``{except: [...]}``); every FITTED object it declares
(``lora`` / ``lora_subtraction`` / ``relu_steer`` / ``composed`` -- not the unsteered object, not a ``lane_free`` one) on a model row that
is served on another transfer kind than the model's ``transfer_lane`` then gets a selector carrying the transfer fit's ``transfer_kind``
and ``fit_model_id`` (``{paper_label, object_kind, transfer_kind: base_to_instruct, fit_model_id: <model>_base}``) while the comparison id
keeps the block's template with the MODEL ROW's transfer kind (``fig1_headline/instruct_direct`` still names the figure). A model with no
``transfer_lane`` (nemotron) and a model row already on the transfer kind (the ``/base_to_instruct`` companions) are untouched; an
``except`` item (matched like an ``annotate`` exception, on the model row's own kind) keeps the served checkpoint's fit for the cells
whose only fit is that one. The flag refuses by name on a block whose model rows carry several transfer kinds: a comparison that
DIFFERENCES the two fits (``fig4_transfer``, ``figQ_transfer``) keeps both. Since a selector is identity, a flipped entry is a NEW
``roster_entry_id`` and the earlier version's id stays in the table -- which is why every entry and the paper_round row
now carry ``roster_version`` (migration 0016): ``v_roster_match`` admits an entry only under the version the round's ``paper_round``
row carries, so a re-shaped entry's old id stops matching (and stops being a gap) the moment the new version loads.

STRICT: a block naming a model, object, eval or comparison the roster's own ``models:`` / ``objects:`` / ``evals:``
sections do not declare, a selector value outside the registry vocabulary, or a ruling with an unknown ``action`` /
``scope_kind`` / comparison glob, REFUSES by name (:class:`RosterRefusal`). Against a database the loader also
reports every model / eval / object / named instrument id the tables do not carry (those records dependency-block
until the dimension lands; they are never refused here).

Rulings carry a normalised ``scope_json.match`` (lists of ids / labels / kinds the flag view matches on; every
present list constrains, a ruling naming nothing covers nothing) beside the human-readable scope, a registry
``ruled_utc`` stamp + ``ruled_utc_precision``, and guards folded into ``expires_when_json``.

PROVENANCE: a ruling ruled at or after :data:`PROVENANCE_REQUIRED_FROM_UTC` must carry a
``provenance`` block — ``{kind: user_direct | relayed | peer_encoded, relayed_by, message_utc, encoded_by, commit,
record, confirmed_by_user}`` — so a criterion relayed second-hand (``relayed``), or declared in committed paper code
(``peer_encoded``), is told apart from one ruled directly (``user_direct``) and can be shown for confirmation.
The block is parked in
``extra_json.provenance`` (the DDL's ``evidence_state`` column is the EXPIRY-evidence state the harvester recomputes
from ``ruling_state`` rows — ``ok | missing | … | not_evaluated`` — and cannot carry provenance; a ``ruling.provenance``
column needs a schema change). ``relayed`` requires ``relayed_by`` (who relayed it) and ``message_utc``. A ruling
after the cutoff with no block, or a kind outside :data:`RULING_PROVENANCE_KINDS`, REFUSES by name. Rulings before
the cutoff (the transcribed seed) carry their provenance in ``ruled_by`` prose and are not re-stamped.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .. import identity
from .. import records as R
from .. import vocab as V

#: uuid5 namespace for roster entries (the draft's ``entry_id_rule``).
NAMESPACE_ROSTER = uuid.UUID("2b6b3a0e-8d9a-4d9f-9f4b-5f1c3e7a9d21")
ROSTER_SCHEMA = "sisteer.registry.roster/1"
ROSTER_DRAFT_SCHEMA = "sisteer.registry.roster_draft/1"
IDENTITY_FIELDS = ("round_id", "comparison_id", "model_id", "object_id", "object_selector_json", "eval_id", "instrument_id",
                   "instrument_selector_json", "judge_id", "judge_selector_json", "strength_rule")
#: The paper's component → the registry eval category.
COMPONENT_TO_CATEGORY = {"static": "static_attack", "bon": "best_of_n", "multi-turn": "multi_turn", "over-refusal": "over_refusal",
                         "petri": "petri", "misalignment": "agentic", "em_organism": "misalignment", "capability": "capability"}
#: The `match` list keys the flag view understands (schema_def: v_paper_current).
MATCH_KEYS = ("run_ids", "model_ids", "object_ids", "object_id_prefixes", "object_kinds", "paper_labels", "object_roles", "eval_ids",
              "eval_categories", "transfer_kinds", "strength_nominals", "operating_point_ids_legacy", "lane_sources_legacy", "views_legacy",
              "campaigns", "pass_labels", "instrument_ids", "token_caps", "decode_sources", "thinking_modes", "harnesses", "item_set_ids",
              "n_shots", "judge_ids", "comparison_ids", "instrument_alternatives")
ALT_KEYS = ("token_caps", "decode_sources", "thinking_modes", "harnesses", "item_set_ids", "n_shots", "scoring_rules", "serving_paths")
#: who a ruling came from — ruled directly (`user_direct`; a `record` may cite the producer's file that carries the
#: choice), relayed second-hand in a message (`relayed`: `relayed_by` + `message_utc`), or COMMITTED CODE declaring
#: the ruling, read off the paper build (`peer_encoded`: `encoded_by` + `commit`).
RULING_PROVENANCE_KINDS = ("user_direct", "relayed", "peer_encoded")
#: every ruling ruled from the cutoff on carries an explicit ``provenance`` block; the seed's earlier rulings
#: (transcribed 2026-09-16 from the paper build) carry it in ``ruled_by`` prose.
PROVENANCE_REQUIRED_FROM_UTC = "2026-09-17T18:00:00.000Z"
_UTC_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UTC_ANY = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?(?:\.(\d{1,6}))?Z$")


class RosterRefusal(ValueError):
    """The roster or a ruling names something unknown, or a value outside the vocabulary."""


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------


def registry_utc(value: str) -> Tuple[str, str]:
    """A recorded ruling time → (registry stamp ``YYYY-MM-DDTHH:MM:SS.mmmZ``, precision ``date`` | ``datetime``).
    A date-only source is midnight UTC of that day with precision ``date``."""
    v = str(value).strip()
    if _UTC_DATE.match(v):
        return v + "T00:00:00.000Z", "date"
    m = _UTC_ANY.match(v)
    if not m:
        raise RosterRefusal(f"ruled_utc {value!r} is neither a date nor an ISO-8601 UTC time")
    date, hh, mm, ss, frac = m.groups()
    ms = (frac or "0").ljust(3, "0")[:3]
    return f"{date}T{hh}:{mm}:{ss or '00'}.{ms}Z", "datetime"


def _as_list(v) -> List:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _canon_number(x):
    """Numbers in `match` lists compare against REAL columns: keep floats as floats, ints as ints."""
    if isinstance(x, bool):
        return int(x)
    return x


def load_yaml(path: str) -> Dict[str, object]:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise RosterRefusal(f"{path}: not a mapping at the top level")
    return data


def read_rulings_jsonl(path: str) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    """(header or None, rulings): the header is the ``_header: true`` line the file may start with."""
    header = None
    out: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RosterRefusal(f"{path}:{i}: not JSON ({exc.msg})") from exc
            if rec.get("_header"):
                header = rec
                continue
            out.append(rec)
    return header, out


# --------------------------------------------------------------------------------------------
# the roster: expansion
# --------------------------------------------------------------------------------------------


@dataclass
class RosterVocab:
    models: Dict[str, Dict[str, object]]
    objects: Dict[str, Dict[str, object]]
    evals: Dict[str, Dict[str, object]]
    judges: Dict[str, Optional[Dict[str, object]]]
    eval_id_map: Dict[str, str]                # paper eval key → registry eval_id
    pins: Dict[str, object]
    comparison_templates: List[str]


def roster_vocab(roster: Mapping[str, object]) -> RosterVocab:
    vocab = roster.get("vocabulary") or {}
    objects = {k: v for k, v in (roster.get("objects") or {}).items() if k != "pins" and isinstance(v, dict)}
    evals = {k: v for k, v in (roster.get("evals") or {}).items() if isinstance(v, dict)}
    return RosterVocab(models=dict(roster.get("models") or {}), objects=objects, evals=evals, judges=dict(roster.get("judges") or {}),
                       eval_id_map=dict((vocab.get("eval_id_from_eval_legacy") or {})), pins=dict((roster.get("objects") or {}).get("pins") or {}),
                       comparison_templates=[str(b.get("comparison_id_template")) for b in roster.get("comparisons") or []])


def _refuse_unknown(kind: str, name: str, known: Iterable[str], where: str) -> None:
    if name not in known:
        raise RosterRefusal(f"{where}: unknown {kind} {name!r}; the roster declares {sorted(known)}")


def _check_vocab(what: str, value, allowed: Sequence, where: str) -> None:
    if value is None:
        return
    if value not in allowed:
        raise RosterRefusal(f"{where}: {what} {value!r} is outside the registry vocabulary {list(allowed)}")


def _model_rows(block_models: Sequence, where: str, vocab: RosterVocab) -> List[Dict[str, object]]:
    """Normalise a block's models into rows of {model_id, transfer_kind, view_legacy, ...} (one per transfer kind)."""
    rows: List[Dict[str, object]] = []
    for m in block_models:
        if isinstance(m, str):
            m = {"model_id": m, "transfer_kind": None}
        if not isinstance(m, dict) or "model_id" not in m:
            raise RosterRefusal(f"{where}: a model item must be a mapping with model_id, got {m!r}")
        mid = str(m["model_id"])
        _refuse_unknown("model", mid, vocab.models, where)
        if vocab.models[mid].get("descoped"):
            raise RosterRefusal(f"{where}: model {mid!r} is descoped in the roster's models section and may not enter a comparison")
        kinds = m.get("transfer_kinds") or [m.get("transfer_kind")]
        views = m.get("views_legacy") or [m.get("view_legacy")] * len(kinds)
        for tk, view in zip(kinds, views):
            tk = "not_applicable" if tk is None else str(tk)
            _check_vocab("transfer_kind", tk, V.TRANSFER_KINDS, where)
            row = {k: v for k, v in m.items() if k not in ("transfer_kinds", "views_legacy", "transfer_kind", "view_legacy")}
            row.update(model_id=mid, transfer_kind=tk, view_legacy=view)
            rows.append(row)
    return rows


def _item(x, key: str) -> Dict[str, object]:
    return dict(x) if isinstance(x, dict) else {key: x}


def transfer_fit_for(model_row: Mapping[str, object], vocab: RosterVocab) -> Optional[Dict[str, object]]:
    """The model's TRANSFER fit (roster v9, ``models.<id>.transfer_lane`` -- the fit on the base checkpoint served on this checkpoint),
    when the block's model row is served on another transfer kind and the model has one; None for a model with no base checkpoint
    (``transfer_lane: null``, nemotron) and for a model row already on the transfer kind (the ``base_to_instruct`` companions)."""
    fit_row = (vocab.models.get(str(model_row["model_id"])) or {}).get("transfer_lane")
    if not isinstance(fit_row, dict) or not fit_row.get("transfer_kind"):
        return None
    if str(fit_row["transfer_kind"]) == str(model_row.get("transfer_kind")):
        return None
    return {"transfer_kind": str(fit_row["transfer_kind"]), "fit_model_id": fit_row.get("fit_model_id"), "view_legacy": fit_row.get("view_legacy")}


def _object_selector(obj_key: str, obj: Mapping[str, object], model_row: Mapping[str, object], carry_rules: Optional[Mapping[str, object]],
                     comparison_id: str, strength_rule: str, vocab: RosterVocab,
                     transfer_fit: Optional[Mapping[str, object]] = None) -> Tuple[Optional[str], Optional[Dict[str, object]], Dict[str, object]]:
    """(object_id, object_selector_json, notes) for one expanded entry. With ``transfer_fit`` (roster v9: the block reads its fitted objects
    on the model's transfer fit) a FITTED object's selector carries the transfer fit's ``transfer_kind`` and ``fit_model_id`` instead of
    the model row's transfer kind; the comparison id, a lane-free object and the unsteered object are untouched."""
    notes: Dict[str, object] = {}
    label = str(obj["paper_label"])
    kind = str(obj["object_kind"])
    if kind == "none":
        return "none", None, notes
    sel: Dict[str, object] = {"paper_label": label, "object_kind": kind}
    carried = bool(obj.get("lane_free"))          # a lane-free object (prompted, published baseline) is read on whichever lane recorded it
    tk = model_row.get("transfer_kind")
    if not carried and tk not in (None, "not_applicable"):
        # a fitted object is matched on the lane it was served on; a carried (lane-free) object on whichever lane recorded it
        sel["transfer_kind"] = tk
        if transfer_fit:
            # the comparison reads the base-checkpoint fit served on this checkpoint where the model has one;
            # the fit model pins the pick to the base-model fit (an IT-model fit served under a base-model spelling would otherwise match)
            tk = str(transfer_fit["transfer_kind"])
            sel["transfer_kind"] = tk
            if transfer_fit.get("fit_model_id"):
                sel["fit_model_id"] = str(transfer_fit["fit_model_id"])
            notes["transfer_fit"] = {"transfer_kind": tk, "fit_model_id": transfer_fit.get("fit_model_id"), "view_legacy": transfer_fit.get("view_legacy"),
                                     "served_transfer_kind": model_row.get("transfer_kind")}
    if carried:
        notes["lane_free"] = True
    notes["object_transfer_kind"] = sel.get("transfer_kind")
    pins = vocab.pins
    mid = str(model_row["model_id"])
    if label == "alphasteer_rebake" and strength_rule == "operating_point":
        pin = (pins.get("alphasteer_rebake_operating") or {}).get(mid)
        if pin and pin.get("object_id"):
            notes["operating_point_id_legacy"] = pin.get("operating_point_id_legacy")
            notes["strength_coefficient"] = pin.get("strength_coefficient")
            if pin.get("inferred"):
                notes["inferred"] = True
                notes["disagreement"] = pin.get("disagreement")
            return str(pin["object_id"]), None, notes
    if label == "arditi" and strength_rule == "operating_point":
        pin = (pins.get("arditi_operating") or {}).get(mid)
        if pin:
            notes["operating_point_id_legacy"] = pin.get("operating_point_id_legacy")
            notes["cell_id_legacy"] = pin.get("cell_id_legacy")
            if pin.get("object_id_prefix"):
                sel["object_id_prefix"] = pin["object_id_prefix"]
    if label == "constitution_lora" and tk == "instruct_direct" and comparison_id.split("/")[0] in ("fig5_frontier", "figQ_frontier"):
        pin = (pins.get("constitution_lora_instruct_direct_fig5") or {}).get(mid)
        if pin:
            notes["fit_lineage_legacy"] = pin.get("fit_lineage_legacy")
            if pin.get("object_id"):
                return str(pin["object_id"]), None, notes
            if pin.get("object_id_prefix"):
                sel["object_id_prefix"] = pin["object_id_prefix"]
    return None, sel, notes


def _judge_selector(eval_row: Mapping[str, object], vocab: RosterVocab) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    jkey = eval_row.get("judge")
    if not jkey:
        return None, None
    _refuse_unknown("judge", str(jkey), vocab.judges, f"evals.{eval_row.get('_key')}")
    j = vocab.judges.get(str(jkey))
    if not j:
        return None, str(jkey)
    model = j.get("judge_model")
    if not model or model == "unverified":
        return None, str(jkey)
    return {"judge_model": str(model)}, str(jkey)


def _instrument_selector(paper_sel: Optional[Mapping[str, object]], where: str) -> Dict[str, object]:
    """The registry selector the flag view matches on, from the paper's CANONICAL-shaped selector: only vocabulary-valued
    fields enter; the paper's own fields (decode_contains, rung, metric, threshold, bon_n, notes) ride the entry's extra."""
    sel: Dict[str, object] = {}
    if not paper_sel:
        return sel
    if paper_sel.get("caps"):
        sel["token_cap_tokens"] = [int(c) for c in paper_sel["caps"]]
    for key, allowed in (("harness", V.HARNESSES), ("thinking_mode", V.THINKING_MODES), ("decode_source", V.DECODE_SOURCES),
                         ("scoring_rule", V.SCORING_RULES), ("serving_path", V.SERVING_PATHS)):
        v = paper_sel.get(key)
        if v is None:
            continue
        if v in allowed:
            sel[key] = v
        elif isinstance(v, str) and " " in v:
            continue    # prose like "on (llama: not_available)": a note, not a selector
        else:
            raise RosterRefusal(f"{where}: instrument selector {key}={v!r} is outside the vocabulary {list(allowed)}")
    if paper_sel.get("item_set"):
        sel["item_set_id"] = str(paper_sel["item_set"])
    if paper_sel.get("n_shot") is not None:
        sel["n_shot"] = int(paper_sel["n_shot"])
    return sel


_INSTRUMENT_ID = re.compile(r"^[0-9a-f]{64}$")


def _named_instruments(eval_item: Mapping[str, object], where: str) -> List[Optional[str]]:
    """The instrument ids an ``evals[]`` item names (``instrument_ids`` list or one ``instrument_id``), each a 64-hex sha256 (the
    registry's ``identity.instrument_id``), no duplicates, exclusive with ``instrument_selector_json_override``; ``[None]`` when the item
    names none (the entry then carries the eval's selector, as before v6)."""
    raw = eval_item.get("instrument_ids") if eval_item.get("instrument_ids") is not None else eval_item.get("instrument_id")
    ids = _as_list(raw)
    if not ids:
        return [None]
    if eval_item.get("instrument_selector_json_override") is not None:
        raise RosterRefusal(f"{where}: instrument_ids and instrument_selector_json_override are exclusive (roster_entry carries an "
                            f"instrument_id XOR an instrument_selector_json)")
    out: List[Optional[str]] = []
    for i in ids:
        sid = str(i)
        if not _INSTRUMENT_ID.match(sid):
            raise RosterRefusal(f"{where}: instrument id {i!r} is not a 64-hex sha256 (identity.instrument_id)")
        if sid in out:
            raise RosterRefusal(f"{where}: instrument id {sid[:12]}… is listed twice")
        out.append(sid)
    return out


def _named_objects(obj_item: Mapping[str, object], obj: Mapping[str, object], where: str) -> List[Optional[str]]:
    """The object ids an ``objects[]`` item names (``object_ids`` list or one ``object_id``; roster v10), each a 64-hex sha256 (the
    registry's ``identity.object_id`` for an artifact), no duplicates, never for the unsteered object or a system instruction (their ids
    are not artifact shas); ``[None]`` when the item names none (the entry then carries the object's selector, as before v10)."""
    raw = obj_item.get("object_ids") if obj_item.get("object_ids") is not None else obj_item.get("object_id")
    ids = _as_list(raw)
    if not ids:
        return [None]
    if str(obj.get("object_kind")) in ("none", "system_instruction"):
        raise RosterRefusal(f"{where}: object {obj_item.get('object')!r} of kind {obj.get('object_kind')!r} cannot be named by id "
                            f"(its identity is not an artifact sha256)")
    out: List[Optional[str]] = []
    for i in ids:
        sid = str(i)
        if not _INSTRUMENT_ID.match(sid):
            raise RosterRefusal(f"{where}: object id {i!r} is not a 64-hex sha256 (identity.object_id)")
        if sid in out:
            raise RosterRefusal(f"{where}: object id {sid[:12]}… is listed twice")
        out.append(sid)
    return out


def _matches(exc: Mapping[str, object], entry: Mapping[str, object], obj_key: str, eval_key: str, model_row: Mapping[str, object]) -> bool:
    def hit(field, value) -> bool:
        want = exc.get(field)
        if want is None:
            return True
        return value in _as_list(want)
    fig = str(entry["comparison_id"]).split("/")[0]
    if exc.get("comparisons") is not None and fig not in _as_list(exc["comparisons"]):
        return False
    if exc.get("comparisons_not") is not None and fig in _as_list(exc["comparisons_not"]):
        return False
    return (hit("model_id", entry["model_id"]) and hit("object", obj_key) and hit("eval", eval_key)
            and hit("transfer_kind", model_row.get("transfer_kind")))


def _transfer_fit_flag(block: Mapping[str, object], where: str) -> Tuple[bool, List[Dict[str, object]]]:
    """The block's ``fitted_objects_use_transfer_fit`` (roster v9): ``true``, or a mapping ``{except: [<exception items>]}`` whose
    items name the (model_id / object / eval / comparisons) cells that keep the served checkpoint's fit -- the cells whose ONLY fit is the
    IT-model one (an `except` item is matched like an `annotate` exception, on the model row's own transfer kind). Returns
    ``(flag, except_items)``. Refuses by name on a block whose model rows carry several transfer kinds (``transfer_kinds``): such a
    comparison DIFFERENCES the two fits and never prefers one (fig4_transfer, figQ_transfer)."""
    raw = block.get("fitted_objects_use_transfer_fit")
    if raw is None or raw is False:
        return False, []
    if raw is not True and not isinstance(raw, dict):
        raise RosterRefusal(f"{where}: fitted_objects_use_transfer_fit must be true or a mapping with `except`, got {raw!r}")
    for m in block.get("models") or []:
        if isinstance(m, dict) and len(m.get("transfer_kinds") or []) > 1:
            raise RosterRefusal(f"{where}: fitted_objects_use_transfer_fit on a comparison whose model {m.get('model_id')!r} carries several "
                                f"transfer kinds {m.get('transfer_kinds')!r}: a comparison of the two fits keeps both and prefers neither")
    items: List[Dict[str, object]] = []
    if isinstance(raw, dict):
        for k in raw:
            if k not in ("except", "reason"):
                raise RosterRefusal(f"{where}: fitted_objects_use_transfer_fit key {k!r} is not `except` or `reason`")
        for x in _as_list(raw.get("except")):
            if not isinstance(x, dict) or not (x.get("model_id") or x.get("object") or x.get("eval") or x.get("comparisons")):
                raise RosterRefusal(f"{where}: an `except` item must name a model_id, object, eval or comparisons, got {x!r}")
            items.append(dict(x))
    return True, items


def _entry_id(payload: Mapping[str, object]) -> str:
    ident = {k: payload.get(k) for k in IDENTITY_FIELDS}
    return str(uuid.uuid5(NAMESPACE_ROSTER, identity.canonical_json(ident)))


@dataclass
class Expansion:
    round_id: str
    paper_round: Dict[str, object]
    entries: List[Dict[str, object]] = field(default_factory=list)
    dispositions: List[Dict[str, object]] = field(default_factory=list)
    counts_by_comparison: Dict[str, int] = field(default_factory=dict)
    declared_counts: Dict[str, int] = field(default_factory=dict)
    disagreements: List[Dict[str, object]] = field(default_factory=list)

    @property
    def inferred(self) -> List[Dict[str, object]]:
        return [e for e in self.entries if e.get("inferred")]


def expand(roster: Mapping[str, object], strict_counts: bool = True) -> Expansion:
    """Expand every comparison block into roster_entry payloads (strict; see the module docstring)."""
    meta = roster.get("meta") or {}
    if meta.get("schema") not in (ROSTER_SCHEMA, ROSTER_DRAFT_SCHEMA):
        raise RosterRefusal(f"roster meta.schema {meta.get('schema')!r} is not {ROSTER_SCHEMA}")
    round_id = str(meta.get("round_id") or (meta.get("paper_round") or {}).get("round_id") or "")
    if not round_id:
        raise RosterRefusal("roster meta.round_id is missing")
    pr = dict(meta.get("paper_round") or {})
    pr.setdefault("round_id", round_id)
    opened, prec = registry_utc(str(pr.get("opened_utc")))
    version = str(meta.get("roster_version") or "v1")
    # the roster version rides the paper_round row and every entry (migration 0016): v_roster_match admits an entry of the open round only
    # under the version its paper_round row carries, so an entry a later version no longer expands (its selector re-shaped) stops matching
    # and stops being a gap once that version loads, instead of staying behind as a second selector for the same key
    paper_round = {"round_id": round_id, "opened_utc": opened, "description": str(pr.get("description") or f"paper round {round_id}"),
                   "paper_build_commit": pr.get("paper_build_commit"), "roster_version": version,
                   "extra": {"opened_utc_precision": pr.get("opened_utc_precision") or prec, "paper_build_branch": meta.get("paper_build_branch"),
                             "paper_build_commit_utc": meta.get("paper_build_commit_utc"), "transcribed_utc": meta.get("transcribed_utc"),
                             "read_method": meta.get("read_method"), "roster_schema": meta.get("schema")}}
    if pr.get("closed_utc"):
        paper_round["closed_utc"] = registry_utc(str(pr["closed_utc"]))[0]
    vocab = roster_vocab(roster)
    for key, obj in vocab.objects.items():
        _check_vocab("paper_label", obj.get("paper_label"), V.PAPER_LABELS, f"objects.{key}")
        _check_vocab("object_kind", obj.get("object_kind"), V.OBJECT_KINDS, f"objects.{key}")
        _check_vocab("object_role", obj.get("object_role"), V.OBJECT_ROLES, f"objects.{key}")
    for key, ev in vocab.evals.items():
        comp = ev.get("component_legacy") or ev.get("eval_category")
        if comp not in COMPONENT_TO_CATEGORY and comp not in V.EVAL_CATEGORIES:
            raise RosterRefusal(f"evals.{key}: component {comp!r} maps to no registry eval category")
    out = Expansion(round_id=round_id, paper_round=paper_round)
    declared = dict((meta.get("expanded_counts_by_comparison") or {}))
    out.declared_counts = {k: int(v) for k, v in declared.items() if k != "total"}
    for bi, block in enumerate(roster.get("comparisons") or []):
        tmpl = str(block.get("comparison_id_template") or "")
        where = f"comparisons[{bi}] {tmpl}"
        if not tmpl:
            raise RosterRefusal(f"comparisons[{bi}]: no comparison_id_template")
        sub_specs: List[Dict[str, object]] = []
        if block.get("objects") is not None and block.get("evals") is not None:
            sub_specs.append({"name": None, "objects": block["objects"], "evals": block["evals"],
                              "strength_rules": block.get("strength_rules") or ["operating_point"], "models": block.get("models")})
        for sb in block.get("sub_blocks") or []:
            sub_specs.append(dict(sb))
        if not sub_specs:
            out.counts_by_comparison.setdefault(tmpl.replace("/{transfer_kind}", ""), 0)
            continue
        exceptions = block.get("exceptions") or []
        block_models = _model_rows(block.get("models") or [], where, vocab)
        transfer_fit_flag, transfer_fit_except = _transfer_fit_flag(block, where)
        for spec in sub_specs:
            sname = spec.get("name")
            swhere = where + (f" sub_block {sname}" if sname else "")
            models = _model_rows(spec["models"], swhere, vocab) if spec.get("models") else block_models
            if not models:
                raise RosterRefusal(f"{swhere}: no models")
            rules = [str(r) for r in (spec.get("strength_rules") or ["operating_point"])]
            for rule in rules:
                if not (rule == "operating_point" or rule.startswith("fixed:") or rule.startswith("ladder:")):
                    raise RosterRefusal(f"{swhere}: strength_rule {rule!r} is not operating_point | fixed:<x> | ladder:<record_id>")
                if rule.startswith("fixed:"):
                    float(rule[6:])
            obj_ann = spec.get("object_annotations") or {}
            eval_ann = spec.get("eval_annotations") or {}
            for mrow in models:
                tk = mrow["transfer_kind"]
                comparison_id = tmpl.replace("{transfer_kind}", tk)
                for oitem in spec.get("objects") or []:
                    o = _item(oitem, "object")
                    okey = str(o.get("object") or o.get("paper_label") or "")
                    _refuse_unknown("object", okey, vocab.objects, swhere)
                    obj = vocab.objects[okey]
                    named_objects = _named_objects(o, obj, f"{swhere} object {okey}")
                    if named_objects != [None] and transfer_fit_flag:
                        raise RosterRefusal(f"{swhere} object {okey}: a named object is not re-shaped by fitted_objects_use_transfer_fit; "
                                            f"name the fit you mean or drop the flag")
                    for eitem in spec.get("evals") or []:
                        e = _item(eitem, "eval_id")
                        ekey = str(e.get("eval_id") or "")
                        _refuse_unknown("eval", ekey, vocab.evals, swhere)
                        ev = dict(vocab.evals[ekey], _key=ekey)
                        named = _named_instruments(e, f"{swhere} eval {ekey}")
                        for rule, iid, named_oid in itertools.product(rules, named, named_objects):
                            # roster v9: the block reads its fitted objects on the model's transfer fit, unless an `except` item names
                            # this (model, object, eval) as one whose only fit is the served checkpoint's (matched on the model row's own kind)
                            fit = transfer_fit_for(mrow, vocab) if transfer_fit_flag else None
                            if fit and any(_matches(x, {"comparison_id": comparison_id, "model_id": mrow["model_id"]}, okey, ekey, mrow)
                                           for x in transfer_fit_except):
                                fit = None
                            if named_oid is not None:
                                # roster v10: the item names the object by id; the entry carries the id and no selector, and is filed under
                                # the model row's transfer kind (the served kind the figure reads it on; the id fixes the fit)
                                oid, osel, onotes = named_oid, None, {"object_named": True, "object_transfer_kind": mrow.get("transfer_kind"),
                                                                       "paper_label": str(obj["paper_label"]), "object_kind": str(obj["object_kind"])}
                                if obj.get("lane_free"):
                                    onotes["lane_free"] = True
                            else:
                                oid, osel, onotes = _object_selector(okey, obj, mrow, spec.get("carry_rules") or block.get("carry_rules"),
                                                                    comparison_id, rule, vocab, transfer_fit=fit)
                            # exceptions (annotations, pins) match the transfer kind the entry's selector carries, not the model row's
                            erow = dict(mrow, transfer_kind=onotes.get("object_transfer_kind") or mrow.get("transfer_kind"))
                            paper_isel = e.get("instrument_selector_json_override") or ev.get("instrument_selector_json") or {}
                            isel = _instrument_selector(paper_isel, f"{swhere} eval {ekey}")
                            jsel, jkey = _judge_selector(ev, vocab)
                            payload: Dict[str, object] = {
                                "round_id": round_id, "comparison_id": comparison_id, "model_id": mrow["model_id"],
                                "object_id": oid, "object_selector_json": osel, "eval_id": vocab.eval_id_map.get(ekey, ekey),
                                "instrument_id": iid, "instrument_selector_json": None if iid is not None else isel, "judge_id": None, "judge_selector_json": jsel,
                                "strength_rule": rule,
                            }
                            payload["roster_entry_id"] = _entry_id(payload)
                            payload["roster_version"] = version       # not an identity field: the version that last expanded this entry (0016)
                            inferred = bool(block.get("inferred")) or bool(spec.get("inferred")) or bool(o.get("inferred")) or bool(e.get("inferred")) \
                                or bool(onotes.get("inferred")) or bool((obj_ann.get(okey) or {}).get("inferred"))
                            drawn = True
                            reasons: List[str] = []
                            disagreement: List[str] = []
                            for src in (block, spec, o, e, obj_ann.get(okey) or {}, eval_ann.get(ekey) or {}, (obj_ann.get(okey) or {}).get(rule) if isinstance((obj_ann.get(okey) or {}).get(rule), dict) else {}):
                                if not isinstance(src, dict):
                                    continue
                                if src.get("reason"):
                                    reasons.append(str(src["reason"]))
                                if src.get("disagreement"):
                                    disagreement.append(str(src["disagreement"]))
                                if src.get("drawn") is False or (isinstance(src.get("extra_json"), dict) and src["extra_json"].get("drawn") is False):
                                    drawn = False
                                if src.get("inferred"):
                                    inferred = True
                            if onotes.get("disagreement"):
                                disagreement.append(str(onotes["disagreement"]))
                            extra: Dict[str, object] = {
                                "transfer_kind": tk, "view_legacy": mrow.get("view_legacy"), "arm_legacy": obj.get("arm_legacy"),
                                "object_role": obj.get("object_role"), "cell_id_legacy": obj.get("cell_id_legacy"), "eval_legacy": ev.get("eval_legacy", ekey),
                                "component_legacy": ev.get("component_legacy"), "eval_category": COMPONENT_TO_CATEGORY.get(str(ev.get("component_legacy")), ev.get("component_legacy")),
                                "instrument_selector_legacy": None if iid is not None else paper_isel, "instrument_named": iid is not None,
                                "judge_legacy": jkey, "figure_legacy": block.get("figure_legacy"),
                                "notebook_call": block.get("notebook_call"), "sub_block": sname, "source_legacy": ev.get("source_legacy"),
                            }
                            extra.update({k: v for k, v in onotes.items() if k not in ("inferred", "disagreement")})
                            if mrow.get("fit_model_id"):
                                extra["fit_model_id"] = mrow["fit_model_id"]
                            if onotes.get("transfer_fit"):
                                extra["fit_model_id"] = onotes["transfer_fit"].get("fit_model_id")     # the base-model fit the selector names
                            extra["roster_version"] = version
                            if mrow.get("carried_from_paper_lane") or spec.get("carried_from_paper_lane"):
                                extra["carried_from_paper_lane"] = True
                            if isinstance(spec.get("extra_json"), dict):
                                extra["sub_block_extra"] = spec["extra_json"]
                            if isinstance(e.get("extra_json"), dict):
                                extra["eval_extra"] = e["extra_json"]
                            # block-level exceptions: annotations, pins and expected dispositions
                            for exc in exceptions:
                                if not isinstance(exc, dict) or exc.get("kind") != "annotate":
                                    continue
                                if not _matches(exc, payload, okey, ekey, erow):
                                    continue
                                if exc.get("inferred"):
                                    inferred = True
                                if exc.get("disagreement"):
                                    disagreement.append(str(exc["disagreement"]))
                                if exc.get("reason"):
                                    reasons.append(str(exc["reason"]))
                                if exc.get("object_pin"):
                                    extra["object_pin"] = exc["object_pin"]
                                    pin = (vocab.pins.get(str(exc["object_pin"])) or {}).get(str(mrow["model_id"]))
                                    if isinstance(pin, dict):
                                        extra["object_pin_detail"] = {k: v for k, v in pin.items() if k in ("fit_lineage_legacy", "object_id_prefix", "object_id", "reason")}
                                if exc.get("disposition_expected"):
                                    extra["disposition_expected"] = exc["disposition_expected"]
                                    extra["disposition_source_ref"] = exc.get("source_ref")
                                    state = str(exc["disposition_expected"])
                                    if state in V.DISPOSITIONS and state in ("declined", "not_applicable"):
                                        out.dispositions.append({
                                            "disposition_id": str(uuid.uuid5(NAMESPACE_ROSTER, "disposition|" + payload["roster_entry_id"])),
                                            "round_id": round_id, "roster_entry_id": payload["roster_entry_id"], "comparison_id": comparison_id,
                                            "model_id": mrow["model_id"], "object_id": oid, "eval_id": payload["eval_id"], "state": state,
                                            "reason": str(exc.get("source_ref") or state), "source_ref": str(exc.get("source_ref") or "roster exception"),
                                            "ruled_utc": opened, "extra": {"paper_label": obj.get("paper_label")}})
                            if reasons:
                                extra["inferred_reason"] = "; ".join(dict.fromkeys(reasons))
                            if disagreement:
                                extra["disagreement"] = "; ".join(dict.fromkeys(disagreement))
                                inferred = True
                            payload.update(source_citation=str(block.get("source_citation") or ""), inferred=int(inferred), drawn=int(drawn), extra=extra)
                            if not payload["source_citation"]:
                                raise RosterRefusal(f"{swhere}: block has no source_citation")
                            out.entries.append(payload)
                            fig_key = tmpl.replace("/{transfer_kind}", "") if "{transfer_kind}" in tmpl else tmpl
                            if "/base_to_instruct" in tmpl:
                                fig_key = tmpl
                            out.counts_by_comparison[fig_key] = out.counts_by_comparison.get(fig_key, 0) + 1
    # dedupe on the identity (the same run may be listed by two sub-blocks with identical fields): keep the first
    seen: Dict[str, Dict[str, object]] = {}
    for e in out.entries:
        if e["roster_entry_id"] in seen:
            prev = seen[e["roster_entry_id"]]
            prev["inferred"] = int(prev["inferred"] or e["inferred"])
            prev["drawn"] = int(prev["drawn"] and e["drawn"])
            continue
        seen[e["roster_entry_id"]] = e
    dropped = len(out.entries) - len(seen)
    out.entries = list(seen.values())
    if strict_counts and out.declared_counts:
        bad = {k: (out.declared_counts[k], out.counts_by_comparison.get(k, 0)) for k in out.declared_counts
               if out.counts_by_comparison.get(k, 0) != out.declared_counts[k]}
        if bad:
            raise RosterRefusal("expanded counts differ from meta.expanded_counts_by_comparison (declared, expanded): "
                                + json.dumps(bad, sort_keys=True) + (f"; {dropped} duplicate identities folded" if dropped else ""))
    out.disagreements = sorted((e for e in out.entries if (e.get("extra") or {}).get("disagreement")),
                               key=lambda e: (str((e["extra"] or {}).get("disagreement")), e["comparison_id"], e["model_id"], e["eval_id"]))
    return out


# --------------------------------------------------------------------------------------------
# rulings: normalisation and strict validation
# --------------------------------------------------------------------------------------------


def normalise_provenance(block, ruled_utc_stamp: str, where: str) -> Optional[Dict[str, object]]:
    """The ``provenance`` block of a ruling, validated. Required (refuses when absent) for a ruling ruled at or
    after :data:`PROVENANCE_REQUIRED_FROM_UTC`; optional but still validated before it. ``kind`` must be one of
    :data:`RULING_PROVENANCE_KINDS`; ``relayed`` must name ``relayed_by`` (the session) and ``message_utc``."""
    required = ruled_utc_stamp >= PROVENANCE_REQUIRED_FROM_UTC
    if block is None:
        if required:
            raise RosterRefusal(f"{where}: provenance missing — a ruling ruled at or after {PROVENANCE_REQUIRED_FROM_UTC} must say "
                                f"whether the user ruled it here (kind user_direct) or a peer session relayed it (kind relayed, with "
                                f"relayed_by + message_utc) so it can be shown to the user for confirmation (D83)")
        return None
    if not isinstance(block, dict):
        raise RosterRefusal(f"{where}: provenance must be an object with a `kind`, got {block!r}")
    kind = block.get("kind")
    if kind not in RULING_PROVENANCE_KINDS:
        raise RosterRefusal(f"{where}: provenance kind {kind!r} is not one of {RULING_PROVENANCE_KINDS}")
    out = dict(block)
    if kind == "relayed":
        for k in ("relayed_by", "message_utc"):
            if not out.get(k):
                raise RosterRefusal(f"{where}: provenance kind 'relayed' needs {k} (the relaying session and the message time)")
        registry_utc(str(out["message_utc"]))            # refuses a malformed time by name
        out.setdefault("confirmed_by_user", False)
    if kind == "peer_encoded":
        for k in ("encoded_by", "commit"):
            if not out.get(k):
                raise RosterRefusal(f"{where}: provenance kind 'peer_encoded' needs {k} (the session whose committed code declares the ruling, and the commit)")
        if not re.fullmatch(r"[0-9a-f]{7,40}", str(out["commit"])):
            raise RosterRefusal(f"{where}: provenance commit {out['commit']!r} is not a git sha")
        out.setdefault("confirmed_by_user", False)
    return out


def normalise_ruling(raw: Mapping[str, object], vocab: Optional[RosterVocab] = None, where: str = "ruling") -> Dict[str, object]:
    """A ruling as the file carries it → the record payload: registry ``ruled_utc`` + precision, guards folded into
    ``expires_when_json``, ``scope_json.match`` built when absent, vocabulary checked, comparison globs checked
    against the roster's comparison templates."""
    r = dict(raw)
    rid = str(r.get("ruling_id") or "")
    where = f"{where} {rid}" if rid else where
    if not rid:
        raise RosterRefusal(f"{where}: ruling_id missing")
    for k in ("round_id", "action", "scope_kind", "scope_json", "reason", "ruled_utc", "ruled_by", "source_ref"):
        if r.get(k) in (None, ""):
            raise RosterRefusal(f"{where}: {k} missing")
    _check_vocab("action", r["action"], V.RULING_ACTIONS, where)
    _check_vocab("scope_kind", r["scope_kind"], V.SCOPE_KINDS, where)
    extra = dict(r.get("extra_json") or r.get("extra") or {})
    stamp, prec = registry_utc(str(r["ruled_utc"]))
    note = extra.pop("ruled_utc_precision", None)
    declared_prec = r.get("ruled_utc_precision")
    if declared_prec is not None and declared_prec not in V.RULED_UTC_PRECISIONS:
        raise RosterRefusal(f"{where}: ruled_utc_precision {declared_prec!r} is not one of {V.RULED_UTC_PRECISIONS}")
    if note is not None and note not in V.RULED_UTC_PRECISIONS:
        extra["ruled_utc_precision_note"] = str(note)
        if str(note).startswith("date"):
            prec = "date"
    if declared_prec == "date" and not stamp.endswith("T00:00:00.000Z"):
        raise RosterRefusal(f"{where}: ruled_utc_precision 'date' declared but ruled_utc {r['ruled_utc']!r} carries a time of day")
    prec = declared_prec or prec
    prov = normalise_provenance(r.get("provenance", extra.pop("provenance", None)), stamp, where)
    if prov is not None:
        extra["provenance"] = prov
    ew = r.get("expires_when_json")
    guards = extra.pop("guards_json", None)
    if guards:
        if ew is None:
            ew = {"kind": "any_of", "conditions": [], "guards": list(guards)}
        elif isinstance(ew, dict) and ew.get("kind") in V.EXPIRY_COMBINATORS:
            ew = dict(ew, guards=list(ew.get("guards") or []) + list(guards))
        elif isinstance(ew, dict):
            ew = {"kind": "any_of", "conditions": [ew], "guards": list(guards)}
    if isinstance(ew, dict):
        kind = ew.get("kind")
        if kind in V.EXPIRY_COMBINATORS:
            for c in list(ew.get("conditions") or []) + list(ew.get("guards") or []):
                _check_vocab("expiry kind", c.get("kind"), V.EXPIRY_KINDS, where)
        else:
            _check_vocab("expiry kind", kind, V.EXPIRY_KINDS, where)
    scope = dict(r["scope_json"]) if isinstance(r["scope_json"], dict) else json.loads(str(r["scope_json"]))
    if "match" not in scope:
        scope["match"] = build_match(scope, str(r["scope_kind"]), vocab)
    match = scope.get("match")
    if match is not None:
        if not isinstance(match, dict):
            raise RosterRefusal(f"{where}: scope_json.match must be an object")
        for k, v in match.items():
            if k not in MATCH_KEYS:
                raise RosterRefusal(f"{where}: scope_json.match key {k!r} is not one the flag matches on {MATCH_KEYS}")
            if not isinstance(v, list) or not v:
                raise RosterRefusal(f"{where}: scope_json.match.{k} must be a non-empty list")
        for k, allowed in (("object_kinds", V.OBJECT_KINDS), ("paper_labels", V.PAPER_LABELS), ("object_roles", V.OBJECT_ROLES),
                           ("transfer_kinds", V.TRANSFER_KINDS), ("eval_categories", V.EVAL_CATEGORIES), ("decode_sources", V.DECODE_SOURCES),
                           ("thinking_modes", V.THINKING_MODES), ("harnesses", V.HARNESSES)):
            for v in match.get(k) or []:
                _check_vocab(k[:-1], v, allowed, where)
        if vocab is not None:
            for v in match.get("model_ids") or []:
                _refuse_unknown("model", str(v), vocab.models, where)
            for v in match.get("eval_ids") or []:
                known = set(vocab.eval_id_map.values()) | set(vocab.evals) | {"tiny_ll", "humaneval_plus", "alpaca_eval", "hellaswag"}
                _refuse_unknown("eval", str(v), known, where)
            for g in match.get("comparison_ids") or []:
                if not comparison_glob_known(str(g), vocab.comparison_templates):
                    raise RosterRefusal(f"{where}: comparison glob {g!r} matches none of the roster's comparisons {vocab.comparison_templates}")
        for alt in match.get("instrument_alternatives") or []:
            if not isinstance(alt, dict) or not alt:
                raise RosterRefusal(f"{where}: an instrument alternative must be a non-empty object")
            for k in alt:
                if k not in ALT_KEYS:
                    raise RosterRefusal(f"{where}: instrument alternative key {k!r} is not one of {ALT_KEYS}")
    payload: Dict[str, object] = {
        "ruling_id": rid, "round_id": str(r["round_id"]), "action": str(r["action"]), "scope_kind": str(r["scope_kind"]), "scope_json": scope,
        "target_json": r.get("target_json") if r.get("target_json") is not None else {}, "expires_when_json": ew,
        "reason": str(r["reason"]), "ruled_utc": stamp, "ruled_utc_precision": prec, "ruled_by": str(r["ruled_by"]),
        "source_ref": str(r["source_ref"]), "extra": extra,
    }
    return payload


def comparison_glob_known(glob: str, templates: Sequence[str]) -> bool:
    import fnmatch
    if not templates:
        return True
    cands = []
    for t in templates:
        if "{transfer_kind}" in t:
            cands += [t.replace("{transfer_kind}", tk) for tk in V.TRANSFER_KINDS]
        else:
            cands.append(t)
    return any(fnmatch.fnmatchcase(c, glob) for c in cands)


_PETRI_BEHAVIOURS = ("petri_cyber_bio_chem_capabilities", "petri_defer_to_users", "petri_delusion_sycophancy", "petri_instruction_evasion",
                     "petri_reasoning_unfaithfulness", "petri_self_preservation")


def _eval_ids(values: Iterable, vocab: Optional[RosterVocab]) -> List[str]:
    out: List[str] = []
    emap = vocab.eval_id_map if vocab else {}
    for v in values:
        v = str(v)
        if v.startswith("petri_<behaviour>"):
            out += list(_PETRI_BEHAVIOURS)
            continue
        out.append(emap.get(v, v))
    return list(dict.fromkeys(out))


def build_match(scope: Mapping[str, object], scope_kind: str, vocab: Optional[RosterVocab]) -> Optional[Dict[str, object]]:
    """Derive the machine-matchable ``match`` object from a human-shaped scope. Returns None when the scope names
    nothing the flag can match on (the ruling then covers no run: documentation, or a reader-side rule)."""
    m: Dict[str, List] = {}

    def put(key, values):
        vals = [_canon_number(v) for v in _as_list(values) if v is not None and not (isinstance(v, str) and " " in v and key != "lane_sources_legacy")]
        if vals:
            m.setdefault(key, [])
            m[key] += [v for v in vals if v not in m[key]]

    put("run_ids", scope.get("run_id") or scope.get("run_ids"))
    put("model_ids", scope.get("model_id") or scope.get("model_ids"))
    put("object_ids", scope.get("object_id") or scope.get("object_ids"))
    put("object_id_prefixes", scope.get("object_id_prefix") or scope.get("object_id_prefixes"))
    put("object_kinds", scope.get("object_kind") or scope.get("object_kinds"))
    labels = scope.get("paper_label") or scope.get("paper_labels")
    if labels and not (isinstance(labels, str) and " " in labels):
        put("paper_labels", labels)
    if scope.get("eval_id") or scope.get("eval_ids"):
        put("eval_ids", _eval_ids(_as_list(scope.get("eval_id") or scope.get("eval_ids")), vocab))
    # the evals a comparison-scoped exclude names as EXCLUDED are the evals it covers (r2v1-exclude-fig5-defence-axis-bon-multiturn's
    # `eval_ids_excluded` / `eval_categories_excluded`); a match built without them stripped the comparison off every matched run (2026-09-20)
    if scope.get("eval_ids_excluded"):
        put("eval_ids", _eval_ids(_as_list(scope.get("eval_ids_excluded")), vocab))
    cats = _as_list(scope.get("eval_category") or scope.get("eval_categories")) + _as_list(scope.get("eval_categories_excluded"))
    if cats:
        put("eval_categories", [COMPONENT_TO_CATEGORY.get(str(c).replace("_", "-") if str(c) in ("multi_turn", "over_refusal") else str(c), c)
                                if c not in V.EVAL_CATEGORIES else c for c in cats])
    tks = _as_list(scope.get("transfer_kind") or scope.get("transfer_kinds"))
    put("transfer_kinds", [t for t in tks if t in V.TRANSFER_KINDS])
    put("strength_nominals", scope.get("strength_nominal") or scope.get("strength_nominals"))
    put("operating_point_ids_legacy", scope.get("operating_point_id_legacy") or scope.get("operating_point_ids_legacy"))
    ls = scope.get("lane_source_legacy") or scope.get("lane_sources_legacy")
    if ls and not (isinstance(ls, str) and ls.upper().startswith("EVERY")):
        put("lane_sources_legacy", ls)
    put("views_legacy", scope.get("view_legacy") or scope.get("views_legacy"))
    put("comparison_ids", scope.get("comparison_ids") or scope.get("comparison_id"))
    cells = scope.get("cells")
    if isinstance(cells, list):
        for c in cells:
            if isinstance(c, dict):
                put("model_ids", c.get("model_id"))
                put("paper_labels", c.get("paper_labels") or c.get("paper_label"))
                put("operating_point_ids_legacy", c.get("operating_point_id_legacy"))
                put("transfer_kinds", [c.get("transfer_kind")] if c.get("transfer_kind") in V.TRANSFER_KINDS else [])
    isel = scope.get("instrument_selector_json")
    if isinstance(isel, dict):
        alts = isel.get("any_of") if isinstance(isel.get("any_of"), list) else [isel]
        out_alts = []
        for a in alts:
            if not isinstance(a, dict):
                continue
            alt: Dict[str, List] = {}
            if a.get("token_cap_tokens") is not None:
                alt["token_caps"] = [int(a["token_cap_tokens"])]
            if a.get("reply_cap_tokens") is not None and "token_caps" not in alt:
                alt["token_caps"] = [int(a["reply_cap_tokens"])]
            if a.get("decode_source") in V.DECODE_SOURCES:
                alt["decode_sources"] = [a["decode_source"]]
            if a.get("thinking_mode") in V.THINKING_MODES:
                alt["thinking_modes"] = [a["thinking_mode"]]
            if a.get("harness") in V.HARNESSES:
                alt["harnesses"] = [a["harness"]]
            if a.get("item_set"):
                alt["item_set_ids"] = [str(a["item_set"])]
            if a.get("n_shot") is not None:
                alt["n_shots"] = [int(a["n_shot"])]
            if alt:
                out_alts.append(alt)
        if out_alts:
            if len(out_alts) == 1 and "instrument_alternatives" not in m:
                for k, v in out_alts[0].items():
                    put(k, v)
            else:
                m["instrument_alternatives"] = out_alts
    return m or None


# --------------------------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------------------------


@dataclass
class RosterLoad:
    round_id: str
    version: str
    paper_round: Dict[str, object]
    entries: List[Dict[str, object]]
    rulings: List[Dict[str, object]]
    dispositions: List[Dict[str, object]]
    header: Optional[Dict[str, object]]
    counts_by_comparison: Dict[str, int]
    disagreements: List[Dict[str, object]]
    unknown_to_db: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        by_action: Dict[str, int] = {}
        for r in self.rulings:
            by_action[str(r["action"])] = by_action.get(str(r["action"]), 0) + 1
        by_prov: Dict[str, int] = {}
        for r in self.rulings:
            k = str(((r.get("extra") or {}).get("provenance") or {}).get("kind") or "in_ruled_by_prose")
            by_prov[k] = by_prov.get(k, 0) + 1
        return {"round_id": self.round_id, "version": self.version, "entries": len(self.entries), "inferred": sum(1 for e in self.entries if e["inferred"]),
                "drawn_false": sum(1 for e in self.entries if not e["drawn"]), "rulings": len(self.rulings), "rulings_by_action": dict(sorted(by_action.items())),
                "rulings_by_provenance": dict(sorted(by_prov.items())),
                "dispositions": len(self.dispositions), "disagreements": len(self.disagreements), "counts_by_comparison": self.counts_by_comparison,
                "unknown_to_db": self.unknown_to_db}


def load(roster_path: str, rulings_path: Optional[str] = None, conn=None, strict_counts: bool = True) -> RosterLoad:
    roster = load_yaml(roster_path)
    exp = expand(roster, strict_counts=strict_counts)
    vocab = roster_vocab(roster)
    header, raw_rulings = read_rulings_jsonl(rulings_path) if rulings_path else (None, [])
    rulings = [normalise_ruling(r, vocab, where=os.path.basename(rulings_path or "")) for r in raw_rulings]
    ids = [r["ruling_id"] for r in rulings]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise RosterRefusal(f"duplicate ruling ids: {dup}")
    for r in rulings:
        if r["round_id"] != exp.round_id:
            raise RosterRefusal(f"ruling {r['ruling_id']} names round {r['round_id']!r}, the roster is round {exp.round_id!r}")
    version = str((roster.get("meta") or {}).get("roster_version") or "v1")
    out = RosterLoad(exp.round_id, version, exp.paper_round, exp.entries, rulings, exp.dispositions, header, exp.counts_by_comparison, exp.disagreements)
    if conn is not None:
        out.unknown_to_db = unknown_to_db(conn, out)
        bad = named_object_disagreements(conn, out)
        if bad:
            raise RosterRefusal("named objects whose object row disagrees with the item that names them (paper_label / object_kind): "
                                + "; ".join(f"{b['object_id'][:12]}… named as {b['named_label']}/{b['named_kind']} is {b['db_label']}/{b['db_kind']} "
                                            f"({b['comparison_id']} {b['model_id']})" for b in bad[:8]) + (f"; +{len(bad) - 8} more" if len(bad) > 8 else ""))
    return out


def named_object_disagreements(conn, load_: RosterLoad) -> List[Dict[str, object]]:
    """Named objects (roster v10) whose ``object`` row carries another ``paper_label`` or ``object_kind`` than the item naming them: an id
    the table does not carry is NOT a disagreement (it is reported under ``unknown_to_db.object_ids`` and dependency-blocks at the fold)."""
    named = {e["object_id"]: e for e in load_.entries if e["object_id"] and (e.get("extra") or {}).get("object_named")}
    if not named:
        return []
    try:
        rows = {str(r[0]): (r[1], r[2]) for r in conn.execute(
            "SELECT object_id, paper_label, object_kind FROM object WHERE object_id IN (" + ",".join("?" * len(named)) + ")", list(named)).fetchall()}
    except Exception:   # noqa: BLE001 -- a database without the table: nothing to disagree with
        return []
    out = []
    for oid, e in sorted(named.items()):
        if oid not in rows:
            continue
        lab, kind = rows[oid]
        if str(lab) != str(e["extra"].get("paper_label")) or str(kind) != str(e["extra"].get("object_kind")):
            out.append({"object_id": oid, "named_label": e["extra"].get("paper_label"), "named_kind": e["extra"].get("object_kind"),
                        "db_label": lab, "db_kind": kind, "comparison_id": e["comparison_id"], "model_id": e["model_id"]})
    return out


def unknown_to_db(conn, load_: RosterLoad) -> Dict[str, List[str]]:
    def ids(table, col):
        try:
            return {str(r[0]) for r in conn.execute(f"SELECT {col} FROM {table}").fetchall()}
        except Exception:   # noqa: BLE001 — a database without the table reports every id unknown
            return set()
    models, evals, objects, instruments = ids("model", "model_id"), ids("eval", "eval_id"), ids("object", "object_id"), ids("instrument", "instrument_id")
    return {"model_ids": sorted({e["model_id"] for e in load_.entries if e["model_id"] not in models}),
            "eval_ids": sorted({e["eval_id"] for e in load_.entries if e["eval_id"] not in evals}),
            "object_ids": sorted({e["object_id"] for e in load_.entries if e["object_id"] and e["object_id"] not in objects}),
            # a NAMED instrument (v6) the table lacks: the entry dependency-blocks at the fold until the minting reader lands (never refused)
            "instrument_ids": sorted({e["instrument_id"] for e in load_.entries if e["instrument_id"] and e["instrument_id"] not in instruments})}


def producer(script: str = "sisteer.registry.adapters.roster", code_sha: str = "unverified") -> Dict[str, object]:
    import socket
    from .. import emit
    return {"script": script, "code_sha": code_sha, "session": emit.session_from_env(), "host": socket.gethostname(), "substrate": "grading_host"}


def to_records(load_: RosterLoad, written_utc: Optional[str] = None, prod: Optional[Mapping[str, object]] = None) -> Dict[str, List[Dict[str, object]]]:
    """Records by type, deterministic ids (uuid5 over the payload) so a re-load is a duplicate, not a second fact."""
    now = written_utc or R.utc_now()
    prod = dict(prod or producer())

    def rec(rt, payload):
        return R.make_record(rt, payload, prod, identity.adapter_record_id(payload, rt), now)
    return {"paper_round": [rec("paper_round", load_.paper_round)],
            "roster_entry": [rec("roster_entry", e) for e in load_.entries],
            "ruling": [rec("ruling", r) for r in load_.rulings],
            "disposition": [rec("disposition", d) for d in load_.dispositions]}


def write_records(root: str, load_: RosterLoad, written_utc: Optional[str] = None, prod: Optional[Mapping[str, object]] = None,
                  subdir: Optional[str] = None) -> List[str]:
    """Write the load as record files under ``<root>/rulings/<round>_<version>/`` (the log's rulings namespace). Files are
    written whole and are byte-deterministic for a given ``written_utc``; an existing identical file is left alone, a
    differing one is refused (a correction is a NEW version directory, never a rewrite)."""
    recs = to_records(load_, written_utc, prod)
    d = os.path.join(root, "rulings", subdir or f"{load_.round_id}_{load_.version}")
    os.makedirs(d, exist_ok=True)
    paths = []
    for seq, rt in enumerate(("paper_round", "roster_entry", "ruling", "disposition"), start=1):
        if not recs[rt]:
            continue
        path = os.path.join(d, R.record_file_name(seq, rt))
        data = "".join(identity.canonical_json(r) + "\n" for r in recs[rt]).encode("utf-8")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                if fh.read() == data:
                    paths.append(path)
                    continue
            raise RosterRefusal(f"{path} exists with different content; load a NEW roster version instead of rewriting the log")
        with open(path, "wb") as fh:
            fh.write(data)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------------------------
# show / diff
# --------------------------------------------------------------------------------------------


def entry_key(e: Mapping[str, object]) -> Tuple:
    sel = e.get("object_selector_json")
    if isinstance(sel, str):
        try:
            sel = json.loads(sel)
        except ValueError:
            sel = None
    obj = e.get("object_id") or ((sel or {}).get("paper_label") if isinstance(sel, dict) else None) or identity.canonical_json(sel)
    return (str(e["comparison_id"]), str(e["model_id"]), str(obj), str(e["eval_id"]), str(e["strength_rule"]))


def db_entries(conn, round_id: str) -> List[Dict[str, object]]:
    rows = conn.execute("SELECT * FROM roster_entry WHERE round_id = ?", (round_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("object_selector_json", "instrument_selector_json", "judge_selector_json", "extra_json"):
            if isinstance(d.get(k), str):
                try:
                    d[k] = json.loads(d[k])
                except ValueError:
                    pass
        d["extra"] = d.pop("extra_json", {}) or {}
        out.append(d)
    return out


def diff_entries(a: Sequence[Mapping[str, object]], b: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Set difference of two roster expansions on the identity key; ``changed`` lists entries present in both whose
    inferred / drawn / citation differ."""
    ka = {e["roster_entry_id"]: e for e in a}
    kb = {e["roster_entry_id"]: e for e in b}
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))
    changed = []
    for k in sorted(set(ka) & set(kb)):
        ea, eb = ka[k], kb[k]
        diffs = {f: (ea.get(f), eb.get(f)) for f in ("inferred", "drawn", "source_citation") if (ea.get(f) or 0) != (eb.get(f) or 0) and str(ea.get(f)) != str(eb.get(f))}
        if diffs:
            changed.append({"roster_entry_id": k, "key": entry_key(ea), "diffs": diffs})
    # the COARSE key (comparison, model, object label / id, eval, strength rule) tells a re-transcription that reshaped a
    # selector from one that changed which measurements a figure reads
    ca = {entry_key(e) for e in a}
    cb = {entry_key(e) for e in b}
    return {"n_a": len(ka), "n_b": len(kb), "only_a": [dict(key=entry_key(ka[k]), roster_entry_id=k) for k in only_a],
            "only_b": [dict(key=entry_key(kb[k]), roster_entry_id=k) for k in only_b], "changed": changed,
            "coarse_only_a": sorted(ca - cb), "coarse_only_b": sorted(cb - ca), "coarse_equal": ca == cb,
            "equal": not only_a and not only_b and not changed}


def show_text(load_: RosterLoad, limit: int = 40) -> str:
    lines = [f"roster {load_.round_id} {load_.version}: {len(load_.entries)} entries over {len(load_.counts_by_comparison)} comparisons; "
             f"{len(load_.rulings)} rulings; {len(load_.dispositions)} dispositions; {len(load_.disagreements)} entries with a build-vs-ruling disagreement"]
    if load_.disagreements:
        lines.append("DISAGREEMENTS FIRST (fixed decision 11; the build's value is kept, inferred=1):")
        seen = {}
        for e in load_.disagreements:
            seen.setdefault(str(e["extra"]["disagreement"]), []).append(e)
        for text, es in seen.items():
            lines.append(f"  [{len(es):4d} entries] {text}")
    lines.append("entries per comparison:")
    for k, n in sorted(load_.counts_by_comparison.items()):
        lines.append(f"  {n:5d}  {k}")
    summary = load_.summary()
    lines.append("rulings by action: " + json.dumps(summary["rulings_by_action"]))
    lines.append("rulings by provenance (D83; `relayed` = a peer session relayed the user's words, awaiting the user's confirmation): "
                 + json.dumps(summary["rulings_by_provenance"]))
    for r in load_.rulings:
        pv = (r.get("extra") or {}).get("provenance") or {}
        if pv.get("kind") == "relayed":
            lines.append(f"  relayed  {r['ruling_id']}  ({r['action']}, {r['scope_kind']}) via {pv.get('relayed_by')} at {pv.get('message_utc')}")
        elif pv.get("kind") == "peer_encoded":
            lines.append(f"  peer_encoded  {r['ruling_id']}  ({r['action']}, {r['scope_kind']}) in {pv.get('encoded_by')}'s commit {pv.get('commit')}")
    if load_.unknown_to_db and any(load_.unknown_to_db.values()):
        lines.append("ids the database does not carry yet (these records dependency-block until the dimension lands; a named instrument until its reader's read window): "
                     + json.dumps(load_.unknown_to_db))
    return "\n".join(lines)
