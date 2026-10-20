"""Dimension seeds.

``configs/registry/{models,evals,item_sets,judges,instruments,si_texts,decode_policy_seed,legacy_spellings}.yaml``
-> ``model`` / ``eval`` / ``item_set`` / ``judge`` / ``instrument`` / ``si_text`` / ``decode_policy`` /
``sampling_mode_policy`` records (deterministic uuid5 ids), plus the ``object`` rows no producer owns — the unsteered
``none`` object and one declared-prompt ``si:<sha>`` object per ``si_text`` row — written under
``<root>/records/dimensions/<revision>/`` — published as ONE directory rename (:func:`publish_seed_records`) and applied by the
harvester in ONE transaction (``harvest.Ingestor.apply_dimension_bundle``), so a reader sees the previous dimension set or
the new one and never a mix. This adapter runs FIRST: every other table's FKs point here.

Rules (the README's citation convention):

* the loader NEVER fills a value. A seed row missing a required column is refused naming the seed file, the row
  and the field (:class:`SeedRefusal`, carrying the validator's reason); an out-of-vocabulary value likewise. The
  fix is always in the YAML, never a default here.
* the literal ``unverified`` is admitted only where the column can spell it: kept verbatim where the column's
  vocabulary / pattern admits it, ``NULL`` where the column is nullable (an INTEGER's only spelling), refused
  otherwise — in every case with a reason in ``field_confidence_json`` (taken from the row's own ``source`` /
  ``*_note`` / ``notes`` text, else the README's known-gaps list is cited).
* the seeds' ``confidence: inferred`` (row- or field-level) is carried into ``field_confidence_json`` as
  ``{confidence: inferred, reason: <inferred_reason>}``.
* readable seed ids for judges and instruments become the spec's hashes (``identity.judge_id`` /
  ``identity.instrument_id``); the readable id is kept in ``extra_json.seed_*_id``.
* an instrument seed row spans several evals (``eval_ids`` / ``item_set_ids``) and, through ``decode_policy_refs``,
  several served models; it expands to one ``instrument`` row per (eval, decode policy) — the decode numbers ARE
  instrument identity (``identity.INSTRUMENT_ID_FIELDS``), so the same eval on two models is two instruments.
* wildcard ``model_id: "*"`` decode rows expand over every SERVED paper model (``paper_status`` in_paper / descoped
  AND ``checkpoint_kind`` instruct / chat — base checkpoints are never served); ``same_values_as`` is checked
  (equal numbers or refuse); ``superseded_by_policy_id`` / instrument refs resolve through :func:`PolicyIndex.resolve_ref`, and
  ``superseded_by_policy_id`` is stated on EVERY decode_policy row (null where the seed names no successor), so a pointer a seed
  drops folds the live row back to NULL instead of standing for ever.
* ``ruling_id`` FKs stay NULL until the ``ruling`` table is seeded (the seed's ruling id rides
  ``extra_json.decode_ruling_ref`` with its text), so no dimension row is dependency-blocked on later work.
* where two sources contradict, the seed keeps BOTH values and the loader takes the drafter's default flag —
  nothing here re-ranks a contradiction.
* judge and instrument ids are minted by ``identity.judge_id`` / ``identity.instrument_id`` from the NORMALISED
  payload (the canonical column names and storage-typed values), the same functions every adapter uses, so a
  judgment's ``judge_id`` and a run's ``instrument_id`` join the dimension rows whoever wrote them.
* every instrument record's ``extra_json`` spells the served models its decode policy binds it to the same way:
  ``model_ids`` (a list, always) plus ``model_id`` when exactly one — a seed entry whose rows state identical decode
  numbers on several models (a greedy, protocol or sends-nothing policy) is ONE model-independent row listing them
  all, because the identity fields are identical (a resolver keyed on ``(seed, eval, model)`` must accept
  ``model_id == m or m in model_ids``).
* the ``object`` rows minted here are the two kinds no adapter derives from a fit artifact: ``none`` (the unsteered
  object) and ``si:<sha>`` (a declared prompt) — one per ``si_text`` row; status ``approved`` when a paper cell
  declares the prompted object (``legacy_spellings.yaml paper_labels`` with ``object_role: prompt``), else
  ``reference``; the campaign is the paper's (``sdfbase_full22``) with the campaign-independence noted.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .. import identity
from .. import records as R
from .. import schema_def as S
from .. import site_forms as SF
from .. import validate as VAL
from .. import vocab as V
from . import si_text as SI
from .base import AdapterRefusal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DEFAULT_SEEDS_DIR = os.path.join(REPO_ROOT, "configs", "registry")
SEED_FILES: Dict[str, str] = {
    "model": "models.yaml", "eval": "evals.yaml", "item_set": "item_sets.yaml", "judge": "judges.yaml",
    "instrument": "instruments.yaml", "si_text": "si_texts.yaml", "decode_policy": "decode_policy_seed.yaml",
    "legacy_spellings": "legacy_spellings.yaml",
}
#: the seeds' eval-category spellings (paperlib components) -> the controlled vocabulary
EVAL_CATEGORY_SPELLINGS: Dict[str, str] = {"static": "static_attack", "bon": "best_of_n"}
#: the seeds' read_at spelling -> the controlled value (the old word is retired)
READ_AT_SPELLINGS: Dict[str, str] = {"final_rung": "ladder_final"}
ADAPTER_NAME = "dimensions"
SCRIPT = "sisteer.registry.adapters.dimensions"
#: the table order records are written in (FK parents first)
TABLE_ORDER: Tuple[str, ...] = ("model", "si_text", "eval", "item_set", "judge", "instrument", "decode_policy", "sampling_mode_policy",
                                "object")
#: the campaign the `none` / prompted `si:<sha>` object rows are filed under (the controlled spelling of the paper's
#: campaign, legacy_spellings.yaml `campaigns`); `extra_json.campaign_note` records that these objects are campaign-independent
OBJECT_SEED_CAMPAIGN = "sdfbase_full22"

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2}:\d{2})(?:\.(\d{1,3}))?Z)?(.*)$")
_REF_VERSION_RE = re.compile(r"^(?P<ref>.+?)@v(?P<version>\d+)$")
_REF_ALT_RE = re.compile(r"^(?P<ref>[^ ]+) \((?P<prefix>[a-z0-9_]+): (?P<tail>\|[^)]+)\)$")


class SeedRefusal(AdapterRefusal):
    """A seed row cannot become a record: names the seed file, the row and the field."""

    def __init__(self, seed_file: str, row: str, field_name: Optional[str], why: str):
        super().__init__(f"{seed_file}:{row}", (f"field {field_name!r}: " if field_name else "") + why)
        self.seed_file = seed_file
        self.row = row
        self.field = field_name


# --------------------------------------------------------------------------------------------
# YAML in
# --------------------------------------------------------------------------------------------


@dataclass
class Seeds:
    seeds_dir: str
    models: List[Dict[str, Any]]
    evals: List[Dict[str, Any]]
    item_sets: List[Dict[str, Any]]
    judges: List[Dict[str, Any]]
    instruments: List[Dict[str, Any]]
    si_texts: List[Dict[str, Any]]
    decode: Dict[str, Any]          # policies, sampling_mode_policy, rulings, citations
    legacy: Dict[str, Any]
    base_commit: str
    file_sha256: Dict[str, str] = field(default_factory=dict)

    def revision_id(self) -> str:
        return identity.sha16("|".join(f"{k}:{v}" for k, v in sorted(self.file_sha256.items())))


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SeedRefusal(os.path.basename(path), "<file>", None, "the seed is not a YAML mapping")
    return data


def _rows(data: Dict[str, Any], key: str, fname: str) -> List[Dict[str, Any]]:
    rows = data.get(key)
    if not isinstance(rows, list):
        raise SeedRefusal(fname, "<file>", key, f"expected a list under {key!r}")
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise SeedRefusal(fname, f"{key}[{i}]", None, "row is not a mapping")
    return rows


def load_seeds(seeds_dir: str = DEFAULT_SEEDS_DIR) -> Seeds:
    import hashlib
    paths = {k: os.path.join(seeds_dir, fn) for k, fn in SEED_FILES.items()}
    for k, p in paths.items():
        if not os.path.isfile(p):
            raise SeedRefusal(SEED_FILES[k], "<file>", None, f"seed file missing: {p}")
    shas = {}
    for k, p in paths.items():
        with open(p, "rb") as fh:
            shas[SEED_FILES[k]] = hashlib.sha256(fh.read()).hexdigest()
    d = {k: _load_yaml(p) for k, p in paths.items()}
    si = (_rows(d["si_text"], "si_texts", SEED_FILES["si_text"]) + list(d["si_text"].get("reference_texts") or [])
          + list(d["si_text"].get("candidate_texts") or []))
    return Seeds(
        seeds_dir=seeds_dir,
        models=_rows(d["model"], "models", SEED_FILES["model"]),
        evals=_rows(d["eval"], "evals", SEED_FILES["eval"]),
        item_sets=_rows(d["item_set"], "item_sets", SEED_FILES["item_set"]),
        judges=_rows(d["judge"], "judges", SEED_FILES["judge"]),
        instruments=_rows(d["instrument"], "instruments", SEED_FILES["instrument"]),
        si_texts=si,
        decode={"policies": _rows(d["decode_policy"], "policies", SEED_FILES["decode_policy"]),
                "sampling_mode_policy": _rows(d["decode_policy"], "sampling_mode_policy", SEED_FILES["decode_policy"]),
                "rulings": dict(d["decode_policy"].get("rulings") or {}),
                "citations": dict(d["decode_policy"].get("citations") or {})},
        legacy=d["legacy_spellings"],
        base_commit=str(d["model"].get("base_commit", "unverified")),
        file_sha256=shas,
    )


# --------------------------------------------------------------------------------------------
# column normalisation
# --------------------------------------------------------------------------------------------


def parse_date_utc(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """``"2026-09-11"`` / ``"2026-09-11T00:00:00Z"`` / ``"2026-08-18 (prose)"`` -> (UTC-with-ms, note).
    Returns ``(None, None)`` for None and ``("unverified", None)`` for the literal."""
    if value is None:
        return None, None
    if value == "unverified":
        return "unverified", None
    m = _DATE_RE.match(str(value).strip())
    if not m:
        return None, f"unparseable date {value!r}"
    day, hms, ms, rest = m.group(1), m.group(2), m.group(3), m.group(4).strip()
    utc = f"{day}T{hms or '00:00:00'}.{(ms or '0').ljust(3, '0')}Z"
    note = "date_only" if hms is None else None
    if rest:
        note = (note + "; " if note else "") + rest.strip("() ")
    return utc, note


def _column(table: str, col: str) -> Optional[S.Column]:
    try:
        return S.TABLE_BY_NAME[table].column(col)
    except KeyError:
        return None


def _admits_unverified(c: S.Column) -> bool:
    if c.vocab:
        return "unverified" in V.VOCABULARIES[c.vocab]
    if c.pattern:
        return c.pattern.endswith("_OR_UNVERIFIED")
    return c.type == "TEXT"


def _source_entry(row: Mapping[str, Any], col: str) -> Any:
    src = row.get("source")
    if isinstance(src, dict):
        return src.get(col)
    return None


def unverified_reason(row: Mapping[str, Any], col: str, base_commit: str) -> str:
    """The reason a seed value is ``unverified``: the row's own words first, the README's known-gaps list last."""
    ent = _source_entry(row, col)
    if isinstance(ent, dict):
        for k in ("reason", "inferred_reason", "value_source"):
            if ent.get(k):
                return str(ent[k])
    for k in (f"{col}_note", f"{col}_reason", f"{col}_absence_evidence"):
        if row.get(k):
            return f"{k}: {row[k]}"
    if isinstance(ent, str) and ent:
        return f"source: {ent}"
    return (f"not recorded in the tree at base_commit {base_commit}; listed under 'Known gaps' in "
            "configs/registry/README.md")


class Normaliser:
    """Maps a seed row's raw values onto one table's columns: dates -> UTC ms, ``unverified`` -> the column's
    spelling, ``confidence`` -> ``field_confidence_json``, FK-to-ruling text -> ``extra_json``."""

    def __init__(self, seed_file: str, row_key: str, base_commit: str):
        self.seed_file = seed_file
        self.row_key = row_key
        self.base_commit = base_commit

    def refuse(self, field_name: Optional[str], why: str) -> SeedRefusal:
        return SeedRefusal(self.seed_file, self.row_key, field_name, why)

    def __call__(self, table: str, payload: Dict[str, Any], row: Mapping[str, Any], extra: Dict[str, Any]
                 ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        fc: Dict[str, Any] = {}
        out: Dict[str, Any] = {}
        cols = set(S.payload_columns(table))
        # field-level confidence from the row's `source` dict
        src = row.get("source")
        if isinstance(src, dict):
            for k, ent in src.items():
                if isinstance(ent, dict) and ent.get("confidence"):
                    fc[k] = {"confidence": str(ent["confidence"]),
                             "reason": str(ent.get("inferred_reason") or ent.get("reason") or ent.get("value_source") or "")}
        if row.get("confidence"):
            fc["row"] = {"confidence": str(row["confidence"]), "reason": str(row.get("inferred_reason") or "")}
        for k, v in payload.items():
            if k == "extra":
                continue
            c = _column(table, k)
            if c is None:
                extra[k] = v
                continue
            if c.fk == ("ruling", "ruling_id") and isinstance(v, str) and v:
                # the ruling table is seeded later: keep the text, never a dangling FK (a blocked seed helps nobody)
                extra[f"{k}_note"] = v
                out[k] = None
                continue
            if k.endswith("_utc") and c.type == "TEXT" and v is not None:
                utc, note = parse_date_utc(v)
                if utc is None:
                    raise self.refuse(k, note or f"bad date {v!r}")
                if note:
                    extra[f"{k}_confidence"] = note
                v = utc
            if v == "unverified":
                fc.setdefault(k, {"confidence": "unverified", "reason": unverified_reason(row, k, self.base_commit)})
                if _admits_unverified(c):
                    out[k] = "unverified"
                elif not c.nn:
                    out[k] = None
                else:
                    raise self.refuse(k, f"column {table}.{k} ({c.type}, NOT NULL) cannot spell 'unverified'; the seed must give a value")
                continue
            out[k] = v
        if fc:
            if "field_confidence_json" not in cols:
                extra["field_confidence"] = fc
            else:
                out["field_confidence_json"] = fc
        return out, extra


# --------------------------------------------------------------------------------------------
# per-table payloads
# --------------------------------------------------------------------------------------------

_MODEL_EXTRA = ("producer_spellings", "bank_presence", "chat_template_hash16", "generation_config_file_sha256",
                "generation_config_note", "generation_config_absence_evidence")


def model_payload(row: Mapping[str, Any], norm: Normaliser) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in row.items():
        if k == "source":
            extra["seed_source"] = v
        elif k in ("confidence", "inferred_reason"):
            continue
        elif k in _MODEL_EXTRA:
            extra[k] = v
        else:
            payload[k] = v
    return norm("model", payload, row, extra)


def eval_payload(row: Mapping[str, Any], norm: Normaliser) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in row.items():
        if k == "producer_spellings":
            payload["producer_spellings_json"] = list(v or [])
        elif k == "eval_category":
            payload[k] = EVAL_CATEGORY_SPELLINGS.get(str(v), v)
            if str(v) in EVAL_CATEGORY_SPELLINGS:
                extra["eval_category_spelling_legacy"] = v
        elif k == "source":
            extra["seed_source"] = v
        elif k in ("confidence", "inferred_reason"):
            continue
        else:
            payload[k] = v
    return norm("eval", payload, row, extra)


def item_set_payload(row: Mapping[str, Any], norm: Normaliser) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in row.items():
        if k == "source":
            extra["seed_source"] = v
        elif k in ("confidence", "inferred_reason"):
            continue
        else:
            payload[k] = v
    return norm("item_set", payload, row, extra)


def judge_backend_value(v: Any, norm: Normaliser) -> str:
    vals = sorted(set(v)) if isinstance(v, list) else [v]
    if len(vals) == 1:
        return str(vals[0])
    if set(vals) == {"batch", "live"}:
        return "batch_or_live"
    raise norm.refuse("judge_backend", f"cannot represent the backend set {vals} in JUDGE_BACKENDS {list(V.JUDGE_BACKENDS)}")


def judge_payload(row: Mapping[str, Any], norm: Normaliser) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Returns (payload, extra, readable seed id). ``judge_id`` becomes the spec's hash."""
    payload: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    seed_id = str(row.get("judge_id"))
    for k, v in row.items():
        if k == "judge_id":
            extra["seed_judge_id"] = v
        elif k == "label_vocabulary":
            if v is not None:
                payload["label_vocabulary_json"] = v
        elif k == "judge_backend":
            payload[k] = judge_backend_value(v, norm)
            if isinstance(v, list):
                extra["judge_backends_seeded"] = list(v)
        elif k == "source":
            extra["seed_source"] = v
        elif k in ("confidence", "inferred_reason"):
            continue
        else:
            payload[k] = v
    payload, extra = norm("judge", payload, row, extra)
    fields = {k: payload.get(k) for k in identity.JUDGE_ID_FIELDS}
    payload["judge_id"] = identity.judge_id(fields)
    return payload, extra, seed_id


# --------------------------------------------------------------------------------------------
# decode policy
# --------------------------------------------------------------------------------------------

PARAM_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
#: the seed's per-row keys that are not columns: kept in extra_json under their own names
_POLICY_EXTRA = ("mode", "params_sent", "same_values_as", "deliberate_departure", "legacy_contracts_row",
                 "effective_from_confidence")


def parse_ref(ref: str) -> Tuple[str, Optional[int]]:
    """``"a|b|c|d@v2"`` -> (``"a|b|c|d"``, 2); no suffix -> version None."""
    m = _REF_VERSION_RE.match(ref)
    if m:
        return m.group("ref"), int(m.group("version"))
    return ref, None


def policy_id_for(ref: str, version: int) -> str:
    return f"{ref}@v{version}"


def served_model_ids(models: Sequence[Mapping[str, Any]]) -> List[str]:
    """The paper's SERVED checkpoints: in_paper / descoped AND instruct / chat (base checkpoints are never served)."""
    return [str(m["model_id"]) for m in models
            if m.get("paper_status") in ("in_paper", "descoped") and m.get("checkpoint_kind") in ("instruct", "chat")]


@dataclass
class PolicyIndex:
    """Every expanded decode policy row (payload + extra) by policy_id, with ref resolution."""
    rows: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = field(default_factory=dict)
    by_ref: Dict[str, List[str]] = field(default_factory=dict)      # ref -> [policy_id]
    unresolved: List[Tuple[str, str, str]] = field(default_factory=list)  # (policy_id, field, ref text)
    no_default_keys: List[str] = field(default_factory=list)              # keys whose live rows are all non-default

    def add(self, payload: Dict[str, Any], extra: Dict[str, Any]) -> None:
        pid = str(payload["policy_id"])
        if pid in self.rows:
            raise SeedRefusal(SEED_FILES["decode_policy"], pid, "policy_id", "duplicate policy (same ref and version)")
        self.rows[pid] = (payload, extra)
        self.by_ref.setdefault(str(payload["policy_ref"]), []).append(pid)

    def resolve_ref(self, text: str, model_id: Optional[str] = None) -> Optional[str]:
        """A seed reference (``ref``, ``ref@vN``, ``<model>|…``, ``*|…``, ``ref (llama: |tail)``) -> policy_id, or None."""
        text = str(text).strip()
        m = _REF_ALT_RE.match(text)
        if m:
            base, prefix, tail = m.group("ref"), m.group("prefix"), m.group("tail")
            if model_id and model_id.startswith(prefix):
                segs = base.split("|")
                tail_segs = tail.strip("|").split("|")
                base = "|".join(segs[: len(segs) - len(tail_segs)] + tail_segs)
            text = base
        if model_id:
            text = text.replace("<model>", model_id)
            if text.startswith("*|"):
                text = model_id + text[1:]
        ref, version = parse_ref(text)
        pids = self.by_ref.get(ref) or []
        if not pids:
            return None
        if version is not None:
            pid = policy_id_for(ref, version)
            return pid if pid in self.rows else None
        if len(pids) == 1:
            return pids[0]
        defaults = [p for p in pids if int(self.rows[p][0].get("is_default") or 0) == 1]
        if len(defaults) == 1:
            return defaults[0]
        raise SeedRefusal(SEED_FILES["decode_policy"], ref, "policy_ref", f"ambiguous bare reference (versions {sorted(pids)}) with no single default")

    def numbers(self, pid: str) -> Dict[str, Any]:
        p = self.rows[pid][0]
        return {k: p.get(k) for k in PARAM_KEYS}


def _expand_policy_rows(seeds: Seeds) -> List[Tuple[Dict[str, Any], str]]:
    """Wildcard rows over the served models; returns (row, expanded_from_or_'') pairs."""
    served = served_model_ids(seeds.models)
    out = []
    for row in seeds.decode["policies"]:
        mid = str(row.get("model_id"))
        if mid == "*":
            for model_id in served:
                r = dict(row)
                r["model_id"] = model_id
                r["policy_ref"] = model_id + str(row["policy_ref"])[1:]
                out.append((r, str(row["policy_ref"])))
        else:
            out.append((row, ""))
    return out


def decode_policy_payloads(seeds: Seeds) -> PolicyIndex:
    fname = SEED_FILES["decode_policy"]
    idx = PolicyIndex()
    rulings = seeds.decode["rulings"]
    pending_refs: List[Tuple[str, str, str, str]] = []   # (policy_id, column|extra key, ref text, model_id)
    for row, expanded_from in _expand_policy_rows(seeds):
        ref_text = str(row.get("policy_ref") or "")
        if not ref_text:
            raise SeedRefusal(fname, str(row.get("model_id")), "policy_ref", "required field 'policy_ref' missing")
        ref, ref_version = parse_ref(ref_text)
        version = int(row.get("policy_version") or 0)
        if ref_version is not None and ref_version != version:
            raise SeedRefusal(fname, ref_text, "policy_version", f"@v{ref_version} in the ref disagrees with policy_version {version}")
        if version < 1:
            raise SeedRefusal(fname, ref_text, "policy_version", "required field 'policy_version' missing")
        pid = policy_id_for(ref, version)
        norm = Normaliser(fname, ref_text, seeds.base_commit)
        # `superseded_by_policy_id` is STATED on every row -- an explicit null where the seed names no successor -- because the reducer
        # keeps a column an event omits (reducer: an omitted key is not a statement), so a row that once carried a successor and no
        # longer does must say so to fold the live pointer back to NULL (the nine best-of-N grandfathered_T1 rows made live again
        # on 2026-10-14; before this the loader wrote the key only when the seed named a successor, and removing the pointer from
        # the seed changed nothing in the database). Stating null on a row that never had a successor changes no value.
        payload: Dict[str, Any] = {"policy_id": pid, "policy_ref": ref, "policy_version": version, "ruling_id": None,
                                   "superseded_by_policy_id": None}
        extra: Dict[str, Any] = {}
        if expanded_from:
            extra["wildcard_expanded_from"] = expanded_from
        for k, v in row.items():
            if k in ("policy_ref", "policy_version"):
                continue
            if k == "is_default_for_lookup":
                payload["is_default"] = v
            elif k == "params_not_sent":
                payload["params_not_sent_json"] = v or {}
            elif k == "eval_category":
                if v is not None:
                    payload[k] = EVAL_CATEGORY_SPELLINGS.get(str(v), v)
                    if str(v) in EVAL_CATEGORY_SPELLINGS:
                        extra["eval_category_spelling_legacy"] = v
            elif k == "ruling_id":
                if v:
                    extra["decode_ruling_ref"] = v
                    if v in rulings:
                        extra["decode_ruling_text"] = rulings[v]
                    else:
                        raise SeedRefusal(fname, ref_text, "ruling_id", f"ruling {v!r} is not in the seed's rulings block")
            elif k == "superseded_by_policy_id":
                if v:
                    pending_refs.append((pid, "superseded_by_policy_id", str(v), str(row["model_id"])))
            elif k == "same_values_as":
                pending_refs.append((pid, "same_values_as", str(v), str(row["model_id"])))
            elif k in _POLICY_EXTRA:
                extra[k] = v
            elif k in ("confidence", "inferred_reason", "source"):
                continue
            else:
                payload[k] = v
        if row.get("legacy_contracts_row"):
            # the flag is accepted; the vocabulary carries the dedicated value so a consumer
            # filtering `decode_source = 'model_card'` never mistakes this hashed legacy identity for a vendor row
            extra["seed_decode_source"] = payload.get("decode_source")
            payload["decode_source"] = "legacy_contracts_row"
        payload, extra = norm("decode_policy", payload, row, extra)
        payload.setdefault("is_default", 1)
        idx.add(payload, extra)
    # references
    for pid, col, text, model_id in pending_refs:
        target = idx.resolve_ref(text, model_id)
        payload, extra = idx.rows[pid]
        if col == "same_values_as":
            if target is None:
                raise SeedRefusal(fname, pid, "same_values_as", f"reference {text!r} names no policy")
            mine, theirs = idx.numbers(pid), idx.numbers(target)
            if mine != theirs:
                raise SeedRefusal(fname, pid, "same_values_as", f"numbers {mine} differ from {target}'s {theirs}")
            extra["same_values_as"] = target
        else:
            if target is None:
                idx.unresolved.append((pid, col, text))
                extra["superseded_by_unresolved"] = text
                payload[col] = None
            else:
                payload[col] = target
    _check_default_uniqueness(idx)
    return idx


def _check_default_uniqueness(idx: PolicyIndex) -> None:
    """At most ONE default per (model, eval-or-category, thinking) among the live rows. A key
    whose live rows are all explicitly non-default has NO default: ``resolve()`` refuses it by name (nobody ruled a
    default) — recorded in ``idx.no_default_keys`` so the loader reports it rather than hiding it."""
    groups: Dict[Tuple[str, str, str], List[str]] = {}
    for pid, (p, _) in idx.rows.items():
        if p.get("superseded_by_policy_id") or idx.rows[pid][1].get("superseded_by_unresolved"):
            continue
        key = (str(p["model_id"]), str(p.get("eval_id") or f"category:{p.get('eval_category')}"), str(p["thinking_mode"]))
        groups.setdefault(key, []).append(pid)
    for key, pids in sorted(groups.items()):
        defaults = [p for p in pids if int(idx.rows[p][0].get("is_default") or 0) == 1]
        if len(defaults) > 1:
            raise SeedRefusal(SEED_FILES["decode_policy"], "|".join(key), "is_default_for_lookup",
                              f"at most one default per (model, eval-or-category, thinking); found {len(defaults)}: {sorted(defaults)}")
        if not defaults:
            idx.no_default_keys.append("|".join(key))


def legacy_model_map(seeds: Seeds) -> Dict[str, str]:
    """Every legacy model spelling -> model_id: legacy_spellings.yaml ``model_keys`` (pipe-separated spellings),
    each model's ``producer_spellings`` and ``hf_repo``. A spelling naming two models resolves to the one whose
    ``paper_status`` ranks first (:func:`vocab.model_spelling_owner`: in_paper, then descoped, then reference -- the
    paper's checkpoint owns its names, so the worked example ``qwen35_4b_demo``, a reference row re-pinning the paper's
    Qwen3.5-4B checkpoints, never captures ``Qwen/Qwen3.5-4B``); two models of the same rank naming one spelling are a
    clash, dropped (never guessed). ``decode_policy.export_policy`` applies the same rule to the export's spellings.

    A spelling in site-placeholder form (``$SHARED/...``) also maps its expansion on this host (:mod:`..site_forms`), the literal
    path the site's runs recorded, unless some row spells that literal itself; ``PolicyFile.model_id`` reads the export alike."""
    status = {str(m["model_id"]): m.get("paper_status") for m in seeds.models}
    claims: Dict[str, Dict[str, Optional[str]]] = {}

    def put(k: str, mid: str) -> None:
        k = k.strip()
        if not k or mid not in status:
            return
        claims.setdefault(k, {})[mid] = status[mid]

    for ent in seeds.legacy.get("model_keys") or []:
        mid = str(ent.get("model_id", ""))
        for k in str(ent.get("legacy", "")).split(" | "):
            put(k, mid)
    for m in seeds.models:
        mid = str(m["model_id"])
        put(mid, mid)
        put(str(m.get("hf_repo", "")), mid)
        for k in m.get("producer_spellings") or []:
            put(str(k), mid)
    out: Dict[str, str] = {}
    for k, claimants in claims.items():
        owner = V.model_spelling_owner(claimants)
        if owner is not None:
            out[k] = owner
    for k in [k for k in out if "$" in k]:
        lit = SF.expansion(k)
        if lit is not None and lit not in claims:
            out[lit] = out[k]
    return out


def sampling_mode_payloads(seeds: Seeds, idx: PolicyIndex) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """``sampling_mode_policy`` rows: every (model_key, mode) pair ``contracts.sampling_for`` accepts. Entries with
    ``refuses: true`` yield no row (absence IS the refusal, as in the old tables); ``model_key: "*"`` (greedy)
    expands over the served models to their ``static_attack`` greedy row — the numbers are category-independent
    (T 0.0, nothing else sent) and the consuming instrument's own greedy row is the run's policy."""
    fname = SEED_FILES["decode_policy"]
    mmap = legacy_model_map(seeds)
    served = served_model_ids(seeds.models)
    out = []
    for ent in seeds.decode["sampling_mode_policy"]:
        mode = ent.get("mode")
        mkey = ent.get("model_key")
        if mkey is None or ent.get("refuses") or mode not in V.SAMPLING_MODES:
            continue
        targets: List[Tuple[str, str]] = []
        if mkey == "*":
            for mid in served:
                targets.append((mid, str(ent["policy_ref"]).replace("<category>", "static")))
        else:
            mid = mmap.get(str(mkey))
            if mid is None:
                raise SeedRefusal(fname, f"sampling_mode_policy[{mkey}|{mode}]", "model_key", f"no model spelled {mkey!r} in models.yaml / legacy_spellings.yaml")
            targets.append((mid, str(ent["policy_ref"])))
        for mid, ref_text in targets:
            pid = idx.resolve_ref(ref_text, mid)
            if pid is None:
                raise SeedRefusal(fname, f"sampling_mode_policy[{mkey}|{mode}]", "policy_ref", f"reference {ref_text!r} names no policy")
            payload = {"model_id": mid, "sampling_mode": mode, "policy_id": pid, "notes": ent.get("source")}
            extra = {"model_key_legacy": mkey if mkey != "*" else mid, "seed_policy_ref": ent.get("policy_ref")}
            out.append((payload, extra))
    return out


# --------------------------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------------------------

_INSTRUMENT_DECODE_COLS = {"temperature": "decode_temperature", "top_p": "decode_top_p", "top_k": "decode_top_k",
                           "min_p": "decode_min_p", "repetition_penalty": "decode_repetition_penalty",
                           "presence_penalty": "decode_presence_penalty"}


def instrument_payloads(row: Mapping[str, Any], seeds: Seeds, idx: PolicyIndex, judge_ids: Mapping[str, str]
                        ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    fname = SEED_FILES["instrument"]
    seed_id = str(row.get("instrument_id"))
    eval_ids = list(row.get("eval_ids") or [])
    set_ids = list(row.get("item_set_ids") or [])
    if not eval_ids:
        raise SeedRefusal(fname, seed_id, "eval_ids", "required field 'eval_ids' missing")
    if not set_ids:
        raise SeedRefusal(fname, seed_id, "item_set_ids", f"eval(s) {eval_ids} named without their item_set_ids — an eval is never named without its version")
    if len(set_ids) == 1:
        set_ids = set_ids * len(eval_ids)
    if len(set_ids) != len(eval_ids):
        raise SeedRefusal(fname, seed_id, "item_set_ids", f"{len(eval_ids)} evals but {len(set_ids)} item sets")
    refs = list(row.get("decode_policy_refs") or [])
    served = served_model_ids(seeds.models)
    # one row per (eval, policy) — or per eval when the instrument declares no per-model decode policy
    policy_targets: List[Optional[str]] = []
    if refs:
        for ref in refs:
            if str(ref).startswith("*|"):
                for mid in served:
                    policy_targets.append(_must_resolve(idx, ref, mid, fname, seed_id))
            else:
                policy_targets.append(_must_resolve(idx, ref, None, fname, seed_id))
    else:
        policy_targets.append(None)
    out = []
    for eval_id, set_id in zip(eval_ids, set_ids):
        for pid in policy_targets:
            norm = Normaliser(fname, f"{seed_id}[{eval_id}{'|' + pid if pid else ''}]", seeds.base_commit)
            payload: Dict[str, Any] = {"eval_id": eval_id, "item_set_id": set_id}
            extra: Dict[str, Any] = {"seed_instrument_id": seed_id, "eval_ids": eval_ids, "item_set_ids": set_ids}
            for k, v in row.items():
                if k in ("instrument_id", "eval_ids", "item_set_ids", "decode_policy_refs", "confidence", "inferred_reason"):
                    continue
                if k == "source":
                    extra["seed_source"] = v
                elif k == "read_at":
                    payload[k] = READ_AT_SPELLINGS.get(str(v), v)
                elif k == "judge_id":
                    if v is None:
                        payload[k] = None
                    elif v in judge_ids:
                        payload[k] = judge_ids[v]
                        extra["seed_judge_id"] = v
                    else:
                        raise SeedRefusal(fname, seed_id, "judge_id", f"judge {v!r} is not in judges.yaml")
                elif k == "auditor_judge_id":
                    extra[k] = v
                    extra["auditor_judge_id_hash"] = judge_ids.get(str(v))
                elif k == "decode_source" and pid is not None:
                    extra["seed_decode_source"] = v
                else:
                    payload[k] = v
            if pid is not None:
                pol, pol_extra = idx.rows[pid]
                extra["decode_policy_id"] = pid
                extra["decode_policy_ref"] = pol["policy_ref"]
                extra["model_id"] = pol["model_id"]
                payload["decode_source"] = pol["decode_source"]
                sent = pol_extra.get("params_sent")
                if isinstance(sent, list):
                    sent_keys = set(sent)
                else:
                    # prose (two producers sent different things): not determinable at the instrument grain —
                    # nothing is copied; each run's decode_sent_json says what it sent (same rule as decode_policy.py)
                    sent_keys = set()
                    extra["decode_params_sent_note"] = sent
                for pk, col in _INSTRUMENT_DECODE_COLS.items():
                    payload[col] = pol.get(pk) if pk in sent_keys else None
                # the policy's thinking ARM is the row's (llama's `not_available` under a declared-`on` instrument);
                # only a declared REAL mode is substituted — a seed `unverified` stays and is refused by the normaliser
                if payload.get("thinking_mode") in V.THINKING_MODES and pol["thinking_mode"] != payload["thinking_mode"]:
                    extra["thinking_mode_declared"] = payload["thinking_mode"]
                    payload["thinking_mode"] = pol["thinking_mode"]
                if pol.get("seed_rule") and pol["seed_rule"] != payload.get("decode_seed_rule"):
                    extra["decode_seed_rule_policy"] = pol["seed_rule"]
            payload, extra = norm("instrument", payload, row, extra)
            payload["instrument_id"] = identity.instrument_id(payload)
            out.append((payload, extra))
    return out


def _must_resolve(idx: PolicyIndex, ref: str, model_id: Optional[str], fname: str, seed_id: str) -> str:
    pid = idx.resolve_ref(ref, model_id)
    if pid is None:
        raise SeedRefusal(fname, seed_id, "decode_policy_refs", f"reference {ref!r}{' for ' + model_id if model_id else ''} names no decode policy")
    return pid


# --------------------------------------------------------------------------------------------
# objects no producer owns: `none` and the declared prompts
# --------------------------------------------------------------------------------------------

_OBJECT_NOT_FIT = {"fit_thinking_render": "not_applicable", "fit_objective": "not_applicable", "fit_positions": "not_applicable",
                   "fit_seed_confidence": "not_applicable", "derivation": "none"}
_OBJECT_CAMPAIGN_NOTE = ("campaign-independent: the object is the absence of an intervention / a declared prompt, filed once under the paper's "
                         "campaign; a run's own `campaign` names where it was drawn")


def prompted_paper_labels(seeds: Seeds) -> Dict[str, str]:
    """``{si_id: paper_label}`` for every prompted paper cell the seeds declare: ``legacy_spellings.yaml paper_labels`` entries
    with ``object_role: prompt`` and an ``si_id`` (the seed's own statement of which SI texts are paper cells)."""
    out: Dict[str, str] = {}
    for ent in seeds.legacy.get("paper_labels") or []:
        if ent.get("object_role") == "prompt" and ent.get("si_id") and ent.get("paper_label") in V.PAPER_LABELS:
            out[str(ent["si_id"])] = str(ent["paper_label"])
    return out


def object_payloads(seeds: Seeds, si_records: Sequence[Mapping[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """The ``object`` rows the seeds own: ``none`` + one ``si:<sha>`` declared-prompt object per ``si_text`` record
    (``identity.object_id``). Returns ``(payload, extra, row_key)`` triples; nothing here is filled — every value is the
    structural fact the CHECK spells (``declared_prompt`` / ``none`` pairing) or read off the si_text row / the seeds."""
    labels = prompted_paper_labels(seeds)
    out: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
    none_payload: Dict[str, Any] = {"object_id": identity.object_id("none"), "object_kind": "none", "paper_label": "none",
                                    "campaign": OBJECT_SEED_CAMPAIGN, "campaign_cell_id_legacy": "none", "object_role": "baseline_none",
                                    "fit_target_kind": "none", "artifact_sha256": "none", "status": "approved",
                                    "notes": "the unsteered object: no intervention applied (spec object_id convention `none`)", **_OBJECT_NOT_FIT}
    out.append((none_payload, {"seed_owner": f"{SCRIPT} (D55-3)", "campaign_note": _OBJECT_CAMPAIGN_NOTE}, "object:none"))
    for rec in si_records:
        p = rec["payload"]
        si_id = str(p["si_id"])
        sha = str(p["si_text_sha256"])
        oid = identity.object_id("system_instruction", si_text_sha256=sha)
        label = labels.get(si_id)
        payload: Dict[str, Any] = {"object_id": oid, "object_kind": "system_instruction", "paper_label": label or "other",
                                   "campaign": OBJECT_SEED_CAMPAIGN, "campaign_cell_id_legacy": f"prompted__{si_id}", "object_role": "prompt",
                                   "fit_target_kind": "declared_prompt", "si_text_sha256": sha, "artifact_sha256": sha,
                                   "status": "approved" if label else "reference",
                                   "notes": (f"the prompted object declaring si_text {si_id} as the system instruction"
                                             + ("" if label else "; no paper cell declares it (status reference)")), **_OBJECT_NOT_FIT}
        extra = {"seed_owner": f"{SCRIPT} (D55-3)", "si_id": si_id, "si_kind": p.get("si_kind"), "campaign_note": _OBJECT_CAMPAIGN_NOTE,
                 "si_token_count_note": "per-model counts live on the si_text row (si_token_count_by_model_json); no single count is copied"}
        if label:
            extra["paper_label_source"] = f"legacy_spellings.yaml paper_labels (object_role prompt, si_id {si_id})"
        out.append((payload, extra, f"object:{oid}"))
    return out


# --------------------------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------------------------


@dataclass
class SeedBundle:
    records: Dict[str, List[Dict[str, Any]]]          # record_type -> records
    judge_ids: Dict[str, str]                         # readable seed id -> hashed judge_id
    instrument_ids: Dict[str, List[str]]              # readable seed id -> hashed instrument ids
    policies: PolicyIndex
    revision_id: str
    warnings: List[str] = field(default_factory=list)

    def all_records(self) -> List[Dict[str, Any]]:
        return [r for t in TABLE_ORDER for r in self.records.get(t, [])]

    def counts(self) -> Dict[str, int]:
        return {t: len(self.records.get(t, [])) for t in TABLE_ORDER}


def _record(record_type: str, payload: Dict[str, Any], extra: Dict[str, Any], producer: Mapping[str, Any],
            written_utc: str, seed_file: str, row_key: str) -> Dict[str, Any]:
    p = dict(payload)
    p["extra"] = extra
    rec = R.make_record(record_type, p, producer, identity.adapter_record_id(p, record_type), written_utc)
    v = VAL.validate_record(rec, f"{seed_file}:{row_key}")
    if isinstance(v, VAL.Malformed):
        raise SeedRefusal(seed_file, row_key, v.field, v.reason)
    return rec


def build_records(seeds: Seeds, producer: Mapping[str, Any], now: Optional[str] = None, verify_si: bool = True) -> SeedBundle:
    """Every dimension record from the seeds, validated (a refusal names file, row and field)."""
    now = now or R.utc_now()
    base_ms = R.utc_to_epoch_ms(now)
    stamp = {t: R.epoch_ms_to_utc(base_ms + i) for i, t in enumerate(TABLE_ORDER)}
    recs: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TABLE_ORDER}
    bc = seeds.base_commit

    for row in seeds.models:
        key = str(row.get("model_id"))
        payload, extra = model_payload(row, Normaliser(SEED_FILES["model"], key, bc))
        recs["model"].append(_record("model", payload, extra, producer, stamp["model"], SEED_FILES["model"], key))

    si_norm = lambda table, payload, row, extra: Normaliser(SEED_FILES["si_text"], str(row.get("si_id")), bc)(table, payload, row, extra)
    si_unverifiable: Dict[str, str] = {}
    for rec in SI.records_for(seeds.si_texts, producer, stamp["si_text"], si_norm, verify=verify_si, unverifiable=si_unverifiable):
        v = VAL.validate_record(rec, f"{SEED_FILES['si_text']}:{rec['payload'].get('si_id')}")
        if isinstance(v, VAL.Malformed):
            raise SeedRefusal(SEED_FILES["si_text"], str(rec["payload"].get("si_id")), v.field, v.reason)
        recs["si_text"].append(rec)
    for payload, extra, key in object_payloads(seeds, recs["si_text"]):
        recs["object"].append(_record("object", payload, extra, producer, stamp["object"], SEED_FILES["si_text"], key))

    for row in seeds.evals:
        key = str(row.get("eval_id"))
        payload, extra = eval_payload(row, Normaliser(SEED_FILES["eval"], key, bc))
        recs["eval"].append(_record("eval", payload, extra, producer, stamp["eval"], SEED_FILES["eval"], key))

    for row in seeds.item_sets:
        key = str(row.get("item_set_id"))
        payload, extra = item_set_payload(row, Normaliser(SEED_FILES["item_set"], key, bc))
        recs["item_set"].append(_record("item_set", payload, extra, producer, stamp["item_set"], SEED_FILES["item_set"], key))

    judge_ids: Dict[str, str] = {}
    for row in seeds.judges:
        key = str(row.get("judge_id"))
        payload, extra, seed_id = judge_payload(row, Normaliser(SEED_FILES["judge"], key, bc))
        if payload["judge_id"] in judge_ids.values():
            other = [k for k, v in judge_ids.items() if v == payload["judge_id"]]
            raise SeedRefusal(SEED_FILES["judge"], key, "judge_id", f"identity collision with {other}: same model, rubric sha, protocol sha and threshold rule")
        judge_ids[seed_id] = payload["judge_id"]
        recs["judge"].append(_record("judge", payload, extra, producer, stamp["judge"], SEED_FILES["judge"], key))

    idx = decode_policy_payloads(seeds)
    for pid, (payload, extra) in idx.rows.items():
        recs["decode_policy"].append(_record("decode_policy", payload, extra, producer, stamp["decode_policy"], SEED_FILES["decode_policy"], pid))
    for payload, extra in sampling_mode_payloads(seeds, idx):
        key = f"{payload['model_id']}|{payload['sampling_mode']}"
        recs["sampling_mode_policy"].append(_record("sampling_mode_policy", payload, extra, producer, stamp["sampling_mode_policy"],
                                                    SEED_FILES["decode_policy"], key))

    instrument_ids: Dict[str, List[str]] = {}
    seen_instruments: Dict[str, Tuple[str, Dict[str, Any], Dict[str, Any]]] = {}   # iid -> (seed_id, payload, extra)
    for row in seeds.instruments:
        seed_id = str(row.get("instrument_id"))
        for payload, extra in instrument_payloads(row, seeds, idx, judge_ids):
            iid = str(payload["instrument_id"])
            if iid in seen_instruments:
                other_seed, _, other_extra = seen_instruments[iid]
                if other_seed != seed_id:
                    raise SeedRefusal(SEED_FILES["instrument"], seed_id, "instrument_id",
                                      f"identity collision with {other_seed}[{payload['eval_id']}] (identical canonical fields)")
                # the same seed row expanded over several models states IDENTICAL decode numbers (a greedy or protocol
                # row): the instrument is model-independent — ONE row, every policy / model it stands for listed
                for k in ("decode_policy_id", "decode_policy_ref", "model_id"):
                    vals = other_extra.setdefault(k + "s", [other_extra.get(k)] if other_extra.get(k) else [])
                    if extra.get(k) and extra[k] not in vals:
                        vals.append(extra[k])
                    other_extra.pop(k, None)
                continue
            seen_instruments[iid] = (seed_id, payload, extra)
            instrument_ids.setdefault(seed_id, []).append(iid)
    for iid, (seed_id, payload, extra) in seen_instruments.items():
        # one spelling for "which served models does this row stand for": `model_ids` is ALWAYS a list beside a
        # singleton `model_id`; the merged model-independent rows already carry the lists and no singular
        for k in ("model_id", "decode_policy_id", "decode_policy_ref"):
            if k in extra and k + "s" not in extra:
                extra[k + "s"] = [extra[k]]
        recs["instrument"].append(_record("instrument", payload, extra, producer, stamp["instrument"], SEED_FILES["instrument"], seed_id))

    warnings = [f"{pid}: {col} reference {text!r} names no seeded policy (left NULL; extra_json.superseded_by_unresolved)"
                for pid, col, text in idx.unresolved]
    warnings += [f"si_text {sid}: sha NOT re-verified on this host ({why}); loaded with the seeded value, field_confidence_json says so"
                 for sid, why in sorted(si_unverifiable.items())]
    warnings += [f"no default decode policy for {key} (every seeded row is explicitly non-default; resolve() refuses this key)"
                 for key in idx.no_default_keys]
    return SeedBundle(recs, judge_ids, instrument_ids, idx, seeds.revision_id(), warnings)


def adapter_legacy_spellings(seeds_dir: str = DEFAULT_SEEDS_DIR, seeds: Optional[Seeds] = None) -> Dict[str, Dict[str, str]]:
    """The ``AdapterContext.legacy_spellings`` the source adapters read: ``model_ids`` — every legacy model
    spelling (HF repo, lane key, producer tag) -> ``model_id`` (:func:`legacy_model_map`); ``si_texts`` — ``si_id`` ->
    ``si_text_sha256`` for every seeded text. Built from the seeds, never typed twice."""
    seeds = seeds or load_seeds(seeds_dir)
    return {"model_ids": legacy_model_map(seeds),
            "si_texts": {str(r["si_id"]): str(r["si_text_sha256"]) for r in seeds.si_texts if r.get("si_id") and r.get("si_text_sha256")}}


def build_seed_bundle(root: str, seeds_dir: str = DEFAULT_SEEDS_DIR, producer: Optional[Mapping[str, Any]] = None,
                      now: Optional[str] = None, verify_si: bool = True) -> SeedBundle:
    """The disk-bound half of a load and nothing else: read the seed YAMLs (and, with ``verify_si``, the SI texts), build and
    validate every record. Takes no lock and writes nothing, so the harvester lock a load takes afterwards covers only the
    publication and the fold (``cli.cmd_dimensions``)."""
    from .. import harvest as H
    seeds = load_seeds(seeds_dir)
    prod = dict(producer or H.harvester_producer(root))
    prod["script"] = SCRIPT
    return build_records(seeds, prod, now, verify_si=verify_si)


def seed_revision_dir(root: str, revision_id: str) -> str:
    from .. import harvest as H
    return os.path.join(H.records_dir(root), ADAPTER_NAME, revision_id)


def publish_seed_records(root: str, bundle: SeedBundle, force: bool = False) -> List[str]:
    """Publish a built bundle under ``<root>/records/dimensions/<revision>/`` as ONE directory: the nine ``<seq>-<table>.jsonl``
    files are written to a staging directory outside ``records/`` (``<root>/staging/dimensions/``) and renamed into place in
    one step, so a harvester cycle that lists ``records/`` meanwhile finds the whole revision or nothing — never five of its
    nine files (the harvester then applies the directory in one transaction, ``harvest.Ingestor.apply_dimension_bundle``). A
    revision already on the log is left alone and ``[]`` returned; with ``force`` its records are APPENDED to the existing
    files (the log is append-only: a file already mirrored is never rewritten; a file gzipped after ingest is appended as a
    further gzip member), file by file — the caller holds the harvester lock across a forced publication so no cycle reads
    the half-appended directory."""
    out_dir = seed_revision_dir(root, bundle.revision_id)
    if os.path.isdir(out_dir):
        if not force:
            return []
        paths = []
        for i, t in enumerate(TABLE_ORDER, start=1):
            if bundle.records.get(t):
                path = os.path.join(out_dir, R.record_file_name(i, t))
                if not os.path.exists(path) and os.path.exists(path + ".gz"):
                    path += ".gz"
                paths.append(R.write_records(path, bundle.records[t]))
        return paths
    staging_root = os.path.join(root, "staging", ADAPTER_NAME)
    staged = os.path.join(staging_root, f"{bundle.revision_id}-{os.getpid()}-{R.utc_now().replace(':', '').replace('-', '').replace('.', '')}")
    if os.path.isdir(staged):
        shutil.rmtree(staged)
    names = []
    for i, t in enumerate(TABLE_ORDER, start=1):
        if bundle.records.get(t):
            names.append(os.path.basename(R.write_records(os.path.join(staged, R.record_file_name(i, t)), bundle.records[t])))
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    try:
        os.rename(staged, out_dir)          # atomic on one filesystem; fails (rather than merges) when the directory appeared meanwhile
    except OSError:
        shutil.rmtree(staged, ignore_errors=True)
        _rmdir_if_empty(staging_root)
        if os.path.isdir(out_dir):
            return []                        # a concurrent load published the same revision: the log has it once
        raise
    _rmdir_if_empty(staging_root)
    return [os.path.join(out_dir, n) for n in names]


def _rmdir_if_empty(path: str) -> None:
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass


def write_seed_records(root: str, seeds_dir: str = DEFAULT_SEEDS_DIR, producer: Optional[Mapping[str, Any]] = None,
                       now: Optional[str] = None, verify_si: bool = True, force: bool = False) -> Tuple[SeedBundle, List[str]]:
    """Build the records and publish them under ``<root>/records/dimensions/<revision>/`` (:func:`build_seed_bundle` then
    :func:`publish_seed_records`). The revision is the sha16 over the seed files' shas, so re-loading UNCHANGED seeds writes
    nothing (idempotent); changed seeds get a new revision directory whose records fold over the old rows."""
    bundle = build_seed_bundle(root, seeds_dir, producer, now, verify_si=verify_si)
    return bundle, publish_seed_records(root, bundle, force=force)
