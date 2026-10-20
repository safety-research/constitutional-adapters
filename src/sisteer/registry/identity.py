"""Run, judgment, computation and object identities — STDLIB ONLY, shared by adapters AND emitters.

Why a producer's own key is not a run: Petri run keys span several behaviour
evals, a grades root spans several attacks and holds up to nine objects under one eval + instrument,
one controls ``CellStore`` holds several evals. The registry grain is therefore one run per
(producer EXECUTION key, producer OBJECT key, eval, instrument)::

    run_id = f"{campaign}:{eval_id}:{instrument_id[:12]}:{sha16(producer_key + '|' + producer_object_key)}"

with the producer's keys kept verbatim in ``run.producer_run_key_legacy`` /
``run.producer_object_key_legacy``. Native emitters and adapters mint the SAME id for the same
resolved spec, so a store and its run are joinable without a lookup table.

Importable on a compute node before ``uv sync``: ``python3 -S -c "import sisteer.registry.identity"``.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

#: uuid5 namespace for deterministic adapter record ids (C24).
RECORD_NAMESPACE = uuid.UUID("6f1b6d2e-3c4a-4b5d-8e9f-0a1b2c3d4e5f")

PRODUCER_KEY_KINDS = ("cell_key", "grades_root", "petri_run_key", "bon_dir", "battery_block", "em_row", "opaque")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha16(text: str) -> str:
    """16 lowercase hex chars of sha256(text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def canonical_json(obj) -> str:
    """Compact, key-sorted JSON; the identity form for every hash in the registry."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require(name: str, value: Optional[str]) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"identity: {name} must be a non-empty string, got {value!r}")
    if ":" in value and name in ("campaign", "eval_id"):
        raise ValueError(f"identity: {name} may not contain ':' (it is a run_id separator): {value!r}")
    return value


def run_id(campaign: str, producer_key: str, producer_object_key: str, eval_id: str, instrument_id: str) -> str:
    """The registry-grain run identity. ``producer_object_key`` is ``'none'`` for an unsteered execution."""
    _require("campaign", campaign)
    _require("producer_key", producer_key)
    _require("producer_object_key", producer_object_key)
    _require("eval_id", eval_id)
    _require("instrument_id", instrument_id)
    return f"{campaign}:{eval_id}:{instrument_id[:12]}:{sha16(producer_key + '|' + producer_object_key)}"


def parse_run_id(rid: str) -> Dict[str, str]:
    parts = rid.split(":")
    if len(parts) != 4:
        raise ValueError(f"identity: not a registry run_id: {rid!r}")
    return {"campaign": parts[0], "eval_id": parts[1], "instrument_prefix": parts[2], "producer_sha16": parts[3]}


def run_dir_name(rid: str) -> str:
    """Filesystem-safe directory name for a run's record files."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", rid)


def judgment_id(run: str, judge: str, session: Optional[str] = None, claimed_utc: Optional[str] = None) -> str:
    """Deterministic judgment identity: uuid5 over run|judge|session|claimed_utc."""
    return str(uuid.uuid5(RECORD_NAMESPACE, "|".join([run, judge, session or "", claimed_utc or ""])))


#: The forms a judgment id is minted in, as ``(form name, session field, time field)`` over the record's OWN
#: payload: the plain form (every adapter: 107,494 of the 108,355 judgment records in the log on 2026-10-03), the claim form
#: (``pending.claim`` and ``hooks.JudgingScope``: ``claimed_by_session`` + ``claimed_utc``, 831) and the producer form
#: (``em_misalign``'s anchor re-judge: ``producer_session`` with no ``claimed_utc``, 12). Every minter binds the id to
#: (``run_id``, ``judge_id``) through :func:`judgment_id`, so :func:`judgment_id_form` is a pure check with no store read, and
#: the validator refuses a ``judgment`` record no form reproduces (``validate.validate_record``): a same-id record stating another
#: ``run_id`` or ``judge_id`` never folds, which makes the two identity columns immutable by construction -- the successor
#: relationship ``registry judgment supersede`` reads at fold time (``reducer``) is then the same on live ingest and on replay.
#: Adding a form is a validator change with a test; the whole-log audit (0 ``run_id`` / ``judge_id`` changes over 11,340 ids)
#: is re-run before every landing.
JUDGMENT_ID_FORMS: Tuple[Tuple[str, Optional[str], Optional[str]], ...] = (
    ("run_judge", None, None),
    ("claimed_by_session", "claimed_by_session", "claimed_utc"),
    ("producer_session", "producer_session", "claimed_utc"),
)


def _form_value(payload: Mapping[str, object], field_name: Optional[str]) -> Optional[str]:
    if field_name is None:
        return None
    value = payload.get(field_name)
    return None if value is None else str(value)


def judgment_id_form(payload: Mapping[str, object]) -> Optional[str]:
    """The name of the FIRST form in :data:`JUDGMENT_ID_FORMS` whose :func:`judgment_id` over the payload's own fields equals its
    ``judgment_id``, else None. An absent or null session / time field hashes as the empty string, exactly as a minter that passed
    None did, so a payload with neither matches the plain form and one with a session alone matches the session-only spelling."""
    jid, run, judge = payload.get("judgment_id"), payload.get("run_id"), payload.get("judge_id")
    if not isinstance(jid, str) or not isinstance(run, str) or not isinstance(judge, str):
        return None
    for name, session_field, time_field in JUDGMENT_ID_FORMS:
        if judgment_id(run, judge, _form_value(payload, session_field), _form_value(payload, time_field)) == jid:
            return name
    return None


def judgment_identity_inputs(payload: Mapping[str, object]) -> Optional[Dict[str, object]]:
    """The payload fields a record must carry to reproduce ``payload``'s judgment id under the form that mints it: ``{}`` for the
    plain form, else the form's session and time fields copied verbatim (``{claimed_by_session, claimed_utc}`` or
    ``{producer_session, claimed_utc}``, a null kept as null); None when no form reproduces the id. What a minimal record about an
    existing judgment (a supersede) must state beside the four required fields so the identity rule admits it."""
    form = judgment_id_form(payload)
    if form is None:
        return None
    _, session_field, time_field = next(f for f in JUDGMENT_ID_FORMS if f[0] == form)
    return {k: payload.get(k) for k in (session_field, time_field) if k is not None}


def computation_id(algorithm: str, version: str, seed_provenance: str) -> str:
    """sha16 over (algorithm, version, seed provenance) — joins the run_metric / run_contrast keys (F9)."""
    return sha16(canonical_json([algorithm, version, seed_provenance]))


def object_id(kind: str, artifact_sha256: Optional[str] = None, si_text_sha256: Optional[str] = None) -> str:
    """``none`` | ``si:<sha256 of the instruction text>`` | the artifact's sha256."""
    if kind == "none":
        return "none"
    if kind == "system_instruction":
        if not si_text_sha256 or len(si_text_sha256) != 64:
            raise ValueError("identity.object_id: a system_instruction object needs its text sha256")
        return f"si:{si_text_sha256}"
    if not artifact_sha256 or len(artifact_sha256) != 64:
        raise ValueError(f"identity.object_id: object kind {kind!r} needs its artifact sha256")
    return artifact_sha256


#: the one object-id prefix that names its kind: :func:`object_id` spells a system-instruction object ``si:<sha256 of the text>``
SI_OBJECT_ID_PREFIX = "si:"


def object_kind_from_id(object_id: object) -> Optional[str]:
    """The object kind an id states by its own spelling (the inverse of :func:`object_id` where one exists): ``none`` for the unsteered
    anchor, ``system_instruction`` for an ``si:<sha>`` id, None for every other id -- a staged artifact's sha names no kind, which is
    on its ``object`` row. What a reader that cannot see the object table (a job-start plan, a served runs file entry) hands the
    rules that read the kind for these two values alone: a prompted object steers no positions, and neither does the anchor."""
    if not isinstance(object_id, str):
        return None
    if object_id == "none":
        return "none"
    if object_id.startswith(SI_OBJECT_ID_PREFIX):
        return "system_instruction"
    return None


#: The canonical fields of an INSTRUMENT identity (EVERY identity column of the `instrument` table — everything but
#: the id itself, notes, extra_json, field_confidence_json and the harvester tail), each with its storage type so two
#: spellings of one value hash alike (0 == 0.0, 1.0 == 1, "[512]" == [512]). sha256 over the canonical JSON (sorted
#: keys, None for an absent field). The seeds loader AND every adapter mint ids through :func:`instrument_id`, so
#: `v_paper_long`'s joins meet. tests/test_registry_core_fixes.py pins this list to the table.
INSTRUMENT_ID_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("eval_id", "TEXT"), ("item_set_id", "TEXT"), ("harness", "TEXT"), ("harness_version", "TEXT"), ("harness_config_hash", "TEXT"),
    ("n_shot", "INTEGER"), ("prompt_template_sha256", "TEXT"), ("cot_prompting", "TEXT"), ("scoring_rule", "TEXT"),
    ("answer_extraction", "TEXT"), ("token_cap_tokens", "INTEGER"), ("token_cap_ladder_json", "JSON"), ("ladder_promotion_rule", "TEXT"),
    ("read_at", "TEXT"), ("reply_cap_tokens", "INTEGER"), ("context_window_tokens", "INTEGER"), ("decode_source", "TEXT"),
    ("decode_temperature", "REAL"), ("decode_top_p", "REAL"), ("decode_top_k", "INTEGER"), ("decode_min_p", "REAL"),
    ("decode_repetition_penalty", "REAL"), ("decode_presence_penalty", "REAL"), ("decode_seed_rule", "TEXT"), ("thinking_mode", "TEXT"),
    ("n_trials", "INTEGER"), ("bon_n", "INTEGER"), ("max_turns", "INTEGER"), ("serving_path", "TEXT"), ("applied_positions_mode", "TEXT"),
    ("attacker_model_id", "TEXT"), ("attacker_decode_json", "JSON"), ("judge_id", "TEXT"), ("prefill_scheme", "TEXT"),
    # migration 0012: prompt padding is INSTRUMENT identity -- a padded static store is another instrument, never a run-level
    # pad column defaulting to 0 (the padded stores differ in cap and window too, so one axis would read the pad as the only difference)
    ("padding_tokens", "INTEGER"), ("padding_identity_sha256", "TEXT"),
)
#: The identity fields that contribute NOTHING when NULL: they are dropped from the canonical JSON before hashing, so every
#: instrument minted before they existed (every live row on 2026-09-22, all unpadded) keeps its id byte for byte -- an id is a join
#: key on every run, judgment and roster entry, and the two new fields must not move it (tests/test_registry_instrument_padding_identity.py
#: proves the invariance on the fixture rows and, read-only, on every live row). A NON-NULL value still discriminates: two instruments
#: that differ only in the pad, or only in the filler contract, hash apart.
INSTRUMENT_ID_OPTIONAL_FIELDS: frozenset = frozenset({"padding_tokens", "padding_identity_sha256"})
#: The keys of a static-padding block that make the FILLER CONTRACT (`sisteer.eval.attacks.static_padding.BLOCK_KEYS` minus the level
#: `tokens`, the schema label and the corpus name): what `padding_identity_sha256` hashes, sorted, as canonical JSON. The level rides
#: `padding_tokens` beside it, so two levels of one contract are two instruments that share the contract hash.
PADDING_IDENTITY_KEYS: Tuple[str, ...] = ("filler_sha256", "mode", "placement", "separator", "tolerance")
_INSTRUMENT_FIELD_TYPES: Dict[str, str] = dict(INSTRUMENT_ID_FIELDS)
#: Shorter spellings a policy / seed row may carry for the same setting.
_INSTRUMENT_FIELD_ALIASES: Dict[str, str] = {
    "temperature": "decode_temperature", "top_p": "decode_top_p", "top_k": "decode_top_k", "min_p": "decode_min_p",
    "repetition_penalty": "decode_repetition_penalty", "presence_penalty": "decode_presence_penalty",
    "token_cap_ladder": "token_cap_ladder_json",
}
#: The canonical fields of a JUDGE identity.
JUDGE_ID_FIELDS: Tuple[str, ...] = ("judge_model", "rubric_sha256", "protocol_sha256", "threshold_rule")


def _typed(name: str, ctype: str, value):
    """Normalise one identity value to its storage type before hashing: REAL → float, INTEGER → int
    (an integral float or a bool accepted), TEXT → str, JSON → the parsed value (a JSON string is decoded); None
    stays None; a value that cannot be read as its type is an error, never silently coerced."""
    if value is None:
        return None
    if ctype == "REAL":
        return float(value)
    if ctype == "INTEGER":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"identity: {name} must be an integer, got {value!r}")
            return int(value)
        return int(value)
    if ctype == "JSON":
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value
    return str(value)


def instrument_identity_fields(fields: Mapping[str, object]) -> Dict[str, object]:
    """The canonical instrument identity as a dict: every INSTRUMENT_ID_FIELDS name (None when absent), alias
    spellings folded onto the column names, every value normalised to its type."""
    seen: Dict[str, Tuple[str, object]] = {}          # canonical name -> (spelling seen, typed value)
    for k, v in fields.items():
        name = _INSTRUMENT_FIELD_ALIASES.get(str(k), str(k))
        if name not in _INSTRUMENT_FIELD_TYPES:
            continue
        tv = _typed(name, _INSTRUMENT_FIELD_TYPES[name], v)
        if tv is None:
            continue
        prev = seen.get(name)
        if prev is not None and prev[1] != tv:
            # an alias and the canonical name (or two aliases) naming DIFFERENT values is not a spelling — refuse,
            # whatever the mapping order; equal values after typed normalisation are one value
            raise ValueError(f"identity.instrument_id: {prev[0]!r} and {str(k)!r} both set {name} with different values "
                             f"({prev[1]!r} vs {tv!r})")
        if prev is None or str(k) == name:
            seen[name] = (str(k), tv)                      # the canonical spelling is the one recorded, deterministically
    return {name: (seen[name][1] if name in seen else _typed(name, ctype, None)) for name, ctype in INSTRUMENT_ID_FIELDS}


def instrument_id(fields: Mapping[str, object]) -> str:
    """sha256 hex over the canonical instrument identity: same fields → same id; any identity field
    differing → a different id; an `instrument` row, a seed row and the alias spellings mint the same id. A field in
    :data:`INSTRUMENT_ID_OPTIONAL_FIELDS` is omitted while NULL, so an unpadded instrument hashes exactly as it
    did before the padding fields existed."""
    ident = instrument_identity_fields(fields)
    for name in INSTRUMENT_ID_OPTIONAL_FIELDS:
        if ident.get(name) is None:
            ident.pop(name, None)
    return sha256_hex(canonical_json(ident).encode("utf-8"))


def padding_identity_sha256(block: Mapping[str, object]) -> str:
    """sha256 hex over the canonical JSON of the block's FILLER CONTRACT (:data:`PADDING_IDENTITY_KEYS`): the same
    corpus, placement, mode, separator and tolerance at two pad levels share this hash and differ in `padding_tokens`.
    Every key must be present and non-null -- a partial block is refused, never hashed with a hole (a hole would mint a
    contract nobody declared)."""
    missing = [k for k in PADDING_IDENTITY_KEYS if block.get(k) is None]
    if missing:
        raise ValueError(f"identity.padding_identity_sha256: the padding block lacks {missing}; every contract key is required")
    return sha256_hex(canonical_json({k: block[k] for k in PADDING_IDENTITY_KEYS}).encode("utf-8"))


def judge_id(judge_model, rubric_sha256: Optional[str] = None, protocol_sha256: Optional[str] = None,
             threshold_rule: Optional[str] = None) -> str:
    """sha256 hex over (judge_model, rubric_sha256, protocol_sha256, threshold_rule) as canonical JSON. Takes the
    four fields positionally, or one mapping carrying them (a `judge` row)."""
    if isinstance(judge_model, Mapping):
        fields = {k: judge_model.get(k) for k in JUDGE_ID_FIELDS}
    else:
        fields = {"judge_model": judge_model, "rubric_sha256": rubric_sha256, "protocol_sha256": protocol_sha256,
                  "threshold_rule": threshold_rule}
    return sha256_hex(canonical_json(fields).encode("utf-8"))


def adapter_record_id(payload: Mapping[str, object], record_type: str) -> str:
    """Deterministic uuid5 record id for adapter-produced records (idempotent backfill, C24)."""
    return str(uuid.uuid5(RECORD_NAMESPACE, record_type + "|" + canonical_json(payload)))


def producer_record_id() -> str:
    """uuid4 for producer-side records (the contract's default)."""
    return str(uuid.uuid4())


# --------------------------------------------------------------------------------------------
# Producer key grammars
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProducerKey:
    """A producer's own key split into its EXECUTION key and its OBJECT discriminator."""
    kind: str
    execution_key: str
    object_key: str
    parts: Dict[str, str] = field(default_factory=dict)


_STORE_KEY_RE = re.compile(
    r"^(?P<object>[^_/][^/]*?)__(?P<tier_or_task>[^/]+?)__t(?P<trial>\d+)__a(?P<attempt>\d+)__(?P<intervention8>[0-9a-f]{8})__(?P<harness8>[0-9a-f]{8})$")
_GRADES_ROOT_RE = re.compile(r"^grades_(?P<tag>[A-Za-z0-9]+)_(?P<cluster>ch|bu|md|dv|rp)_(?P<model_transfer>[A-Za-z0-9_]+)$")
_BATTERY_BLOCK_RE = re.compile(r"^(?P<model_transfer>[^:]+):(?P<object>[^:]+)$")


def split_producer_key(kind: str, key: str, object_key: Optional[str] = None,
                       scope: Optional[Mapping[str, str]] = None) -> ProducerKey:
    """Split a producer's key into (execution key, object key) for one of the known grammars.

    ``cell_key`` — ``<object>__<tier_or_task>__t<trial>__a<attempt>__<intervention8>__<harness8>``
    (``capeval.store.cell_key``): the object key is ``<object>@<intervention8>``; a controls store
    yields one run per control present in its rows, so the caller passes the control's object
    label as ``object_key`` to override.

    ``grades_root`` — ``grades_<tag>_<cluster>_<model_transfer>``: one run per (aggregate file,
    object entry); the caller passes the aggregate's object entry key as ``object_key`` (``'none'``
    for the unsteered entry).

    ``petri_run_key`` — one run per (behaviour eval, stats-block object id) plus the pooled eval;
    the object key repeats across tags, so the tag (``scope['tag']``) joins it.

    ``bon_dir`` — one run per (rung / pass, object): ``object_key`` is the store's object label.

    ``battery_block`` — ``'<model_transfer>:<object>'`` (strength battery): object key from the block.

    ``em_row`` — an EM panel row: ``object_key`` is the panel's object label; ``scope['target']``
    and ``scope['behaviour']`` are recorded.

    ``opaque`` — any other producer key; ``object_key`` is required.
    """
    if kind not in PRODUCER_KEY_KINDS:
        raise ValueError(f"identity.split_producer_key: unknown grammar {kind!r}; known: {PRODUCER_KEY_KINDS}")
    _require("key", key)
    scope = dict(scope or {})
    if kind == "cell_key":
        m = _STORE_KEY_RE.match(key)
        if not m:
            raise ValueError(f"identity.split_producer_key: {key!r} does not match the cell_key grammar")
        parts = m.groupdict()
        obj = object_key or f"{parts['object']}@{parts['intervention8']}"
        return ProducerKey(kind, key, obj, parts)
    if kind == "grades_root":
        m = _GRADES_ROOT_RE.match(key)
        if not m:
            raise ValueError(f"identity.split_producer_key: {key!r} does not match the grades_root grammar")
        if not object_key:
            raise ValueError("identity.split_producer_key: a grades root needs the aggregate's object entry as object_key")
        return ProducerKey(kind, key, object_key, m.groupdict())
    if kind == "petri_run_key":
        if not object_key:
            raise ValueError("identity.split_producer_key: a Petri run key needs the stats block's object id as object_key")
        tag = scope.get("tag")
        obj = f"{tag}:{object_key}" if tag else object_key
        return ProducerKey(kind, key, obj, {"tag": tag or ""})
    if kind == "bon_dir":
        if not object_key:
            raise ValueError("identity.split_producer_key: a BoN store needs its object label as object_key")
        return ProducerKey(kind, key, object_key, {"rung": scope.get("rung", ""), "pass_label": scope.get("pass_label", "")})
    if kind == "battery_block":
        m = _BATTERY_BLOCK_RE.match(key)
        if not m:
            raise ValueError(f"identity.split_producer_key: {key!r} does not match the battery_block grammar")
        parts = m.groupdict()
        return ProducerKey(kind, key, object_key or parts["object"], parts)
    if kind == "em_row":
        if not object_key:
            raise ValueError("identity.split_producer_key: an EM panel row needs its object label as object_key")
        return ProducerKey(kind, key, object_key, {"target": scope.get("target", ""), "behaviour": scope.get("behaviour", "")})
    if not object_key:
        raise ValueError("identity.split_producer_key: an opaque key needs an explicit object_key")
    return ProducerKey(kind, key, object_key, {})


def run_ids_for_producer_key(campaign: str, pk: ProducerKey, evals: Iterable[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
    """All run ids one producer key yields: {(eval_id, instrument_id): run_id}."""
    return {(e, i): run_id(campaign, pk.execution_key, pk.object_key, e, i) for e, i in evals}
