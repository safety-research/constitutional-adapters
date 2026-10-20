"""Record validator: ``validate_record(rec, path) -> Validated | Malformed``.

Malformed (refused, listed on the status page, never silently skipped) vs dependency-blocked
(well-formed, parent row not yet ingested — queued and re-evaluated; F4) are DIFFERENT outcomes:
this module decides the first; :func:`parent_references` names the rows the harvester checks for the second.

Steps: envelope → required fields per type → the strictness rule (an eval named without its
``item_set_id`` / ``instrument_id`` is refused naming the field, never defaulted) → vocabulary per
controlled field (same tuples as the DDL CHECKs) → ``unverified`` only with a reason in
``field_confidence_json`` → secret-shaped keys refused → unknown keys parked in ``extra_json``,
never dropped → ``run_patch`` field-set rules → item-scale whole-row rule → the token-cap ladder facts of a
``run`` record agree with each other, read as the producer stated them (:func:`run_cap_ladder_refusal`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from . import identity
from . import records as R
from . import schema_def as S
from . import site_aliases as SA
from . import vocab as V

SECRET_KEY_RE = re.compile(
    r"(?i)(?:(^|_)(api_?key|apikey|secret|password|passwd|bearer|private_key|"
    r"(auth|access|hf|wandb|session|refresh|oauth|service)_?token)(_|$)|^token$)")


#: Record shas (``records.record_sha``) the validator admits although a rule added AFTER they were written refuses their shape. The
#: log is append-only and a rebuild must fold it to the live tables byte for byte, so a new rule with historical hits names them
#: here instead of bending for every future record; the list is re-measured by the whole-log audit before every landing and
#: grows only by audit. Today, the judgment identity rule (``identity.judgment_id_form``): the 18
#: ``abandoned`` release records ``sisteer.registry.pending`` wrote under ``records/claims/`` before ``release`` and the forced
#: ``claim`` learned to re-state the claim's own ``claimed_utc`` -- each stamps ``claimed_utc`` with the release time while its id was
#: minted over the claim's, so no form reproduces it (the plan's audit of 2026-10-02 found 4; the re-run before this commit, 2026-10-03
#: 03:4xZ, 18 -- the live base checkout kept filing them, as the plan's risk table says). The token-cap ladder rule
#: (:func:`run_cap_ladder_refusal`) adds nothing: 0 of the 107,778 ``run`` records in the log on 2026-10-03 fail it. The
#: placeholder-code_sha rule (:func:`run_code_sha_refusal`) adds the FOUR ``run`` records of the two synthetic test runs a
#: registration test folded into the live registry on 2026-10-02 (``code_sha`` forty ``a``s; both runs since withdrawn by
#: owner-filed ``run_state`` records, the files kept, the log immutable) -- the whole-log audit re-run before the commit (2026-10-03
#: 12:5xZ, read-only: 7,059 ``-run`` files, 109,088 ``run`` records) finds exactly those four and no other. The whole-log audit before the
#: closing commit (``scripts/registry/audit_new_validator_rules.py``, 2026-10-04 05:1xZ and again before the review fixes) adds TWELVE
#: more judgment records to the identity rule's list: ``superseded`` and ``complete`` records written 2026-10-03 13:53-14:01Z about eight
#: ``sdfbase_full22`` agentic / strongreject_bon runs by the judging drivers' supersession path (``worklist.superseded_judgment_payloads``
#: and ``own_partial_judgments`` as they stand on the ``stage-e-run`` branch at d9b9cf17, through ``scripts/registry/grade_daemon.py`` and
#: ``scripts/sdfbase/full22/misalign/misalign_judge.py``) and by the one-off correction tool ``impl/tools/g39_corrections.py`` (committed
#: nowhere): each re-states a CLAIM-minted judgment id with ``run_id`` and ``judge_id`` alone and none of the claim's identity inputs
#: (``claimed_by_session`` + ``claimed_utc``, both on the folded row), which no form reproduces. They folded live (the base checkout has
#: no identity rule) and the live rows stand; the writers' repair -- copy ``identity.judgment_identity_inputs(row)`` into every record
#: about an existing judgment, as ``supersede.assemble`` and ``pending.release`` do -- lives with the code that writes them
#: (``scripts/registry/audit_new_validator_rules.py`` audits the whole log against every rule). A sha here is admitted by EVERY rule, so an
#: entry is a measured historical record, never a producer's escape hatch.
GRANDFATHERED_RECORD_SHAS: frozenset = frozenset({
    "0028e3f5b5fc68e7335bfe3d9d946bc93de5f202aafb51a8fdcf3db766128309",   # placeholder code_sha run record
    "039c52a6687d1d644da51acbb87ffd2f8af5b0e0289f59082094d738ae6dde0e",
    "138635c792b7d30eb46f604a55fec2fc3a9134d939e69363df0cf4dedd4582d6",
    "24e766b4149939b168ef6ebce7a2d6eda1f3e84eba64056d086a5993c91fe111",
    "37cf9e48d1ca456c4240d55e470082f65ece8ab3f6b82c31ed1d998399698ca6",
    "3cd031c66deb990f5ab4c7f059f3731488df2b3f589feb798d0a626a2679ae7c",
    "3f0dda69e3c8c1d8ce3f70dbeeeca7e43685d6de50d86e3602c3c8dba0512b58",
    "48b18001f33e35ec0f7e292fd2461e6d4b83debd91c484630a47ee508e133e93",
    "4dc25cca98b4a01b2eedbd49aec0a51e0acf225529650f300ca9c9269a7c6d4d",   # placeholder code_sha run record
    "842fd4384e0bb767f5c8cab9a66740f8496ac2f058ae2efe778479bffb5ee015",
    "91cd32788200cf0ff5b5b21f7da3bb8630c31828f5474a5312f13ae23389f60c",
    "93136b2e6c27fafb958118771c32db9718ed053d04b5187aac4b6994b232f1a9",
    "9417c9a3b106665119061a3607d91a9e5641aa105fcb619a93e0970c45611b50",
    "98fe6d0d3da73bc9b918911dd20fb33ec76ccb16526399b87ce1bd8031f068f4",   # placeholder code_sha run record
    "9aee2757f926a59bd195e38928358b2c4dddb82c49df17d6c61be3789a7a4d72",
    "9f1d76a0108474497a56f8f1bbba1f221083ef106710fafb39c7d523cc7efdbe",
    "af7a5e6f51ab16599d6dc7c861fab9487b9ec768e1a728c75fef5620fd4ce855",
    "b59823cdd3751a787992ac24ecd948a5b9196a46bf1a704be80dc06fb98d15b3",   # placeholder code_sha run record
    "bb33c7c0ef9e3be73777b9941ef82b591a6713fbc02792f58c034dfa2c0e5232",
    "cf0de023e574d4e2fcf6789152e0c3fbae628936bc4377bf624fd1a9b07f67a5",
    "dcf4e0b222efbbfc9875b01b809584cea8ad7540b1fa5aa1e5eec940a867c606",
    "ea6b6f024465350ccbc4b4afef2105fc6c0698f0463322cd4ccf21ea57dd18ee",
    # the twelve supersession / correction records about claim-minted judgment ids (audit of 2026-10-04)
    "348d2811d06e538f80fee9923bec2f3eef6c46424126904fea19bb70dace1790",   # superseded, grade_daemon.py
    "5564a9b8d9cfc9afab4ae014dad864ce6befb9e88b15c1adcbf71408bb4b63fe",   # complete, g39_corrections.py
    "5c1a0463eb2e91932db86f5220dcbd9ca625891400962d10a623811a270272e1",   # superseded, grade_daemon.py
    "7363618e1a23a877d860f87c15fe7922450642792b8ca435a104379b6a0f2d14",   # superseded, misalign_judge.py
    "7be197f39f8b018e3997dcf5ccadf1bbc2f5ee640ae797437c2435c629eaab06",   # superseded, grade_daemon.py
    "8672d82038a891d5ca16c097e586ef5bac97b0f0159fed5fb7269bb83ab21d2b",   # complete, g39_corrections.py
    "9418067e067a8274469eb866abbd0f93c3e00b329ea02d98845da34be325fd8d",   # superseded, grade_daemon.py
    "a08574e1a76862ba7b67494623fcd8e7ef7850b1bc1e0cac2a4a3c2bc9bced35",   # complete, g39_corrections.py
    "b160df32aeb74f4d5c056e6bacb52bb3096eecea48e4a3b4d115730e0c3e917c",   # superseded, grade_daemon.py
    "beba9dd22ad94bf21779a4afe44087da0f11751a9a94fc01d7c8ae928f0b5a59",   # complete, g39_corrections.py
    "c3023006c4c9ec2b9b4f7300a23ade29786d7d970823d5476d58dd2deca540f0",   # complete, g39_corrections.py
    "c96d4ea382f505ac886ff6b3f5a49df0e6a3d381e99a806dec1b66801b78afa0",   # superseded, misalign_judge.py
})


@dataclass
class Validated:
    record_type: str
    record_id: str
    written_utc: str
    producer: Dict[str, object]
    payload: Dict[str, object]          # normalised: known columns only, JSON columns serialised
    extra: Dict[str, object]            # unknown keys, parked (-> extra_json)
    table: Optional[str]                # the row's table (None for effect types)
    sha: str = ""
    uri: str = ""

    @property
    def order_key(self) -> Tuple[str, int, str]:
        """The record's slot in the fold's order, ``(written_utc, tie_rank, record_sha256)`` -- the same key ``reducer.order_key``
        gives a history event and ``reducer.ORDER_SQL`` reads off ``row_history``: ``tie_rank`` is 1 for a ``run_state`` to a
        lateral/terminal state (``vocab.LIFECYCLE_LATERAL``), else 0. The harvester applies a transaction's run-scale records in
        this order (the fold itself is order-independent; the apply order and the fold's order agree by construction)."""
        rank = 1 if self.record_type == "run_state" and str(self.payload.get("state")) in V.LIFECYCLE_LATERAL else 0
        return (self.written_utc, rank, self.sha)


@dataclass
class Malformed:
    field: Optional[str]
    reason: str
    path: str
    record_type: Optional[str] = None
    sha: str = ""
    uri: str = ""

    def __str__(self) -> str:
        where = f" (field {self.field})" if self.field else ""
        return f"refused {self.path}{where}: {self.reason}"


# --------------------------------------------------------------------------------------------
# column knowledge derived from the schema definition
# --------------------------------------------------------------------------------------------

_PATTERN_RE = {
    "SHA1_OR_UNVERIFIED": re.compile(r"^([0-9a-f]{40}|unverified)$"),
    "SHA256_OR_UNVERIFIED": re.compile(r"^([0-9a-f]{64}|unverified)$"),
    "SHA256": re.compile(r"^[0-9a-f]{64}$"),
    "STAMP16_OR_UNVERIFIED": re.compile(r"^([0-9a-f]{16}|unverified)$"),      # the judging code's stamp on a refusal fact
    "UTC_MS": R.UTC_MS_RE,
    "UTC_MS_OR_UNVERIFIED": re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z|unverified)$"),
    "FIT_LR_SELECTED_BY": re.compile(r"^(search|default|manual|unverified|inherited_from:.+)$"),
    "SUBSAMPLE_RULE": re.compile(r"^(full|fixed_ids_file|seeded_sample:.+|paired_with:.+)$"),
    "STRENGTH_RULE": re.compile(r"^(operating_point|fixed:.+|ladder:.+)$"),
}


@dataclass(frozen=True)
class ColumnPlan:
    """Everything :func:`check_column_value` needs about one column, computed once at import (F8)."""
    vocab: Optional[str]
    allowed: Optional[frozenset]
    pattern: Optional[str]
    regex: Optional["re.Pattern[str]"]
    ctype: str
    is_json: bool
    has_field_confidence: bool
    #: a site-label column (``site_aliases.LABEL_VOCABS``): its vocabulary check reads the value through the site's alias table
    label: bool = False


def _plan(t: S.Table, c: S.Column) -> ColumnPlan:
    return ColumnPlan(c.vocab, frozenset(V.VOCABULARIES[c.vocab]) if c.vocab else None, c.pattern,
                      _PATTERN_RE.get(c.pattern) if c.pattern else None, c.type,
                      c.name.endswith("_json") and c.type == "TEXT", "field_confidence_json" in t.column_names,
                      c.vocab in SA.LABEL_VOCABS)


_COLUMN_PLAN: Dict[Tuple[str, str], ColumnPlan] = {(t.name, c.name): _plan(t, c) for t in S.TABLES for c in t.columns}
#: Per table: the payload columns, the JSON columns among them, and (F7) the keys the secret scan may skip —
#: every column name of the schema, verified non-secret-shaped once here rather than per record.
_PAYLOAD_COLS: Dict[str, frozenset] = {t.name: frozenset(S.payload_columns(t.name)) for t in S.TABLES}
_JSON_COLS: Dict[str, frozenset] = {t.name: frozenset(c for c in _PAYLOAD_COLS[t.name] if _COLUMN_PLAN[(t.name, c)].is_json)
                                    for t in S.TABLES}
_ALL_COLUMN_NAMES: frozenset = frozenset(c.name for t in S.TABLES for c in t.columns)
#: the metric-naming guard: the same list the DDL's `*_metric_id_not_generic` CHECKs render
_METRIC_ID_GUARDED: frozenset = frozenset(S.METRIC_ID_GUARDED)
_BANNED_METRIC_IDS: frozenset = frozenset(V.banned_metric_ids())
_SAFE_KEYS: Dict[str, frozenset] = {t.name: _PAYLOAD_COLS[t.name] | {"fields", "verify", "observation_keys"} for t in S.TABLES}
_FKS: Dict[str, Tuple[Tuple[str, str, str], ...]] = {t.name: tuple(S.foreign_keys(t.name)) for t in S.TABLES}
for _name in _ALL_COLUMN_NAMES:
    assert not SECRET_KEY_RE.search(_name), f"schema column {_name!r} is secret-shaped; the validator would refuse every record"


def table_for(record_type: str) -> Optional[str]:
    return R.RECORD_TYPE_TABLE.get(record_type)


def _is_json_col(table: str, col: str) -> bool:
    return col in _JSON_COLS.get(table, frozenset())


def _scan_secret_keys(obj, prefix: str = "", safe: frozenset = frozenset()) -> Optional[str]:
    """The first secret-shaped key in a nested payload (dotted path), or None. Keys in ``safe`` (the table's own
    column names, verified at import) skip the regex; their values are still scanned."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in safe and isinstance(k, str) and SECRET_KEY_RE.search(k):
                return prefix + k
            if isinstance(v, (dict, list)):
                hit = _scan_secret_keys(v, prefix + str(k) + ".", safe)
                if hit:
                    return hit
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                hit = _scan_secret_keys(v, prefix + f"[{i}].", safe)
                if hit:
                    return hit
    return None


def check_column_value(table: str, col: str, value, payload: Mapping[str, object]) -> Optional[str]:
    """Vocabulary / pattern / type check of one column value; returns a reason or None."""
    if value is None:
        return None
    plan = _COLUMN_PLAN.get((table, col))
    if plan is None:
        return None
    # shape FIRST: a list or object where a scalar belongs is Malformed, never a TypeError in the cycle
    ctype = plan.ctype
    if ctype == "INTEGER":
        if not isinstance(value, int):
            return f"{col}={value!r} must be an integer"
    elif ctype == "REAL":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{col}={value!r} must be a number"
    elif ctype == "TEXT":
        if not isinstance(value, str):
            return f"{col}={value!r} must be a string" + (" (JSON columns take an object, serialised canonically)" if plan.is_json else "")
    elif ctype == "BLOB":
        if not isinstance(value, (bytes, str)):
            return f"{col}={value!r} must be bytes"
    # a site label is checked as the label its spelling stands for (site_aliases.canonical); the record keeps the spelling it was
    # written with, and the fold maps it again before it writes the live row, so a rebuild of the log equals the migrated database
    if plan.allowed is not None and (SA.canonical(value) if plan.label else value) not in plan.allowed:
        return f"{col}={value!r} is not in {plan.vocab} {list(V.VOCABULARIES[plan.vocab])}"
    if (table, col) in _METRIC_ID_GUARDED and value in _BANNED_METRIC_IDS:
        return (f"{col}={value!r} is a banned metric spelling (a bare generic name, or a degeneration statistic under an unqualified "
                f"name — degeneration belongs in run_degeneration by measure and scope; see vocab.banned_metric_ids() and "
                f"docs/registry/vocabulary.md 'Metric ids')")
    pat = plan.pattern
    if plan.regex is not None:
        if not plan.regex.match(value):
            return f"{col}={value!r} does not match {pat}"
    elif pat == "JSON":
        try:
            json.loads(value)
        except json.JSONDecodeError:
            return f"{col} is not valid JSON"
    elif pat == "NONNEG" and value < 0:
        return f"{col}={value!r} must be >= 0"
    if value == "unverified" and plan.has_field_confidence:
        fc = payload.get("field_confidence_json")
        if isinstance(fc, str):
            try:
                fc = json.loads(fc)
            except json.JSONDecodeError:
                fc = None
        entry = (fc or {}).get(col) if isinstance(fc, dict) else None
        if not (isinstance(entry, dict) and entry.get("confidence") == "unverified" and str(entry.get("reason", "")).strip()):
            return (f"{col}='unverified' requires field_confidence_json[{col!r}] = "
                    "{'confidence': 'unverified', 'reason': <non-empty>}")
    return None


def _normalise_payload(table: str, payload: Mapping[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Split payload into known columns (JSON columns serialised canonically) and parked extras."""
    cols = _PAYLOAD_COLS[table]
    json_cols = _JSON_COLS[table]
    known: Dict[str, object] = {}
    extra: Dict[str, object] = {}
    for k, v in payload.items():
        if k == "extra":
            if isinstance(v, dict):
                extra.update(v)
            elif v is not None:
                extra["extra"] = v
            continue
        if k in cols:
            if k in json_cols and v is not None and not isinstance(v, str):
                v = R.canonical_json(v)
            if isinstance(v, bool):
                v = int(v)
            known[k] = v
        else:
            extra[k] = v
    if "extra_json" in known and isinstance(known["extra_json"], str):
        try:
            existing = json.loads(known["extra_json"])
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(extra)
                extra = merged
        except json.JSONDecodeError:
            pass
        known.pop("extra_json", None)
    return known, extra


def _nonfinite_repr(value, hit: str) -> str:
    """The offending value as text (``inf`` / ``-inf`` / ``nan``) for the refusal reason, found again by its path."""
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", hit):
        try:
            value = value[int(part[1:-1])] if part.startswith("[") else value[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return "non-finite"
    return repr(value)


def validate_envelope(rec: Mapping[str, object], path: str):
    rt = rec.get("record_type")
    if rt not in R.RECORD_TYPES:
        return Malformed("record_type", f"unknown record_type {rt!r}", path, None)
    if rec.get("schema_version") != R.SCHEMA_VERSION:
        return Malformed("schema_version", f"schema_version {rec.get('schema_version')!r} != {R.SCHEMA_VERSION}", path, rt)
    rid = rec.get("record_id")
    if not isinstance(rid, str) or not rid:
        return Malformed("record_id", "record_id missing or empty", path, rt)
    w = rec.get("written_utc")
    if not isinstance(w, str) or not R.UTC_MS_RE.match(w):
        return Malformed("written_utc", f"written_utc {w!r} is not ISO-8601 UTC with milliseconds (…Z)", path, rt)
    prod = rec.get("producer")
    if not isinstance(prod, dict):
        return Malformed("producer", "producer block missing", path, rt)
    for k in ("script", "code_sha", "host", "substrate"):
        if k not in prod or prod[k] in (None, ""):
            return Malformed(f"producer.{k}", f"producer.{k} missing", path, rt)
    if not (R.SHA1_RE.match(str(prod["code_sha"])) or prod["code_sha"] == "unverified"):
        return Malformed("producer.code_sha", f"producer.code_sha {prod['code_sha']!r} is not 40-hex or 'unverified'", path, rt)
    if SA.canonical(prod["substrate"]) not in V.SUBSTRATES:
        return Malformed("producer.substrate", f"producer.substrate {prod['substrate']!r} not in SUBSTRATES {list(V.SUBSTRATES)}", path, rt)
    payload = rec.get("payload")
    if not isinstance(payload, dict):
        return Malformed("payload", "payload missing or not an object", path, rt)
    # a record carries finite numbers only: NaN / ±Infinity parse in Python but are not JSON, and would
    # fail the json_valid CHECK of every *_json column they reach — an IntegrityError that ended the cycle out of a
    # revision's staging transaction, and a silent drop (INSERT OR IGNORE) in row_history. Refused here, by field path.
    for key, value in rec.items():
        hit = R.first_nonfinite(value, "" if key == "payload" else str(key))
        if hit is not None:
            shown = _nonfinite_repr(payload if key == "payload" else rec, hit)
            return Malformed(hit, f"{hit}={shown} is not finite: NaN and Infinity are not JSON (RFC 8259; SQLite json_valid "
                                  "refuses them) — a number that does not exist is a measurement_state, never a value", path, rt)
    effect = R.EFFECTS.get(rt)
    table = R.RECORD_TYPE_TABLE.get(rt) or effect.columns_table  # type: ignore[union-attr]
    hit = _scan_secret_keys(payload, safe=_SAFE_KEYS[table])
    if hit:
        return Malformed(hit, f"payload key {hit!r} is secret-shaped; records carry no credentials", path, rt)
    return None


def fit_target_pairing_problem(row: Mapping[str, object]) -> Optional[Tuple[str, str]]:
    """The writer-side mirror of the ``object`` table's ``object_fit_target_kind_pairing`` CHECK
    (``schema_def.render_fit_target_check`` over ``vocab.FIT_TARGET_RULES``): ``(column, reason)`` for the first rule the row
    breaks — the admitted ``fit_objective`` / ``derivation`` values, the equalities, the required non-NULL and the required
    NULL columns, NULL-ness exactly as SQL reads it — or None. An unknown ``fit_target_kind`` is the vocabulary check's to
    name, not this one's. Until 2026-10-02 the validator did not enforce the pairing, so a record the DB CHECK would refuse
    passed the spool with the cluster-side evidence reading "emitted" and died, unread, in ``record_refusal`` (the native
    derived emitters' objective ``ce`` against the rule's ``not_applicable``: qwen35_35b_a3b's paired write, stage E)."""
    kind = row.get("fit_target_kind")
    rule = V.FIT_TARGET_RULES.get(str(kind)) if kind is not None else None
    if rule is None:
        return None
    tag = "object_fit_target_kind_pairing"
    obj = row.get("fit_objective")
    if obj not in rule["fit_objective"]:
        return ("fit_objective", f"fit_target_kind {kind!r} admits fit_objective {list(rule['fit_objective'])}, got {obj!r} ({tag})")
    der = row.get("derivation")
    if der not in rule["derivation"]:
        return ("derivation", f"fit_target_kind {kind!r} admits derivation {list(rule['derivation'])}, got {der!r} ({tag})")
    for col, val in rule["equals"]:
        if row.get(col) != val:
            return (col, f"fit_target_kind {kind!r} requires {col} = {val!r}, got {row.get(col)!r} ({tag})")
    for col in rule["nonnull"]:
        if row.get(col) is None:
            return (col, f"fit_target_kind {kind!r} requires {col} to be set ({tag})")
    for col in rule["null"]:
        if row.get(col) is not None:
            return (col, f"fit_target_kind {kind!r} requires {col} to be NULL, got {str(row.get(col))[:80]!r} ({tag})")
    return None


def validate_record(rec: Mapping[str, object], path: str = "<memory>", sha: Optional[str] = None):
    """The envelope, then the payload's required fields and value checks, plus the run_patch / whole-row rules; pure (no DB)."""
    bad = validate_envelope(rec, path)
    if bad:
        bad.sha = sha or ""
        return bad
    rt = str(rec["record_type"])
    payload: Dict[str, object] = dict(rec["payload"])  # type: ignore[arg-type]
    required = R.REQUIRED_FIELDS[rt]

    # item-scale records carry the WHOLE row: name every missing column at once
    if rt in R.ITEM_SCALE_RECORD_TYPES:
        missing = [c for c in R.ITEM_WHOLE_ROW_COLUMNS[rt] if c not in payload]
        if missing:
            return Malformed(",".join(missing),
                             f"item-scale record must carry the whole row; missing columns: {missing}", path, rt, sha or "")

    # required fields
    for f in required:
        if rt == "object" and f in R.OBJECT_OMITTABLE_WITH_ROW and f not in payload:
            # an object record may OMIT `status` -- the producer's statement, which a mirror must not restate. Whether a row exists
            # for the id is a fold-time fact: the harvester dependency-blocks the record while none does (never a refusal), and the fold
            # keeps the folded value when one does. An explicit null is not an omission (refused below).
            continue
        if f not in payload or payload[f] is None or payload[f] == "":
            if rt in ("run", "instrument", "roster_entry") and f in ("item_set_id", "instrument_id") and payload.get("eval_id"):
                return Malformed(f, f"eval {payload.get('eval_id')!r} named without its {f} — an eval is never named without its version",
                                 path, rt, sha or "")
            if rt == "object" and f in R.OBJECT_OMITTABLE_WITH_ROW:
                return Malformed(f, f"object.{f} is null: omit the field to keep the folded {f} (D721; a null is not a statement and the "
                                    "column is NOT NULL), or state a value", path, rt, sha or "")
            return Malformed(f, f"required field {f!r} missing", path, rt, sha or "")

    if rt == "run_patch":
        return _validate_run_patch(rec, payload, path, sha or "")
    if rt == "observation":
        keys = payload.get("observation_keys")
        if not isinstance(keys, list) or not keys:
            return Malformed("observation_keys", "observation_keys must be a non-empty list", path, rt, sha or "")
        if payload.get("table_name") not in R.RUN_SCALE_TABLES:
            return Malformed("table_name", f"observation.table_name must be a run-scale table, got {payload.get('table_name')!r}",
                             path, rt, sha or "")
        return Validated(rt, str(rec["record_id"]), str(rec["written_utc"]), dict(rec["producer"]),
                         payload, {}, None, sha or "", path)
    if rt == "item_withdrawal":
        return _validate_item_withdrawal(rec, payload, path, sha or "")

    effect = R.EFFECTS.get(rt)
    table = table_for(rt) or effect.columns_table  # type: ignore[union-attr]
    known, extra = _normalise_payload(table, payload)
    if effect is not None and effect.strict_keys:
        known = {k: payload[k] for k in effect.required}
        known.update({k: payload[k] for k in effect.optional if payload.get(k) not in (None, "")})
        extra = {}
    for col, val in known.items():
        reason = check_column_value(table, col, val, known)
        if reason:
            return Malformed(col, reason, path, rt, sha or "")
    if rt == "judgment" and (sha or "") not in GRANDFATHERED_RECORD_SHAS:
        bad = _judgment_id_refusal(known)
        if bad:
            return Malformed("judgment_id", bad, path, rt, sha or "")
    if rt == "object":
        # the two lineage columns, stated as a column OR carried under extra (the fold lifts the extra key when no column is stated,
        # records.OBJECT_LIFTED_FROM_EXTRA), are refused by the same two rules either way: a named_role outside the vocabulary, a parent that
        # is the object itself. The column form was checked above by check_column_value; the extra form is checked here. Whole-log audit
        # before the rule (2026-10-03, read-only; 3,607 object records): 195 carry extra.named_role, every one in the vocabulary; 195 carry
        # extra.parent_object_id, 4 distinct parents, all in the log, none the object itself -- so historical replay is unchanged.
        bad = _object_lineage_refusal(known, extra)
        if bad:
            return Malformed(bad[0], bad[1], path, rt, sha or "")
    bad_unstated = _check_unstated_columns(table, known, extra)
    if bad_unstated:
        return Malformed("extra.unstated_columns", bad_unstated, path, rt, sha or "")
    bad_shape = _check_union_column_shapes(table, known)
    if bad_shape:
        return Malformed(bad_shape[0], bad_shape[1], path, rt, sha or "")
    bad_union = _check_union_columns(table, known, extra)
    if bad_union:
        return Malformed("extra.union_columns", bad_union, path, rt, sha or "")
    bad_same_bytes = _check_same_bytes(table, extra)
    if bad_same_bytes:
        return Malformed(f"extra.{SAME_BYTES_KEY}", bad_same_bytes, path, rt, sha or "")
    if rt == "run_state" and known.get("state") == "ingested":
        return Malformed("state", "run_state may not set 'ingested' — only the harvester's run_ingest record does", path, rt, sha or "")
    if rt == "object":
        # the fit_target_kind pairing (F16), refused HERE so a writer learns it before the spool — the DB CHECK is the
        # last line, and a record that dies there after the cluster evidence says "emitted" is invisible to the emitting job
        pairing = fit_target_pairing_problem(known)
        if pairing is not None:
            return Malformed(pairing[0], pairing[1], path, rt, sha or "")
        # hook-fix-2: an `object` record whose extra names ITSELF as its parent is the per-epoch
        # re-emission of an already-published object (the best-epoch restore gives the published object an epoch's bytes); folded,
        # it would overwrite the published object's provenance and primary location under the same natural key. Refused by name;
        # the fixed hook records such an epoch as a role membership on the canonical object instead.
        ex = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}
        if ex and str(ex.get("parent_object_id") or "") == str(known.get("object_id") or payload.get("object_id") or ""):
            return Malformed("extra.parent_object_id", "object record names itself as parent_object_id — a per-epoch re-emission of the "
                             "published object under its own object_id (the epoch's bytes ARE the published object's); it would overwrite "
                             "the published object's provenance; record the epoch as a role on the canonical object instead (hook-fix-2)",
                             path, rt, sha or "")
    if rt == "artifact":
        # hook-fix-2 (second round): the pre-fix hook's per-epoch artifact — built by object_from_sidecar
        # over the epoch dir — carried extra {object_id: <its own sha>, sidecars} and NO role; when the epoch's bytes ARE the
        # published object's, that record shares the published artifact's natural key and, folded, overwrites its primary
        # location. A record is only distinguishable by what it CARRIES, so an artifact under an epoch-checkpoint dir must
        # carry provenance that ties it to a published object OTHER than itself: extra.parent_object_id != artifact_id (the
        # fixed hook's sidecar-built epoch artifact) or extra.role == epoch_checkpoint with extra.object_id != artifact_id (the
        # index-derived epoch artifact). Anything else under epoch_ckpts/ is refused by name — the pre-fix duplicate first of
        # all; a pre-fix DISTINCT epoch artifact too (conservative: it is re-emitted with the marker by the corrective
        # re-emission). The published artifact itself (uri_primary at the cell root) is untouched by this rule.
        ex = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}
        aid = str(known.get("artifact_id") or payload.get("artifact_id") or "")
        uri = str(known.get("uri_primary") or payload.get("uri_primary") or "")
        if re.search(r"/epoch_ckpts/epoch_\d+/", uri):
            parent = str(ex.get("parent_object_id") or "") if ex else ""
            via_index = bool(ex) and ex.get("role") == "epoch_checkpoint" and str(ex.get("object_id") or "") not in ("", aid)
            if not ((parent and parent != aid) or via_index):
                return Malformed("uri_primary", f"artifact record at an epoch-checkpoint path ({uri.rsplit('/', 3)[-3] if uri.count('/') >= 3 else uri}/…) "
                                 "without provenance tying it to a published object other than itself (extra.parent_object_id, or "
                                 "extra.role epoch_checkpoint + extra.object_id) — the pre-hook-fix-2 per-epoch duplicate shape, which shares "
                                 "the published artifact's natural key when the epoch's bytes are the published object's and would overwrite "
                                 "its primary location; re-emit from the fixed hook (hook-fix-2)", path, rt, sha or "")
    if rt in V.ANALYSIS_RECORD_TYPES:
        # the two pairing rules the tables' CHECKs enforce, refused HERE by name so the filer learns them before the spool (a record
        # that dies at the CHECK is refused as 'CHECK constraint failed', naming nothing)
        bad = analysis_pairing_refusal(rt, known)
        if bad:
            return Malformed(bad[0], bad[1], path, rt, sha or "")
    if rt == "run" and known.get("lifecycle_state") not in ("queued", "running", "generated"):
        return Malformed("lifecycle_state",
                         f"a run record carries its identity and the state its store proves at that moment (queued | running | generated), "
                         f"got {known.get('lifecycle_state')!r}; later states (judged, failed, cancelled, superseded, withdrawn) move only "
                         "through run_state records and 'ingested' only through the harvester's run_ingest", path, rt, sha or "")
    if rt == "run" and (sha or "") not in GRANDFATHERED_RECORD_SHAS:
        # The three facts AS THE PRODUCER STATED THEM (`payload`), never `known`'s rendering of them: _normalise_payload turns a boolean into
        # its integer for every INTEGER column (the flag columns' convention), so a `token_cap_rung_index: true` reached this rule as 1,
        # matched a [4096, 8192] ladder and an 8,192 cap, and folded a level-1 fact the producer never stated -- which the live-row
        # re-check (`registry cap-check`) then had no way to see. Replay-neutral by audit: 0 of the
        # 107,799 `run` records in the log on 2026-10-03 state a boolean in any of the three fields, and an admitted integer is the same
        # value raw and normalized (a string or float never reaches here: the column rule above refuses it first).
        bad = run_cap_ladder_refusal(payload.get("token_cap_tokens"), payload.get("token_cap_rung_index"), payload.get("token_cap_ladder_json"))
        if bad:
            return Malformed(bad[0], bad[1], path, rt, sha or "")
        # The run's own `code_sha` as the producer stated it (a TEXT column is the same raw and normalized; the column rule above has
        # already refused anything but forty hex digits or 'unverified', so this rule sees only those). Payload-only: the envelope's
        # producer.code_sha is the envelope check's business and no other kind's column is read.
        placeholder = run_code_sha_refusal(payload.get("code_sha"))
        if placeholder:
            return Malformed("code_sha", placeholder, path, rt, sha or "")
    return Validated(rt, str(rec["record_id"]), str(rec["written_utc"]), dict(rec["producer"]), known, extra,
                     table_for(rt), sha or "", path)


def analysis_pairing_refusal(record_type: str, known: Mapping[str, object]) -> Optional[Tuple[str, str]]:
    """The pairing rules of the analysis kinds, as the tables' named CHECKs state them: an `analysis` files a report uri and its
    sha together or neither; an `analysis_input` names its one input in `input_id` AND in the FK column of its kind, the other FK column
    null. Returns ``(field, reason)`` or None."""
    if record_type == "analysis":
        uri, sha = known.get("report_uri"), known.get("report_sha256")
        if (uri is None) != (sha is None):
            field_name = "report_sha256" if sha is None else "report_uri"
            return (field_name, "analysis.report_uri and analysis.report_sha256 are filed together or not at all: a report the filer could "
                                "not hash needs its sha declared in the spec (never a guess), and a sha needs the file it is of")
        return None
    if record_type == "analysis_input":
        kind, input_id = known.get("input_kind"), known.get("input_id")
        run_id, object_id = known.get("run_id"), known.get("object_id")
        if kind == "run" and not (run_id == input_id and object_id is None):
            return ("run_id", f"analysis_input of kind 'run' names its run in input_id AND run_id (got input_id={input_id!r}, run_id={run_id!r}) "
                              f"with object_id null (got {object_id!r})")
        if kind == "object" and not (object_id == input_id and run_id is None):
            return ("object_id", f"analysis_input of kind 'object' names its object in input_id AND object_id (got input_id={input_id!r}, "
                                 f"object_id={object_id!r}) with run_id null (got {run_id!r})")
    return None


#: forty identical hex digits -- the shape a synthetic producer block states where a checkout states its commit
_PLACEHOLDER_SHA1_RE = re.compile(r"^([0-9a-f])\1{39}$")


def run_code_sha_refusal(code_sha) -> Optional[str]:
    """Pure: a ``run`` record's ``code_sha`` is never a placeholder -- forty identical hex digits (``"a" * 40``,
    ``"0" * 40``), the shape a test's synthetic producer block states where a real checkout states ``git rev-parse HEAD``. Returns the
    reason or None. The ruling is payload-only: it reads the run's own ``code_sha`` column, never the envelope's ``producer.code_sha``
    (which the envelope check admits as any 40-hex) and no other kind's column. A checkout that cannot state its sha states
    ``unverified`` with a ``field_confidence_json`` entry, which the column rule governs, not this one. Incident: two synthetic test runs
    folded into the live registry on 2026-10-02 (a registration test called ``register --from-config`` with no ``--spool``), ``code_sha``
    forty ``a``s; the record log is immutable, so their four records are grandfathered by sha (:data:`GRANDFATHERED_RECORD_SHAS`) and
    this rule refuses the next one by name. Whole-log audit before the rule landed (2026-10-03, read-only; 7,059 ``-run`` files,
    109,088 ``run`` records): exactly those four records carry a placeholder, 71,818 state ``unverified``, no other does. The synthetic
    corpus states derived shas since this rule (``synth.code_sha``), so the rule catches the incident's shape -- a producer block left at
    its placeholder -- not every escaped test record; the suite-wide spool fence remains the guard."""
    if isinstance(code_sha, str) and _PLACEHOLDER_SHA1_RE.match(code_sha):
        return (f"code_sha={code_sha!r} is a placeholder (forty identical hex digits), never a checkout's commit: a run record states the "
                "sha of the code that produced it (`git rev-parse HEAD`), or 'unverified' with a field_confidence_json entry (D650). A "
                "placeholder is the mark of a test's synthetic producer block reaching the log")
    return None


def run_cap_ladder_refusal(token_cap_tokens, token_cap_rung_index, token_cap_ladder_json) -> Optional[Tuple[str, str]]:
    """Pure: the token-cap ladder facts a ``run`` states must agree with each other. ``token_cap_tokens``
    is the cap the row was DRAWN at -- registry-wide, one run per level of a token-cap ladder, the producer key naming the level and the
    paper labeling from the column -- so a record that states WHICH level the row sits at (``token_cap_rung_index``) must state the
    ladder it indexes (``token_cap_ladder_json``: a non-empty JSON array of strictly ascending positive integers), an index inside it, and
    a cap equal to the indexed level. Returns ``(field, reason)`` naming the field that disagrees, or None. A record that states no index
    -- the legacy shapes: a ladder with no index, an empty ladder on a multi-level instrument -- is admitted as it always was; the
    instrument cross-check (the half of the ruling that needs a store read) is ``registry cap-check``'s report, never a fold-time read.
    ``token_cap_ladder_json`` may be the canonical text (the validator's normalized column, a live row) or the array itself. Whole-log
    audit before the rule landed (2026-10-03, read-only): 15,685 of 107,778 ``run`` records state an index and every one conforms;
    ``run_patch`` records naming any of the three fields: 0 (``token_cap_tokens`` is identity; the other two are not patchable).
    :func:`validate_record` hands this rule the payload's values as the producer stated them -- not the normalizer's, which turns a
    boolean into its integer for every INTEGER column -- so a boolean index, or a boolean cap beside a stated index, is refused here as
    not an integer rather than folded as 1 or 0 (0 such records in the log on 2026-10-03)."""
    idx = token_cap_rung_index
    if idx is None:
        return None
    if isinstance(idx, bool) or not isinstance(idx, int):
        return ("token_cap_rung_index", f"token_cap_rung_index={idx!r} must be an integer (the position of the run's level in its ladder, D648)")
    ladder = token_cap_ladder_json
    if isinstance(ladder, str):
        try:
            ladder = json.loads(ladder)
        except json.JSONDecodeError as exc:
            return ("token_cap_ladder_json", f"token_cap_ladder_json is not valid JSON ({exc.msg} at char {exc.pos}); a run that states "
                                             f"token_cap_rung_index={idx} states the ladder it indexes (D648)")
    if ladder is None or ladder == []:
        return ("token_cap_ladder_json", f"token_cap_rung_index={idx} is stated with no token_cap_ladder_json: a ladder level index needs the "
                                         "ladder it indexes -- a run at a level of a token-cap ladder states the ladder, its index and the "
                                         "cap it was drawn at (D648); a run with one fixed cap states no index")
    if not isinstance(ladder, list):
        return ("token_cap_ladder_json", f"token_cap_ladder_json must be a JSON array of strictly ascending positive integers (the caps of the "
                                         f"ladder's levels, D648), got {type(ladder).__name__}")
    for i, cap in enumerate(ladder):
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            return ("token_cap_ladder_json", f"token_cap_ladder_json={R.canonical_json(ladder)} must be a JSON array of strictly ascending "
                                             f"positive integers (the caps of the ladder's levels, D648): element {i} is {cap!r}")
        if i and cap <= ladder[i - 1]:
            return ("token_cap_ladder_json", f"token_cap_ladder_json={R.canonical_json(ladder)} must be strictly ascending (the caps of the "
                                             f"ladder's levels, lowest first, D648): element {i} ({cap}) is not above element {i - 1} ({ladder[i - 1]})")
    if idx < 0 or idx >= len(ladder):
        return ("token_cap_rung_index", f"token_cap_rung_index={idx} is outside token_cap_ladder_json={R.canonical_json(ladder)} "
                                        f"({len(ladder)} level{'s' if len(ladder) != 1 else ''}, indexed 0..{len(ladder) - 1}; D648)")
    if isinstance(token_cap_tokens, bool):
        # before the equality: True == 1 in Python, so a boolean cap beside a ladder whose indexed level is 1 would pass it
        return ("token_cap_tokens", f"token_cap_tokens={token_cap_tokens!r} must be an integer (the cap the row was drawn at, D648); a boolean "
                                    f"is not a cap, whatever integer the ladder's level {idx} holds")
    if ladder[idx] != token_cap_tokens:
        return ("token_cap_tokens", f"token_cap_tokens={token_cap_tokens!r} is not the ladder level it indexes: "
                                    f"token_cap_ladder_json={R.canonical_json(ladder)}[token_cap_rung_index={idx}] = {ladder[idx]}. A run's "
                                    "cap is the cap it was drawn at, one run per ladder level (D648); a row drawn at another level registers "
                                    "under its own producer key with that level's index and cap, never by relabeling this one")
    return None


def _object_lineage_refusal(known: Mapping[str, object], extra: Mapping[str, object]) -> Optional[Tuple[str, str]]:
    """Pure: the lineage facts of an ``object`` record -- `named_role` and `parent_object_id`, read from the stated column else from
    the parked extra (the fold lifts the extra key, so the rule holds for both spellings). Returns ``(field, reason)`` or None: a
    `named_role` outside ``vocab.OBJECT_NAMED_ROLES`` (the column's own CHECK list) and a `parent_object_id` equal to the object's own id
    are refused by name; a non-string value of either is refused as its shape. Whether the parent EXISTS is the harvester's dependency
    check (``parent_references``), never a refusal."""
    for col in R.OBJECT_LIFTED_FROM_EXTRA:
        value = known[col] if col in known else extra.get(col)
        if value is None:
            continue
        if not isinstance(value, str):
            return (col, f"{col}={value!r} must be a string" + (" (an object id)" if col == "parent_object_id" else " (a named role, D696)"))
        if col == "named_role" and value not in V.OBJECT_NAMED_ROLES:
            return (col, f"named_role={value!r} is not in OBJECT_NAMED_ROLES {list(V.OBJECT_NAMED_ROLES)} (D696; stated as a column or under extra alike)")
        if col == "parent_object_id" and value == known.get("object_id"):
            return (col, f"parent_object_id={value!r} is the object's own id: an epoch-checkpoint child names the fit it belongs to, never itself (D696)")
    return None


def _judgment_id_refusal(known: Mapping[str, object]) -> Optional[str]:
    """The identity rule of a ``judgment`` record: its ``judgment_id`` must equal
    ``identity.judgment_id(run_id, judge_id, S, C)`` for one of the forms in use (``identity.JUDGMENT_ID_FORMS``: none,
    ``claimed_by_session`` + ``claimed_utc``, ``producer_session`` + ``claimed_utc``), else the reason, naming every form tried with
    the values it hashed. Pure (no store read). A same-id record stating another ``run_id`` or ``judge_id`` fails every form, so the two
    identity columns of a judgment can never change by a later record and the successor relationship a supersede reads at fold time
    is the same on live ingest and on replay. Historical conformance: 108,337 of the 108,355 judgment records in the log on 2026-10-03;
    the 18 that fail are release records grandfathered by sha (:data:`GRANDFATHERED_RECORD_SHAS`), and the whole-log audit of
    2026-10-04 (107,850 judgment records) found 12 more -- the supersession / correction records -- grandfathered the same way."""
    if identity.judgment_id_form(known) is not None:
        return None
    run, judge = known.get("run_id"), known.get("judge_id")
    tried = ", ".join(
        f"({name}: " + ", ".join(f"{f}={known.get(f)!r}" for f in (sf, tf) if f) + ")" if sf else f"({name})"
        for name, sf, tf in identity.JUDGMENT_ID_FORMS)
    return (f"judgment_id {known.get('judgment_id')!r} is not derived from this record's run_id / judge_id: no identity form over its own "
            f"fields reproduces it -- identity.judgment_id({run!r}, {judge!r}, session, claimed_utc) tried as {tried}. A judgment id is "
            "bound to its run and judge by every minter (D639), so a record whose id was minted elsewhere, or a same-id record naming "
            "another run or judge, never folds")


#: The record-level non-statement marker (``reducer.UNSTATED_COLUMNS_KEY``): admitted on the tables whose fold honors it
#: (``reducer.UNSTATED_TABLES``: ``object``, and ``run`` since a later follow-up -- a stats reader's record onto the export's run row),
#: as a list of column names of that table, never the key and never the flag's carrier (:data:`UNSTATED_CARRIER_COLUMN`), each carried in
#: the payload (the record supplies the value it does not state -- a column it omits needs no marker, omission already keeps the folded
#: value; a null is carried on the refresh tables only).
UNSTATED_COLUMNS_KEY = "unstated_columns"
UNSTATED_TABLES: Tuple[str, ...] = ("object", "run")
#: The tables on which a marked column may be carried as NULL (mirrors ``reducer.UNSTATED_REFRESH_TABLES``; pinned equal by the tests): a
#: marked null is a stats reader's WITHDRAWAL of a value it supplied earlier -- the fold clears a flagged value and moves nothing over an
#: owner's or a standing null. On ``object`` a null marked column is refused as before: the fold has
#: no withdrawal rule (it skips over any standing value), and ``status``, its only marked column, is NOT NULL.
UNSTATED_REFRESH_TABLES: Tuple[str, ...] = ("run",)
#: The column a marked column's provenance rides in (``reducer.CONFIDENCE_COLUMN``, spelled here because the reducer imports THIS module;
#: pinned equal by the tests). It is the CARRIER of the fold's per-column ``unstated`` flag, never itself a marked column, so a marker that
#: NAMES it is refused here rather than dropped at fold time: a ``run_patch`` stating nothing but an empty
#: ``field_confidence_json`` with the carrier marked passed validation, the reducer dropped the carrier from the marker, and the empty
#: object folded as an ordinary statement -- erasing the standing entries, the reason an ``unverified`` value requires included, while the
#: column that reason explains kept its value. The reducer's drop stays as the belt behind this refusal. Replay-neutral: no ``run`` record
#: in the live log carries the marker (the first writer is on this branch), and the ``object`` records that do -- the bank's of 2026-10-03
#: -- name ``status`` alone.
UNSTATED_CARRIER_COLUMN = "field_confidence_json"
#: The record-level UNION marker (``reducer.UNION_COLUMNS_KEY``; the fourth review round of the analysis filing, 2026-10-09): admitted on
#: the tables with a union column (``reducer.UNION_COLUMNS``: `artifact`, its `locations_json`), as a non-empty list of that table's
#: union columns, each carried in the payload as a JSON list -- the entries this record ADDS to the standing list (a record that adds
#: nothing omits the column; omission keeps the folded value). Since the fourth closing pass (2026-10-09) the fold unions those columns on
#: EVERY record of the table, marker or none, so the marker names what the fold does anyway; a malformed one is still refused by name.
#: Spelled here because the reducer imports THIS module; pinned equal by the tests.
#:
#: The SHAPE of a union column is checked on every record of its table, marker or none (:func:`_check_union_column_shapes`; the seventh
#: closing pass, 2026-10-09): a non-null `locations_json` that is not a JSON list -- an object, a scalar, a string encoding either -- is
#: refused by name, naming the column. The fold's union (``reducer.union_json_list``) reads a non-list as NO entries, so before this a
#: hand-written artifact record carrying its one location as a JSON object passed the validator whenever it carried no marker, and folded
#: as an accepted artifact whose location had vanished -- evidence lost with no refusal. Whole-log audit at the rule's landing (read-only,
#: 2026-10-09): 214,114 `artifact` records across the whole log (507,462 record files scanned, the 14,283 `*-artifact.jsonl` files holding
#: every one), every `locations_json` a JSON list (none omitted, none null, none any other shape), so historical replay is unchanged and
#: nothing is grandfathered.
UNION_COLUMNS_KEY = "union_columns"
UNION_COLUMNS: Dict[str, Tuple[str, ...]] = {"artifact": ("locations_json",)}

#: ``analysis_artifact.extra.same_bytes`` (``analysis.SAME_BYTES_KEY``, spelled here because that module imports the harvester, which
#: imports this one; pinned equal by the tests): the further entries of ONE filing whose bytes are the link's digest, a list of objects
#: ``{uri, role, page_dir?, relpath?, page_url?, declared_sha256?}``. The readers (`analysis.PAGE_URLS_SQL`, the status page) pull
#: `page_url` out of every entry with `json_extract`, which raises `malformed JSON` on a bare string -- so one accepted link whose entries
#: were path strings failed `registry analysis list` whole and showed the status page's Analyses section as a query error (the fourth
#: closing pass, 2026-10-09). The readers now skip an entry that is not an object; a NEW link is refused by name here when the key is
#: carried in any other shape, so nothing malformed enters the table from now on.
SAME_BYTES_KEY = "same_bytes"
SAME_BYTES_TABLE = "analysis_artifact"


def _check_unstated_columns(table: Optional[str], known: Mapping[str, object], extra: Mapping[str, object]) -> Optional[str]:
    """The reason ``extra.unstated_columns`` is malformed, or None (absent, or well-formed)."""
    if UNSTATED_COLUMNS_KEY not in extra:
        return None
    cols = extra[UNSTATED_COLUMNS_KEY]
    if table not in UNSTATED_TABLES:
        return f"unstated_columns is admitted on {', '.join(UNSTATED_TABLES)} records only (the fold honors it there), not on {table!r}"
    if not isinstance(cols, list) or not cols or not all(isinstance(c, str) and c for c in cols):
        return "unstated_columns must be a non-empty list of column names"
    allowed = set(_PAYLOAD_COLS[table])
    keys = set(R.NATURAL_KEYS[table])
    for c in cols:
        if c in keys:
            return f"unstated column {c!r} is the natural key: a record cannot carry its key as a non-statement"
        if c == UNSTATED_CARRIER_COLUMN:
            return (f"unstated column {c!r} is the carrier of the per-column unstated flag, never a marked column itself: the fold reads "
                    f"every marked column's provenance out of it, so a record states the entries it has evidence for (a stats reader keeps "
                    f"those of its marked columns) or omits the column -- a marked carrier folds as an ordinary statement and erases the "
                    f"standing entries, the reason an 'unverified' value requires included, while the column that reason explains stands")
        if c not in allowed:
            return f"unstated column {c!r} is not a column of {table}"
        if c not in known:
            return f"unstated column {c!r} is not carried in the payload: a column the record omits needs no marker (omission keeps the folded value)"
        if known[c] is None and table not in UNSTATED_REFRESH_TABLES:
            return (f"unstated column {c!r} is not carried in the payload (carried as null): on {table} a marked null is no statement and "
                    f"withdraws nothing -- omit the column (omission keeps the folded value) or state a value")
    if len(set(cols)) != len(cols):
        return "unstated_columns repeats a column"
    return None


def _check_union_column_shapes(table: Optional[str], known: Mapping[str, object]) -> Optional[Tuple[str, str]]:
    """``(column, reason)`` for the first union column of ``table`` (:data:`UNION_COLUMNS`) the payload carries NON-NULL in any shape but a
    JSON list -- an object, a scalar, a string encoding either -- or None (no union column, or every one carried is a list or null). The
    column arrives here as the text :func:`_normalise_payload` made of it (an object or list serialized canonically; a string as given), so
    the text is parsed and its top-level type is what counts. A null adds nothing to the union and is admitted; an omitted column keeps the
    folded value. Marker or none: the fold unions the column on every record of the table and reads a non-list as no entries, so a
    non-list would fold as an accepted record whose location had vanished (the seventh closing pass, 2026-10-09)."""
    cols = UNION_COLUMNS.get(table or "")
    if not cols:
        return None
    for c in cols:
        val = known.get(c)
        if val is None:
            continue
        parsed: object = val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
            except json.JSONDecodeError:
                return (c, f"{c} is not valid JSON")
        if not isinstance(parsed, list):
            shape = "an object" if isinstance(parsed, dict) else f"a {type(parsed).__name__}"
            return (c, f"{c} must carry a JSON list of the locations this record adds to the standing list (the fold unions the column on "
                       f"every {table} record and reads any other shape as no entries, so {shape} would fold as an accepted record whose "
                       f"location had vanished), got {shape}: {R.canonical_json(parsed)[:120]}")
    return None


def _check_union_columns(table: Optional[str], known: Mapping[str, object], extra: Mapping[str, object]) -> Optional[str]:
    """The reason ``extra.union_columns`` is malformed, or None (absent, or well-formed). The column's own SHAPE is
    :func:`_check_union_column_shapes`'s, checked before this on every record, marker or none; the list clause below stays as the belt."""
    if UNION_COLUMNS_KEY not in extra:
        return None
    cols = extra[UNION_COLUMNS_KEY]
    if table not in UNION_COLUMNS:
        return f"union_columns is admitted on {', '.join(UNION_COLUMNS)} records only (the fold honors it there), not on {table!r}"
    if not isinstance(cols, list) or not cols or not all(isinstance(c, str) and c for c in cols):
        return "union_columns must be a non-empty list of column names"
    for c in cols:
        if c not in UNION_COLUMNS[table]:
            return (f"union column {c!r} is not a list column of {table} the fold unions (the union columns of {table} are "
                    f"{', '.join(UNION_COLUMNS[table])})")
        if c not in known or known[c] is None:
            return (f"union column {c!r} is not carried in the payload: a record that adds nothing to the list omits the column "
                    f"(omission keeps the folded value)")
        val = known[c]
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                val = None
        if not isinstance(val, list):
            return f"union column {c!r} must carry a JSON list (the entries this record adds to the standing list)"
    if len(set(cols)) != len(cols):
        return "union_columns repeats a column"
    return None


def _check_same_bytes(table: Optional[str], extra: Mapping[str, object]) -> Optional[str]:
    """The reason ``extra.same_bytes`` on an `analysis_artifact` link is malformed, or None (absent, or well-formed: a list of objects,
    each with a non-empty string `uri` and a `role` from the analysis artifact roles, `page_url` a string when carried). On any other
    table the key is free-form metadata and is not checked."""
    if table != SAME_BYTES_TABLE or SAME_BYTES_KEY not in extra:
        return None
    entries = extra[SAME_BYTES_KEY]
    if not isinstance(entries, list):
        return (f"same_bytes must be a JSON list of entries {{uri, role, page_dir?, relpath?, page_url?}}, got {type(entries).__name__}: the "
                "listing and the status page read `page_url` out of every entry")
    for i, e in enumerate(entries):
        where = f"same_bytes[{i}]"
        if not isinstance(e, dict):
            return f"{where} must be an object {{uri, role, page_dir?, relpath?, page_url?}}, got {type(e).__name__} (a bare path is not an entry)"
        uri = e.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            return f"{where}.uri must be a non-empty string naming the further entry's path"
        role = e.get("role")
        if role not in V.ANALYSIS_ARTIFACT_ROLES:
            return f"{where}.role must be one of {'|'.join(V.ANALYSIS_ARTIFACT_ROLES)}, got {role!r}"
        if e.get("page_url") is not None and not isinstance(e.get("page_url"), str):
            return f"{where}.page_url must be a string or null"
    return None


def _validate_run_patch(rec, payload, path, sha):
    rt = "run_patch"
    fields = payload.get("fields") or {}
    verify = payload.get("verify") or {}
    if not isinstance(fields, dict) or not isinstance(verify, dict):
        return Malformed("fields", "run_patch.fields / verify must be objects", path, rt, sha)
    if not fields and not verify:
        return Malformed("fields", "run_patch carries neither fields nor verify", path, rt, sha)
    for k in fields:
        if k in R.RUN_VERIFIED_FIELDS:
            return Malformed(k, f"{k!r} is an immutable field: send it under 'verify', never under 'fields'", path, rt, sha)
        if k in R.RUN_IDENTITY_FIELDS:
            return Malformed(k, f"{k!r} is an identity/lifecycle field and may not be patched (state moves only through run_state)",
                             path, rt, sha)
        if k in R.RUN_DERIVED_FIELDS:
            # the decode-policy pair is a stamped derivation (decode_match.stamp_run_rows rewrites it from the run's recorded
            # decode fields every cycle); a patched value would not survive the cycle that folded it, so it is refused by name
            return Malformed(k, f"{k!r} is derived from the run's decode fields by decode_match.stamp_run_rows every cycle and may not "
                                f"be patched: patch the fields it is derived from ({', '.join(f for f in R.RUN_PATCHABLE_FIELDS if f.startswith('decode_'))})",
                             path, rt, sha)
        if k not in R.RUN_PATCHABLE_FIELDS:
            return Malformed(k, f"{k!r} is not in RUN_PATCHABLE_FIELDS", path, rt, sha)
    for k in verify:
        if k not in R.RUN_VERIFIED_FIELDS:
            return Malformed(k, f"{k!r} is not in RUN_VERIFIED_FIELDS", path, rt, sha)
    if "extra_json" in fields:
        # fields.extra_json is the ONE place a patch's stamps and provenance ride: a value that carries no keys (a
        # serialized array or scalar, text that is not JSON, a non-mapping) is refused by name HERE, before the fold -- never dropped
        # by _normalise_payload's merge below, which keeps only an object's keys and used to drop such a value silently while the
        # patch's typed columns folded without the writer's stamps. Replay-neutral: the whole-log
        # audit at landing found 0 run_patch records with such a value (records.run_patch_extra_json).
        _own, reason = R.run_patch_extra_json(fields["extra_json"])
        if reason:
            return Malformed("extra_json", reason, path, rt, sha)
    known, extra = _normalise_payload("run", fields)
    for col, val in known.items():
        reason = check_column_value("run", col, val, known)
        if reason:
            return Malformed(col, reason, path, rt, sha)
    for col, val in verify.items():
        reason = check_column_value("run", col, val, verify)
        if reason:
            return Malformed(col, reason, path, rt, sha)
    # the non-statement marker rides a patch's `fields.extra_json` into the effect's extra and the fold honors it on a `patch` event
    # (reducer.UNSTATED_TABLES): the same rules as on a full record -- a list of carried, non-key run columns -- or the patch is refused by
    # name, never folded -- a string marker was ignored and a malformed list silently suppressed the patch's own `notes`
    # statement.
    bad_unstated = _check_unstated_columns("run", known, extra)
    if bad_unstated:
        return Malformed("extra.unstated_columns", bad_unstated, path, rt, sha)
    out_payload = {"run_id": payload["run_id"], "fields": known, "verify": dict(verify)}
    return Validated(rt, str(rec["record_id"]), str(rec["written_utc"]), dict(rec["producer"]), out_payload, extra,
                     None, sha, path)


#: The item_result defaults an item key of an ``item_withdrawal`` may leave out (the DDL defaults / sentinels; ``reducer._key_of``).
_ITEM_KEY_DEFAULTS: Dict[str, object] = {"trial_index": 0, "sample_index": 0, "turn_index": S.SENTINELS["turn_index"],
                                        "judgment_id": S.SENTINELS["judgment_id"]}


def _validate_item_withdrawal(rec, payload, path, sha):
    """An ``item_withdrawal``: ``{run_id, items: [{item_id, metric_id, trial_index?, sample_index?, turn_index?, judgment_id?}, …],
    reason, evidence_uri?, belongs_to_run_id?}``. Refused BY NAME: ``items`` not a non-empty list; an item key that is not an object,
    carries a field outside the six key fields, lacks ``item_id`` / ``metric_id``, carries a non-integer index or a non-string
    ``judgment_id``, or repeats an earlier key; a ``reason`` that is not a string (the required-field pass already refused a missing or
    empty one). An absent index takes the item_result default (trial 0, sample 0, turn -1, judgment ''), and the normalized payload
    carries every key whole. Whether the run or a named row EXISTS is the harvester's dependency check, never a refusal."""
    rt = "item_withdrawal"
    items = payload.get("items")
    fields = ", ".join(R.ITEM_WITHDRAWAL_KEY_FIELDS)
    if not isinstance(items, list) or not items:
        return Malformed("items", f"items must be a non-empty list of item keys {{{fields}}}", path, rt, sha)
    if not isinstance(payload.get("reason"), str):
        return Malformed("reason", "reason must be a non-empty string (free text: why the rows are retracted)", path, rt, sha)
    allowed = set(R.ITEM_WITHDRAWAL_KEY_FIELDS)
    keys: List[Dict[str, object]] = []
    seen: Dict[tuple, int] = {}
    for i, k in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(k, dict):
            return Malformed(where, f"an item key must be an object {{{fields}}}, got {type(k).__name__}", path, rt, sha)
        unknown = sorted(set(k) - allowed)
        if unknown:
            return Malformed(f"{where}.{unknown[0]}", f"unknown item-key field {unknown[0]!r}: an item key carries exactly {{{fields}}}",
                             path, rt, sha)
        for f in R.ITEM_WITHDRAWAL_KEY_REQUIRED:
            if not isinstance(k.get(f), str) or not k.get(f):
                return Malformed(f"{where}.{f}", f"item key field {f!r} missing or not a non-empty string", path, rt, sha)
        norm: Dict[str, object] = {}
        for f in R.ITEM_WITHDRAWAL_KEY_FIELDS:
            val = k.get(f, _ITEM_KEY_DEFAULTS.get(f))
            if f in ("trial_index", "sample_index", "turn_index"):
                if isinstance(val, bool) or not isinstance(val, int):
                    return Malformed(f"{where}.{f}", f"{f}={val!r} must be an integer", path, rt, sha)
            elif not isinstance(val, str):
                return Malformed(f"{where}.{f}", f"{f}={val!r} must be a string", path, rt, sha)
            norm[f] = val
        tup = tuple(norm[f] for f in R.ITEM_WITHDRAWAL_KEY_FIELDS)
        if tup in seen:
            return Malformed(where, f"{where} repeats items[{seen[tup]}] (the same item key twice)", path, rt, sha)
        seen[tup] = i
        keys.append(norm)
    rest = {k: v for k, v in payload.items() if k != "items"}
    known, extra = _normalise_payload("item_withdrawal", rest)
    for col, val in known.items():
        reason = check_column_value("item_withdrawal", col, val, known)
        if reason:
            return Malformed(col, reason, path, rt, sha)
    out_payload: Dict[str, object] = {"run_id": known["run_id"], "items": keys, "reason": known["reason"]}
    for col in R.EFFECTS[rt].optional:
        if known.get(col) is not None:
            out_payload[col] = known[col]
    return Validated(rt, str(rec["record_id"]), str(rec["written_utc"]), dict(rec["producer"]), out_payload, extra, None, sha, path)


# --------------------------------------------------------------------------------------------
# Dependencies (the harvester's FK pre-check; F4)
# --------------------------------------------------------------------------------------------


def parent_references(v: Validated) -> List[Tuple[str, str, str]]:
    """[(parent_table, parent_column, parent_id)] this record needs present before it can fold."""
    rt = v.record_type
    p = v.payload
    refs: List[Tuple[str, str, str]] = []
    effect = R.EFFECTS.get(rt)
    if effect is not None:
        refs += [(ptable, pcol, str(p[key])) for ptable, pcol, key in effect.parents]
        if rt == "run_state" and p.get("state") == "judged":
            refs.append(("judgment", "run_id", str(p["run_id"])))   # `judged` needs the judgment row (trigger)
        if rt == "run_patch":
            # a foreign key carried in the patch's fields (judge_id, decode_policy_id, …) is a parent too:
            # absent → dependency-blocked, never refused by the FK at apply time
            fields = p.get("fields") or {}
            for col, ptable, pcol in _FKS["run"]:
                val = fields.get(col)
                if val is not None and not (ptable == "run" and str(val) == str(p["run_id"])):
                    refs.append((ptable, pcol, str(val)))
        return refs
    table = v.table
    assert table is not None
    own_key = R.NATURAL_KEYS[table][0]
    sentinels = S.SENTINELS
    for col, ptable, pcol in _FKS[table]:
        val = p.get(col)
        if val is None or (col in sentinels and val == sentinels[col]):
            continue
        if ptable == table and str(val) == str(p.get(own_key)):
            continue  # self-reference to its own key
        refs.append((ptable, pcol, str(val)))
    if table == "object":
        # a parent carried under extra (the fold lifts it) is a dependency too -- blocked until the parent folds, never refused for it.
        # Consulted only when the record states NO column at all (`col not in p`), exactly the fold's test (reducer.lift_object_fields):
        # a stated top-level null resets the lineage and outranks a stale extra key, so a record clearing `parent_object_id` beside an old
        # `extra.parent_object_id` naming a parent that never folds is no dependency -- it would have been blocked indefinitely on a value
        # the fold would not have set
        for col, ptable, pcol in _FKS[table]:
            if col in R.OBJECT_LIFTED_FROM_EXTRA and col not in p and isinstance(v.extra.get(col), str) and v.extra[col] != p.get(own_key):
                refs.append((ptable, pcol, str(v.extra[col])))
    return refs
